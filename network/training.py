import math
import os

import pytorch_msssim
import pytorch_ssim
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as skl
from tqdm.auto import tqdm as prog
import matplotlib.pyplot as plt
from numpy import savetxt
import statistics as s
import numpy as np
import sys
import torchvision.transforms.functional as tf

sys.path.append(os.getcwd())

import shutup

shutup.please()

from dataset_provider import (
    get_loader
)

from dataset.helper.dataset_helper import (
    split
)

from network.helper.network_helper import (
    batch_size,
    num_workers,
    device,
    pin_memory
)

from pytorchtools import (
    EarlyStopping
)

from torch.autograd import Variable

from unet_fanned.model import UNET_FANNED

from pytorch_msssim.ssim import SSIM


def train(epoch, loader, loss_fn, optimizer, scaler, model):
    torch.enable_grad()
    model.train()

    loop = prog(loader)

    running_loss = []
    running_mae = []
    running_mse = []

    for batch_index, (data, target, dataframepath) in enumerate(loop):
        optimizer.zero_grad(set_to_none=True)

        data = data.to(device)
        data = model(data, data)

        target = target.unsqueeze(1).to(device)
        target = tf.resize(target, size=data.shape[2:])

        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()

        #outlier_file.write(dataframepath[0] + " " + str(loss_value) + "\n")

        data = data.view(-1).detach().cpu()
        target = target.view(-1).detach().cpu()

        running_mae.append(loss_value)
        running_mse.append(skl.mean_squared_error(target, data))

        loop.set_postfix(info="Epoch {}, train, loss={:.5f}".format(epoch, loss_value))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_mae), s.mean(running_mse)


def valid(epoch, loader, loss_fn, model):
    model.eval()

    loop = prog(loader)

    running_loss = []
    running_mae = []
    running_mse = []

    for batch_index, (data, target, dataframepath) in enumerate(loop):
        data = data.to(device)
        data = model(data, data)

        target = target.unsqueeze(1).to(device)
        target = tf.resize(target, size=data.shape[2:])

        with torch.no_grad():
            loss = loss_fn(data, target)

        loss_value = loss.item()

        #outlier_file.write(dataframepath[0] + " " + str(loss_value) + "\n")

        data = data.view(-1).detach().cpu()
        target = target.view(-1).detach().cpu()

        running_mae.append(loss_value)
        running_mse.append(skl.mean_squared_error(target, data))

        loop.set_postfix(info="Epoch {}, valid, loss={:.5f}".format(epoch, loss_value))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_mae), s.mean(running_mse)


def run(num_epochs, lr, epoch_to_start_from):
    model = UNET_FANNED(in_channels=4, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)
    path_train = split['train'][1]
    path_validation = split['validation'][1]

    epochs_done = 0

    overall_training_loss = []
    overall_validation_loss = []

    overall_training_mae = []
    overall_validation_mae = []
    overall_training_mse = []
    overall_validation_mse = []

    path = "{}_{}_{}_{}_nearn_500_512/".format(
        "results",
        str(loss_fn.__class__.__name__),
        str(optimizer.__class__.__name__),
        str(UNET_FANNED.__qualname__)
    )

    #training_outlier_file = open(os.path.join(path, "helper/training_outlier_detection.txt"), mode="a+")
    #validation_outlier_file = open(os.path.join(path, "helper/validation_outlier_detection.txt"), mode="a+")

    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.isfile(path + "model_epoch" + str(epoch_to_start_from) + ".pt"):
        checkpoint = torch.load(path + "model_epoch" + str(epoch_to_start_from) + ".pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_done = checkpoint['epoch']
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_mae = checkpoint['training_maes']
        overall_training_mse = checkpoint['training_mses']
        overall_validation_mae = checkpoint['validation_maes']
        overall_validation_mse = checkpoint['validation_mses']
        early_stopping = checkpoint['early_stopping']
    else:
        model.to(device)

    train_loader = get_loader(path_train, batch_size, num_workers, pin_memory, amount=0)
    validation_loader = get_loader(path_validation, batch_size, num_workers, pin_memory, amount=0)

    for epoch in range(epochs_done + 1, num_epochs + 1):
        training_loss, training_mae, training_mse = train(epoch, train_loader, loss_fn, optimizer, scaler, model)
        validation_loss, validation_mae, validation_mse = valid(epoch, validation_loader, loss_fn, model)

        overall_training_loss.append(training_loss)
        overall_validation_loss.append(validation_loss)

        overall_training_mae.append(training_mae)
        overall_training_mse.append(training_mse)

        overall_validation_mae.append(validation_mae)
        overall_validation_mse.append(validation_mse)

        early_stopping(validation_loss, model)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': overall_training_loss,
            'validation_losses': overall_validation_loss,
            'training_maes': overall_training_mae,
            'training_mses': overall_training_mse,
            'validation_maes': overall_validation_mae,
            'validation_mses': overall_validation_mse,
            'early_stopping': early_stopping
        }, path + "model_epoch" + str(epoch) + ".pt")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        model.to(device)

        metrics = np.array([
            overall_training_loss,
            overall_validation_loss,
            overall_training_mae,
            overall_training_mse,
            overall_validation_mae,
            overall_validation_mse
        ])

        savetxt(path + "metrics.csv", metrics, delimiter=',', header="tloss,vloss,tmae,tmse,tnan,vmae,vmse", fmt='%s')

    plt.figure()
    plt.plot(overall_training_loss, 'b', label="Loss training")
    plt.plot(overall_validation_loss, 'r', label="Loss validation")
    plt.savefig(path + "losses.png")
    plt.show()

    metrics = np.array([
        overall_training_loss,
        overall_validation_loss,
        overall_training_mae,
        overall_training_mse,
        overall_validation_mae,
        overall_validation_mse
    ])

    savetxt(path + "metrics.csv", metrics, delimiter=',', header="tloss,vloss,tmae,tmse,vmae,vmse", fmt='%s')


if __name__ == '__main__':
    run(num_epochs=100, lr=3e-04, epoch_to_start_from=0)
