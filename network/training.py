import os
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as skl
import skimage.metrics as ski
from tqdm.auto import tqdm as prog
import matplotlib.pyplot as plt
from numpy import savetxt
import statistics as s
import numpy as np
import sys
import torchvision.transforms.functional as tf

from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.append(os.getcwd())

import shutup
shutup.please()

from network.provider.dataset_provider import (
    get_loader
)

from dataset.helper.dataset_helper import (
    split,
    cutting_length
)

from network.helper.network_helper import (
    batch_size,
    num_workers,
    device,
    pin_memory
)

from network.provider.pytorchtools import (
    EarlyStopping
)

from network.metrics.zncc import (
    zncc
)

from unet_fanned.model import UNET_FANNED


def train(epoch, loader, loss_fn, optimizer, scaler, model, mse, ssim):
    torch.enable_grad()
    model.train()

    loop = prog(loader)

    running_loss = []
    running_mae = []
    running_mse = []
    running_ssim = []
    running_zncc = []

    for batch_index, (data, target, dataframepath) in enumerate(loop):
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)

        data[data < 0] = 0
        target[target < 0] = 0

        data = model(data)
        data[data < 0] = 0

        target = target.unsqueeze(1).to(device)

        data = tf.center_crop(data, [cutting_length, cutting_length])
        target = tf.center_crop(target, [cutting_length, cutting_length])

        with torch.cuda.amp.autocast():
            loss = loss_fn(data, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loss_value = loss.item()

        running_mae.append(loss_value)
        running_mse.append(mse(data, target).item())
        running_zncc.append(zncc(data, target).item())
        running_ssim.append(ssim(data, target).item())

        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Epoch {}, train, loss={:.5f}".format(epoch, loss_value))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_mae), \
           s.mean(running_mse), s.mean(running_ssim), \
           s.mean(running_zncc)


def valid(epoch, loader, loss_fn, model, mse, ssim):
    model.eval()

    loop = prog(loader)

    running_loss = []
    running_mae = []
    running_mse = []
    running_ssim = []
    running_zncc = []

    for batch_index, (data, target, dataframepath) in enumerate(loop):
        data[data < 0] = 0
        target[target < 0] = 0

        data = data.to(device)
        data = model(data)

        data[data < 0] = 0

        target = target.unsqueeze(1).to(device)

        data = tf.center_crop(data, [cutting_length, cutting_length])
        target = tf.center_crop(target, [cutting_length, cutting_length])

        with torch.no_grad():
            loss = loss_fn(data, target)

        loss_value = loss.item()

        running_mae.append(loss_value)
        running_mse.append(mse(data, target).item())
        running_zncc.append(zncc(data, target).item())
        running_ssim.append(ssim(data, target).item())

        mse.reset()
        ssim.reset()

        loop.set_postfix(info="Epoch {}, valid, loss={:.5f}".format(epoch, loss_value))
        running_loss.append(loss_value)

    return s.mean(running_loss), s.mean(running_mae), \
           s.mean(running_mse), s.mean(running_ssim), \
           s.mean(running_zncc)


def run(num_epochs, lr, epoch_to_start_from):
    torch.cuda.empty_cache()

    model = UNET_FANNED(in_channels=4, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)
    path_train = split['train'][1]
    path_validation = split['validation'][1]

    torch_mse = MeanSquaredError().to(device)
    torch_ssim = StructuralSimilarityIndexMeasure(kernel_size=(5, 5)).to(device)

    epochs_done = 0

    overall_training_loss = []
    overall_validation_loss = []

    overall_training_mae = []
    overall_validation_mae = []
    overall_training_mse = []
    overall_validation_mse = []
    overall_training_ssim = []
    overall_validation_ssim = []
    overall_training_zncc = []
    overall_validation_zncc = []

    path = "{}_{}_{}_{}_v1/".format(
        "results",
        str(loss_fn.__class__.__name__),
        str(optimizer.__class__.__name__),
        str(UNET_FANNED.__qualname__)
    )

    if not os.path.isdir(path):
        os.mkdir(path)

    if os.path.isfile(path + "model_epoch" + str(epoch_to_start_from) + ".pt") and epoch_to_start_from > 0:
        checkpoint = torch.load(path + "model_epoch" + str(epoch_to_start_from) + ".pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_done = checkpoint['epoch']
        overall_training_loss = checkpoint['training_losses']
        overall_validation_loss = checkpoint['validation_losses']
        overall_training_mae = checkpoint['training_maes']
        overall_training_mse = checkpoint['training_mses']
        overall_training_ssim = checkpoint['training_ssims']
        overall_training_zncc = checkpoint['training_znccs']
        overall_validation_mae = checkpoint['validation_maes']
        overall_validation_mse = checkpoint['validation_mses']
        overall_validation_ssim = checkpoint['validation_ssims']
        overall_validation_zncc = checkpoint['validation_znccs']
        early_stopping = checkpoint['early_stopping']
    else:
        if epoch_to_start_from == 0:
            model.to(device)
        else:
            raise Exception("No model_epoch" + str(epoch_to_start_from) + ".pt found")

    model.to(device)

    train_loader = get_loader(path_train, batch_size, num_workers, pin_memory, amount=0)
    validation_loader = get_loader(path_validation, batch_size, num_workers, pin_memory, amount=0)

    for epoch in range(epochs_done + 1, num_epochs + 1):
        training_loss, training_mae, training_mse, training_ssim, training_zncc = train(epoch, train_loader, loss_fn,
                                                                                        optimizer, scaler, model,
                                                                                        torch_mse, torch_ssim)
        torch.cuda.empty_cache()

        validation_loss, validation_mae, validation_mse, validation_ssim, validation_zncc = valid(epoch,
                                                                                                  validation_loader,
                                                                                                  loss_fn, model,
                                                                                                  torch_mse, torch_ssim)
        torch.cuda.empty_cache()

        overall_training_loss.append(training_loss)
        overall_validation_loss.append(validation_loss)

        overall_training_mae.append(training_mae)
        overall_validation_mae.append(validation_mae)

        overall_training_mse.append(training_mse)
        overall_validation_mse.append(validation_mse)

        overall_training_ssim.append(training_ssim)
        overall_validation_ssim.append(validation_ssim)

        overall_training_zncc.append(training_zncc)
        overall_validation_zncc.append(validation_zncc)

        early_stopping(validation_loss, model)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': overall_training_loss,
            'validation_losses': overall_validation_loss,
            'training_maes': overall_training_mae,
            'training_mses': overall_training_mse,
            'training_ssims': overall_training_ssim,
            'training_znccs': overall_training_zncc,
            'validation_maes': overall_validation_mae,
            'validation_mses': overall_validation_mse,
            'validation_ssims': overall_validation_ssim,
            'validation_znccs': overall_validation_zncc,
            'early_stopping': early_stopping
        }, path + "model_epoch" + str(epoch) + ".pt")

        model.to(device)

        metrics = np.array([
            overall_training_loss,
            overall_validation_loss,
            overall_training_mae,
            overall_training_mse,
            overall_training_ssim,
            overall_training_zncc,
            overall_validation_mae,
            overall_validation_mse,
            overall_validation_ssim,
            overall_validation_zncc,
        ], dtype='object')

        savetxt(path + "metrics.csv", metrics, delimiter=',',
                header="tloss,vloss,tmae,tmse,tssim,tzncc,vmae,vmse,vssim,vzncc", fmt='%s')

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    run(num_epochs=100, lr=5e-06, epoch_to_start_from=0)
