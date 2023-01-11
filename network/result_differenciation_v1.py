import os.path

import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm as prog
from PIL import Image


from provider.dataset_provider import (
    get_loader
)
from helper.network_helper import (
    num_workers,
    pin_memory,
    device
)

import provider.pytorchtools as pytorchtools

from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError
)

from torchmetrics.image import StructuralSimilarityIndexMeasure

from metrics.zncc import zncc
from metrics.ssim import custom_ssim

from unet_fanned.model_v1 import UNET_FANNED

import torchvision.transforms.functional as tf

import sys
sys.path.append(os.getcwd())

import shutup
shutup.please()


def test(amount, model_path, test_data_path):
    unet = UNET_FANNED(in_channels=4, out_channels=1)
    unet.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    unet.to(device)

    unet.eval()
    torch.no_grad()

    loader = get_loader(test_data_path, 1, num_workers, pin_memory, amount=amount, shuffle=False)
    c = 0

    if not os.path.exists("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/results"):
        os.mkdir("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/results")

    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure(kernel_size=(5, 5)).to(device)

    walking_mae = 0

    matrix = [
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
        [[], [], [], [], []],
    ]

    loop = prog(loader)

    for (data, target, src_path) in loop:
        data = data.to(device)

        data[data < 0] = 0
        target[target < 0] = 0

        prediction = unet(data)
        prediction[prediction < 0] = 0

        target = target.unsqueeze(1)

        prediction = tf.center_crop(prediction, [500, 500]).cpu()
        target = tf.center_crop(target, [500, 500]).cpu()

        torch_mae = mae(prediction, target).item()
        torch_mse = mse(prediction, target).item()
        torch_zncc = zncc(prediction, target).item()
        torch_ssim = custom_ssim(prediction, target).item()
        torch_medae = torch.median(torch.abs(prediction - target)).item()

        mae.reset()
        mse.reset()
        ssim.reset()

        target = target.squeeze(0).squeeze(0).detach().numpy()

        target_building_density = building_density(target)
        target_build_height = building_height(target)

        class_index = matrix_index((target_building_density, target_build_height))

        matrix[class_index][0].append(torch_mae)
        matrix[class_index][1].append(torch_mse)
        matrix[class_index][2].append(torch_ssim)
        matrix[class_index][3].append(torch_zncc)
        matrix[class_index][4].append(torch_medae)

        c += 1

    file = open("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/results/mae2.txt", "w+")

    for index in range(9):
        density, height = matrix_index_rev(index)

        file.write("Density: {}, Height: {} | MAE: {}, MSE: {}, SSIM: {}, ZNCC: {}, MEDAE: {}\n".format(
            density,
            height,
            str(s.mean(matrix[index][0])),
            str(s.mean(matrix[index][1])),
            str(s.mean(matrix[index][2])),
            str(s.mean(matrix[index][3])),
            str(s.mean(matrix[index][4]))
        ))
    file.close()


def building_density(dsm):
    density = round((dsm >= 1).sum()  / 500**2 * 100)

    if density > 65:
        return 2
    elif density > 10:
        return 1
    else:
        return 0



def building_height(dsm):
    dsm[dsm > 100] = 100
    height = round(dsm.max())

    if height > 45:
        return 2
    elif height > 15:
        return 1
    else:
        return 0


def matrix_index(tupel):
    if tupel == (0, 0):
        return 0
    elif tupel == (0, 1):
        return 1
    elif tupel == (0, 2):
        return 2
    elif tupel == (1, 0):
        return 3
    elif tupel == (1, 1):
        return 4
    elif tupel == (1, 2):
        return 5
    elif tupel == (2, 0):
        return 6
    elif tupel == (2, 1):
        return 7
    elif tupel == (2, 2):
        return 8


def matrix_index_rev(index):
    if index == 0:
        return 0, 0
    if index == 1:
        return 0, 1
    if index == 2:
        return 0, 2
    if index == 3:
        return 1, 0
    if index == 4:
        return 1, 1
    if index == 5:
        return 1, 2
    if index == 6:
        return 2, 0
    if index == 7:
        return 2, 1
    if index == 8:
        return 2, 2

if __name__ == '__main__':
    test(
        0,
        "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/model_epoch18.pt",
        "/home/fkt48uj/nrw/dataset/data/test/"
    )

