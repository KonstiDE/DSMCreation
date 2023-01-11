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
    unet.load_state_dict(torch.load(model_path)['model_state_dict'])
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

    running_mae = []
    running_mse = []
    running_ssim = []
    running_zncc = []
    running_median = []

    loop = prog(loader)

    for (data, target, src_path) in loop:
        data = data.to(device)

        data[data < 0] = 0
        target[target < 0] = 0

        prediction = unet(data)
        prediction[prediction < 0] = 0

        target = target.unsqueeze(1).to(device)

        prediction = tf.center_crop(prediction, [500, 500])
        target = tf.center_crop(target, [500, 500])

        target_building_density = building_desity(target)

        running_mae.append(mae(prediction, target).item())
        running_mse.append(mse(prediction, target).item())
        running_zncc.append(zncc(prediction, target).item())
        running_ssim.append(custom_ssim(prediction, target).item())
        running_median.append(torch.median(torch.abs(prediction - target)).item())

        mae.reset()
        mse.reset()
        ssim.reset()

        prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
        target = target.squeeze(0).squeeze(0).detach().cpu()



        walking_mae += running_mae[-1]

        plt.savefig(
            "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/results/" + os.path.basename(src_path[0]) + ".png"
        )
        plt.close(fig)

        c += 1

        loop.set_postfix(info="MAE={:.4f}".format(walking_mae / c))

    file = open("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/results/mae1.txt", "w+")
    file.write("MAE: {}, MSE: {}, SSIM: {}, ZNCC: {}, MEDAE: {}".format(
        str(s.mean(running_mae)),
        str(s.mean(running_mse)),
        str(s.mean(running_ssim)),
        str(s.mean(running_zncc)),
        str(s.mean(running_median))
    ))
    file.close()


def building_desity(dsm):
    return torch.numel()


def building_height(dsm):
    return


if __name__ == '__main__':
    test(
        0,
        "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/model_epoch18.pt",
        "/home/fkt48uj/nrw/dataset/data/test/"
    )

