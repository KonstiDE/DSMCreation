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

from unet_fanned.model_v2 import UNET_FANNED

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

    if not os.path.exists("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/results"):
        os.mkdir("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/results")

    mae = MeanAbsoluteError().to(device)
    mse = MeanSquaredError().to(device)

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

        prediction = unet(data, data)
        prediction[prediction < 0] = 0

        target = target.unsqueeze(1).to(device)

        prediction = tf.center_crop(prediction, [500, 500])
        target = tf.center_crop(target, [500, 500])

        running_mae.append(mae(prediction, target).item())
        running_mse.append(mse(prediction, target).item())
        running_zncc.append(zncc(prediction, target).item())
        running_ssim.append(custom_ssim(prediction, target).item())
        running_median.append(torch.median(torch.abs(data - target)).item())

        mae.reset()
        mse.reset()

        prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
        target = target.squeeze(0).squeeze(0).detach().cpu()

        data = data.squeeze(0).cpu().numpy()
        red = data[0]
        red_normalized = (red * (1 / red.max()))
        green = data[1]
        green_normalized = (green * (1 / green.max()))
        blue = data[2]
        blue_normalized = (blue * (1 / blue.max()))

        beauty = np.dstack((blue_normalized, green_normalized, red_normalized))

        fig, axs = plt.subplots(1, 3, figsize=(21, 5))

        im = axs[0].imshow(beauty)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])
        # plt.colorbar(im, ax=axs[0])

        im = axs[1].imshow(prediction, cmap="viridis")
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        im.set_clim(0, max(prediction.max(), target.max()))
        plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(target, cmap="viridis")
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        plt.colorbar(im, ax=axs[2])

        fig.suptitle("MAE: {:.3f}, MSE: {:.3f}, SSIM: {:.3f}, ZNCC: {:.3f}, MEDAE: {:.3f}".format(
            running_mae[-1],
            running_mse[-1],
            running_ssim[-1],
            running_zncc[-1],
            running_median[-1]
        ), fontsize=24)

        walking_mae += running_mae[-1]

        plt.savefig(
            "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/results/" + os.path.basename(src_path[0]) + ".png"
        )
        plt.close(fig)

        c += 1

        loop.set_postfix(info="MAE={:.4f}".format(walking_mae / c))

    file = open("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/results/mae1.txt", "w+")
    file.write("MAE: {}, MSE: {}, SSIM: {}, ZNCC: {}, MEDAE: {}".format(
        str(s.mean(running_mae)),
        str(s.mean(running_mse)),
        str(s.mean(running_ssim)),
        str(s.mean(running_zncc)),
        str(s.mean(running_median))
    ))
    file.close()


if __name__ == '__main__':
    test(
        0,
        "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/model_epoch19.pt",
        "/home/fkt48uj/nrw/dataset/data/test/"
    )

