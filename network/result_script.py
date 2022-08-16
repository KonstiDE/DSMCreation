import os.path

import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm as prog

from provider.dataset_provider import (
    get_dataset
)
import provider.pytorchtools as pytorchtools

from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError
)

from torchmetrics.image import StructuralSimilarityIndexMeasure

from metrics.zncc import (
    zncc
)

from unet_fanned.model import UNET_FANNED

import torchvision.transforms.functional as tf

import sys
sys.path.append(os.getcwd())

import shutup
shutup.please()


def test(amount, model_path, test_data_path):
    unet = UNET_FANNED(in_channels=4, out_channels=1)
    unet.load_state_dict(torch.load(model_path)['model_state_dict'])
    unet.to("cuda:0")

    unet.eval()
    torch.no_grad()

    loader = get_dataset(test_data_path, amount)
    c = 0

    if not os.path.exists("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/results"):
        os.mkdir("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/results")

    mae = MeanAbsoluteError().to("cuda:0")
    mse = MeanSquaredError().to("cuda:0")
    ssim = StructuralSimilarityIndexMeasure(kernel_size=(5, 5)).to("cuda:0")

    walking_mae = 0

    running_mae = []
    running_mse = []
    running_ssim = []
    running_zncc = []

    loop = prog(loader)

    for (data, target, src_path) in loop:
        data = data.unsqueeze(0).to("cuda:0")

        data[data < 0] = 0
        target[target < 0] = 0

        prediction = unet(data)

        prediction[prediction < 0] = 0

        running_mae.append(mae(prediction, target).item())
        running_mse.append(mse(prediction, target).item())
        running_zncc.append(zncc(prediction, target).item())
        running_ssim.append(ssim(prediction, target).item())

        mae.reset()
        mse.reset()
        ssim.reset()

        prediction = prediction.squeeze(0).squeeze(0)
        target = target.squeeze(0).squeeze(0)

        prediction = prediction.detach().cpu()
        target = target.detach().cpu()

        data = data.squeeze(0).cpu()
        red = data[0]
        red *= 1 / red.max()
        green = data[1]
        green *= 1 / green.max()
        blue = data[2]
        blue *= 1 / blue.max()

        beauty = np.dstack((blue, green, red))

        fig, axs = plt.subplots(1, 3)

        im = axs[0].imshow(beauty)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])
        # plt.colorbar(im, ax=axs[0])

        im = axs[1].imshow(prediction, cmap="viridis")
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(target, cmap="viridis")
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        plt.colorbar(im, ax=axs[2])

        fig.suptitle("MAE: {}, MSE: {},\nSSIM: {}, ZNCC: {}".format(
            str(running_mae[-1]),
            str(running_mse[-1]),
            str(running_ssim[-1]),
            str(running_zncc[-1])
        ))

        walking_mae += running_mae[-1]

        plt.savefig("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/results/" + os.path.basename(src_path) + ".png")
        plt.close(fig)

        c += 1

        loop.set_postfix(info="MAE={:.2f}".format(walking_mae / c))

    file = open("/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/results/mae.txt", "w+")
    file.write("MAE: {}, MSE: {}, SSIM: {}, ZNCC: {}".format(
        str(s.mean(running_mae)),
        str(s.mean(running_mse)),
        str(s.mean(running_ssim)),
        str(s.mean(running_zncc))
    ))
    file.close()


if __name__ == '__main__':
    test(
        0,
        "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/model_epoch1.pt",
        "/home/fkt48uj/nrw/dataset/data/test/"
    )
