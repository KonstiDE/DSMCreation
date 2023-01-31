import math

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.metrics as skl
import warnings
from tqdm.auto import tqdm as prog

import torch
import torch.nn as nn

from unet_fanned.model_v1 import UNET_FANNED
from provider.dataset_provider import get_dataset

import importlib.util
import sys
spec_v1 = importlib.util.spec_from_file_location("module.name", "/home/fkt48uj/nrw/network/unet_fanned/model_v1.py")
foo_v1 = importlib.util.module_from_spec(spec_v1)
sys.modules["module.name"] = foo_v1
spec_v1.loader.exec_module(foo_v1)
unet_v1 = foo_v1.UNET_FANNED(in_channels=4, out_channels=1)

spec_v2 = importlib.util.spec_from_file_location("module.name", "/home/fkt48uj/nrw/network/unet_fanned/model_v2.py")
foo_v2 = importlib.util.module_from_spec(spec_v2)
sys.modules["module.name"] = foo_v2
spec_v2.loader.exec_module(foo_v2)
unet_v2 = foo_v2.UNET_FANNED(in_channels=4, out_channels=1)

spec_v4 = importlib.util.spec_from_file_location("module.name", "/home/fkt48uj/nrw/network/unet_fanned/model_v4.py")
foo_v4 = importlib.util.module_from_spec(spec_v4)
sys.modules["module.name"] = foo_v4
spec_v4.loader.exec_module(foo_v4)
unet_v4 = foo_v4.UNET_FANNED(in_channels=4, out_channels=1)


warnings.filterwarnings("ignore")

DATA_PATH = "/home/fkt48uj/nrw/dataset/data/test/"
MODEL_PATH_V1 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/model_epoch18.pt"
MODEL_PATH_V2 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/model_epoch19.pt"
MODEL_PATH_V4 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v4/model_epoch24.pt"
BATCH_SIZE = 1
DEVICE = "cuda:0"
px = 1 / plt.rcParams['figure.dpi']


def crop_center(array, crop):
    y, x = array.shape
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    return array[starty:starty + crop, startx:startx + crop]


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def perform_tests(loader, models, multiencoders, sample_ids=None):
    if sample_ids is None:
        sample_ids = [1, 2]

    for height in range(35, 45):
        fig, axs = plt.subplots(len(sample_ids), 2 + len(models), figsize=(29, height))

        h = 0

        for sample_id in sample_ids:
            first_done = False

            for i in range(len(models)):
                data, target, dataframe_path = loader.__getitem_by_name__(sample_id)

                data = data.cpu()
                target = target.cpu()

                data[data < 0] = 0
                target[target < 0] = 0

                if not first_done:
                    first_done = True

                    data = data.squeeze(0).cpu()
                    red = crop_center(data[0].numpy(), 500)
                    red_normalized = (red * (1 / red.max()))
                    green = crop_center(data[1].numpy(), 500)
                    green_normalized = (green * (1 / green.max()))
                    blue = crop_center(data[2].numpy(), 500)
                    blue_normalized = (blue * (1 / blue.max()))

                    beauty = np.dstack((blue_normalized, green_normalized, red_normalized))

                    target = crop_center(target, 500)

                    im = axs[h, 0].imshow(beauty)
                    axs[h, 0].set_xticklabels([])
                    axs[h, 0].set_yticklabels([])

                    im = axs[h, 1].imshow(target, cmap="viridis")
                    axs[h, 1].set_xticklabels([])
                    axs[h, 1].set_yticklabels([])
                    cbar = plt.colorbar(im, ax=axs[h, 1])
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(26)

                data = data.unsqueeze(0)

                if multiencoders[i]:
                    prediction = models[i](data, data).squeeze(0).squeeze(0).detach().cpu()
                else:
                    prediction = models[i](data).squeeze(0).squeeze(0).detach().cpu()

                prediction[prediction < 0] = 0

                prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
                target = target.squeeze(0).squeeze(0).detach().cpu()

                target = crop_center(target, 500)
                prediction = crop_center(prediction, 500)

                mae = skl.mean_absolute_error(target, prediction)
                mse = skl.mean_squared_error(target, prediction)

                im = axs[h, 2 + i].imshow(prediction, cmap="viridis")
                axs[h, 2 + i].set_xticklabels([])
                axs[h, 2 + i].set_yticklabels([])
                cbar = plt.colorbar(im, ax=axs[h, 2 + i])
                im.set_clim(0, max(prediction.max(), target.max()))
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(26)
                axs[h, 2 + i].set_xlabel("MAE: {:.2f}\nRMSE: {:.2f}".format(mae, math.sqrt(mse)), fontsize=30)

            h += 1

        plt.tight_layout()
        plt.savefig("/home/fkt48uj/nrw/results/visual_results_{}.png".format(height), dpi=400)


def setup():
    test_loader = get_dataset(DATA_PATH, amount=0)

    unet_v1.load_state_dict(torch.load(MODEL_PATH_V1, map_location='cpu')['model_state_dict'])
    unet_v2.load_state_dict(torch.load(MODEL_PATH_V2, map_location='cpu')['model_state_dict'])
    unet_v4.load_state_dict(torch.load(MODEL_PATH_V4, map_location='cpu')['model_state_dict'])

    unet_v1.eval()
    unet_v2.eval()
    unet_v4.eval()

    perform_tests(
        test_loader,
        [unet_v1, unet_v4, unet_v2],
        [False, True, True],
        [
            #urban:
            "ndom50_32350_5684_1_nw_2019_9~SENTINEL2X_20190215-000000-000_L3A_T32ULB_C_V1-2.npz",
            "ndom50_32345_5699_1_nw_2018_10~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz",
            #suburban:
            "ndom50_32340_5690_1_nw_2018_1~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz",
            "ndom50_32336_5697_1_nw_2018_12~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz",
            #idustrial:
            "ndom50_32312_5747_1_nw_2018_6~SENTINEL2X_20180515-000000-000_L3A_T32ULC_C_V1-2.npz",
            #rural/countryside:
            "ndom50_32352_5753_1_nw_2018_14~SENTINEL2X_20180315-000000-000_L3A_T32ULC_C_V1-2.npz",
            #vegetation:
            "ndom50_32351_5650_1_nw_2019_8~SENTINEL2X_20190615-000000-000_L3A_T32ULB_C_V1-2.npz",
        ]
    )


if __name__ == '__main__':
    setup()
