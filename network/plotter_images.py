import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.metrics as skl
import warnings
from tqdm.auto import tqdm as prog

import torch
import torch.nn as nn

from network.unet_fanned.model_v1 import UNET_FANNED
from provider.dataset_provider import get_dataset


import importlib.util
import sys
spec = importlib.util.spec_from_file_location("module.name", "/home/fkt48uj/nrw/")
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)



warnings.filterwarnings("ignore")

DATA_PATH = "/home/fkt48uj/nrw/dataset/data/test/"
MODEL_PATH_V1 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/model_epoch18.pt"
MODEL_PATH_V2 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/model_epoch19.pt"
MODEL_PATH_V3 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/model_epoch18.pt"
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
                data, target, dataframe_path = loader.__getitem__(sample_id)

                data = data.cpu()
                target = target.cpu()

                data[data < 0] = 0
                target[target < 0] = 0

                if not first_done:
                    first_done = True

                    data = data.squeeze(0).cpu().numpy()
                    red = data[0]
                    red_normalized = (red * (255 / red.max())).astype(np.uint8)
                    green = data[1]
                    green_normalized = (green * (255 / green.max())).astype(np.uint8)
                    blue = data[2]
                    blue_normalized = (blue * (255 / blue.max())).astype(np.uint8)

                    beauty = np.dstack((red_normalized, green_normalized, blue_normalized))

                    target = crop_center(target, 512)
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

                prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
                target = target.squeeze(0).squeeze(0).detach().cpu()

                mae = skl.mean_absolute_error(target, prediction)
                mse = skl.mean_squared_error(target, prediction)

                im = axs[h, 2 + i].imshow(prediction, cmap="viridis")
                axs[h, 2 + i].set_xticklabels([])
                axs[h, 2 + i].set_yticklabels([])
                cbar = plt.colorbar(im, ax=axs[h, 2 + i])
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(26)
                axs[h, 2 + i].set_xlabel("MAE: {:.2f}\nMSE: {:.2f}".format(mae, mse), fontsize=30)

            h += 1

        plt.tight_layout()
        plt.savefig("/home/fkt48uj/nrw/results/visual_results_{}.png".format(height))


def setup():
    test_loader = get_dataset(DATA_PATH)

    unet_v1 = UNET_FANNED(in_channels=4, out_channels=1)
    unet_v1.load_state_dict(torch.load(MODEL_PATH_V1, map_location='cpu')['model_state_dict'])

    unet_v2 = UNET_FANNED(in_channels=4, out_channels=1)
    unet_v2.load_state_dict(torch.load(MODEL_PATH_V2, map_location='cpu')['model_state_dict'])

    unet_v3 = UNET_FANNED(in_channels=4, out_channels=1)
    unet_v3.load_state_dict(torch.load(MODEL_PATH_V3, map_location='cpu')['model_state_dict'])

    perform_tests(test_loader, [unet_v1, unet_v2, unet_v3], [False, True, True], [1, 2, 3, 4, 5, 6, 7, 8])


if __name__ == '__main__':
    setup()
