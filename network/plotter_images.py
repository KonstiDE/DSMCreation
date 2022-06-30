import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.metrics as skl
import warnings
from tqdm.auto import tqdm as prog

import torch
import torch.nn as nn

from provider.dataset_provider import get_dataset

from unet_fanned.model import UNET_FANNED

warnings.filterwarnings("ignore")

DATA_PATH = "/home/fkt48uj/nrw/dataset/data/test/"
MODEL_PATH_V1 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v1/"
MODEL_PATH_V2 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v2/"
MODEL_PATH_V3 = "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_v3/"
BATCH_SIZE = 1
DEVICE = "cuda:0"
px = 1/plt.rcParams['figure.dpi']


def crop_center(array, crop):
    y, x = array.shape
    startx = x//2 - (crop//2)
    starty = y//2 - (crop//2)
    return array[starty:starty + crop, startx:startx + crop]


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def perform_tests(loaders, models, classifications, sample_ids=None):

    if sample_ids is None:
        sample_ids = [1, 2]

    for height in range(37, 38):
      fig, axs = plt.subplots(len(sample_ids), 2 + len(models), figsize=(29, height))

      h = 0

      for sample_id in sample_ids:
          first_done = False

          for i in range(len(loaders)):
              data, target = loaders[i].__getitem__(sample_id)

              data = data.cpu()
              target = target.cpu()

              if not first_done:
                  first_done = True
                  target = target.squeeze(0).squeeze(0).detach()
                  numpy_data = data.numpy()

                  datadata = np.dstack((
                      normalize(crop_center(numpy_data[2], 388)),
                      normalize(crop_center(numpy_data[1], 388)),
                      normalize(crop_center(numpy_data[0], 388)),
                  ))
                  im = axs[h, 0].imshow(datadata)
                  axs[h, 0].set_xticklabels([])
                  axs[h, 0].set_yticklabels([])

                  im = axs[h, 1].imshow(target, cmap="viridis")
                  axs[h, 1].set_xticklabels([])
                  axs[h, 1].set_yticklabels([])
                  cbar = plt.colorbar(im, ax=axs[h, 1])
                  for t in cbar.ax.get_yticklabels():
                      t.set_fontsize(26)

              data = data.unsqueeze(0)

              if classifications[i]:
                  prediction = torch.argmax(models[i](data), dim=1, keepdim=True).squeeze(0).squeeze(0).detach().cpu()
              else:
                  prediction = models[i](data).squeeze(0).squeeze(0).detach().cpu()

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
      plt.savefig("drive/MyDrive/visual_results_{}.png".format(height))


def setup():
    test_loader = get_dataset(DATA_PATH)

    unet_v1 = UNET_FANNED(in_channels=4, out_channels=1)
    unet_v1.load_state_dict(torch.load(MODEL_PATH_V1)['model_state_dict'])

    unet_v2 = UNET_FANNED(in_channels=4, out_channels=1)
    unet_v2.load_state_dict(torch.load(MODEL_PATH_V1)['model_state_dict'])

    unet_v3 = UNET_FANNED(in_channels=4, out_channels=1)
    unet_v3.load_state_dict(torch.load(MODEL_PATH_V1)['model_state_dict'])
    unet_v3.to(DEVICE)

    perform_tests(test_loader, [unet_v1, unet_v2, unet_v3], [True, True, False], [463, 501, 380, 585, 467, 379, 356])


if __name__ == '__main__':
    setup()
