import os.path

import matplotlib.pyplot as plt
import torch
import sklearn.metrics as skl
from tqdm.auto import tqdm as prog

from provider.dataset_provider import (
    get_dataset
)
import provider.pytorchtools as pytorchtools

from unet_fanned.model import UNET_FANNED

import torchvision.transforms.functional as tf


def test(amount, model_path, test_data_path):
    unet = UNET_FANNED(in_channels=4, out_channels=1).to("cuda:0")
    unet.load_state_dict(torch.load(model_path)['model_state_dict'])

    unet.eval()
    torch.no_grad()

    loader = get_dataset(test_data_path)
    c = 0

    if not os.path.exists("results"):
        os.mkdir("results")

    walking_mae = 0

    loop = prog(loader)

    outlier_file = open("results/test_outliers_in_action.txt", "w+")

    for (data, target, src_path) in loop:
        if 0 < amount <= c:
            break

        data = data.unsqueeze(0).to("cuda:0")
        prediction = unet(data, data)

        target = target.unsqueeze(0).unsqueeze(0)
        target = tf.resize(target, size=prediction.shape[2:])

        target = target.squeeze(0).squeeze(0).detach().cpu()
        prediction = prediction.squeeze(0).squeeze(0).detach().cpu()

        mae = skl.mean_absolute_error(target, prediction)

        data = data.squeeze(0)[0].cpu()

        fig, axs = plt.subplots(1, 3)

        im = axs[0].imshow(data, cmap="Reds_r")
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])
        #plt.colorbar(im, ax=axs[0])

        im = axs[1].imshow(prediction, cmap="viridis")
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        #plt.colorbar(im, ax=axs[1])

        im = axs[2].imshow(target, cmap="viridis")
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        #plt.colorbar(im, ax=axs[2])

        outlier_file.write(src_path + "\t\t" + str(mae) + "\n")

        fig.suptitle("MAE: " + str(mae))

        plt.savefig("results/" + os.path.basename(src_path) + ".png")
        plt.close(fig)

        c += 1
        walking_mae += mae

        loop.set_postfix(info="MAE={:.2f}".format(walking_mae / c))

    file = open("results/mae.txt", "w+")
    file.write(str(walking_mae / c))
    file.close()
    outlier_file.close()


if __name__ == '__main__':
    test(
        0,
        "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_FANNED_nearn_500_1024_potency_attention/model_epoch17.pt",
        "/home/fkt48uj/nrw/dataset/data/test/"
    )
