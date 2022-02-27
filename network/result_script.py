import os.path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from network.dataset_provider import (
    get_dataset
)
from network.unet_bachelor.model import UNET_BACHELOR

import torchvision.transforms.functional as tf


def test(amount, model_path, test_data_path):
    unet = UNET_BACHELOR(in_channels=4, out_channels=1).cpu()
    unet.load_state_dict(torch.load(model_path)['model_state_dict'])

    unet.eval()
    torch.no_grad()

    loader = get_dataset(test_data_path)
    c = 0

    if not os.path.exists("results"):
        os.mkdir("results")

    for (data, target, src_path) in loader:
        if 0 < amount <= c:
            break

        data = data.unsqueeze(0).cpu()
        prediction = unet(data)

        target = target.unsqueeze(0).unsqueeze(0)

        target = tf.resize(target, size=prediction.shape[2:])

        prediction = prediction.squeeze(0).squeeze(0).detach().cpu()
        target = target.squeeze(0).squeeze(0).detach().cpu()

        # mae = skl.mean_absolute_error(target, prediction)
        # mse = skl.mean_squared_error(target, prediction)

        # print(mae)
        # print(mse)

        fig = plt.figure()
        plt.imshow(target, cmap="viridis")
        plt.colorbar()
        plt.savefig("results/" + os.path.basename(src_path) + "_target.png")
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(prediction, cmap="viridis")
        plt.colorbar()
        plt.savefig("results/" + os.path.basename(src_path) + "_pred.png")
        plt.close(fig)

        c += 1


if __name__ == '__main__':
    test(
        30,
        "/home/fkt48uj/nrw/results_L1Loss_Adam_UNET_BACHELOR/model.pt",
        "/home/fkt48uj/nrw/dataset/data/test/"
    )
