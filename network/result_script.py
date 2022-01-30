import os
import random

import rasterio as rio
import matplotlib.pyplot as plt
import sklearn.metrics as skl
import torch
import network.pytorchtools as pytorchtools

from network.unet_complete_padding_1000_to_1000.model import (
    UNET
)

from network.dataset_builder import (
    get_dataset
)

from network.helper.network_helper import (
    device
)


def test(amount, model_path, test_data_path):
    unet = UNET(in_channels=4, out_channels=1).cpu()
    unet.load_state_dict(torch.load(model_path)['model_state_dict'])

    loader = get_dataset(test_data_path, 100)
    c = 0

    for (data, target) in loader:
        if c >= amount:
            break

        data = data.unsqueeze(0).cpu()
        target = target.cpu()

        target = target.squeeze(0).squeeze(0).detach()
        prediction = unet(data).squeeze(0).squeeze(0).detach().cpu()

        print(prediction.shape)

        mae = skl.mean_absolute_error(target, prediction)
        mse = skl.mean_squared_error(target, prediction)

        print(mae)
        print(mse)

        plt.imshow(target, cmap="viridis")
        plt.show()

        plt.imshow(prediction, cmap="viridis")
        plt.show()

        c += 1


if __name__ == '__main__':
    test(10, "/home/fkt48uj/nrw/results_MSELoss_Adam/model_regression.pt", "/home/fkt48uj/nrw/dataset/data/test/")
