import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show


def view_data_frame(path):
    if path.endswith(".npz"):
        data_frame = np.load(path, allow_pickle=True)

        blue = data_frame["red"]
        green = data_frame["green"]
        red = data_frame["blue"]
        nir = data_frame["nir"]
        dom = data_frame["dom"]

        print(blue.shape)
        print(green.shape)
        print(red.shape)
        print(nir.shape)
        print(dom.shape)

        plt.imshow(red, cmap='Reds_r')
        plt.colorbar()
        plt.title("Red")
        plt.show()

        plt.imshow(green, cmap='Greens_r')
        plt.colorbar()
        plt.title("Green")
        plt.show()

        plt.imshow(blue, cmap='Blues_r')
        plt.colorbar()
        plt.title("Blue")
        plt.show()

        plt.imshow(nir, cmap='Purples_r')
        plt.colorbar()
        plt.title("Near Infrared")
        plt.show()

        plt.imshow(dom, cmap='viridis')
        plt.title("nDSM")
        plt.colorbar()
        plt.show()

    else:
        print("{} is not a valid data frame".format(path))


if __name__ == '__main__':
    path = "/home/fkt48uj/nrw/dataset/data/train/"

    view_data_frame(path + "ndom50_32479_5738_1_nw_2017_1~SENTINEL2B_20170815-000000-000_L3A_T32UMC_C_V1-2.npz")
