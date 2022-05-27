import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show


def view_data_frame(path):
    if path.endswith(".npz"):
        data_frame = np.load(path, allow_pickle=True)

        blue = data_frame["arr_" + str(0)]
        green = data_frame["arr_" + str(1)]
        red = data_frame["arr_" + str(2)]
        nir = data_frame["arr_" + str(3)]
        dom = data_frame["arr_" + str(4)]

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
    path = "/home/fkt48uj/nrw/dataset/output/"

    view_data_frame(path + "ndom50_32307_5672_1_nw_2018_3~SENTINEL2X_20180515-000000-000_L3A_T32ULB_C_V1-2.npz")
