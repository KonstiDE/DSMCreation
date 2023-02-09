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
        dom = data_frame["arr_" + str(4)]

        print(dom.shape)

        red_normalized = (red * (255 / red.max())).astype(np.uint8)
        green_normalized = (green * (255 / green.max())).astype(np.uint8)
        blue_normalized = (blue * (255 / blue.max())).astype(np.uint8)

        beauty = np.dstack((red_normalized, green_normalized, blue_normalized))

        fig = plt.figure()
        plt.imshow(beauty)
        plt.title("beauty")
        plt.savefig("data.png", dpi=600)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(dom, cmap='viridis')
        plt.title("nDSM")
        plt.savefig("dsm.png", dpi=600)
        plt.close(fig)

    else:
        print("{} is not a valid data frame".format(path))


if __name__ == '__main__':
    path = "/home/fkt48uj/nrw/dataset/data/test/"

    view_data_frame(path + "ndom50_32403_5800_1_nw_2019_4~SENTINEL2X_20190615-000000-000_L3A_T32ULC_C_V1-2.npz")
