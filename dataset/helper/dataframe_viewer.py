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

        plt.imshow(red, cmap='Reds')
        plt.colorbar()
        plt.title("Red")
        plt.show()

        plt.imshow(green, cmap='Greens')
        plt.colorbar()
        plt.title("Green")
        plt.show()

        plt.imshow(blue, cmap='Blues')
        plt.colorbar()
        plt.title("Blue")
        plt.show()

        plt.imshow(nir, cmap='Purples')
        plt.colorbar()
        plt.title("Near Infrared")
        plt.show()

        red = red / np.max(red)
        green = green / np.max(green)
        blue = blue / np.max(blue)

        composite = np.stack((red, green, blue))
        show(composite)

    else:
        print("{} is not a valid data frame".format(path))


if __name__ == '__main__':
    view_data_frame("C:/Users/Caipi/PycharmProjects/NRW/ndom50_32342_5729_1_nw_2018~SENTINEL2X_20180515-000000-000_L3A_T32ULC_C_V1-2.npz")
