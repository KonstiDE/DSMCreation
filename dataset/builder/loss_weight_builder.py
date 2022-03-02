import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show


def provide_weight_array(path):
    for file in os.listdir(path):
        data_frame = np.load(os.path.join(path, file), allow_pickle=True)

        blue = data_frame["arr_" + str(0)]
        green = data_frame["arr_" + str(1)]
        red = data_frame["arr_" + str(2)]
        nir = data_frame["arr_" + str(3)]
        dom = data_frame["arr_" + str(4)]




if __name__ == '__main__':
    provide_weight_array("/home/fkt48uj/nrw/dataset/data/train")
