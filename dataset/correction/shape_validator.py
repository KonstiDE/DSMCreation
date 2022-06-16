import os
import numpy as np
import sys
sys.path.append(os.getcwd())

from dataset.helper.dataset_helper import (
    output_path,
    cutting_length
)


def valid_shapes(path):
    c = 0
    for file in os.listdir(path):
        data_frame = np.load(os.path.join(path, file))

        blue = data_frame["arr_" + str(0)]
        green = data_frame["arr_" + str(1)]
        red = data_frame["arr_" + str(2)]
        nir = data_frame["arr_" + str(3)]
        dom = data_frame["arr_" + str(4)]

        if blue.shape[0] == blue.shape[1] and green.shape[0] == green.shape[1] and red.shape[0] == red.shape[1] and nir.shape[0] == nir.shape[1] and dom.shape[0] == dom.shape[1]:
            ""
        else:
            print(os.path.join(path, file))
            # os.remove(os.path.join(path, file))
            c += 1
    print(c)


if __name__ == '__main__':
    valid_shapes("/home/fkt48uj/nrw/dataset/data/train")
