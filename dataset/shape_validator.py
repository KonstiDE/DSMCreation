import os
import numpy as np
import sys
sys.path.append(os.getcwd())

from dataset.helper.dataset_helper import (
    output_path
)


def valid_shapes(path):
    c = 0
    for file in os.listdir(path):
        data_frame = np.load(os.path.join(path, file))

        blue = data_frame["arr_" + str(0)]

        if blue.shape[0] != blue.shape[1]:
            print("Stuff would be broken with " + file)
            # os.remove(os.path.join(path, file))
            c += 1
    print(c)


if __name__ == '__main__':
    valid_shapes("/home/fkt48uj/nrw/dataset/output/")
