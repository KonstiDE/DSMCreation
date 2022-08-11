import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt


def detection(paths):
    c = 0
    outlier_file = open("outliers_unchecked.txt", "w+")

    for path in paths:
        files = os.listdir(path)
        t = len(files)
        n = 0

        for file in files:
            c += 1

            data_frame = np.load(os.path.join(path, file), allow_pickle=True)

            red = data_frame["red"]
            green = data_frame["green"]
            blue = data_frame["blue"]
            nir = data_frame["nir"]
            dom = data_frame["dom"]

            isOk = True

            if blue.shape[0] != blue.shape[1] or green.shape[0] != green.shape[1] or\
                    red.shape[0] != red.shape[1] or nir.shape[0] != nir.shape[1] or\
                    dom.shape[0] != dom.shape[1]:
                isOk = False
            if np.sum(dom < -12) / (len(dom) * len(dom)) > 0.2:
                isOk = False
            elif np.any(dom < -50):
                isOk = False
            elif np.max(dom) < 0:
                isOk = False
            elif np.min(dom) > 10:
                isOk = False
            elif abs(np.min(dom) - np.max(dom)) > 400:
                isOk = False
            elif np.all(blue == 0) or np.all(green == 0) or np.all(red == 0) or np.all(nir == 0):
                isOk = False

            if not isOk:
                n += 1
                outlier_file.write(os.path.join(path, file) + "\n")
                print("[" + str(c) + "/" + str(t) + "]::(" + str(n) + ") Error with " + file)

    outlier_file.close()


if __name__ == '__main__':
    detection(
        [
            "/home/fkt48uj/nrw/dataset/data/train/",
            "/home/fkt48uj/nrw/dataset/data/validation/",
            "/home/fkt48uj/nrw/dataset/data/test/"
        ]
    )

