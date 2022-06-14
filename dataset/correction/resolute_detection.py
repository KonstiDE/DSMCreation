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

            blue = data_frame["arr_" + str(0)]
            green = data_frame["arr_" + str(1)]
            red = data_frame["arr_" + str(2)]
            nir = data_frame["arr_" + str(3)]
            dom = data_frame["arr_" + str(4)]

            isOk = True

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
