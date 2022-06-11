import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt


def detection(paths):
    c = 0
    outlier_file = open("outliers_unchecked.txt", "w+")

    for path in paths:
        files = os.listdir(path)

        for file in files:
            data_frame = np.load(os.path.join(path, file), allow_pickle=True)

            blue = data_frame["arr_" + str(0)]
            green = data_frame["arr_" + str(1)]
            red = data_frame["arr_" + str(2)]
            nir = data_frame["arr_" + str(3)]
            dom = data_frame["arr_" + str(4)]

            if not np.all(blue == 0) and not np.all(green == 0) and not np.all(red == 0) and not np.all(nir == 0):
                if not np.any(dom < -10):
                    if not np.max(dom) < 0:
                        if not np.min(dom) > 10:
                            "Do nothing"
                        else:
                            c += 1
                            outlier_file.write(os.path.join(path, file) + "\n")
                            print("[" + str(c) + "] Error with " + file)
                    else:
                        c += 1
                        outlier_file.write(os.path.join(path, file) + "\n")
                        print("[" + str(c) + "] Error with " + file)
                else:
                    c += 1
                    outlier_file.write(os.path.join(path, file) + "\n")
                    print("[" + str(c) + "] Error with " + file)
            else:
                c += 1
                outlier_file.write(os.path.join(path, file) + "\n")
                print("[" + str(c) + "] Error with " + file)

    outlier_file.close()


if __name__ == '__main__':
    detection(
        [
            # "/home/fkt48uj/nrw/dataset/data/train/",
            # "/home/fkt48uj/nrw/dataset/data/validation/",
            "/home/fkt48uj/nrw/dataset/data/test/"
        ]
    )
