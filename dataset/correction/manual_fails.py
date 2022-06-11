import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt


def show(path):
    new_file = open("test_outliers.txt", "a+")
    file = open("test_outlier_detection.txt", "r")

    index = 0
    lines = new_file.readlines()
    for line in file.readlines():
        index += 1

        if index > -1:
            split = line.split(" ")
            loss_value = split[1]

            if float(loss_value) > 5:
                print(index)
                data_frame = np.load(os.path.join(path, split[0]), allow_pickle=True)

                blue = data_frame["arr_" + str(0)]
                green = data_frame["arr_" + str(1)]
                red = data_frame["arr_" + str(2)]
                nir = data_frame["arr_" + str(3)]
                dom = data_frame["arr_" + str(4)]

                if not lines.__contains__(split[0]):
                    if not np.any(dom < -10):
                        if np.any(dom < -5) or np.min(dom) > 10:
                            print(np.min(dom))

                            plt.imshow(dom, cmap='viridis')
                            plt.title("nDSM")
                            plt.colorbar()
                            plt.show()

                            character = input()

                            if character == "":
                                print("yes")
                            else:
                                print("no")
                                new_file.write(split[0] + "\n")
                    else:
                        new_file.write(split[0] + "\n")

    new_file.close()


if __name__ == '__main__':
    show("/home/fkt48uj/nrw/dataset/data/test/")
