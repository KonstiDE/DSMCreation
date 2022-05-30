import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt


def show(path):
    new_file = open("outliers.txt", "a+")
    file = open("../network/helper/outliers_detection.txt", "r")

    index = 0
    lines = new_file.readlines()
    for line in file.readlines():
        index += 1

        if index > 100:
            split = line.split(" ")
            loss_value = split[1]

            if float(loss_value) > 50:
                print(index)
                data_frame = np.load(os.path.join(path, split[0]), allow_pickle=True)

                dom = data_frame["arr_" + str(4)]

                if not lines.__contains__(split[0]):
                    if not np.any(dom < -50):
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
    show("/home/fkt48uj/nrw/dataset/output/")
