import os.path

import numpy as np
import matplotlib.pyplot as plt


def check():
    if not os.path.isfile("/home/fkt48uj/nrw/outliers_checked_stayed.txt"):
        with open("/home/fkt48uj/nrw/outliers_checked_stayed.txt", 'w'):
            pass

    if not os.path.isfile("/home/fkt48uj/nrw/outliers_checked_sorted_out.txt"):
        with open("/home/fkt48uj/nrw/outliers_checked_sorted_out.txt", 'w'):
            pass

    new_file_stay_read = open("/home/fkt48uj/nrw/outliers_checked_stayed.txt", "r+")
    new_lines_stayed = new_file_stay_read.readlines()
    new_file_stay_read.close()

    new_file_sorted_out_read = open("/home/fkt48uj/nrw/outliers_checked_sorted_out.txt", "r+")
    new_lines_sorted_out = new_file_sorted_out_read.readlines()
    new_file_sorted_out_read.close()

    new_file_stay = open("/home/fkt48uj/nrw/outliers_checked_stayed.txt", "a+")
    new_file_out = open("/home/fkt48uj/nrw/outliers_checked_sorted_out.txt", "a+")
    file = open("/home/fkt48uj/nrw/outliers_unchecked.txt", "r")

    index = 0
    for line in file.readlines():

        if new_lines_stayed.__contains__(line) or new_lines_sorted_out.__contains__(line):
            continue

        data_frame = np.load(line.replace("\n", ""), allow_pickle=True)

        blue = data_frame["arr_" + str(0)]
        green = data_frame["arr_" + str(1)]
        red = data_frame["arr_" + str(2)]
        nir = data_frame["arr_" + str(3)]
        dom = data_frame["arr_" + str(4)]

        index += 1

        if not np.all(blue == 0) and not np.all(green == 0) and not np.all(red == 0) and not np.all(nir == 0) and\
                not(abs(np.min(dom) - np.max(dom) > 400)) and not np.all(dom < 0) and not np.any(dom < -1000):

            print(index)
            print(line.replace("\n", ""))

            fig, axs = plt.subplots(1, 5)

            im = axs[0].imshow(blue, cmap="Blues_r")
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
            # plt.colorbar(im, ax=axs[0])

            im = axs[1].imshow(green, cmap="Greens_r")
            axs[1].set_xticklabels([])
            axs[1].set_yticklabels([])
            # plt.colorbar(im, ax=axs[0])

            im = axs[2].imshow(red, cmap="Reds_r")
            axs[2].set_xticklabels([])
            axs[2].set_yticklabels([])
            # plt.colorbar(im, ax=axs[0])

            im = axs[3].imshow(nir, cmap="Purples_r")
            axs[3].set_xticklabels([])
            axs[3].set_yticklabels([])
            # plt.colorbar(im, ax=axs[0])

            im = axs[4].imshow(dom, cmap="viridis")
            axs[4].set_xticklabels([])
            axs[4].set_yticklabels([])
            plt.colorbar(im, ax=axs[4])

            fig.show()

            character = input()

            if character == "":
                print("stay")
                new_file_stay.write(line)
            else:
                new_file_out.write(line)
                print("got out")

        else:
            new_file_stay.write(line)

    new_file_stay.close()
    new_file_out.close()


if __name__ == '__main__':
    check()
