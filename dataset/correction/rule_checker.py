import numpy as np
import matplotlib.pyplot as plt


def check():
    new_file = open("outliers_checked.txt", "w+")
    file = open("/home/fkt48uj/nrw/outliers_unchecked.txt", "r")

    index = 0
    for line in file.readlines():
        index += 1

        data_frame = np.load(line.replace("\n", ""), allow_pickle=True)

        blue = data_frame["arr_" + str(0)]
        green = data_frame["arr_" + str(1)]
        red = data_frame["arr_" + str(2)]
        nir = data_frame["arr_" + str(3)]
        dom = data_frame["arr_" + str(4)]

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
            new_file.write(line)
        else:
            print("got out")

    new_file.close()


if __name__ == '__main__':
    check()
