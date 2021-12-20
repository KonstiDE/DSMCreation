import os
import random

from dataset.helper.helper import (
    split,
    path_train,
    path_validation,
    path_test
)

import shutil


def move_into_split():

    list_of_files = os.listdir("../../")
    list_of_files = [filename for filename in list_of_files if '.npz' in filename]
    files_amount = len(list_of_files)

    for split_directory, split_percentage in split.items():
        max_moving = int(files_amount * split_percentage)

        for counter in range(0, max_moving):
            data_frame = random.choice(list_of_files)

            if ".npz" in data_frame:
                shutil.move("../../" + data_frame, "../data/" + split_directory + "/")
                list_of_files.remove(data_frame)


if __name__ == '__main__':
    move_into_split()
