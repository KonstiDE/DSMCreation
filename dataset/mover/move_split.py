import os
import random
import sys
sys.path.append(os.getcwd())

from dataset.helper.dataset_helper import (
    split,
    output_path
)

import shutil


def move_into_split():

    list_of_files = os.listdir(output_path)
    list_of_files = [filename for filename in list_of_files if '.npz' in filename]
    files_amount = len(list_of_files)

    for split_type, split_tuple in split.items():
        if not os.path.isdir(split_tuple[1]):
            print(split_tuple[1] + " does not exist")
            return

    for split_type, split_tuple in split.items():
        split_percentage = split_tuple[0]
        split_path = split_tuple[1]

        max_moving = int(files_amount * split_percentage)

        for counter in range(0, max_moving):
            data_frame = random.choice(list_of_files)

            if ".npz" in data_frame:
                shutil.move(os.path.join(output_path, data_frame), os.path.join(split_path, data_frame))
                list_of_files.remove(data_frame)


if __name__ == '__main__':
    move_into_split()
