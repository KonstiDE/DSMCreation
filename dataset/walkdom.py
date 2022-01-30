import os

from modifier.transformer import (
    transform_coordinate_system
)

from modifier.cutter import (
    cut
)

from helper.dataset_helper import (
    dom_path,
    sen_example,
    cutting_length
)


def walkDom():
    for root, dirs, files in os.walk(dom_path, topdown=True):
        for name in files:
            if not file_already_transformed(files, name, cutting_length):
                result_path, result_tile = transform_coordinate_system(name, dom_path, sen_example)
                cut(result_path, result_tile, cutting_length)
                print("Cut up and transformed " + name)
            else:
                print("Already build transformed for " + name)


def file_already_transformed(current_files, name, side_length):

    m = int(2000 / side_length)

    c = 0
    for i in range(int(m * m)):
        if "cut_transformed_" + removesuffix(name, ".tif") + "_" + str(i) + ".tif" in current_files:
            c += 1

    if c == m * m:
        return True

    if "cut_transformed_" in name:
        return True

    return False


def removesuffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


if __name__ == '__main__':
    walkDom()
