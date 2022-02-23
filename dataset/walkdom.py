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
            if not file_already_transformed(files, name):
                result_path, result_tile = transform_coordinate_system(name, dom_path, sen_example)
                cut(result_path, result_tile, cutting_length)
                print("Cut up and transformed " + name)
            else:
                print("Already build transformed for " + name)


def file_already_transformed(current_files, name):
    if ("cut_transformed_" + name.removesuffix(".tif") + "_0.tif" in current_files and
            "cut_transformed_" + name.removesuffix(".tif") + "_1.tif" in current_files and
            "cut_transformed_" + name.removesuffix(".tif") + "_2.tif" in current_files and
            "cut_transformed_" + name.removesuffix(".tif") + "_3.tif" in current_files) or \
            "cut_transformed_" in name:
        return True

    return False


if __name__ == '__main__':
    walkDom()
