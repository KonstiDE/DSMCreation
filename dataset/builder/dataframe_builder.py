import os
import rasterio as rio
import sys
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.getcwd())

from dataset.modifier.resample import (
    resampleWindow
)

from dataset.modifier.test_position import (
    test_position
)

from dataset.helper.dataset_helper import (
    size_out
)

from dataset.modifier.cropper import (
    center_crop
)

from dataset.modifier.extender import (
    mirrow_extrapolate
)


def build_data_frame(sentinel_option_folder, tile):
    # Blue, Infrared, Green, Red
    data_frame = {}

    for file in os.listdir(sentinel_option_folder):
        if file.__contains__("B2.tif"):
            append_dict_with_code(data_frame, tile, sentinel_option_folder, file, code="blue")

        elif file.__contains__("B3.tif"):
            append_dict_with_code(data_frame, tile, sentinel_option_folder, file, code="green")

        elif file.__contains__("B4.tif"):
            append_dict_with_code(data_frame, tile, sentinel_option_folder, file, code="red")

        elif file.__contains__("B8.tif"):
            append_dict_with_code(data_frame, tile, sentinel_option_folder, file, code="nir")

    dom = rio.open(tile).read(1)
    data_frame["dom"] = mirrow_extrapolate(Image.fromarray(dom), thickness=6)

    return data_frame


def append_dict_with_code(frame, tile, sentinel_option_folder, file, code):
    window = test_position(tile, os.path.join(sentinel_option_folder, file))
    window_resampled = resampleWindow(window)
    frame[code] = mirrow_extrapolate(Image.fromarray(window_resampled), thickness=6)
