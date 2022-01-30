import os
import rasterio as rio
import sys
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


def build_data_frame(sentinel_option_folder, tile):
    # Blue, Green, Red, Infrared
    data_frame = []

    for file in os.listdir(sentinel_option_folder):
        if '.tif' in file and (
                'B2' in file or
                'B3' in file or
                'B4' in file or
                ('B8' in file and 'B8A' not in file)
        ):

            window = test_position(tile, os.path.join(sentinel_option_folder, file))
            window_resampled = resampleWindow(window)

            data_frame.append(window_resampled)

    dom = rio.open(tile).read(1)
    dom = center_crop(dom, size_out, size_out)
    data_frame.append(dom)

    return data_frame

