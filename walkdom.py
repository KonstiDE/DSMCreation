import os
import numpy as np
import rasterio as rio
import pygeodesy
import rasterio.crs
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import matplotlib.pyplot as plt

from transformer import (
    transform_coordinate_system
)

from cutter import (
    cut
)

from helper import (
    dom_path,
    sen_example,
    cutting_length
)


def walkDom():
    for root, dirs, files in os.walk(dom_path, topdown=False):
        for name in files:
            if "transformed_" not in name:
                result_path, result_tile = transform_coordinate_system(name, dom_path, sen_example)
                cut(result_path, result_tile, cutting_length)


if __name__ == '__main__':
    walkDom()
