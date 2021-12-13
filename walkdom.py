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

domnrwpath = "D:/domnrw/"
sentinel = "B:/sennrw/U/LB/2020/SENTINEL2X_20200515-000000-000_L3A_T32ULB_C_V1-2/" \
           "SENTINEL2X_20200515-000000-000_L3A_T32ULB_C_V1-2_FRC_B2.tif"


def walkDom():
    for root, dirs, files in os.walk(domnrwpath, topdown=False):
        for name in files:
            if "transformed_" not in name:
                result_path, result_tile = transform_coordinate_system(name, domnrwpath, sentinel)
                cut(result_path, result_tile)


if __name__ == '__main__':
    walkDom()
