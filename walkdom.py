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


domnrwpath = "D:/domnrw/"


def walkDom():
    for root, dirs, files in os.walk(domnrwpath, topdown=False):
        for name in files:
            if "transformed_" not in name:
                transform_coordinate_system(name, domnrwpath, sentinel)


if __name__ == '__main__':
    walkDom()
