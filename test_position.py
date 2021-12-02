import os
import numpy as np
import rasterio as rio
import pygeodesy
import rasterio.crs
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import matplotlib.pyplot as plt


def test_position(tile_path, sentinel_path):
    sentinel = rio.open(sentinel_path)
    tile = rio.open(tile_path)

    sentinel_transform = sentinel.transform

    window = sentinel.read(1, window=from_bounds(
       tile.bounds.left,
       tile.bounds.bottom,
       tile.bounds.right,
       tile.bounds.top,
       sentinel_transform
    ))

    return window
