import rasterio as rio
from rasterio.enums import Resampling
import scipy.ndimage as resample

import matplotlib.pyplot as plt

from ..helper.dataset_helper import (
    upsampling_multiplier_sentinel,
    upsampling_technique
)


def resampleWindow(window):
    return resample.zoom(window, upsampling_multiplier_sentinel, order=upsampling_technique)
