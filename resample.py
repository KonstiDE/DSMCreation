import rasterio as rio
from rasterio.enums import Resampling
import scipy.ndimage as resample

import matplotlib.pyplot as plt


def resampleWindow(window):
    window_resampled = resample.zoom(window, 20, order=3)

    plt.imshow(window_resampled)
    plt.show()

    return window_resampled
