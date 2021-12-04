import rasterio as rio
from rasterio.enums import Resampling
import scipy.ndimage as resample

import matplotlib.pyplot as plt


def resampleWindow(window):
    print(window.shape)
    exit(32)
    #plt.imshow(window)
    #plt.show()

    window_resampled = resample.zoom(window, 20, order=3)

    print(window_resampled.shape)
    #plt.imshow(window_resampled)
    #plt.show()
