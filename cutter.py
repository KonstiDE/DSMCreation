from PIL import Image
from itertools import product

import os
import rasterio as rio
import numpy as np
from osgeo import gdal

import pygeodesy

import shutil
from subprocess import Popen, PIPE, STDOUT


def cut(tilepath, tilename):
    out_path = tilepath
    output_filename = tilename

    tile_size_x = 1000
    tile_size_y = 1000

    ds = gdal.Open(os.path.join(tilepath, tilename))
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    c = 0
    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + " " + str(tilepath) + str(tilename) + " " + str(out_path) + "cut_" + str(output_filename.replace(".tif", "")) + "_" + str(c) + ".tif"
            os.system(com_string)
            c += 1


if __name__ == '__main__':
    tile()
