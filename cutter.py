from PIL import Image
from itertools import product

import os
import rasterio as rio
import numpy as np
from osgeo import gdal

import pygeodesy


def tile():
    in_path = 'B:/nrw/nrw_raw/'
    input_filename = 'ndom50_32280_5652_1_nw_2019.tif'

    out_path = in_path
    output_filename = 'ndom50_32280_5652_1_nw_2019_'

    tile_size_x = 1000
    tile_size_y = 1000

    profile = rio.open(os.path.join(in_path, input_filename)).profile
    latitude = profile['transform'][2]
    longitude = profile['transform'][5]

    print(profile)

    c = pygeodesy.Utm('32', 'N', latitude, longitude, band='U').toMgrs()
    print(c)

    #print(m)
    exit(3)

    ds = gdal.Open(os.path.join(in_path, input_filename))
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    for i in range(0, xsize, tile_size_x):
        for j in range(0, ysize, tile_size_y):
            com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(in_path) + str(input_filename) + " " + str(out_path) + str(output_filename) + str(i) + "_" + str(j) + ".tif"
            os.system(com_string)


if __name__ == '__main__':
    tile()
