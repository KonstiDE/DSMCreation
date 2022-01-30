import rasterio as rio
import pygeodesy
import os
import numpy as np
import rasterio.crs
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling, calculate_default_transform


def transform_coordinate_system(tilename, tilepathonly, senpath):
    tile = rio.open(os.path.join(tilepathonly, tilename))

    sen = rio.open(senpath)

    dst_affine, w, h = calculate_default_transform(
        tile.crs, sen.crs, tile.width, tile.height, *tile.bounds
    )

    profile = tile.profile
    profile['crs'] = sen.crs
    profile['transform'] = dst_affine

    file_new = os.path.join(tilepathonly, 'transformed_' + tilename)

    with rio.open(file_new, 'w', **profile) as dst:
        dst.write(tile.read(1), 1)

    tile.close()

    return tilepathonly, 'transformed_' + tilename
