import os
import rasterio as rio


def test():
    x = rio.open("B:/nrw/nrw_raw/ndom50_32280_5652_1_nw_2019.tif")

    print(x.profile)


if __name__ == '__main__':
    test()
