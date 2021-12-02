import os
import rasterio as rio
import csv
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from test_position import (
    test_position
)


def findMatchingForTile(tile, pathToSen, pathtoDom, domMetaFile):
    with open(domMetaFile, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for i, line in enumerate(reader):
            filename, date_of_measurement = line[0], line[2]
            tile = pathtoDom + "transformed_" + filename + ".tif"

            if os.path.isfile(tile) and filename == "ndom50_32342_5729_1_nw_2018":
                month_of_measurement = date_of_measurement.split("-")[1]

                subfolderpath_to_go = ""

                path_for_position_test_lb = os.path.join(pathToSen, "LB", "positiontest")
                path_for_position_test_lc = os.path.join(pathToSen, "LC", "positiontest")
                path_for_position_test_mb = os.path.join(pathToSen, "MB", "positiontest")
                path_for_position_test_mc = os.path.join(pathToSen, "MC", "positiontest")

                positions_to_test = [
                    path_for_position_test_lb,
                    path_for_position_test_lc,
                    path_for_position_test_mb,
                    path_for_position_test_mc
                ]

                for path_to_test in positions_to_test:
                    senfile_to_test = os.listdir(path_to_test)[0]
                    window = test_position(tile, os.path.join(path_to_test, senfile_to_test))

                    if len(window) != 0:
                        plt.imshow(window)
                        plt.show()
                        subfolderpath_to_go = path_to_test
                        break

                print(subfolderpath_to_go)

                exit(69)


if __name__ == '__main__':
    findMatchingForTile("", "B:/sennrw/U/", "D:/domnrw/", "meta/ndom_nw.csv")
