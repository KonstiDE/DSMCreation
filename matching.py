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

from resample import resampleWindow


def findMatchingForTile(tile, pathToSen, pathtoDom, domMetaFile):
    data_set_counter = 0

    with open(domMetaFile, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for i, line in enumerate(reader):
            if i > 0:
                filename, date_of_measurement = line[0], line[2]
                tile = pathtoDom + "transformed_" + filename + ".tif"

                if os.path.isfile(tile):
                    date = date_of_measurement.split("-")  # yyyy-mm-dd

                    month_of_measurement_ndom = date[1]
                    year_of_measurement_ndom = date[0]

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
                    # Test if a position of sentinel is available via positiontest dirs
                    position_test_passed = False

                    for path_to_test in positions_to_test:
                        senfile_to_test = os.listdir(path_to_test)[0]
                        window = test_position(tile, os.path.join(path_to_test, senfile_to_test))

                        if len(window) != 0 and window != []:
                            subfolderpath_to_go = path_to_test.replace("\positiontest", "/")
                            position_test_passed = True
                            break

                    # Time matching
                    if position_test_passed:
                        found_sentinel_match = False

                        for root, dirs, files in os.walk(subfolderpath_to_go, topdown=False):
                            for directory in dirs:
                                if 'SENTINEL' in directory:

                                    datestring_sentinel = directory[11:19]
                                    year_of_measurement_sentinel = datestring_sentinel[0:4]
                                    month_of_measurement_sentinel = datestring_sentinel[4:6]

                                    if year_of_measurement_ndom == year_of_measurement_sentinel and \
                                            month_of_measurement_ndom == month_of_measurement_sentinel:
                                        sentinel_option = os.path.join(root, directory)

                                        # Blue, Green, Red, Infrared
                                        data_frame = []

                                        for file in os.listdir(sentinel_option):
                                            if '.tif' in file and (
                                                    'B2' in file or
                                                    'B3' in file or
                                                    'B4' in file or
                                                    ('B8' in file and 'B8A' not in file)
                                            ):
                                                window = test_position(tile, os.path.join(sentinel_option, file))

                                                data_frame.append(window)

                                                resampleWindow(window)

                                        # Saved dataframe here

                                        data_set_counter += 1
                                        found_sentinel_match = True
                                        break

                            if found_sentinel_match:
                                break

                # else:
                # print("Tile " + filename + " not downloaded or not in " + pathtoDom + " or not transformed!")

    print("Created " + str(data_set_counter) + " data_sets")


if __name__ == '__main__':
    findMatchingForTile("", "B:/sennrw/U/", "D:/domnrw/", "meta/ndom_nw.csv")
