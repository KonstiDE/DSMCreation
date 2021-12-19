import os

import numpy as np
import rasterio as rio
import csv

import warnings

warnings.filterwarnings("ignore")

from dataset.modifier.test_position import (
    test_position
)

from dataset.modifier.resample import resampleWindow

from dataset.helper.helper import (
    dom_path,
    sen_path,
    meta_file
)

from dataset.builder.dataframe_builder import (
    build_data_frame
)


def createMatching():
    data_set_counter = 0

    with meta_file as f:
        reader = csv.reader(f, delimiter=";")
        for i, line in enumerate(reader):
            c = 0
            if i > 0:
                if c <= 3:
                    filename, date_of_measurement = line[0], line[2]
                    tile = dom_path + "cut_transformed_" + filename + "_" + str(c) + ".tif"
                    c += 1

                    if os.path.isfile(tile):
                        date = date_of_measurement.split("-")  # yyyy-mm-dd

                        month_of_measurement_ndom = date[1]
                        year_of_measurement_ndom = date[0]

                        subfolderpath_to_go = ""

                        path_for_position_test_lb = os.path.join(sen_path, "U", "LB", "positiontest")
                        path_for_position_test_lc = os.path.join(sen_path, "U", "LC", "positiontest")
                        path_for_position_test_mb = os.path.join(sen_path, "U", "MB", "positiontest")
                        path_for_position_test_mc = os.path.join(sen_path, "U", "MC", "positiontest")

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

                                            data_frame = build_data_frame(sentinel_option, tile)

                                            np.savez_compressed("dataframe.npz", *data_frame)
                                            with open("dataframe.npz") as f2:
                                                f2.__setattr__("sen", directory)
                                                f2.__setattr__("dom", filename)

                                            exit(32)

                                            data_set_counter += 1
                                            found_sentinel_match = True
                                            break

                                if found_sentinel_match:
                                    break

                    #else:
                        #print("Tile " + filename + " not downloaded or not in " + dom_path + " or not transformed!")

    print("Created " + str(data_set_counter) + " data_sets")


if __name__ == '__main__':
    createMatching()
