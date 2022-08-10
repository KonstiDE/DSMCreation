import os

import numpy as np
import rasterio as rio
import csv

import warnings

warnings.filterwarnings("ignore")

from modifier.test_position import (
    test_position
)

from helper.dataset_helper import (
    dom_path,
    sen_path,
    meta_file,
    output_path,
    positiontest_LB,
    positiontest_LC,
    positiontest_MB,
    positiontest_MC,
    cutting_length
)

from builder.dataframe_builder import (
    build_data_frame
)


def createMatching():
    data_set_counter = 0
    time_counter = 0
    current_output_patch = os.listdir(output_path)
    m = 2000 / cutting_length

    with meta_file as f:
        reader = csv.reader(f, delimiter=";")
        for i, line in enumerate(reader):
            if i > 0:
                for c in range(int(m * m)):
                    filename, date_of_measurement = line[0], line[2]

                    if not already_created(filename, c + 1, current_output_patch):

                        tile = dom_path + "cut_transformed_" + filename + "_" + str(c) + ".tif"
                        c += 1

                        if os.path.isfile(tile):
                            date = date_of_measurement.split("-")  # yyyy-mm-dd

                            month_of_measurement_ndom = date[1]
                            year_of_measurement_ndom = date[0]

                            subfolderpath_to_go = ""

                            positions_to_test = [positiontest_LB, positiontest_LC, positiontest_MB, positiontest_MC]
                            # Test if a position of sentinel is available via positiontest dirs
                            position_test_passed = False

                            for path_to_test in positions_to_test:
                                senfile_to_test = os.listdir(path_to_test)[0]
                                window = test_position(tile, os.path.join(path_to_test, senfile_to_test))

                                if len(window) != 0 and window != []:

                                    subfolderpath_to_go = path_to_test.replace("positiontest", "")
                                    if subfolderpath_to_go.endswith("/") or subfolderpath_to_go.startswith("\\"):
                                        subfolderpath_to_go = subfolderpath_to_go[:-1]

                                    position_test_passed = True
                                    break

                            # Time matching
                            if position_test_passed:
                                found_sentinel_match = False

                                for root, dirs, files in os.walk(subfolderpath_to_go):
                                    for directory in dirs:
                                        if 'SENTINEL' in directory:

                                            datestring_sentinel = directory[11:19]
                                            year_of_measurement_sentinel = datestring_sentinel[0:4]
                                            month_of_measurement_sentinel = datestring_sentinel[4:6]

                                            if year_of_measurement_ndom == year_of_measurement_sentinel and \
                                                    month_of_measurement_ndom == month_of_measurement_sentinel:
                                                sentinel_option = os.path.join(root, directory)

                                                data_frame = build_data_frame(sentinel_option, tile)

                                                np.savez_compressed(
                                                    os.path.join(output_path, filename + "_" + str(c) + "~" + directory + ".npz"),
                                                    red=data_frame["red"],
                                                    green=data_frame["green"],
                                                    blue=data_frame["blue"],
                                                    nir=data_frame["nir"],
                                                    dom=data_frame["dom"]
                                                )
                                                print("Passed positional and time matching for " + filename + "_" + str(c) + " with index " + str(c) + ". Building up " + filename + "_" + str(c) + "~" + directory + ".npz now")

                                                data_set_counter += 1
                                                found_sentinel_match = True
                                                break

                                        if found_sentinel_match:
                                            break

                                if not found_sentinel_match:
                                    print(filename + " passed the positional matching but not the time matching.")
                                    time_counter += 1
                            else:
                                print("No positional matching found for " + filename + "_" + str(c))
                        else:
                            print("Tile " + filename + " not downloaded or not in " + dom_path + " or not transformed!")
                    else:
                        print("Dataframe for " + filename + "_" + str(c) + " already exists")

    print("Created " + str(data_set_counter) + " data_sets")
    print("Thereby, matched " + str(time_counter) + " position_wise, but not time wise")


def already_created(filename, c, output_patch):
    for f in output_patch:
        if filename + "_" + str(c) in f and ".npz" in f:
            return True
    return False


if __name__ == '__main__':
    createMatching()
