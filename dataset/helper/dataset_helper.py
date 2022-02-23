import os

sen_path = "/home/nrw_data/sennrw/"  # Path to the sentinel data. The subfolder of sennrw then contains LB,LC,MB,LC
dom_path = "/home/nrw_data/domnrw/"  # Path to the ndsm data. The subfolder of domnrw then contains all .tif files of the ndsm
meta_file = open("/home/fkt48uj/nrw/dataset/meta/ndom_nw.csv")  # Path to the meta file. It contains the timestamps of the ndsm data frames
sen_example = "/home/fkt48uj/nrw/sennrw/LB/2020/SENTINEL2X_20200515-000000-000_L3A_T32ULB_C_V1-2/" \
              "SENTINEL2X_20200515-000000-000_L3A_T32ULB_C_V1-2_FRC_B2.tif"  # Path to a completely random sentinel tile
output_path = "/home/fkt48uj/nrw/dataset/output/"  # Path of the output directory of the .npz files produces by the matching

positiontest_LB = os.path.join(sen_path, "LB", "positiontest")
positiontest_LC = os.path.join(sen_path, "LC", "positiontest")
positiontest_MB = os.path.join(sen_path, "MB", "positiontest")
positiontest_MC = os.path.join(sen_path, "MC", "positiontest")

split = {
    'train': (0.7, "/home/fkt48uj/nrw/dataset/data/train/"),
    'validation': (0.2, "/home/fkt48uj/nrw/dataset/data/validation/"),
    'test': (0.1, "/home/fkt48uj/nrw/dataset/data/test/")
}  # Split of training, validation and test and their directories

size_in = 1000
size_out = 1000

cutting_length = 1000  # The ndsm have a size of 2000x2000 pixels, too big for most neural networks. The cutter
# script generates smaller tiles with the size of cutting_length x cutting_length.

upsampling_multiplier_sentinel = 20  # Upsample factor from sentinel to ndsm, to not modify this if using our data
upsampling_technique = 3  # Upsampling technique code. 3 = cubic



