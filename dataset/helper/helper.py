sen_path = "B:/sennrw/"
dom_path = "D:/domnrw/"
meta_file = open("C:/Users/Caipi/PycharmProjects/NRW/dataset/meta/ndom_nw.csv")
sen_example = "B:/sennrw/U/LB/2020/SENTINEL2X_20200515-000000-000_L3A_T32ULB_C_V1-2/" \
              "SENTINEL2X_20200515-000000-000_L3A_T32ULB_C_V1-2_FRC_B2.tif"


output_train = "C:/Users/Caipi/PycharmProjects/NRW/dataset/data/train"
output_validation = "C:/Users/Caipi/PycharmProjects/NRW/dataset/data/validation"
output_test = "C:/Users/Caipi/PycharmProjects/NRW/dataset/data/test"
split = {'train': 0.7,
         'validation': 0.2,
         'test': 0.1
        }

cutting_length = 1000

upsampling_multiplier_sentinel = 20
upsampling_technique = 3
