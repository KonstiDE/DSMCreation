import os.path

import statistics as s
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm as prog
from PIL import Image


from provider.dataset_provider import (
    get_loader
)
from helper.network_helper import (
    num_workers,
    pin_memory
)

from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError
)


import torchvision.transforms.functional as tf

import sys
sys.path.append(os.getcwd())

import shutup
shutup.please()

from jenkspy import JenksNaturalBreaks

def test(amount, test_data_path):
    loader = get_loader(test_data_path, 1, num_workers, pin_memory=False, amount=amount, shuffle=False)

    loop = prog(loader)

    density_dist = {}
    height_dist = {}

    for (data, target, src_path) in loop:
        target[target < 0] = 0

        target = tf.center_crop(target, [500, 500]).squeeze(0).numpy()

        build_density = round((target >= 1).sum()  / 500**2 * 100)
        build_height = round(np.quantile(target, 0.95))

        if density_dist.get(build_density) is None:
            density_dist[build_density] = 1
        else:
            density_dist[build_density] = density_dist.get(build_density) + 1

        if height_dist.get(build_height) is None:
            height_dist[build_height] = 1
        else:
            height_dist[build_height] = height_dist.get(build_height) + 1

    print(density_dist)
    print(height_dist)

if __name__ == '__main__':
    density_dist = {13: 682, 2: 1288, 59: 427, 15: 662, 55: 499, 4: 1140, 78: 246, 65: 356, 23: 552, 16: 612, 37: 543,
                    36: 519, 3: 1208, 71: 267, 81: 240, 50: 563, 27: 540, 0: 2775, 22: 574, 57: 483, 24: 506, 9: 831,
                    98: 225, 49: 550, 8: 893, 64: 392, 6: 1017, 93: 238, 1: 1634, 21: 565, 25: 570, 91: 241, 17: 575,
                    96: 241, 35: 543, 14: 655, 97: 229, 66: 337, 28: 510, 38: 540, 5: 939, 74: 246, 26: 518, 100: 130,
                    39: 493, 12: 690, 84: 241, 18: 591, 33: 525, 94: 239, 87: 218, 69: 312, 43: 513, 70: 285, 73: 250,
                    10: 823, 7: 879, 40: 512, 41: 597, 76: 251, 99: 212, 46: 542, 52: 500, 11: 815, 90: 224, 31: 491,
                    30: 535, 56: 510, 29: 485, 47: 525, 32: 557, 75: 261, 88: 228, 20: 557, 61: 419, 58: 472, 44: 519,
                    54: 523, 51: 575, 77: 283, 42: 524, 89: 271, 83: 241, 19: 586, 45: 502, 53: 542, 63: 396, 62: 400,
                    48: 491, 79: 241, 34: 504, 80: 230, 68: 328, 92: 246, 82: 215, 85: 229, 95: 244, 60: 461, 72: 284,
                    86: 232, 67: 331}

    lists = sorted(density_dist.items())
    x, y = zip(*lists)
    plt.figure()
    plt.plot(x, y)
    #plt.show()

    height_dist_90 = {5: 1043, 0: 9186, 16: 951, 13: 1022, 1: 6067, 20: 1192, 3: 1145, 26: 1374, 19: 1096, 24: 1442,
                      32: 306, 18: 1070, 10: 1243, 17: 1008, 8: 1635, 2: 1417, 27: 1255, 30: 676, 15: 1019, 23: 1414,
                      21: 1261, 4: 944, 35: 89, 11: 1134, 7: 1810, 9: 1503, 28: 1101, 31: 484, 29: 862, 6: 1318, 37: 23,
                      12: 1110, 22: 1423, 14: 1018, 25: 1429, 33: 182, 34: 131, 39: 7, 36: 44, 90: 1, 41: 3, 193: 1,
                      38: 7, 50: 2, 45: 1, 53: 1, 121: 1}

    height_dist_70 = {0: 18530, 7: 886, 1: 8951, 5: 1165, 17: 724, 23: 891, 2: 2328, 11: 542, 28: 272, 4: 1533,
                      22: 1017, 6: 1059, 26: 556, 20: 927, 25: 683, 33: 30, 24: 794, 30: 127, 21: 951, 12: 541, 10: 549,
                      3: 2293, 15: 560, 19: 905, 8: 746, 35: 6, 14: 505, 18: 820, 27: 386, 13: 555, 16: 587, 9: 699,
                      31: 67, 29: 190, 32: 58, 34: 11, 189: 1, 37: 1, 41: 1, 39: 1, 52: 1, 36: 1, 38: 1}

    height_dist_95 = {15: 1245, 0: 5823, 21: 1428, 1: 4390, 5: 928, 28: 1515, 19: 1285, 22: 1423, 33: 439, 14: 1230,
                      24: 1600, 10: 1473, 17: 1287, 13: 1307, 27: 1530, 7: 1258, 30: 1132, 11: 1295, 32: 686, 25: 1686,
                      4: 845, 36: 116, 9: 1651, 26: 1658, 8: 1536, 16: 1294, 29: 1296, 12: 1309, 18: 1240, 20: 1340,
                      3: 919, 31: 881, 23: 1527, 38: 38, 2: 1151, 6: 1078, 34: 299, 35: 198, 40: 8, 37: 67, 100: 1,
                      43: 4, 193: 1, 41: 5, 39: 18, 44: 1, 45: 1, 51: 1, 42: 2, 47: 1, 46: 1, 58: 1, 60: 1, 121: 1,
                      49: 1}

    lists0 = sorted(height_dist_95.items())
    x, y = zip(*lists0)
    plt.figure()
    plt.plot(x, y)
    #plt.show()

    lists2 = sorted(height_dist_90.items())
    x, y = zip(*lists2)
    plt.figure()
    plt.plot(x, y)
    #plt.show()

    lists3 = sorted(height_dist_70.items())
    x, y = zip(*lists3)
    plt.figure()
    plt.plot(x, y)
    #plt.show()

    density_values = []
    height_values = []

    for k, v in lists:
        density_values.append(v)
    for k, v in lists0:
        height_values.append(v)

    jnb = JenksNaturalBreaks(3)
    jnb.fit(density_values)
    print(len(jnb.groups_[2]))
    print(len(jnb.groups_[2]) + len(jnb.groups_[1]))

    jnb = JenksNaturalBreaks(3)
    jnb.fit(height_values)
    print(len(jnb.groups_[2]))
    print(len(jnb.groups_[2]) + len(jnb.groups_[1]))

