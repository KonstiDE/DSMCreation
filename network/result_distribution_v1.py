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


def test(amount, test_data_path):

    loader = get_loader(test_data_path, 1, num_workers, pin_memory=False, amount=amount, shuffle=False)

    loop = prog(loader)

    density_dist = {}
    height_dist = {}

    for (data, target, src_path) in loop:
        target[target < 0] = 0
        target[target > 100] = 100

        target = tf.center_crop(target, [500, 500]).squeeze(0).numpy()

        build_density = round((target >= 1).sum()  / 500**2 * 100)
        build_height = round(target.max())

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
    density_dist = {13: 682, 2: 1288, 59: 427, 15: 662, 55: 499, 4: 1140, 78: 246, 65: 356, 23: 552, 16: 612, 37: 543, 36: 519,
     3: 1208, 71: 267, 81: 240, 50: 563, 27: 540, 0: 2775, 22: 574, 57: 483, 24: 506, 9: 831, 98: 225, 49: 550, 8: 893,
     64: 392, 6: 1017, 93: 238, 1: 1634, 21: 565, 25: 570, 91: 241, 17: 575, 96: 241, 35: 543, 14: 655, 97: 229,
     66: 337, 28: 510, 38: 540, 5: 939, 74: 246, 26: 518, 100: 130, 39: 493, 12: 690, 84: 241, 18: 591, 33: 525,
     94: 239, 87: 218, 69: 312, 43: 513, 70: 285, 73: 250, 10: 823, 7: 879, 40: 512, 41: 597, 76: 251, 99: 212, 46: 542,
     52: 500, 11: 815, 90: 224, 31: 491, 30: 535, 56: 510, 29: 485, 47: 525, 32: 557, 75: 261, 88: 228, 20: 557,
     61: 419, 58: 472, 44: 519, 54: 523, 51: 575, 77: 283, 42: 524, 89: 271, 83: 241, 19: 586, 45: 502, 53: 542,
     63: 396, 62: 400, 48: 491, 79: 241, 34: 504, 80: 230, 68: 328, 92: 246, 82: 215, 85: 229, 95: 244, 60: 461,
     72: 284, 86: 232, 67: 331}

    lists = sorted(density_dist.items())
    x, y = zip(*lists)
    plt.figure()
    plt.plot(x, y)
    plt.show()

    height_dist = {24: 1811, 48: 88, 32: 2147, 18: 1212, 23: 1677, 2: 383, 28: 1988, 29: 1929, 21: 1522, 34: 1945, 71: 17, 10: 345,
     26: 1895, 17: 1081, 38: 1436, 39: 1224, 27: 1971, 33: 2012, 22: 1603, 44: 271, 25: 1901, 14: 672, 64: 18, 35: 1983,
     41: 749, 30: 2020, 1: 927, 47: 101, 42: 578, 4: 170, 45: 190, 58: 27, 16: 947, 40: 1024, 36: 1826, 31: 2043,
     78: 16, 20: 1404, 46: 140, 15: 794, 19: 1279, 37: 1678, 6: 174, 11: 376, 67: 38, 0: 322, 51: 68, 43: 403, 3: 178,
     13: 566, 49: 65, 12: 430, 53: 50, 8: 215, 5: 160, 100: 182, 9: 302, 59: 31, 83: 3, 56: 35, 52: 52, 61: 27, 7: 201,
     63: 25, 68: 22, 77: 8, 65: 27, 74: 12, 80: 19, 62: 28, 91: 3, 60: 25, 50: 72, 54: 44, 69: 13, 57: 34, 85: 9, 79: 6,
     70: 22, 89: 7, 66: 23, 86: 4, 55: 35, 99: 5, 72: 8, 81: 7, 92: 3, 75: 17, 76: 12, 95: 4, 82: 3, 97: 3, 90: 2,
     87: 5, 73: 12, 88: 4, 84: 2, 96: 1, 98: 2, 93: 1}

    lists2 = sorted(height_dist.items())
    x, y = zip(*lists2)
    plt.figure()
    plt.plot(x, y)
    plt.show()

    exit(7)

    test(
        0,
        "/home/fkt48uj/nrw/dataset/data/test/"
    )

