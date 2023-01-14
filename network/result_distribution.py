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

import ast

from jenkspy import JenksNaturalBreaks

def test(amount, test_data_path):
    loader = get_loader(test_data_path, 1, num_workers, pin_memory=False, amount=amount, shuffle=False)

    loop = prog(loader)

    density_dist = []
    height_dist = []

    for (data, target, src_path) in loop:
        target[target < 0] = 0

        target = tf.center_crop(target, [500, 500]).squeeze(0).numpy()

        build_density = (target >= 1).sum() / 500**2 * 100
        build_height = np.quantile(target, 0.95)

        density_dist.append(build_density)
        height_dist.append(build_height)

    print(density_dist)
    print(height_dist)

    file = open("/home/fkt48uj/nrw/density_pts.txt", "w+")
    file.write(str(density_dist))
    file.close()

    file = open("/home/fkt48uj/nrw/height_pts.txt", "w+")
    file.write(str(height_dist))
    file.close()

if __name__ == '__main__':
    #test(0, "/home/fkt48uj/nrw/dataset/data/test/")
    #exit(8)

    density_pts = open("/home/fkt48uj/nrw/density_pts.txt", "r").readline()
    density_dist = ast.literal_eval(density_pts)
    plt.figure()
    plt.plot(sorted(density_dist))
    plt.show()

    height_pts = open("/home/fkt48uj/nrw/height_pts.txt", "r").readline()
    height_dist_95 = ast.literal_eval(height_pts)
    plt.figure()
    plt.plot(sorted(height_dist_95))
    plt.show()

    jnb = JenksNaturalBreaks(3)
    jnb.fit(density_dist)
    print(jnb.breaks_)

    jnb = JenksNaturalBreaks(3)
    jnb.fit(height_dist_95)
    print(jnb.breaks_)

