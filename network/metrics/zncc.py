import datetime
import math
import time

import numpy as np


def zncc(img1, img2, eps=0.00001):
    avg1 = np.mean(img1)
    avg2 = np.mean(img2)
    first = 1 / (np.std(img1) * np.std(img2) + eps)

    s = 0
    for p1, p2 in zip(img1, img2):
        s += first * (p1 - avg1) * (p2 - avg2)

    return s / len(img1)


if __name__ == "__main__":
    A = np.random.randn(512, 512).flatten()
    B1 = np.random.randn(512, 512).flatten()

    stamp = time.time() * 1000
    znccv = zncc(A, B1)
    print("{}:: That took {}ms".format(znccv, time.time() * 1000 - stamp))

    # print(zncc(A, B2))
