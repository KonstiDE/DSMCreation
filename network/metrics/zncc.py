import datetime
import math
import time

import statistics
import numpy as np
import torch


def zncc(img1, img2, eps=0.00001):
    avg1 = torch.mean(img1)
    avg2 = torch.mean(img2)

    first = 1 / (torch.std(img1) * torch.std(img2) + eps)

    s = 0
    for p1, p2 in zip(img1, img2):
        s += first * (p1 - avg1) * (p2 - avg2)

    return s.item() / img1.nelement()


if __name__ == "__main__":
    stamp = time.time() * 1000

    A = torch.randn([512, 512]).flatten()
    B1 = torch.randn([512, 512]).flatten()

    znccs = []
    for i in range(0, 1):
        znccs.append(zncc(A, B1))

    print("{}:: That took {}ms".format(statistics.mean(znccs), time.time() * 1000 - stamp))

    # print(zncc(A, B2))
