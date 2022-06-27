import datetime
import math
import time

import statistics
import numpy as np
import torch


def zncc(img1, img2, eps=0.00001):
    stamp1 = time.time() * 1000

    avg1 = img1.mean()
    avg2 = img2.mean()

    print("Avg took: {}ms".format(time.time() * 1000 - stamp1))

    stamp1 = time.time() * 1000
    first = 1 / (img1.std() * img2.std() + eps)
    print("Std took: {}ms".format(time.time() * 1000 - stamp1))

    stamp1 = time.time() * 1000
    s = 0
    #for p1, p2 in zip(img1, img2):
        #s += first * (p1 - avg1) * (p2 - avg2)

    torch.sum((img1 - avg1) * (img2[t] - avg2))

    print("loop took: {}ms".format(time.time() * 1000 - stamp1))

    return s.item() / img1.nelement()


if __name__ == "__main__":
    a = np.random.randn(10)
    b = np.random.randn(10)

    avg1 = np.mean(a)
    avg2 = np.mean(b)

    std1 = np.std(a)
    std2 = np.std(b)

    s = 0
    for p1, p2 in zip(a, b):
        s += (p1 - avg1) * (p2 - avg2)
    print(s)

    for p1, p2 in zip(a, b):
        s += np.sqrt(std1) * np.sqrt(std2)
    print(s)


    #A = torch.randn([512, 512]).cuda().flatten()
    #B1 = torch.randn([512, 512]).cuda().flatten()

    #stamp = time.time() * 1000

    #znccs = []
    #for i in range(0, 1):
        #znccs.append(zncc(A, B1))

    #print("{}:: That took {}ms".format(statistics.mean(znccs), time.time() * 1000 - stamp))

    # print(zncc(A, B2))
