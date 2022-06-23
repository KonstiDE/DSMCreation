import math
import numpy as np


def average(img):
    shape = img.shape
    sum = 0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            sum += img[i][j]

    return float(sum) / img.size


def standard_deviation(img, avg):
    shape = img.shape
    sum = 0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            sum += (img[i][j] - avg)**2

    return math.sqrt(float(sum) / img.size)


def zncc(img1, img2):
    shape = img1.shape

    if shape != img2.shape:
        raise Exception("Images have the be the same shape")

    avg1 = average(img1)
    avg2 = average(img2)
    deviation1 = standard_deviation(img1, avg1)
    deviation2 = standard_deviation(img2, avg2)

    s = 0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            s += (img1[i][j] - avg1) * (img2[i][j] - avg2)

    return float(s) / (img1.size * deviation1 * deviation2)


if __name__ == "__main__":
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 7]])
    print(zncc(A, B1))
    print(zncc(A, B2))
