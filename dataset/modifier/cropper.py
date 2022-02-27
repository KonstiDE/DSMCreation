import os
import numpy as np


def center_crop(img,cropx,cropy):
    x, y = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


if __name__ == '__main__':
    a = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(a.shape)
    print(center_crop(a, 3, 3))
