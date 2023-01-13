import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from network.helper.network_helper import device

import shutup
shutup.please()

from torchmetrics.image import StructuralSimilarityIndexMeasure


ssim_metric = StructuralSimilarityIndexMeasure(kernel_size=(5, 5)).to(device)


def custom_ssim(data, target):
    # BxCxWxH

    ssims = 0
    for i in range(len(data)):
        ssim_metric.data_range = max(data[i].max() - data[i].min(), target[i].max() - target[i].min()).item()
        ssims += ssim_metric(data[i].unsqueeze(0), target[i].unsqueeze(0))
        ssim_metric.reset()

    return ssims / len(data)


if __name__ == '__main__':
    torch_ssim = StructuralSimilarityIndexMeasure(kernel_size=(5, 5))

    A = torch.ones([8, 1, 500, 500])
    B = torch.randn([8, 1, 500, 500])

    print(custom_ssim(A, B))

    s1 = custom_ssim(A[0].unsqueeze(0), B[0].unsqueeze(0))
    s2 = custom_ssim(A[1].unsqueeze(0), B[1].unsqueeze(0))
    s3 = custom_ssim(A[2].unsqueeze(0), B[2].unsqueeze(0))
    s4 = custom_ssim(A[3].unsqueeze(0), B[3].unsqueeze(0))
    s5 = custom_ssim(A[4].unsqueeze(0), B[4].unsqueeze(0))
    s6 = custom_ssim(A[5].unsqueeze(0), B[5].unsqueeze(0))
    s7 = custom_ssim(A[6].unsqueeze(0), B[6].unsqueeze(0))
    s8 = custom_ssim(A[7].unsqueeze(0), B[7].unsqueeze(0))

    print((s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / 8)
