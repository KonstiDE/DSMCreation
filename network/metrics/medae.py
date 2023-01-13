import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

import shutup
shutup.please()


def custom_medae(data, target):
    # BxCxWxH

    medaes = 0
    for i in range(len(data)):
        medaes += torch.median(torch.abs(data[i].unsqueeze(0) - target[i].unsqueeze(0)))

    return medaes / len(data)


if __name__ == '__main__':
    A = torch.ones([8, 1, 500, 500])
    B = torch.randn([8, 1, 500, 500])

    print(torch.median(torch.abs(A - B)).item())

    s1 = torch.median(torch.abs(A[0].unsqueeze(0) - B[0].unsqueeze(0))).item()
    s2 = torch.median(torch.abs(A[1].unsqueeze(0) - B[1].unsqueeze(0))).item()
    s3 = torch.median(torch.abs(A[2].unsqueeze(0) - B[2].unsqueeze(0))).item()
    s4 = torch.median(torch.abs(A[3].unsqueeze(0) - B[3].unsqueeze(0))).item()
    s5 = torch.median(torch.abs(A[4].unsqueeze(0) - B[4].unsqueeze(0))).item()
    s6 = torch.median(torch.abs(A[5].unsqueeze(0) - B[5].unsqueeze(0))).item()
    s7 = torch.median(torch.abs(A[6].unsqueeze(0) - B[6].unsqueeze(0))).item()
    s8 = torch.median(torch.abs(A[7].unsqueeze(0) - B[7].unsqueeze(0))).item()

    print((s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / 8)
