import time
import torch
import statistics as s


def zncc(img1, img2, eps=0.00001):
    # [B, C, W, H]
    znccs = 0
    numel = 1 / torch.numel(img1 / len(img1))
    for i in range(len(img1)):
        avg1 = torch.mean(img1[i])
        avg2 = torch.mean(img2[i])

        first = 1 / (torch.std(img1[i]) * torch.std(img2[i]) + eps)

        znccs += torch.mul(torch.sum(torch.mul(first, torch.mul(torch.sub(img1[i], avg1), torch.sub(img2[i], avg2)))), numel)
    return znccs


if __name__ == "__main__":
    A = torch.randn([8, 1, 500, 500])
    B = torch.randn([8, 1, 500, 500])

    print(zncc(A, B))

    s1 = zncc(A[0].unsqueeze(0), B[0].unsqueeze(0))
    s2 = zncc(A[1].unsqueeze(0), B[1].unsqueeze(0))
    s3 = zncc(A[2].unsqueeze(0), B[2].unsqueeze(0))
    s4 = zncc(A[3].unsqueeze(0), B[3].unsqueeze(0))
    s5 = zncc(A[4].unsqueeze(0), B[4].unsqueeze(0))
    s6 = zncc(A[5].unsqueeze(0), B[5].unsqueeze(0))
    s7 = zncc(A[6].unsqueeze(0), B[6].unsqueeze(0))
    s8 = zncc(A[7].unsqueeze(0), B[7].unsqueeze(0))

    print((s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / 8)
