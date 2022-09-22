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
    A = torch.randn([8, 1, 512, 512]).cuda()
    B = torch.randn([8, 1, 512, 512]).cuda()

    stamp = time.time() * 1000

    value = zncc(A, B)

    print("{}:: That took {}ms".format(value, time.time() * 1000 - stamp))
