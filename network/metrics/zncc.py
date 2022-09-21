import time
import torch


def zncc(img1, img2, eps=0.00001):
    avg1 = torch.mean(img1)
    avg2 = torch.mean(img2)

    first = 1 / (torch.std(img1) * torch.std(img2) + eps)

    img1 = torch.sub(img1, avg1)
    img2 = torch.sub(img2, avg2)

    return torch.div(torch.sum(torch.mul(first, torch.mul(torch.sub(img1, avg1), torch.sub(img2, avg2)))), torch.numel(img1))


if __name__ == "__main__":
    A = torch.randn([8, 1, 512, 512]).cuda()
    B = torch.randn([8, 1, 512, 512]).cuda()

    stamp = time.time() * 1000

    value = zncc(A, B)

    print("{}:: That took {}ms".format(value, time.time() * 1000 - stamp))
