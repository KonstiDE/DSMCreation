import os
import torch


def main():
    device = "cuda:0"
    x = torch.randn(10, 10).to(device)

    torch.cuda.set_device(device)
    print(torch.cuda.memory_reserved())

    while True:
        ""


if __name__ == '__main__':
    main()
