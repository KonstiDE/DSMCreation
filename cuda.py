import os
import torch


def main():
    torch.cuda.set_device("cuda:0")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
