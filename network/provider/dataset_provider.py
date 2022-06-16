import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class NrwDataSet(Dataset):
    def __init__(self, npz_dir, amount):
        outlier_file = open("/home/fkt48uj/nrw/outliers_checked_stayed.txt")
        outliers = [os.path.basename(line.rstrip()) for line in outlier_file]

        c = 0
        self.dataset = []
        for file in os.listdir(npz_dir):
            if not outliers.__contains__(file):

                self.dataset.append(os.path.join(npz_dir, file))

                if amount > 0:
                    c += 1
                    if c >= amount:
                        break

        outlier_file.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dataframepath = self.dataset[index]
        dataframe = np.load(dataframepath, allow_pickle=True)

        blue = dataframe["arr_" + str(0)]
        green = dataframe["arr_" + str(1)]
        red = dataframe["arr_" + str(2)]
        nir = dataframe["arr_" + str(3)]
        dom = dataframe["arr_" + str(4)]

        sentinel = np.stack((red, green, blue, nir))

        sentinel = torch.Tensor(sentinel)
        dsm = torch.Tensor(dom)

        return sentinel, dsm, dataframepath


def get_loader(npz_dir, batch_size, num_workers=4, pin_memory=True, shuffle=True, amount=0):
    train_ds = NrwDataSet(npz_dir, amount)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return train_loader


def get_dataset(npz_dir):
    return NrwDataSet(npz_dir, amount=0)
