import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import albumentations as A


train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5)
    ]
)


class NrwDataSet(Dataset):
    def __init__(self, npz_dir, augmented=False):
        training_outlier_file = open("/home/fkt48uj/nrw/network/training_outlier_gte19.txt")
        training_outliers = [os.path.basename(line.rstrip()) for line in training_outlier_file]

        validation_outlier_file = open("/home/fkt48uj/nrw/network/validation_outlier_gte19.txt")
        validation_outliers = [os.path.basename(line.rstrip()) for line in validation_outlier_file]

        self.dataset = []
        for file in os.listdir(npz_dir):
            if not training_outliers.__contains__(file) and not validation_outliers.__contains__(file):
                self.dataset.append(os.path.join(npz_dir, file))
                self.augmented = augmented

        training_outlier_file.close()
        validation_outlier_file.close()

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


def get_loader(npz_dir, batch_size, num_workers=2, pin_memory=True, augmented=False, shuffle=True):
    train_ds = NrwDataSet(npz_dir, augmented)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return train_loader


def get_dataset(npz_dir):
    return NrwDataSet(npz_dir)
