from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5)
    ]
)


class DsmDataset(Dataset):
    def __init__(self, npz_dir, augmented=False):
        self.data = np.load(npz_dir, allow_pickle=True)
        self.augmented = augmented

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        blue = data_frame["arr_" + str(0)]
        green = data_frame["arr_" + str(1)]
        red = data_frame["arr_" + str(2)]
        nir = data_frame["arr_" + str(3)]

        sentinel = np.stack((red, green, blue, nir)).astype(np.float32)
        dsm = self.data["arr_" + str(index)][4]

        sentinel = torch.Tensor(sentinel)
        dsm = torch.Tensor(dsm)

        return sentinel, dsm


def get_loader(npz_dir, batch_size, num_workers=2, pin_memory=True, augmented=False, shuffle=True):
    train_ds = DsmDataset(npz_dir, augmented)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )
    return train_loader


def get_dataset(npz_dir):
    return DsmDataset(npz_dir)