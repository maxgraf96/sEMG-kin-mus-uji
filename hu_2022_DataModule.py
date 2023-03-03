import os.path
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data
from constants import DATALOADER_NUM_WORKERS
from hyperparameters import BATCH_SIZE
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class Hu2022Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "data"):
        super().__init__()
        # Load data from numpy array
        self.data = np.load(data_dir + os.path.sep + 'hu_2022_raw.npy')

        print("---")
        print("Dataset length: ", self.__len__())
        print("---")

    def __getitem__(self, index):
        data_length_no_augmentation = len(self.data) // 100
        acc_index = index
        if index >= data_length_no_augmentation:
            acc_index = index % data_length_no_augmentation

        # Get item from dataframe
        item = self.data[acc_index * 100 : (acc_index + 1) * 100]
        sample = torch.tensor(item[:, :8], dtype=torch.float32)
        label = torch.tensor(item[:, 8:], dtype=torch.float32)

        # Data augmentation:
        # If the index is > than len(self.data), then roll the tensors accordingly
        # This simulates the effect of rotating the myo armband "by 1 sensor"
        if index >= data_length_no_augmentation:
            # Get roll factor
            percentage = index / (data_length_no_augmentation * 8)
            # Map float percentage from range [0...100] to [0...7]
            shift_factor = int(percentage * 8)
            # Roll along sensor axis
            sample = torch.roll(sample, shift_factor, 1)
            label = torch.roll(label, shift_factor, 1)


        # # Unsqueeze sample
        sample.unsqueeze(0)
        # Pad samples with zeros to make them from shape (100, 8) to (100, 16)
        sample = torch.nn.functional.pad(sample, (0, 8), "constant", 0)
        # Pad label with zeros to make them from shape (100, 15) to (100, 16)
        label = torch.nn.functional.pad(label, (0, 1), "constant", 0)

        return {'sample': sample, 'label': label}

    def __len__(self):
        return len(self.data) // 100 * 8


class Hu2022DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = Hu2022Dataset(self.data_dir)
        self.train = None
        self.val = None

    def setup(self, stage: str):
        # Split dataset into train and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train, self.val = random_split(self.dataset, [train_size, val_size])

        # Non-random split
        # self.train = torch.utils.data.Subset(self.dataset, range(train_size))
        # self.val = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS)


if __name__ == '__main__':
    dm = Hu2022DataModule(data_dir="/import/c4dm-datasets-ext/hu-22-sEMG-myo/processed")

