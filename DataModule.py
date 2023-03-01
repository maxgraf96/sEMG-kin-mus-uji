import ast
import os.path

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from constants import DATALOADER_NUM_WORKERS
from hyperparameters import BATCH_SIZE


class KINDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_dir: str = "data"):
        super().__init__()
        self.data = pd.read_pickle(data_dir + os.path.sep + 'KIN_MUS_UJI_preprocessed_' + mode + '.pkl')

        print("---")
        print("Dataset length: ", self.__len__())
        print("---")

        self.mode = mode

    def __getitem__(self, index):
        acc_index = index
        if index >= len(self.data):
            acc_index = index % len(self.data)

        # Get item from dataframe
        item = self.data.iloc[acc_index]
        sample = torch.tensor(item.EMG_data, dtype=torch.float32)
        label = torch.tensor(item.Kinematic_data, dtype=torch.float32)

        # If the index is > than len(self.data), then roll the tensors accordingly
        if index >= len(self.data):
            # Get roll factor
            percentage = index / (len(self.data) * 7)
            # Map float percentage from range [0...100] to [0...7]
            shift_factor = int(percentage * 7)
            # Roll along sensor axis
            sample = torch.roll(sample, shift_factor, 1)
            label = torch.roll(label, shift_factor, 1)


        # Architecture specific preprocessing
        if self.mode == "cnn":
            sample.unsqueeze(0)
            label.unsqueeze(0)

        elif self.mode == "transformer":
            # Unsqueeze sample
            sample.unsqueeze(0)
            # Pad samples with zeros to make them from shape (100, 7) to (100, 18)
            sample = torch.nn.functional.pad(sample, (0, 11), 'constant', 0)

        return {'sample': sample, 'label': label}

    def __len__(self):
        return len(self.data) * 7


class KINDataModule(pl.LightningDataModule):
    def __init__(self, mode, data_dir: str = "data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = KINDataset(mode, self.data_dir)
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
    dm = KINDataModule()

