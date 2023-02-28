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
        self.data = pd.read_pickle(data_dir + os.path.sep + 'KIN_MUS_UJI_preprocessed.pkl')
        # Only keep first BATCH_SIZE samples
        # self.data = self.data.iloc[:BATCH_SIZE * 10]

        # Make a noise tensor with 100 random values around 0 with variance 0.01
        # self.noise = (torch.randn(100, 7) * 0.01).numpy().tolist()

        print("---")
        print("Dataset length: ", len(self.data))
        print("---")

        self.mode = mode

    def __getitem__(self, index):
        # Get item from dataframe
        item = self.data.iloc[index]
        emg = item.EMG_data

        sample = torch.tensor(emg, dtype=torch.float32)
        label = torch.tensor(item.Kinematic_data, dtype=torch.float32)

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
        return len(self.data)


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

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS)


if __name__ == '__main__':
    dm = KINDataModule()

