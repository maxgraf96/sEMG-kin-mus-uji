import ast
import os.path

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import random_split, DataLoader

from constants import DATALOADER_NUM_WORKERS

class KINDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "data"):
        super().__init__()
        self.data = pd.read_pickle(data_dir + os.path.sep + 'KIN_MUS_UJI_preprocessed.pkl')
        print("Dataset length: ", len(self.data))

    def __getitem__(self, index):
        # Get item from dataframe
        item = self.data.iloc[index]
        emg = item.EMG_data
        sample = torch.tensor(emg, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(item.Kinematic_data, dtype=torch.float32)

        return {'sample': sample, 'label': label}

    def __len__(self):
        return len(self.data)


class KINDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = KINDataset(self.data_dir)
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
