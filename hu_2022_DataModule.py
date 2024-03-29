import os.path
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data
from constants import DATALOADER_NUM_WORKERS
from hyperparameters import DATASET_SHIFT_SIZE, INFORMER_PREDICTION_LENGTH, SEQ_LEN, LOSS_INDICES
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

data_seq_len = SEQ_LEN - 1
means_stds_file = "hu_2022_means_stds.npz"

def load_means_stds():
        # Load means and stds from disk
        data = np.load(means_stds_file)
        # Samples
        means_s = data['means_s']
        stds_s = data['stds_s']
        # Labels
        means_l = data['means_l'][LOSS_INDICES]
        stds_l = data['stds_l'][LOSS_INDICES]

        return means_s, stds_s, means_l, stds_l

class Hu2022Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "data", model_type="transformer"):
        super().__init__()
        self.model_type = model_type
        # Load data from numpy array
        self.data = np.load(data_dir + os.path.sep + 'hu_2022_raw.npy')
        self.sos_token = torch.ones(1, 16) * -100
        self.eos_token = torch.ones(1, 16) * 100

        # Take sEMG of some 10000 rows
        # testdata = self.data[100000:210000, :8]
        # # # Convert to csv with no header
        # df = pd.DataFrame(testdata)
        # # # Save to csv with no header
        # df.to_csv('hu_2022_testdata_sEMG.csv', header=False, index=False)

        print("---")
        print("Dataset length: ", self.__len__())
        print("---")

    def load_means_stds(self):
        # Load means and stds from disk
        self.means_s, self.stds_s, self.means_l, self.stds_l = load_means_stds()

    def __getitem__(self, index):
        data_length_no_augmentation = len(self.data) // data_seq_len
        acc_index = index
        # if index >= data_length_no_augmentation:
            # acc_index = index % data_length_no_augmentation

        # Get item from dataframe
        # item = self.data[acc_index * data_seq_len : (acc_index + 1) * data_seq_len]

        start = acc_index * DATASET_SHIFT_SIZE
        item = self.data[start : start + data_seq_len]

        sample = item[:, :8]
        label = item[:, 8:]
        # Take only relevant indices from label
        label = label[:, LOSS_INDICES]
        
        # # Normalize 
        sample = (sample - self.means_s) / self.stds_s
        label = (label - self.means_l) / self.stds_l

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

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


        if self.model_type == "transformer":
            # Pad samples with zeros to make them from shape (data_seq_len, 8) to (data_seq_len, 16)
            sample = torch.nn.functional.pad(sample, (0, 8), "constant", 0)
            # Concat sos and eos tokens
            sample = torch.cat((self.sos_token, sample), 0)

            # Pad label with zeros to make them from shape (data_seq_len, 15) to (data_seq_len, 16) and add tokens
            label = torch.nn.functional.pad(label, (0, 1), "constant", 0)
            # label = torch.cat((self.sos_token, label, self.eos_token), 0)
            label = torch.cat((self.sos_token, label), 0)
            return {'sample': sample, 'label': label}
        
        elif self.model_type == "lstm_attention":
            # Pad samples with zeros to make them from shape (data_seq_len, 8) to (data_seq_len, 16)
            sample = torch.nn.functional.pad(sample, (0, 8), "constant", 0)
            # Pad label with zeros to make them from shape (data_seq_len, 15) to (data_seq_len, 16) and add tokens
            label = torch.nn.functional.pad(label, (0, 1), "constant", 0)
            # return {'sample': sample, 'label': label[-1:, :]}
            return {'sample': sample, 'label': label}
        
        elif self.model_type == "fc":
            return {'sample': sample, 'label': label[-1:, :]}
        
        elif self.model_type == "transformer_encoder":
            # Pad samples with zeros to make them from shape (data_seq_len, 8) to (data_seq_len, 16)
            sample = torch.nn.functional.pad(sample, (0, 8), "constant", 0)
            return {'sample': sample, 'label': label[-1:, :]}

        elif self.model_type == "informer":
            # Pad samples with zeros to make them from shape (data_seq_len, 8) to (data_seq_len, 16)
            # sample = torch.nn.functional.pad(sample, (0, 8), "constant", 0)

            # Pad label with zeros to make them from shape (data_seq_len, 15) to (data_seq_len, 16) and add tokens
            # label = torch.nn.functional.pad(label, (0, 1), "constant", 0)
            # return {'sample': sample, 'label': label[-INFORMER_PREDICTION_LENGTH:, :]}
            return {'sample': sample, 'label': label}
        
        elif self.model_type == "cnn":
            return {'sample': sample, 'label': label[-1:]}

        

    def __len__(self):
        # return len(self.data) // data_seq_len * 8
        # return (len(self.data) // DATASET_SHIFT_SIZE - data_seq_len) * 8
        return len(self.data) // DATASET_SHIFT_SIZE - data_seq_len


class Hu2022DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64, model_type="transformer"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = Hu2022Dataset(self.data_dir, model_type=model_type)
        self.train = None
        self.val = None

    def setup(self, stage: str):
        # Split dataset into train and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train, self.val = random_split(self.dataset, [train_size, val_size])

        # Get mean and std of the training set column wise and save them to disk
        samples = self.train.dataset.data[:, :8]
        labels = self.train.dataset.data[:, 8:]
        means_s = np.mean(samples, axis=0)
        stds_s = np.std(samples, axis=0)
        means_l = np.mean(labels, axis=0)
        stds_l = np.std(labels, axis=0)

        # Save all to disk as one file
        np.savez(means_stds_file, means_s=means_s, stds_s=stds_s, means_l=means_l, stds_l=stds_l)
        self.dataset.load_means_stds()

        # Take five samples from the validation set and export them to csv
        for i in range(20):
            item = self.val[i]
            sample = item['sample']
            label = item['label']
            sample = sample.numpy()
            label = label.numpy()
            sample = np.concatenate((sample, label), axis=1)
            if i == 0:
                samples = sample
            else:
                samples = np.concatenate((samples, sample), axis=0)

        df = pd.DataFrame(samples)
        df.to_csv('hu_2022_valdata_sEMG.csv', header=False, index=False)

        # Non-random split
        # self.train = torch.utils.data.Subset(self.dataset, range(train_size))
        # self.val = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=DATALOADER_NUM_WORKERS)


if __name__ == '__main__':
    dm = Hu2022DataModule(data_dir="/import/c4dm-datasets-ext/hu-22-sEMG-myo/processed")
    dm.setup(stage="fit")

