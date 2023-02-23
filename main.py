import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from DataModule import KINDataModule
from sEMGEncoderDecoder import SEMGEncoderDecoder

if __name__ == "__main__":
    data_module = KINDataModule("data", 64)

    model = SEMGEncoderDecoder()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, data_module)
