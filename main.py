import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from DataModule import KINDataModule
from sEMGEncoderDecoder import SEMGEncoderDecoder

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    data_module = KINDataModule("data", 2048)

    model = SEMGEncoderDecoder()

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs = 30,
        accelerator = "gpu", 
        devices = 1,
        # strategy = "dp",
        logger = wandb_logger
        )
    trainer.fit(model, data_module)
