import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from DataModule import KINDataModule
from hyperparameters import BATCH_SIZE
from sEMGRNN import SEMGRNN

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    data_module = KINDataModule("data", BATCH_SIZE)

    model = SEMGRNN()

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=500,
        check_val_every_n_epoch=5,
        accelerator="gpu",
        devices=1,
        # strategy="dp",
        logger=wandb_logger
        )
    trainer.fit(model, data_module)
