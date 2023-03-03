import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from DataModule import KINDataModule
from hu_2022_DataModule import Hu2022DataModule
from hyperparameters import BATCH_SIZE
from sEMGTransformer import SEMGTransformer

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # Kin-mus-uji dataset
    # data_module = KINDataModule(mode="transformer", data_dir="data", batch_size=BATCH_SIZE)
    
    # Hu 2022 dataset
    data_module = Hu2022DataModule(data_dir="/import/c4dm-datasets-ext/hu-22-sEMG-myo/processed", batch_size=BATCH_SIZE)

    model = SEMGTransformer()

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=200,
        check_val_every_n_epoch=5,
        accelerator="gpu",
        devices=1,
        # strategy="dp",
        logger=wandb_logger
        )
    trainer.fit(model, data_module)

    torch.save(model.state_dict(), "model_hu_2022.pt")