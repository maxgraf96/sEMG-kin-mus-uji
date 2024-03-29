import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from hu_2022_DataModule import Hu2022DataModule
from hyperparameters import BATCH_SIZE
from sEMGInformer import SEMGInformer, model_name

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    # Hu 2022 dataset
    data_module = Hu2022DataModule(
        data_dir="/import/c4dm-datasets-ext/hu-22-sEMG-myo/processed", 
        batch_size=BATCH_SIZE,
        model_type="informer"
        )

    model = SEMGInformer()

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=600,
        check_val_every_n_epoch=5,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        # num_sanity_val_steps=0  # Disable validation at the start of training
        )
    trainer.fit(model, data_module)

    trainer.save_checkpoint(model_name + ".ckpt")
    torch.save(model.state_dict(), model_name + ".pt")