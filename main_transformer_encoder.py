import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from hu_2022_DataModule import Hu2022DataModule
from hyperparameters import BATCH_SIZE
from sEMGTransformerEncoder import SEMGTransformerEncoder, model_name

torch.set_float32_matmul_precision("high")
# torch._dynamo.config.verbose=True

if __name__ == "__main__":
    # Hu 2022 dataset
    data_module = Hu2022DataModule(
        data_dir="/import/c4dm-datasets-ext/hu-22-sEMG-myo/processed", 
        batch_size=BATCH_SIZE, 
        model_type="transformer_encoder"
        )

    model = SEMGTransformerEncoder()

    # Use torch 2.0 compile
    # model = torch.compile(model)

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=100,
        check_val_every_n_epoch=5,
        accelerator="gpu",
        devices=1,
        # strategy="dp",
        logger=wandb_logger
        )
    trainer.fit(model, data_module)

    torch.save(model.state_dict(), model_name + ".pt")