import torch
from torch import optim, nn, utils, Tensor
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers.activations import NewGELUActivation

from ResBlock import ResBlock
from hyperparameters import BATCH_SIZE, SEQ_LEN, LOSS_INDICES

model_name = "model_hu_2022_cnnfc"
dropout = 0.05

base_lr = 5e-3

conv_channels = [32, 64, 1024]
seq_len = SEQ_LEN - 1

class SEMGCNNFC(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.act = NewGELUActivation()
        self.conv1 = nn.Conv2d(1, conv_channels[0], (2, 8), stride=(2))
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d((2, 1))
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], (2, 1), 2)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])

        self.fc = nn.Sequential(
            nn.Linear(conv_channels[1], 128),
            self.act,
            nn.Linear(128, len(LOSS_INDICES))
        )

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 1

        # Lightning stuff
        self.training_step_outputs = []


    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.pool(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.act(z)
        z = self.dropout(z)

        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return z


    def loss_function(self, x, y):
        loss = nn.MSELoss()(x, y)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = torch.unsqueeze(batch['sample'], 1)
        y = torch.squeeze(batch['label'])

        z = self.forward(x)
        loss = self.loss_function(z, y)
        self.log("train_loss", loss)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.unsqueeze(batch['sample'], 1)
        y = torch.squeeze(batch['label'])

        z = self.forward(x)
        loss = self.loss_function(z, y)

        z_comp = z[:3, 0].cpu().detach().numpy()
        y_comp = y[:3, 0].cpu().detach().numpy()
        print(z_comp, y_comp)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1)
        # Every n epochs reduce learning rate by a factor of 10
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, gamma=0.5)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
        }

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory


if __name__ == "__main__":
    model = SEMGCNNFC()
    input_tensor = torch.rand(100, 1, 10, 8)
    y = model(input_tensor)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



