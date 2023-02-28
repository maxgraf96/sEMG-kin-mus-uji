import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl

from ResBlock import ResBlock
from hyperparameters import BATCH_SIZE

dropout = 0.05

base_lr = 0.1

class SEMGEncoderDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 2), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((2, 2)),
            # ---
            nn.Conv2d(32, 128, (3, 2), 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((2, 2)),
            # ---
            nn.Conv2d(128, 256, (3, 1), 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((2, 1))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 1), 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=(2, 6)),
            # ---
            nn.ConvTranspose2d(128, 32, (2, 3), 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=(1.995, 2)),
            # ---
            nn.ConvTranspose2d(32, 1, (2, 2), 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=(2, 1.1))
        )

        self.fc = nn.Sequential(
            nn.Linear(48, 18)
        )

        self.resblocks = nn.Sequential(
            ResBlock(256, 1, 1),
            ResBlock(256, 1, 1),
            ResBlock(256, 1, 1),
            ResBlock(256, 1, 1),
            ResBlock(256, 1, 1),
        )

        # Indices of the 14 joints we want to predict as per paper
        self.loss_indices = [5, 8, 10, 14, 7, 9, 12, 16, 6, 11, 15, 2, 3, 4]
        # subtract 1 from the indices
        self.loss_indices = [i - 1 for i in self.loss_indices]

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 0

    def loss_function(self, x, y):
        # Take MSE loss on the 14 indices
        loss = nn.MSELoss()(x[:, :, :, self.loss_indices], y[:, :, :, self.loss_indices]) / len(self.loss_indices)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['sample']
        y = batch['label']

        z = self.forward(x)
        loss = self.loss_function(z, y)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        z = self.forward(x)
        loss = self.loss_function(z, y)

        for i in self.loss_indices:
            z_comp = z[:3, 0, 0, i].cpu().detach().numpy()
            y_comp = y[:3, 0, 0, i].cpu().detach().numpy()
            print(i, z_comp, y_comp)

        self.log("val_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.9)
        return self.optimizer

    def training_epoch_end(self, outputs):
        self.scheduler.step()

        # Reset learning rate if loss below threshold
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 20 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * 0.95 ** self.cycle_count
                print("Loss below threshold, new learning rate: ", param_group['lr'])
            self.cycle_count += 1

    def print_sizes(self, input_tensor):
        output = input_tensor
        for m in self.encoder.children():
            output = m(output)
            print(m, output.shape)

        # print()
        # print("---")
        # print("Resblocks")
        # print("---")
        # print()
        # for m in self.resblocks.children():
        #     output = m(output)
        #     print(m, output.shape)

        print()
        print("---")
        print("Decoder")
        print("---")
        print()

        for m in self.decoder.children():
            output = m(output)
            print(m, output.shape)
        return output

    def forward(self, x):
        z = self.encoder(x)
        z = self.resblocks(z)
        z = self.decoder(z)
        return z


if __name__ == "__main__":
    model = SEMGEncoderDecoder()
    input_tensor = torch.rand(64, 1, 100, 7)
    model.print_sizes(input_tensor)
    # y = model(input_tensor)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



