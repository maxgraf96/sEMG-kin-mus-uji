import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl

from ResBlock import ResBlock

dropout = 0.05


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
            nn.Upsample(scale_factor=(1, 1)),
            # ---
            nn.ConvTranspose2d(128, 32, (3, 2), 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=(1, 1)),
            # ---
            nn.ConvTranspose2d(32, 1, (3, 2), 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=(1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(48, 18)
        )

        self.resblocks = nn.Sequential(
            ResBlock(256, 10, 1),
            ResBlock(256, 10, 1),
            ResBlock(256, 10, 1),
            ResBlock(256, 10, 1),
            ResBlock(256, 10, 1),
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['sample']
        y = batch['label']

        z = self.encoder(x)
        z = self.resblocks(z)
        z = self.decoder(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)

        loss = nn.functional.mse_loss(z, y, reduction='mean')
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        z = self.encoder(x)
        z = self.resblocks(z)
        z = self.decoder(z)
        z = z.view(z.size(0), -1)
        z = self.fc(z)

        loss = nn.functional.mse_loss(z, y, reduction='mean')
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def print_sizes(self, input_tensor):
        output = input_tensor
        for m in self.encoder.children():
            output = m(output)
            print(m, output.shape)

        print()
        print("---")
        print("Resblocks")
        print("---")
        print()
        for m in self.resblocks.children():
            output = m(output)
            print(m, output.shape)

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
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return z


if __name__ == "__main__":
    model = SEMGEncoderDecoder()
    input_tensor = torch.rand(64, 1, 100, 7)
    model.print_sizes(input_tensor)
    # y = model(input_tensor)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



