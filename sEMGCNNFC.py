import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl

from ResBlock import ResBlock
from hyperparameters import BATCH_SIZE

model_name = "model_hu_2022_cnnfc"
dropout = 0.05

base_lr = 1e-2

conv_channels = [128, 256, 1024]

class SEMGCNNFC(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(1, conv_channels[0], (20, 8), stride=(2))
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d((2, 1))
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], (2, 1), 2)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], (2, 1), 2)
        self.bn3 = nn.BatchNorm2d(conv_channels[2])

        self.last_entry_linear = nn.Sequential(
            nn.Linear(8, 32),
            self.act,
            nn.Linear(32, 15)
        )


            # nn.Conv2d(128, 256, (3, 1), 1),
            # nn.BatchNorm2d(256),
            # nn.Tanh(),
            # nn.Dropout(dropout),
            # nn.MaxPool2d((2, 1))
        # )

        self.fc = nn.Sequential(
            nn.Linear(conv_channels[2] * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 15)
        )

        self.final_fc = nn.Sequential(
            nn.Linear(30, 64),
            nn.Tanh(),
            nn.Linear(64, 15)
        )

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 1

        # Lightning stuff
        self.training_step_outputs = []


    def forward(self, x):
        last = torch.squeeze(self.last_entry_linear(x[:, :, -1, :]))
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.pool(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.pool(z)

        z = self.conv3(z)
        z = self.bn3(z)
        z = self.act(z)
        z = self.dropout(z)

        z = z.view(z.size(0), -1)
        z = self.fc(z)
        z = torch.cat((z, last), 1)
        z = self.final_fc(z)
        return z


    def loss_function(self, x, y):
        loss = nn.MSELoss()(x, y) / 15
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
        self.optimizer = optim.Adam(self.parameters(), lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.9)
        return self.optimizer

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory

        if self.trainer.current_epoch < 30:
            self.scheduler.step()

            # Reset learning rate if loss below threshold
            # if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 5 == 0:
            #     for param_group in self.optimizer.param_groups:
            #         prev_lr = param_group['lr']
            #         param_group['lr'] = prev_lr * 0.90 ** self.cycle_count
            #         print("Learning rate was: ", prev_lr)
            #         print("New learning rate: ", param_group['lr'])
            #     self.cycle_count += 1


if __name__ == "__main__":
    model = SEMGCNNFC()
    input_tensor = torch.rand(100, 1, 100, 8)
    y = model(input_tensor)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



