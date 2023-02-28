import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from torch.autograd import Variable
import torch.nn.functional as F

from ResBlock import ResBlock
from hyperparameters import BATCH_SIZE

dropout = 0.05


class SEMGRNN(pl.LightningModule):
    def __init__(self):
        super(SEMGRNN, self).__init__()
        self.input_dim = 7
        self.hidden_dim = 64
        self.n_layers = 12
        self.output_dim = 18

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        # Indices of the 14 joints we want to predict as per paper
        self.loss_indices = [5, 8, 10, 14, 7, 9, 12, 16, 6, 11, 15, 2, 3, 4]
        # subtract 1 from the indices
        self.loss_indices = [i - 1 for i in self.loss_indices]

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.fc(lstm_out[:, -1])
        return y_pred

    def loss_function(self, x, y):
        # Take MSE loss on the 14 indices
        loss = nn.MSELoss()(x[:, self.loss_indices], y[:, self.loss_indices])
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

        # For the first 3 elements, get the values from indices 5, 8, 10 and 14 on the cpu and compare
        for i in self.loss_indices:
            z_comp = z[:3, i].cpu().detach().numpy()
            y_comp = y[:3, i].cpu().detach().numpy()
            print(i, z_comp, y_comp)

        self.log("val_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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


if __name__ == "__main__":
    model = SEMGRNN()
    input_tensor = torch.rand(64, 100, 7)
    # model.print_sizes(input_tensor)
    y = model(input_tensor)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



