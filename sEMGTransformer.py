import math

import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from torch.autograd import Variable

from ResBlock import ResBlock
from hyperparameters import BATCH_SIZE

dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_lr = 1e-2


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class SEMGTransformer(pl.LightningModule):
    def __init__(self):
        super(SEMGTransformer, self).__init__()
        self.d_model = 36
        self.input_dim = 18
        self.n_layers = 4
        self.output_dim = 18

        self.positional_encoder = PositionalEncoding(
            dim_model=self.d_model, dropout_p=dropout, max_len=BATCH_SIZE
        )
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=12, num_encoder_layers=self.n_layers,
                                          num_decoder_layers=self.n_layers, dim_feedforward=self.output_dim, dropout=dropout,
                                          activation='relu')
        self.fc = nn.Linear(self.d_model, self.output_dim)

        # Indices of the 14 joints we want to predict as per paper
        self.loss_indices = [5, 8, 10, 14, 7, 9, 12, 16, 6, 11, 15, 2, 3, 4]
        # subtract 1 from the indices
        self.loss_indices = [i - 1 for i in self.loss_indices]

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 0.0

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.fc(transformer_out)
        out = out.permute(1, 0, 2)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1).to(device)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def loss_function(self, x, y):
        # Take MSE loss on the 14 indices
        loss = nn.MSELoss()(x[:, :, self.loss_indices], y[:, :, self.loss_indices]) / len(self.loss_indices)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['sample']
        y = batch['label']

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        # y_input = y[:, :-1]
        # y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)

        z = self.forward(x, y, tgt_mask=tgt_mask)
        loss = self.loss_function(z, y)
        self.log("train_loss", loss, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        # y_input = y[:, :-1]
        # y_expected = y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)

        z = self.forward(x, y, tgt_mask=tgt_mask)
        loss = self.loss_function(z, y)

        # For the first 3 elements, get the values from indices 5, 8, 10 and 14 on the cpu and compare
        for i in self.loss_indices:
            z_comp = z[0, :3, i].cpu().detach().numpy()
            y_comp = y[0, :3, i].cpu().detach().numpy()
            print(i, z_comp, y_comp)

        self.log("val_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1.0e-9, lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.5)
        return self.optimizer

    def training_epoch_end(self, outputs):
        if self.trainer.current_epoch > 30:
            return
        
        self.scheduler.step()

        # Reset learning rate if loss below threshold
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 5 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * 0.95 ** self.cycle_count
                print("Loss below threshold, new learning rate: ", param_group['lr'])
            self.cycle_count += 1.5


if __name__ == "__main__":
    model = SEMGTransformer()
    input_tensor = torch.rand(1024, 100, 18)
    target = torch.rand(1024, 100, 18)
    sequence_length = target.size(1)
    tgt_mask = model.get_tgt_mask(sequence_length)
    # model.print_sizes(input_tensor)
    y = model(input_tensor, target, tgt_mask=tgt_mask)

    print(y.shape)