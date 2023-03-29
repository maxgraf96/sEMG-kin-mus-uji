from functools import cached_property
import math
import torch
from torch import optim, nn
import pytorch_lightning as pl

from hyperparameters import BATCH_SIZE, SEQ_LEN

dropout = 0.05

# device = torch.device("cpu")

model_name = "model_hu_2022"

base_lr = 5e-3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # # Encoding - From formula
        # pos_encoding = torch.zeros(1, max_len, dim_model)
        # positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        # division_term = torch.exp(
        #     torch.arange(0, dim_model, 2) * (-torch.log(torch.tensor([10000.0]))) / dim_model)  # 1000^(2i/dim_model)

        # # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        # pos_encoding[:, :, 0::2] = torch.sin(positions_list * division_term)

        # # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        # pos_encoding[:, :, 1::2] = torch.cos(positions_list * division_term)

        # # Saving buffer (same as parameter without gradients needed)
        # pos_encoding = pos_encoding.transpose(0, 1)
        # self.register_buffer("pos_encoding", pos_encoding)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Residual connection + pos encoding
        # return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SEMGTransformer(pl.LightningModule):
    def __init__(self, batch_first=False):
        super(SEMGTransformer, self).__init__()
        self.batch_first = batch_first
        
        self.n_layers = 2
        
        # Kin-mus-uji config
        # self.d_model = 18
        # self.input_dim = 18
        # self.output_dim = 18

        self.d_model = 16
        self.input_dim = 16
        self.output_dim = 16

        self.positional_encoder = PositionalEncoding(
            d_model=self.d_model, dropout_p=dropout, max_len=5000
        )
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=1, num_encoder_layers=self.n_layers,
                                          num_decoder_layers=self.n_layers, dim_feedforward=self.output_dim, dropout=dropout,
                                          activation='relu', batch_first=self.batch_first)
        self.fc = nn.Linear(self.d_model, self.output_dim)

        self.tgt_mask = None

        # Kin-mus-uji dataset stuff
        # Indices of the 14 joints we want to predict as per kin-mus-uji datasetpaper
        # self.loss_indices = [5, 8, 10, 14, 7, 9, 12, 16, 6, 11, 15, 2, 3, 4]
        # subtract 1 from the indices
        # self.loss_indices = [i - 1 for i in self.loss_indices]

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 0.0

        # Lightning stuff
        self.training_step_outputs = []

    # Rename this to "forward" for training
    def forward(self, src, tgt, tgt_mask): #, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        if not self.batch_first:
            src = src.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=None, tgt_key_padding_mask=None)
        out = self.fc(transformer_out)
        if not self.batch_first:
            out = out.permute(1, 0, 2)
        # out = out.permute(1, 0, 2)

        return out
    
    # Use this for tracing and exporting the model
    # def forward(self, x, y_input, tgt_mask):
    #     y_input = torch.zeros(1, 1, 16, dtype=torch.float32)#.to(device)

    #     for _ in torch.arange(40):
    #         # print(i)
    #         # Get source mask
    #         # tgt_mask = self.get_tgt_mask(y_input.size(1))#.to(device)
    #         tgt_mask = self.tgt_mask[:y_input.size(1), :y_input.size(1)]
            
    #         pred = self.forward_pass(x, y_input, tgt_mask=tgt_mask)[:, -1:, :]

    #         # if i == 0:
    #             # y_input = pred
    #         # else:
    #         # Concatenate previous input with predicted best word
    #         y_input = torch.cat((y_input, pred), dim=1)

    #     return y_input
    
    def get_tgt_mask(self, size: int):
        if self.tgt_mask is not None:
            return self.tgt_mask[:size, :size]
        
        # Generates a square matrix where each row allows one word more to be seen
        mask = torch.triu(torch.ones([size, size], dtype=torch.float32)).T.to(device)  # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        self.tgt_mask = mask
        return mask

    # def create_pad_mask(self, matrix: Tensor, pad_token: int):
    #     # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    #     # [False, False, False, True, True, True]
    #     return (matrix == pad_token)

    def loss_function(self, x, y):
        # Kin-mus-uji loss
        # Take MSE loss on the 14 indices
        # loss = nn.MSELoss()(x[:, :, self.loss_indices], y[:, :, self.loss_indices]) / len(self.loss_indices)

        # Hu 2022 loss
        # Take MSE loss on the whole output, not the dummy token at the end
        loss = nn.MSELoss()(x[:, :, :self.output_dim - 1], y[:, :, :self.output_dim - 1]) / 15

        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['sample']
        y = batch['label']

        # y_input = y
        # y_expected = y

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)

        z = self.forward(x, y_input, tgt_mask=tgt_mask)
        loss = self.loss_function(z, y_expected)
        
        # self.log("train_loss", loss, on_step=True)
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        # y_input = y
        # y_expected = y

         # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1, :]
        y_expected = y[:, 1:, :]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length)

        z = self.forward(x, y_input, tgt_mask=tgt_mask)
        loss = self.loss_function(z, y_expected)

        # For the first element, get the first 3 relevant predicted joint angles on the cpu and compare
        for i in range(3):
            # Kin-mus-uji val logging
            # z_comp = z[0, 0, self.loss_indices[i]].cpu().detach().numpy()
            # y_comp = y[0, 0, self.loss_indices[i]].cpu().detach().numpy()
            
            # Hu 2022 val logging
            z_comp = z[0, 0, i].cpu().detach().numpy()
            y_comp = y_expected[0, 0, i].cpu().detach().numpy()

            print(i, z_comp, y_comp)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1.0e-9, lr=base_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.5)
        return self.optimizer

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory

        # If last epoch loss is below threshold, fix learning rate
        # if self.trainer.logged_metrics['train_loss'] < 3 or self.trainer.current_epoch > 30:
        # if self.trainer.current_epoch > 30:
            # return
        
        # Try this
        # if self.trainer.current_epoch < 30:
            # self.scheduler.step()

        # Reset learning rate if loss below threshold
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 5 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * 0.95 ** self.cycle_count
                print("Learning rate below threshold, new learning rate: ", param_group['lr'])
            self.cycle_count += 1.5


if __name__ == "__main__":
    model = SEMGTransformer().to(device)
    # model = torch.compile(model)

    # Hu 2022 config
    input_tensor = torch.rand(2, SEQ_LEN, 16).to(device)
    target = torch.rand(2, SEQ_LEN, 16).to(device)

    sequence_length = target.size(1)
    tgt_mask = model.get_tgt_mask(sequence_length)
    
    # model.print_sizes(input_tensor)
    y = model(input_tensor, target, tgt_mask=tgt_mask)

    print(y.shape)