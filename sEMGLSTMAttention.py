import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl

from ResBlock import ResBlock
from hyperparameters import BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "model_hu_2022_lstm_attention"
dropout = 0.05

base_lr = 5e-3


# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class SEMGLSTMAttention(pl.LightningModule):
    def __init__(self):
        super(SEMGLSTMAttention, self).__init__()
        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 1

        embed_dim = 16
        num_classes = 16
        num_layers = 2
        hidden_dim = 32
        dropout = 0.1

        self.dropout = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=1, 
                            bidirectional=True)
        self.atten1 = Attention(hidden_dim * 2, batch_first=True) # 2 is bidrectional
        self.lstm2 = nn.LSTM(input_size=hidden_dim * 2,
                            hidden_size=hidden_dim,
                            num_layers=1, 
                            bidirectional=True)
        self.atten2 = Attention(hidden_dim*2, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim * num_layers * 2, hidden_dim * num_layers * 2),
                                 nn.BatchNorm1d(hidden_dim * num_layers * 2),
                                 nn.ReLU()) 
        self.fc2 = nn.Linear(hidden_dim * num_layers * 2, num_classes)
        self.fcnew = nn.Linear(hidden_dim * 2, num_classes)

        # Lightning stuff
        self.training_step_outputs = []

    
    def forward(self, x, x_len, is_training=True):
        if is_training:
            x = self.dropout(x)

        x_len = x_len.to(torch.device("cpu"))
        
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, _ = self.atten1(x, lengths) # skip connect

        out2, (h_n, c_n) = self.lstm2(out1)
        y, lengths = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        # y, y_attentions = self.atten2(y, lengths)

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # return y
        
        # z = torch.cat([x, y], dim=1)
        if is_training:
            # z = self.fc1(self.dropout(z))
            # z = self.fc2(self.dropout(z))
            z = self.fcnew(self.dropout(y))
        else:
            # z = self.fc1(z)
            # z = self.fc2(z)
            z = self.fcnew(self.dropout(y))
        return z

    def loss_function(self, x, y):
        loss = nn.MSELoss()(x, y) / 15
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['sample']
        # y = torch.squeeze(batch['label'])
        y = torch.squeeze(batch['label'])
        seq_len = (torch.ones((x.shape[0])) * x.shape[1]).to(device)
        # seq_len = torch.ones((x.shape[0])) * x.shape[1]
        seq_len = seq_len.type(torch.int64)

        z = self.forward(x, seq_len)
        loss = self.loss_function(z, y)
        self.log("train_loss", loss)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        # y = torch.squeeze(batch['label'])
        y = torch.squeeze(batch['label'])
        seq_len = (torch.ones((x.shape[0])) * x.shape[1]).to(device)
        # seq_len = (torch.ones((x.shape[0])) * x.shape[1])
        # Change dtype to int64
        seq_len = seq_len.type(torch.int64)

        z = self.forward(x, seq_len)
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

         # Try this
        if self.trainer.current_epoch < 30:
            self.scheduler.step()

        # Reset learning rate if loss below threshold
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 5 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * 0.95 ** self.cycle_count
                print("Learning rate below threshold, new learning rate: ", param_group['lr'])
            self.cycle_count += 1.5


if __name__ == "__main__":
    model = SEMGLSTMAttention().to(device)
    input_tensor = torch.rand(512, 1001, 16).to(device)
    seq_len = (torch.ones((input_tensor.shape[0])) * input_tensor.shape[1]).to(device)

    y = model(input_tensor, seq_len)

    # Print number of trainable parameters
    print("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))



