import torch
from torch import optim, nn
import pytorch_lightning as pl

from hyperparameters import BATCH_SIZE, SEQ_LEN

dropout = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_name = "model_hu_2022_transformer_encoder"

base_lr = 5e-3

class SEMGTransformerEncoder(pl.LightningModule):
    def __init__(self):
        super(SEMGTransformerEncoder, self).__init__()
        self.d_model = 16

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=1
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.d_model * 10, 15)

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 0.0

        # Lightning stuff
        self.training_step_outputs = []

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(-1, 10 * self.d_model)
        out = self.relu(z)
        out = self.fc(z)
        out = out.view(-1, 1, 15)
        return out

    def loss_function(self, x, y):
        # Take MSE loss on the whole output, not the dummy token at the end
        loss = nn.MSELoss()(x, y) / 15

        return loss

    def training_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        out = self.forward(x)
        loss = self.loss_function(out, y)
        
        # self.log("train_loss", loss, on_step=True)
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        out = self.forward(x)
        loss = self.loss_function(out, y)
        
        # self.log("train_loss", loss, on_step=True)
        self.training_step_outputs.append(loss)

        # For the first element, get the first 3 relevant predicted joint angles on the cpu and compare
        for i in range(3):
            # Hu 2022 val logging
            z_comp = out[0, 0, i].cpu().detach().numpy()
            y_comp = y[0, 0, i].cpu().detach().numpy()

            print(i, z_comp, y_comp)

        # self.log("val_loss", loss, on_step=True)
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
        if self.trainer.current_epoch > 30:
            return
        
        self.scheduler.step()

        # Reset learning rate if loss below threshold
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 5 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * 0.95 ** self.cycle_count
                print("Learning rate below threshold, new learning rate: ", param_group['lr'])
            self.cycle_count += 1.5


if __name__ == "__main__":
    model = SEMGTransformerEncoder()
    # model = torch.compile(model)

    # Hu 2022 config
    input_tensor = torch.rand(1, 10, 16)

    y = model(input_tensor)

    print(y.shape)