import torch
import pytorch_lightning as pl
from transformers import InformerConfig, InformerModel
from torch import optim, nn
from torch.optim import AdamW

from hyperparameters import BATCH_SIZE, INFORMER_PREDICTION_LENGTH, SEQ_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "model_hu_2022_informer"
base_lr = 3e-3

class SEMGInformer(pl.LightningModule):
    def __init__(self, inference=False, batch_size=BATCH_SIZE):
        super(SEMGInformer, self).__init__()

        self.prediction_length = INFORMER_PREDICTION_LENGTH
        self.lags_sequence = [1]
        self.loss_indices = [0, 1, 3, 4, 7, 8, 11, 12]

        configuration = InformerConfig(
            prediction_length=self.prediction_length,
            lags_sequence=self.lags_sequence,
            context_length=SEQ_LEN - 1 - max(self.lags_sequence),
            num_time_features=16,
            input_size=16,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=1,
            decoder_attention_heads=1,
            scaling=None,
            output_hidden_states=True,
            return_dict_in_generate=True,
            attention_type="full",
            attention_dropout=0.05,
        )

        # self.conv1d = nn.Conv1d(16, 16, 2, 1, padding="same")
        self.model = InformerModel(configuration)
        # if not inference else InformerModel.from_pretrained(model_name + ".ckpt", config=configuration)
        
        self.fc = nn.Linear(configuration.d_model, 16)
        self.act = nn.GELU()

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 0.0

        seq_len = SEQ_LEN - 1
        input_dim = configuration.input_size
        self.time_features = torch.linspace(0, 1, seq_len).reshape(1, seq_len, 1)
        self.time_features = self.time_features.repeat(batch_size, 1, input_dim).to(device)
        # self.time_features = torch.ones(batch_size, seq_len, input_dim).to(device)
        self.past_observed_mask = torch.ones(batch_size, seq_len, input_dim, dtype=torch.bool).to(device)

        # Lightning stuff
        self.training_step_outputs = []

    # def forward(self, past_values, past_time_features, past_observed_mask, future_values=None, future_time_features=None):
    def forward(self, past_values, past_time_features, future_values, future_time_features):
        # z = self.conv1d(past_values.permute(0, 2, 1))
        # past_values = z.permute(0, 2, 1)
        model_output = self.model(past_values=past_values, 
                          past_time_features=past_time_features, 
                          past_observed_mask=self.past_observed_mask, 
                          future_values=future_values,
                          future_time_features=future_time_features,
                          output_hidden_states=True
                          )
        lhs = self.act(model_output.last_hidden_state)
        output = self.fc(lhs)
        return output
    
    def generate(self, past_values, past_time_features, future_time_features, past_observed_mask):
        future_values = torch.ones(past_values.shape[0], self.prediction_length, past_values.shape[2]).to(device)
        model_output = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
            output_hidden_states=True,
            )
        lhs = self.act(model_output.last_hidden_state)
        output = self.fc(lhs)
        return output

    def loss_function(self, x, y):
        # Hu 2022 loss
        # Take MSE loss on the whole output, not the dummy token at the end
        # loss = nn.MSELoss()(x[:, :, :15], y[:, :, :15]) / 15
        loss = nn.MSELoss()(x[:, :, self.loss_indices], y[:, :, self.loss_indices]) / len(self.loss_indices)

        return loss

    def training_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        seq_len = SEQ_LEN - 1
        bs = x.shape[0]
        input_dim = x.shape[2]
        output_dim = y.shape[2]

        # future_values = torch.zeros(bs, self.prediction_length, output_dim).to(device)
        future_values = y

        future_time_features = torch.linspace(0, 1, self.prediction_length).reshape(1, self.prediction_length, 1)
        future_time_features = future_time_features.repeat(bs, 1, output_dim).to(device)

        output = self.forward(past_values=x, past_time_features=self.time_features, past_observed_mask=self.past_observed_mask, future_values=future_values, future_time_features=future_time_features)

        loss = self.loss_function(output, y)

        # self.log("train_loss", loss, on_step=True)
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label']

        seq_len = SEQ_LEN - 1
        bs = x.shape[0]
        input_dim = x.shape[2]
        output_dim = y.shape[2]

        future_values = y

        future_time_features = torch.linspace(0, 1, self.prediction_length).reshape(1, self.prediction_length, 1)
        future_time_features = future_time_features.repeat(bs, 1, output_dim).to(device)

        output = self.forward(past_values=x, past_time_features=self.time_features, past_observed_mask=self.past_observed_mask, future_values=future_values, future_time_features=future_time_features)

        loss = self.loss_function(output, y)

        # For the first element, get the first 3 relevant predicted joint angles on the cpu and compare
        # for i in range(3):
        #     # Hu 2022 val logging
        #     z_comp = output[0, 0, i].cpu().detach().numpy()
        #     y_comp = y[0, 0, i].cpu().detach().numpy()

        #     print(i, z_comp, y_comp)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.95), eps=1.0e-9, lr=base_lr)
        self.optimizer = AdamW(self.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1)
        # Every 30 epochs reduce learning rate by a factor of 10
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, gamma=0.8)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
        }

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory

        # If last epoch loss is below threshold, fix learning rate
        # if self.trainer.logged_metrics['train_loss'] < 3 or self.trainer.current_epoch > 30:
        # if self.trainer.current_epoch > 30:
            # return
        
        # # Reset learning rate if loss below threshold
        # if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 5 == 0:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = base_lr * 0.95 ** self.cycle_count
        #         print("Learning rate below threshold, new learning rate: ", param_group['lr'])
        #     self.cycle_count += 1.5



if __name__ == "__main__":
    model = SEMGInformer().to(device)
    
    configuration = model.model.config
    
    seq_len = SEQ_LEN - 1
    input_dim = configuration.input_size
    output_dim = input_dim
    bs = 2

    # Hu 2022 config
    input_tensor = torch.rand(bs, seq_len, input_dim).to(device)
    # For time features (like positional encoding) use a monotonic sequence that goes from 0 to 1
    # over seq_len steps
    time_features = torch.linspace(0, 1, seq_len).reshape(1, seq_len, 1).to(device)
    time_features = time_features.repeat(bs, 1, input_dim)
    past_observed_mask = torch.ones(bs, seq_len, input_dim, dtype=torch.bool).to(device)

    future_values = torch.rand(bs, model.prediction_length, output_dim).to(device)
    future_time_features = torch.ones(bs, model.prediction_length, output_dim).to(device)

    # model.print_sizes(input_tensor)
    output, hs_output = model(
        past_values=input_tensor, 
        past_time_features=time_features, 
        past_observed_mask=past_observed_mask,
        future_values=future_values,
        future_time_features=future_time_features
    )

    print(hs_output.shape)