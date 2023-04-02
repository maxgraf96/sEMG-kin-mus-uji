import numpy as np
import torch
import pytorch_lightning as pl
from transformers import InformerConfig, InformerModel
from transformers.activations import NewGELUActivation
from torch import optim, nn
from torch.optim import AdamW

from hyperparameters import BATCH_SIZE, INFORMER_PREDICTION_LENGTH, LOSS_INDICES, SEQ_LEN
from hu_2022_DataModule import load_means_stds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "model_hu_2022_informer"
base_lr = 2e-5

class SEMGInformer(pl.LightningModule):
    def __init__(self, batch_size=BATCH_SIZE):
        super(SEMGInformer, self).__init__()

        seq_len = SEQ_LEN - 1
        self.prediction_length = INFORMER_PREDICTION_LENGTH
        # self.lags_sequence = [1, 100]
        self.lags_sequence = [1]
        self.num_features = len(LOSS_INDICES)

        configuration = InformerConfig(
            prediction_length=self.prediction_length,
            lags_sequence=self.lags_sequence,
            context_length=seq_len - max(self.lags_sequence),
            num_time_features=self.num_features,
            input_size=self.num_features,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=1,
            decoder_attention_heads=1,
            scaling=None,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # attention_type="full",
            attention_dropout=0.1,
            activation_function="gelu_new",
        )

        self.loss_fn = nn.L1Loss()
        self.model = InformerModel(configuration)
        self.act = NewGELUActivation()
        self.fc = nn.Sequential(
            nn.Linear(configuration.d_model, configuration.d_model * 2),
            self.act,
            nn.Linear(configuration.d_model * 2, configuration.d_model * 2),
            self.act,
            nn.Linear(configuration.d_model * 2, self.num_features)
        )

        self.optimizer = None
        self.scheduler = None
        self.cycle_count = 0.0
        self.should_val_print = True

        input_dim = configuration.input_size
        self.past_time_features = torch.linspace(0, 1, seq_len).reshape(1, seq_len, 1)
        self.past_time_features = self.past_time_features.repeat(batch_size, 1, input_dim).to(device)
        # self.past_time_features = torch.ones(batch_size, self.seq_len, input_dim).to(device)
        
        self.future_time_features = torch.linspace(0, 1, self.prediction_length).reshape(1, self.prediction_length, 1)
        self.future_time_features = self.future_time_features.repeat(batch_size, 1, input_dim).to(device)
        # self.future_time_features = torch.ones(batch_size, self.prediction_length, input_dim).to(device)
        self.past_observed_mask = torch.ones(batch_size, seq_len, input_dim, dtype=torch.bool).to(device)

        # Lightning stuff
        self.training_step_outputs = []
        self.means_s, self.stds_s, self.means_l, self.stds_l = load_means_stds()

    def forward(self, past_values, past_time_features, future_values=None, future_time_features=None):
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
    
    def generate(self, past_values, past_time_features, future_time_features):
        future_values = torch.ones(past_values.shape[0], self.prediction_length, past_values.shape[2]).to(device)
        model_output = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=self.past_observed_mask,
            future_values=future_values,
            output_hidden_states=True,
            )
        lhs = self.act(model_output.last_hidden_state)
        output = self.fc(lhs)
        return output

    def loss_function(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label'][:, -INFORMER_PREDICTION_LENGTH:, :]
        output = self.forward(
            past_values=x, 
            past_time_features=self.past_time_features, 
            future_values=y, 
            future_time_features=self.future_time_features
            )

        loss = self.loss_function(output, y)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample']
        y = batch['label'][:, -INFORMER_PREDICTION_LENGTH:, :]
        future_values = torch.zeros(x.shape[0], self.prediction_length, x.shape[2]).to(device)
        output = self.forward(
            past_values=x, 
            past_time_features=self.past_time_features,
            future_values=future_values,
            future_time_features=self.future_time_features
            )

        diffs = []
        z_comp = output.cpu().detach().numpy()
        y_comp = y.cpu().detach().numpy()

        # Denormalize
        z_comp = z_comp * self.stds_l + self.means_l
        y_comp = y_comp * self.stds_l + self.means_l

        diffs = np.abs(z_comp - y_comp)

        if self.should_val_print:
            print("Validation")
            print("z_comp:", z_comp[0, 50])
            print("y_comp:", y_comp[0, 50])
            self.should_val_print = False
            
        # np.set_printoptions(precision=3)
        # print("Average diff:", np.mean(diffs))
        # print("-----------------------------------------")

        loss = self.loss_function(output, y)
        self.log("val_loss", loss)
        self.log("avg_val_diff", np.mean(diffs))
        self.log("avg_val_diff_std", np.std(diffs))
        return loss

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.9)
        return self.optimizer

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory
        # Trigger printing of validation set values only for the first val batch
        self.should_val_print = True

        if self.trainer.current_epoch < 80:
            self.scheduler.step()

        # Reset learning rate if loss below threshold
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch < 100 and self.trainer.current_epoch % 10 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = base_lr * 0.96 ** self.cycle_count
                print("Learning rate below threshold, new learning rate: ", param_group['lr'])
            self.cycle_count += 1

        # Save model every 100 epochs
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 100 == 0:
            torch.save(self.state_dict(), model_name + ".pt")


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

    future_values = torch.rand(bs, model.prediction_length, output_dim).to(device)
    future_time_features = torch.ones(bs, model.prediction_length, output_dim).to(device)

    # model.print_sizes(input_tensor)
    output, hs_output = model(
        past_values=input_tensor, 
        past_time_features=time_features, 
        future_values=future_values,
        future_time_features=future_time_features
    )

    print(hs_output.shape)