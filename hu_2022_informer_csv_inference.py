import numpy as np
from tqdm import tqdm
from hyperparameters import SEQ_LEN, LOSS_INDICES
import torch
import pandas as pd
from transformers import GenerationConfig
from sEMGInformer import SEMGInformer, model_name
from hu_2022_DataModule import load_means_stds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__ == "__main__":
    model = SEMGInformer(batch_size=1).to(device)
    model.load_state_dict(torch.load(model_name + ".pt"))
    model.eval()

    # data = np.load('/import/c4dm-datasets-ext/hu-22-sEMG-myo/processed/hu_2022_raw.npy')
    # data = data[:1000]
    # df = pd.DataFrame(data)

    # Load data from csv
    # Our recorded data
    # df = pd.read_csv("data/RawEMG-2023-03-22-13.15.34.csv", skiprows=0, usecols=range(8), dtype=float)
    
    # Data from Hu et al. 2022 validation set - updates every time we train the model
    df = pd.read_csv("hu_2022_valdata_sEMG.csv", header=None, usecols=range(8), dtype=float)

    num_features = 8

    # Take all rows and first 8 columns and convert to tensor
    input_tensor = torch.tensor(df.iloc[:, :num_features].values, dtype=torch.float32)

    # Add batch dimension and reshape from (4000, num_features) to (1, SEQ_LEN, num_features)
    input_tensor = input_tensor.reshape(1, -1, num_features).to(device)

    configuration = model.model.config
    seq_len = SEQ_LEN - 1
    input_dim = configuration.input_size
    output_dim = input_dim
    bs = 1

    past_time_features = torch.linspace(0, 1, seq_len).reshape(1, seq_len, 1).to(device)
    past_time_features = past_time_features.repeat(bs, 1, input_dim)
    
    # future_time_features = torch.linspace(0, 1, model.prediction_length).reshape(1, model.prediction_length, 1)
    # future_time_features = future_time_features.repeat(bs, 1, output_dim).to(device)
    future_time_features = torch.ones(bs, seq_len, output_dim).to(device)

    result = torch.zeros((bs, 1, output_dim)).to(device)

    # Normalise data
    means_s, stds_s, means_l, stds_l = load_means_stds()
    # Convert to tensor
    means_s_t = torch.tensor(means_s, dtype=torch.float32).to(device)
    stds_s_t = torch.tensor(stds_s, dtype=torch.float32).to(device)
    means_l_t = torch.tensor(means_l, dtype=torch.float32).to(device)
    stds_l_t = torch.tensor(stds_l, dtype=torch.float32).to(device)
    input_tensor = (input_tensor - means_s_t) / stds_s_t

    with torch.no_grad():
        # for length in tqdm(range(1, input_tensor.shape[1] - seq_len)):
        for start_idx in tqdm(range(0, input_tensor.shape[1] - seq_len + 1, seq_len)):
        # for length in tqdm(range(1, 500)):
            past_values = input_tensor[:, start_idx:start_idx + seq_len, :]

            output = model.generate(
                past_values=past_values, 
                past_time_features=past_time_features, 
                future_time_features=future_time_features,
            )

            # Denormalise data
            output = (output * stds_l_t) + means_l_t

            if start_idx == 0:
                result = output
            else:
                # result = torch.cat((result, output[:, -1:, :]), dim=1)
                result = torch.cat((result, output), dim=1)
            

        print("Torch result:")
        torch_result = result.cpu().detach().numpy()
        print(torch_result)
        torch_result = torch_result.reshape(-1, num_features)

        # Compare torch result with Hu 2022 val data
        val_df = pd.read_csv("hu_2022_valdata_sEMG.csv", header=None, dtype=float)
        val_df = val_df.iloc[:, 8:]
        val_df = val_df.to_numpy()
        # Denormalise data
        val_df = (val_df * stds_l) + means_l
        # Get MAE of relevant indices
        print("MAE:")
        print(np.mean(np.abs(torch_result - val_df)))

        # Concat both together
        concatenated_result = np.concatenate((torch_result, val_df), axis=1)

        # Convert to csv
        df = pd.DataFrame(concatenated_result)
        df.to_csv("data/RawEMG-2023-03-22-13.15.34-predicted.csv", index=True, header=False)
