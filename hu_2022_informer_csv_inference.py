from tqdm import tqdm
from hyperparameters import SEQ_LEN
import torch
import pandas as pd
from transformers import GenerationConfig
from sEMGInformer import SEMGInformer, model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__ == "__main__":
    model = SEMGInformer(inference=True).to(device)
    model.load_state_dict(torch.load(model_name + ".pt"))
    model.eval()

    # Load data from csv
    # Our recorded data
    df = pd.read_csv("data/RawEMG-2023-03-22-13.15.34.csv", skiprows=0, usecols=range(8), dtype=float)
    # Data from Hu et al. 2022 validation set - updates every time we train the model
    # df = pd.read_csv("hu_2022_valdata_sEMG.csv", usecols=range(8), dtype=float)

    # Take all rows and first 8 columns and convert to tensor
    input_tensor = torch.tensor(df.iloc[:, :8].values, dtype=torch.float32)[:4000, :]
    # Add padding
    input_tensor = torch.nn.functional.pad(input_tensor, (0, 8), "constant", 0)

    # Add batch dimension and reshape from (4000, 16) to (1, SEQ_LEN, 16)
    input_tensor = input_tensor.reshape(1, -1, 16).to(device)

    configuration = model.model.config
    seq_len = SEQ_LEN - 1
    input_dim = configuration.input_size
    output_dim = input_dim
    bs = 1

    time_features = torch.linspace(0, 1, seq_len).reshape(1, seq_len, 1).to(device)
    time_features = time_features.repeat(bs, 1, input_dim)
    
    past_observed_mask = torch.ones(bs, seq_len, input_dim, dtype=torch.bool).to(device)
    
    future_time_features = torch.linspace(0, 1, model.prediction_length).reshape(1, model.prediction_length, 1)
    future_time_features = future_time_features.repeat(bs, 1, output_dim).to(device)

    result = torch.zeros((bs, 1, output_dim)).to(device)

    with torch.no_grad():
        for length in tqdm(range(1, input_tensor.shape[1] - seq_len)):
        # for length in tqdm(range(1, 500)):
            past_values = input_tensor[:, length:length + seq_len, :]

            output = model.generate(
                past_values=past_values, 
                past_time_features=time_features, 
                future_time_features=future_time_features,
                past_observed_mask=past_observed_mask, 
            )

            result = torch.cat((result, output[:, -1:, :]), dim=1)
            

        print("Torch result:")
        torch_result = result.cpu().detach().numpy()[:, :, :15]
        print(torch_result)

        # Convert to csv
        df = pd.DataFrame(torch_result.reshape(-1, 15))
        df.to_csv("data/RawEMG-2023-03-22-13.15.34-predicted.csv", index=True, header=False)
