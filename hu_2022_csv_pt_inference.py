from tqdm import tqdm
from hyperparameters import SEQ_LEN
from sEMGTransformer import SEMGTransformer, model_name
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__ == "__main__":
    model = SEMGTransformer().to(device)
    # model = torch.compile(model, backend="cpu")
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

    sos_token = (torch.ones(input_tensor.shape[0], 1, 16) * -100).to(device)
    # Add sos token to every sequence
    input_torch = torch.cat((sos_token, input_tensor), axis=1)
    x = input_torch[:, :SEQ_LEN, :]  # First samples for autoregressive prediction
    tgt_torch = torch.tensor((sos_token)).to(device)
    # tgt_mask_torch = model.get_tgt_mask(SEQ_LEN).to(device)
    tgt_mask_torch = model.get_tgt_mask(input_tensor.shape[1]).to(device)

    seq_len = SEQ_LEN - 1

    with torch.no_grad():
        # # Initial autoregressive prediction
        for length in tqdm(range(1, SEQ_LEN)):
        # for length in tqdm(range(1, input_tensor.shape[1])):
            pred = model(x, tgt_torch, tgt_mask_torch[:length, :length])
            # pred = model(input_tensor, tgt_torch, tgt_mask_torch[:length, :length])
            tgt_torch = torch.cat((tgt_torch, pred[:, -1:, :]), axis=1)

        # Now continue prediction
        window_size = SEQ_LEN - 1
        temp_tgt_mask = tgt_mask_torch[:window_size, :window_size]
        for index in tqdm(range(1, input_tensor.shape[1] - window_size)):
            temp_input = input_torch[:, index : index + window_size, :]
            temp_tgt = tgt_torch[:, index : -1, :]

            # Add sos token to every sequence
            temp_input = torch.cat((sos_token, temp_input), axis=1)
            temp_tgt = torch.cat((sos_token, temp_tgt), axis=1)

            pred = model(temp_input, temp_tgt, temp_tgt_mask)
            tgt_torch = torch.cat((tgt_torch, pred[:, -1:, :]), axis=1)

        print("Torch result:")
        torch_result = tgt_torch.cpu().detach().numpy()[:, :, :15]
        print(torch_result)

        # Convert to csv
        df = pd.DataFrame(torch_result.reshape(-1, 15))
        df.to_csv("data/RawEMG-2023-03-22-13.15.34-predicted.csv", index=True, header=False)
