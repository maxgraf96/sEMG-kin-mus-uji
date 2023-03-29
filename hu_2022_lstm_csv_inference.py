from tqdm import tqdm
from hyperparameters import SEQ_LEN
from sEMGLSTMAttention import SEMGLSTMAttention, model_name
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__ == "__main__":
    model = SEMGLSTMAttention().to(device)
    # model = torch.compile(model, backend="cpu")
    model.load_state_dict(torch.load(model_name + ".pt"))
    model.eval()

    # Load data from csv
    df = pd.read_csv("data/RawEMG-2023-03-22-13.15.34.csv", skiprows=0, usecols=range(8), dtype=float)
    # Take all rows and first 8 columns and convert to tensor
    input_tensor = torch.tensor(df.iloc[:, :8].values, dtype=torch.float32)[:4000, :]
    # Add padding
    input_tensor = torch.nn.functional.pad(input_tensor, (0, 8), "constant", 0)

    # Add batch dimension and reshape from (4000, 16) to (1, SEQ_LEN, 16)
    input_tensor = input_tensor.reshape(1, -1, 16).to(device)

    seq_len = SEQ_LEN - 1

    with torch.no_grad():
        seq_lengths = (torch.ones((1)) * seq_len).to(device)
        for index in tqdm(range(0, input_tensor.shape[1] - seq_len)):
            x = input_tensor[:, index : index + seq_len, :]
            pred = model(x, seq_lengths, is_training=False)
            if index == 0:
                tgt_torch = pred
            else:
                tgt_torch = torch.cat((tgt_torch, pred), axis=0)

        print("Torch result:")
        torch_result = tgt_torch.cpu().detach().numpy()[:, :15]
        print(torch_result)

        # Convert to csv
        df = pd.DataFrame(torch_result.reshape(-1, 15))
        df.to_csv("data/RawEMG-2023-03-22-13.15.34-predicted.csv", index=False, header=False)
