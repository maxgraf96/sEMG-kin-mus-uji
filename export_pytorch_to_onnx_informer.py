import time
import torch
import onnx

from sEMGInformer import SEMGInformer, model_name
from hyperparameters import SEQ_LEN

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

if __name__ == '__main__':
    model = SEMGInformer(batch_size=1).to(torch.device("cpu"))
    # model = torch.compile(model, backend="cpu")
    model.load_state_dict(torch.load(model_name + ".pt"))
    model.eval()

    configuration = model.model.config
    
    seq_len = SEQ_LEN - 1
    input_dim = configuration.input_size
    output_dim = input_dim
    bs = 1

    past_values = torch.rand(bs, seq_len, input_dim)
    past_time_features = torch.linspace(0, 1, seq_len).reshape(1, seq_len, 1)
    past_time_features = past_time_features.repeat(bs, 1, input_dim)
    # past_observed_mask = torch.ones(bs, seq_len, input_dim, dtype=torch.bool)

    future_time_features = torch.ones(bs, model.prediction_length, output_dim)
    future_values = torch.ones(past_values.shape[0], model.prediction_length, past_values.shape[2])
    
    output = model(
        past_values=past_values, 
        past_time_features=past_time_features, 
        # past_observed_mask=past_observed_mask,
        future_values=future_values,
        future_time_features=future_time_features
    )

    # Take time
    current_time = time.time()

    # Export to ONNX
    print("Exporting to ONNX...")
    filepath = model_name + ".onnx"
    
    model.to_onnx(
        filepath, 
        (past_values, past_time_features, future_values, future_time_features), 
        export_params=True, 
        opset_version=16, 
        input_names=["past_values", "past_time_features", "future_values", "future_time_features"], 
        output_names=["output"],
        # dynamic_axes={"tgt": {1: "sequence_length"}, "tgt_mask": {0: "height", 1: "width"}, "output": {1: "sequence_length"}},
        verbose=True)

    # Print time difference in seconds
    print("Time taken:", time.time() - current_time)

    print("Success! Loading ONNX model...")
    # Load with onnx
    onnx_model = onnx.load(model_name + ".onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model loaded successfully!")
    # Print onnx inputs and outputs
    print("Inputs:")
    print(onnx_model.graph.input)
    print("Outputs:")
    print(onnx_model.graph.output)
    # print(onnx.helper.printable_graph(onnx_model.graph))