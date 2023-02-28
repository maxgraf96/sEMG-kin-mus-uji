import torch
# import onnx
import coremltools as ct

from sEMGTransformer import SEMGTransformer

if __name__ == '__main__':
    model = SEMGTransformer()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    
    input_sample = torch.randn((1024, 100, 18))
    tgt_sample = torch.randn((1024, 100, 18))
    
    # Trace model
    traced_model = model.to_torchscript(method="trace", example_inputs=(input_sample, tgt_sample))
    out = traced_model(input_sample, tgt_sample)
    print(out.shape)

    # Convert to Core ML program using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_sample.shape), ct.TensorType(shape=tgt_sample.shape)]
    )
    # Save the converted model.
    model.save("model.mlpackage")

    # Export to ONNX
    # filepath = "model.onnx"
    # model.to_onnx(filepath, (input_sample, tgt_sample), export_params=True)

    # Load with onnx
    # onnx_model = onnx.load("model.onnx")
    # onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))