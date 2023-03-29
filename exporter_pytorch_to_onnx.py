import time
import torch
import onnx

from sEMGTransformer import SEMGTransformer, model_name
from hyperparameters import SEQ_LEN

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

if __name__ == '__main__':
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
        torch.backends.cuda.enable_math_sdp(False)
        def sdp_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            L, S = Q.size(-2), K.size(-2)
            scale_factor = 1# / sqrt(Q.size(-1)) if scale is None else scale
            if attn_mask is not None:
            # attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
                attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask is not None and attn_mask.dtype==torch.bool else attn_mask
                attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax(Q @ K.transpose(-2, -1) * scale_factor, dim=-1)
            # Don't need dropout for inference
            # attn_weight = torch.nn.functional.dropout(attn_weight, dropout_p)
            return attn_weight @ V
        
        torch.nn.functional.scaled_dot_product_attention = sdp_attention
        model = SEMGTransformer().to(torch.device("cpu"))
        # model = torch.compile(model, backend="cpu")
        model.load_state_dict(torch.load(model_name + ".pt"))

        # Hu 2022 config
        input_sample = torch.randn((1, SEQ_LEN, 16))#.to(device)
        tgt_sample = torch.randn((1, SEQ_LEN, 16))#.to(device)
        tgt_mask_sample = model.get_tgt_mask(SEQ_LEN)
        
        # Script model
        # Scripting does not work because transformer layers are not yet supported

        # Take time
        current_time = time.time()

        # Export to ONNX
        print("Exporting to ONNX...")
        filepath = model_name + ".onnx"
        
        # torch.onnx.export(
        #     model, 
        #     (input_sample, tgt_sample, tgt_mask_sample), 
        #     filepath, 
        #     export_params=True, 
        #     opset_version=16, 
        #     input_names=["input", "tgt", "tgt_mask"], 
        #     output_names=["output"],
        #     dynamic_axes={"tgt": {1: "sequence_length"}, "tgt_mask": {0: "height", 1: "width"}, "output": {1: "sequence_length"}},
        #     verbose=True)

        model.to_onnx(
            filepath, 
            (input_sample, tgt_sample, tgt_mask_sample), 
            export_params=True, 
            opset_version=16, 
            input_names=["input", "tgt", "tgt_mask"], 
            output_names=["output"],
            dynamic_axes={"tgt": {1: "sequence_length"}, "tgt_mask": {0: "height", 1: "width"}, "output": {1: "sequence_length"}},
            verbose=True)

        # Print time difference in seconds
        print("Time taken:", time.time() - current_time)

        print("Success! Loading ONNX model...")
        # Load with onnx
        onnx_model = onnx.load(model_name + ".onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX model loaded successfully!")
        # print(onnx.helper.printable_graph(onnx_model.graph))