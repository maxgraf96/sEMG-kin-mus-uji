import torch
import onnxruntime
import numpy as np
import onnx2tf
import tensorflow as tf
from tensorflow.lite.python import interpreter as tflite_interpreter

# class Model(torch.nn.Module):
#     def forward(self, x, y):
#         return {
#             "add": x + y,
#             "sub": x - y,
#         }

# # Let's double check what PyTorch gives us
# model = Model()
# pytorch_output = model.forward(10, 2)
# print("[PyTorch] Model Predictions:", pytorch_output)

# # First, export the above model to ONNX
# torch.onnx.export(
#     Model(),
#     {"x": 10, "y": 2},
#     "model.onnx",
#     opset_version=16,
#     input_names=["x", "y"],
#     output_names=["add", "sub"],
# )

# # And check its output
# session = onnxruntime.InferenceSession("model.onnx")
# onnx_output = session.run(["add", "sub"], {"x": np.array(10), "y": np.array(2)})
# print("[ONNX] Model Outputs:", [o.name for o in session.get_outputs()])
# print("[ONNX] Model Predictions:", onnx_output)

# # Now, let's convert the ONNX model to TF
onnx2tf.convert(
    input_onnx_file_path="model.onnx",
    output_folder_path="model.tf",
    output_signaturedefs=True,
    copy_onnx_input_output_names_to_tflite=False,
    non_verbose=True,
)



# converter = tf.lite.TFLiteConverter.from_saved_model("model.tf")
# converter.optimizations = [tf.lite.Optimize.DEFAULT] # optional
# converter.target_spec.supported_types = [tf.float16] # optional
# tflite_file_path = "model.tflite"
# with open(tflite_file_path, "wb") as f:
#     f.write(converter.convert())


# Load a TF model and allocate tensors.
tf.lite.experimental.Analyzer.analyze(model_path="model.tflite",
                                      model_content=None,
                                      gpu_compatibility=False)



# Now, test the newer TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
siglist = interpreter.get_signature_list()
print(siglist)


tf_lite_model = interpreter.get_signature_runner()
tt_lite_output = tf_lite_model(
    x=tf.constant((10,), dtype=tf.int64),
    y=tf.constant((2,), dtype=tf.int64),
)
print("[TFLite] Model Predictions:", tt_lite_output)