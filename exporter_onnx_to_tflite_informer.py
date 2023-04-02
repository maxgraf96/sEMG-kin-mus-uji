import os
from hyperparameters import SEQ_LEN
import tensorflow as tf
import onnx2tf

from onnx_tf.backend import prepare
from sEMGInformer import model_name

model_tf_folder = model_name + "_tf"
tflite_file_path = os.path.join(model_tf_folder, model_name + "_correct.tflite")

if __name__ == "__main__":

    # This call converts both to tf and tflite, but the tflite model is faulty (no input/output names)
    onnx2tf.convert(
        input_onnx_file_path=model_name + ".onnx",
        output_folder_path=model_tf_folder,
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        keep_shape_absolutely_input_names=["past_values", "past_time_features", "future_values", "future_time_features"],
        non_verbose=False,
    )

    # Hence, we need to convert the tf model to tflite separately
    converter = tf.lite.TFLiteConverter.from_saved_model(model_tf_folder)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # optional
    converter.target_spec.supported_types = [tf.float16] # optional
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        # tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
    with open(tflite_file_path, "wb") as f:
        f.write(converter.convert())

    tf.lite.experimental.Analyzer.analyze(model_path=tflite_file_path,
                                        model_content=None,
                                        gpu_compatibility=False)
    
    # exit()

    # Load a TF model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
    
    signature_list = interpreter.get_signature_list()
    print(signature_list)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input shapes and types
    print("Input details:")
    for entry in input_details:
        print(entry['name'], entry['index'], entry['shape'], entry['dtype'])
    
    # Print output shapes and types
    print("Output details:")
    for entry in output_details:
        print(entry['name'], entry['index'], entry['shape'], entry['dtype'])

    # return shape is currently seq_len 10 because we limited the input_seq_len to 10
    seq_len = SEQ_LEN - 1
    feature_dim = 16
    past_values = tf.random.uniform((1, seq_len, feature_dim), dtype=tf.float32)
    past_time_features = tf.random.uniform((1, seq_len, feature_dim), dtype=tf.float32)
    future_values = tf.random.uniform((1, seq_len, feature_dim), dtype=tf.float32)
    future_time_features = tf.random.uniform((1, seq_len, feature_dim), dtype=tf.float32)

    # !!! In tflite we have to resize the input tensors to the correct shape for all axes that have dynamic size !!!!!
    # interpreter.resize_tensor_input(tgt_mask_details['index'], tgt_mask_data.shape)
    # interpreter.resize_tensor_input(tgt_details['index'], tgt_data.shape)
    interpreter.allocate_tensors()
    
    # Set data
    interpreter.set_tensor(input_details[0]['index'], past_values)
    interpreter.set_tensor(input_details[1]['index'], future_values)
    interpreter.set_tensor(input_details[2]['index'], past_time_features)
    interpreter.set_tensor(input_details[3]['index'], future_time_features)

    # Take time
    import time
    start = time.time()
    for i in range(SEQ_LEN):
        interpreter.invoke()
        output_shape = interpreter.get_tensor(output_details[0]['index']).shape
    end = time.time()
    print("Time:", end - start)

    print("[TFLite] Model output shape:", output_shape)