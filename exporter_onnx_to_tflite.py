import os
from hyperparameters import SEQ_LEN
import tensorflow as tf
import onnx2tf

from onnx_tf.backend import prepare
from sEMGTransformer import model_name

model_tf_folder = model_name + "_tf"
tflite_file_path = os.path.join(model_tf_folder, model_name + "_correct.tflite")

if __name__ == "__main__":

    # This call converts both to tf and tflite, but the tflite model is faulty (no input/output names)
    onnx2tf.convert(
        input_onnx_file_path=model_name + ".onnx",
        output_folder_path=model_tf_folder,
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
        keep_shape_absolutely_input_names=["input", "tgt", "tgt_mask"],
        non_verbose=False,
    )

    # Hence, we need to convert the tf model to tflite separately
    converter = tf.lite.TFLiteConverter.from_saved_model(model_tf_folder)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # optional
    converter.target_spec.supported_types = [tf.float16] # optional
    with open(tflite_file_path, "wb") as f:
        f.write(converter.convert())

    tf.lite.experimental.Analyzer.analyze(model_path=tflite_file_path,
                                        model_content=None,
                                        gpu_compatibility=False)

    # Load a TF model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
    
    signature_list = interpreter.get_signature_list()
    print(signature_list)

    output_details = interpreter.get_output_details()[0]  # Model has single output.
    
    tgt_mask_details = interpreter.get_input_details()[0]
    input_details = interpreter.get_input_details()[1]
    tgt_details = interpreter.get_input_details()[2]
    # Print input shapes and types
    print("Input details:")
    print("name:", tgt_mask_details['name'], "index:", tgt_mask_details['index'], "shape:", tgt_mask_details['shape'], "type:", tgt_mask_details['dtype'])
    print("name:", input_details['name'], "index:", input_details['index'], "shape:", input_details['shape'], "type:", input_details['dtype'])
    print("name:", tgt_details['name'], "index:", tgt_details['index'], "shape:", tgt_details['shape'], "type:", tgt_details['dtype'])

    # return shape is currently seq_len 10 because we limited the input_seq_len to 10
    input_data = tf.random.uniform((1, SEQ_LEN, 16), dtype=tf.float32)
    tgt_data = tf.random.uniform((1, SEQ_LEN, 16), dtype=tf.float32)
    tgt_mask_data = tf.random.uniform((SEQ_LEN, SEQ_LEN), dtype=tf.float32)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! In tflite we have to resize the input tensors to the correct shape for all axes that have dynamic size !!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    interpreter.resize_tensor_input(tgt_mask_details['index'], tgt_mask_data.shape)
    interpreter.resize_tensor_input(tgt_details['index'], tgt_data.shape)
    interpreter.allocate_tensors()
    
    # Set data
    interpreter.set_tensor(tgt_mask_details['index'], tgt_mask_data)
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.set_tensor(tgt_details['index'], tgt_data)

    # Take time
    import time
    start = time.time()
    for i in range(SEQ_LEN):
        interpreter.invoke()
        output_shape = interpreter.get_tensor(output_details['index']).shape
    end = time.time()
    print("Time:", end - start)

    print("[TFLite] Model output shape:", output_shape)