import os
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
from sEMGTransformer import SEMGTransformer, model_name
from hyperparameters import SEQ_LEN

model_tf_folder = model_name + "_tf"
tflite_file_path = os.path.join(model_tf_folder, model_name + "_correct.tflite")

sos_token = torch.unsqueeze(torch.ones(1, 16) * -100, 0)

if __name__ == "__main__":
    # ===================================================== DATA =======================================================
    input_tensor = np.random.rand(1, SEQ_LEN - 1, 16).astype(np.float32)
    tgt_mask_tensor = np.ones((SEQ_LEN, SEQ_LEN)).astype(np.float32)

    # ===================================================== TORCH ======================================================
    # model = SEMGTransformer().to(torch.device("cpu"))
    # # model = torch.compile(model, backend="cpu")
    # model.load_state_dict(torch.load(model_name + ".pt"))
    
    # input_torch = torch.cat((sos_token, torch.tensor((input_tensor))), axis=1)
    # tgt_torch = torch.tensor((sos_token))

    # for length in range(1, SEQ_LEN):
    #     tgt_mask_torch = torch.ones((length, length), dtype=torch.float32)
    #     pred = model(input_torch, tgt_torch, tgt_mask_torch)
    #     tgt_torch = torch.cat((tgt_torch, pred[:, -1:, :]), axis=1)

    # print("Torch result:")
    # torch_result = tgt_torch.detach().numpy()
    # print(torch_result)


    # # ===================================================== ONNX =======================================================
    session = ort.InferenceSession(model_name + ".onnx")
    inputs = session.get_inputs()
    output_name = session.get_outputs()[0].name
    input_numpy = np.concatenate((sos_token.numpy(), input_tensor), axis=1)
    
    tgt_numpy = sos_token.numpy()

    for length in range(1, SEQ_LEN):
        tgt_mask_tensor = np.ones((length, length)).astype(np.float32)
        
        result = session.run([output_name], {inputs[0].name: input_numpy, inputs[1].name: tgt_numpy, inputs[2].name: tgt_mask_tensor})

        result_numpy = np.array(result[0])
        result_numpy = result_numpy[:, -1:, :]
        tgt_numpy = np.concatenate((tgt_numpy, result_numpy), axis=1)

    onnx_result = tgt_numpy
        
    print("ONNX result:")
    print(onnx_result)

    # # Check if torch and onnx results are equal
    # torch_onnx_close = np.allclose(torch_result, onnx_result, rtol=0.5)
    # # Print last 10 values of both predictions
    # print("Torch prediction last 10 values:")
    # print(torch_result[:, -10:, 0])
    # print("ONNX prediction last 10 values:")
    # print(onnx_result[:, -10:, 0])
    # print("Torch and ONNX results are equal:", torch_onnx_close)
    # # Calculate the mean absolute error
    # torch_onnx_mae = np.mean(np.abs(torch_result - onnx_result))
    # print("Torch and ONNX MAE:", torch_onnx_mae)

    # exit the script here
    # exit()


    # ===================================================== TFLite =====================================================
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

    # return shape is currently SEQ_LEN 10 because we limited the input_seq_len to 10
    input_data = tf.convert_to_tensor(input_tensor, dtype=np.float32)
    tgt_mask_data = tf.convert_to_tensor(tgt_mask_tensor, dtype=tf.float32)


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! In tflite we have to resize the input tensors to the correct shape for all axes that have dynamic size !!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tgt_tf = sos_token.numpy()
    input_tf = tf.convert_to_tensor(np.concatenate((sos_token.numpy(), input_tensor), axis=1), dtype=tf.float32)

    for length in range(1, SEQ_LEN):
        tgt_mask_tensor = np.ones((length, length)).astype(np.float32)
        # mask = np.tril(np.ones((length, length)), k=0).astype(np.float32)
        # # Set zeros to -inf and ones to 0
        # tgt_mask_tensor = np.where(mask == 0, -np.inf, 0).astype(np.float32)
        # tgt_mask_tensor = np.where (mask == 1, 0, tgt_mask_tensor).astype(np.float32)

        tgt_data = tf.convert_to_tensor(tgt_tf, dtype=tf.float32)
        tgt_mask_data = tf.convert_to_tensor(tgt_mask_tensor, dtype=tf.float32)

        interpreter.resize_tensor_input(tgt_mask_details['index'], tgt_mask_data.shape)
        interpreter.resize_tensor_input(tgt_details['index'], tgt_data.shape)
        interpreter.allocate_tensors()
        
        # Set data
        interpreter.set_tensor(tgt_mask_details['index'], tgt_mask_data)
        interpreter.set_tensor(input_details['index'], input_tf)
        interpreter.set_tensor(tgt_details['index'], tgt_data)

        interpreter.invoke()
        prediction_tflite = interpreter.get_tensor(output_details['index'])

        prediction_tflite = prediction_tflite[:, -1:, :]
        tgt_tf = np.concatenate((tgt_tf, prediction_tflite), axis=1)
    
    tflite_result = tgt_tf
    print("TFLite reusult:")
    print(tflite_result)

    # ===================================================== COMPARISON =====================================================
    print("ONNX and TFLite prediction equal:", np.allclose(onnx_result, tflite_result, rtol=3.0))