import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="KWS_custom.tflite")
interpreter.allocate_tensors()

# Get tensor details (tensors represent weights, biases, etc.)
tensor_details = interpreter.get_tensor_details()

# Print the tensor details to understand the structure
for tensor in tensor_details:
    print(f"Tensor Name: {tensor['name']}, Shape: {tensor['shape']}, Type: {tensor['dtype']}")

# Now, let's extract and print the weights and biases
weights_and_biases = {}

# Get the weights/biases from the model (you can filter based on tensor names, e.g., weights and biases usually contain 'weight' or 'bias' in the name)
for tensor in tensor_details:
    if "weight" in tensor['name'].lower() or "bias" in tensor['name'].lower():
        tensor_data = interpreter.get_tensor(tensor['index'])
        weights_and_biases[tensor['name']] = tensor_data
        print(f"Tensor: {tensor['name']}, Data: {tensor_data[:10]}...")  # Print only the first 10 values for brevity

# Save the weights and biases to a file for future use
np.save("weights_and_biases.npy", weights_and_biases)

# Load the saved weights and biases
weights_and_biases = np.load("weights_and_biases.npy", allow_pickle=True).item()

# Generate C array format for weights and biases
def generate_c_array(name, data):
    c_array = f"const int8_t {name}[] = {{"
    c_array += ", ".join(map(str, data.astype(np.int8)))  # Ensure the data is in int8 format
    c_array += "};\n"
    return c_array

# Example: Generate C code for one of the tensors
for tensor_name, tensor_data in weights_and_biases.items():
    c_code = generate_c_array(tensor_name, tensor_data)
    print(c_code)  # Print the C array code

