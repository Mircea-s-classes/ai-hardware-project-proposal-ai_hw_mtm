import numpy as np
import tensorflow as tf
import tempfile
import soundfile as sf
import librosa
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

#Constants
tf.keras.backend.set_floatx('float32')
INPUT_SIZE = 50
CONV_KERNEL_SIZE_X = 5
CONV_OUTPUT_SIZE_X = 46  #After applying 5x1 kernel to 50 input (valid padding)
FC_SIZE = 32
OUTPUT_SIZE = 1  #Output size for binary classification

#Sigmoid and ReLU activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    if(x<=0):
        return 0.0 
    else:
        return x.astype(np.float32)

def depthwise_conv2d(input, kernel, input_width, kernel_width):
    #Calculate the output width (valid padding)
    output_width = input_width - kernel_width + 1
    output = np.zeros(output_width)  #Convolution output (1D)

    for i in range(output_width):
        sum = 0.0
        for k in range(kernel_width):
            sum += input[i + k] * kernel[k]  #Perform 1D convolution (sliding window)
        #print("running sum: ", sum)
        output[i] = relu(sum)  #Apply ReLU after convolution
    return output

#Fully connected layer with ReLU activation
def fully_connected_relu(input, weights, bias, output, input_size, output_size):
    for i in range(output_size):
        sum = bias[i]  # Start with the bias for this output unit
        for j in range(input_size):
            weight_index = i + j * output_size  # Calculate index for row-major (C-order) flattened weights
            sum += input[j] * weights[weight_index]  # Access weight from the flattened array
        output[i] = relu(sum)  # Apply ReLU activation function
    return output


#Fully connected layer with Sigmoid activation
def fully_connected_sigmoid(input, weights, bias, output, input_size, output_size):
    for i in range(output_size):
        sum_value = bias  # Start with the bias for this output unit
        for j in range(input_size):
            weight_index = i + j * output_size  # Calculate index for row-major (C-order) flattened weights
            sum_value += input[j] * weights[weight_index]  # Access weight from the flattened array
        output[i] = sigmoid(sum_value)  # Apply Sigmoid activation function
    return output


#Example input and kernel
input = np.array([-444.95856, 58.542336, 18.05262, 30.173512, 0.6866538, 15.617227, -12.520357, 3.9046752, -3.627293, 6.5801973, -8.113564, 13.387428, -12.955448, 5.8316474, -4.4946904, -2.3746157, -5.7689943, 2.4510553, -8.2580185, -0.13677117, -2.0417533, 2.2710068, -3.1866713, -1.6654011, -3.1574187, 0.053224698, -5.149588, 0.23926407, -4.5624046, -2.6701326, -1.4743314, -2.0951657, -1.0312055, -1.6219964, -4.158239, -0.04374194, -0.48332465, -1.1609437, 0.90399617, -0.28432813, -3.1150825, -1.4177009, -2.0603538, -1.39741, -1.2998035, 0.017326394, -2.6161382, -1.4018486, -1.5016721, 0.40286532
])
input=input.reshape(50,)
#print("input:", input) 
kernel = np.genfromtxt('conv_weights_float.txt', delimiter=',', dtype=np.float32)
#print("kernel: ", kernel)
conv_output = np.zeros((CONV_OUTPUT_SIZE_X, 1))  #Convolution output (1D)
flattened = np.zeros(CONV_OUTPUT_SIZE_X, dtype=float)  #Flattened size
#Weights and biases (example)
fc_bias = np.genfromtxt('fc_bias_float.txt', delimiter=',')
#print("fc_Bias: ", fc_bias)
output_weights = np.genfromtxt('fc_weights2_float.txt', delimiter=',', dtype=np.float32)
#print("output weights: ", output_weights)
output_bias = np.genfromtxt('fc_bias2_float.txt', delimiter=',', dtype=np.float32)
#print("output_bias", output_bias)
fc_weights = np.genfromtxt('fc_weights_float.txt', delimiter=',', dtype=np.float32)
#Output
fc_output = np.zeros(FC_SIZE)
output = np.zeros(OUTPUT_SIZE)


#Perform convolution and flatten
conv_output = depthwise_conv2d(input, kernel, INPUT_SIZE, CONV_KERNEL_SIZE_X)
flattened = conv_output.flatten()
fully_connected_relu(flattened, fc_weights, fc_bias, fc_output, CONV_OUTPUT_SIZE_X, FC_SIZE)
fully_connected_sigmoid(fc_output, output_weights, output_bias, output, FC_SIZE, OUTPUT_SIZE)

print("Final output:", output)

########################END OF MY CUSTOM IMPLEMENTATION#########################################


#Define the model with the same architecture
model = models.Sequential([

    layers.Reshape((1, 50, 1), input_shape=(50,)),  #Reshape input

    #Depthwise Convolutional Layer
    layers.DepthwiseConv2D(kernel_size=(1, 5), depth_multiplier=1, padding='valid', activation='relu', use_bias=False),

    #Flatten the output from 4D to 1D
    layers.Flatten(),

    #Fully connected layer with 32 units
    layers.Dense(32, activation='relu'),

    #Output layer with sigmoid (binary classification)
    layers.Dense(1, activation='sigmoid')  #'Yes' vs 'Not Yes' classification

])

#Load the weights from saved files
conv_weightsTF = np.genfromtxt("conv_weights_float.txt", delimiter=',')
fc_weightsTF = np.genfromtxt("fc_weights_float.txt", delimiter=',')
fc_biasTF = np.genfromtxt("fc_bias_float.txt", delimiter=',')
fc_weights2TF = np.genfromtxt("fc_weights2_float.txt", delimiter=',')
fc_bias2TF = np.genfromtxt("fc_bias2_float.txt", delimiter=',')

conv_weightsTF = conv_weightsTF.reshape(1, 5, 1, 1)
fc_weightsTF = fc_weightsTF.reshape(46, 32)
fc_weights2TF = fc_weights2TF.reshape(32, 1)
fc_bias2TF = fc_bias2TF.reshape(1,)


#print("Conv Weights:", conv_weightsTF)
#print("FC Weights:", fc_weightsTF)
#print("FC Bias:", fc_biasTF)



#Set the weights in the model
model.layers[1].set_weights([conv_weightsTF])  #Convolutional layer weights
model.layers[3].set_weights([fc_weightsTF, fc_biasTF])  #Fully connected layer 1 weights and bias
model.layers[4].set_weights([fc_weights2TF, fc_bias2TF])  #Fully connected layer 2 weights and bias

#Add a batch dimension to the input
input_with_batch = np.expand_dims(input, axis=0)  #Shape will be (1, 50)

#Explicitly call the model with the input to ensure initialization
model(input_with_batch)
#Explicitly call the model with the input to ensure initialization
model.build(input_shape=(None, 50))  #This ensures the model is initialized

#Create a new model to output intermediate results

intermediate_layer_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        model.layers[1].output,  #Convolutional layer output
        model.layers[2].output,  #Flatten layer output
        model.layers[3].output,  #Fully connected (ReLU) layer output
        model.layers[4].output,  #Fully connected (Sigmoid) layer output
    ]
)

#Get the intermediate outputs for the input
example_input = np.expand_dims(input, axis=(0, 2))  #Add batch and channel dimensions (Shape: (1, 50, 1))

intermediate_outputs = intermediate_layer_model.predict(example_input)

#Print the outputs of each layer
conv_output_tf = intermediate_outputs[0]
flattened_tf = intermediate_outputs[1]
fc_relu_output_tf = intermediate_outputs[2]
final_output_tf = intermediate_outputs[3]

#print("TensorFlow Conv Output:", conv_output_tf)
#print("TensorFlow Flattened Output:", flattened_tf)
#print("TensorFlow FC ReLU Output:", fc_relu_output_tf)

# Get weights and biases of the 3rd layer (Dense layer)
#weights, biases = model.layers[3].get_weights()
#Print weights and biases
#print("Weights:", weights)
#print("Biases:", biases)

# Example for first unit in FC layer
# custom_sum = fc_bias[0] + np.sum(flattened * fc_weights[:, 0])
# print("Custom Sum:", custom_sum)
# tf_sum = np.dot(flattened_tf, model.layers[3].get_weights()[0][:, 0]) + model.layers[3].get_weights()[1][0]
# print("TensorFlow Sum:", tf_sum)

print("Custom FC ReLU Output:", fc_output)
print("TensorFlow FC ReLU Output:", fc_relu_output_tf)

print("TensorFlow Final Output:", final_output_tf)
print("(Custom) Final output:", output)




