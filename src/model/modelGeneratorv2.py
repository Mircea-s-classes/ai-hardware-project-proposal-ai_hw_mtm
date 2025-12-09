import tensorflow as tf
import numpy as np
from pydub import AudioSegment
import tempfile
import soundfile as sf
import librosa
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load the Speech Commands dataset
dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True)

# Extract the train and test sets
train_data = dataset['train'].take(1000)
test_data = dataset['test'].take(1000)

# Define the model
model = models.Sequential([
    # Reshape input from [490] to [49, 10, 1]
    layers.Reshape((49, 10, 1), input_shape=(490,)),

    # Depthwise Convolutional Layer (1 input channel, 4 output channels, 5x4 kernel size)
    layers.DepthwiseConv2D(kernel_size=(5, 4), depth_multiplier=1, padding='valid', activation='relu', input_shape=(49, 10, 1)),

    # Flatten the output from 4D to 1D
    layers.Flatten(),

    # Fully connected layer with 4 units
    layers.Dense(4, activation='relu'),

    # Output layer with sigmoid (binary classification)
    layers.Dense(1, activation='sigmoid')  # Using sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Function to extract features from the audio file
def extract_features(audio_data, sample_rate=16000):
    # Create a temporary file to save the audio clip
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        # Convert tensor to numpy array
        audio_array = audio_data.numpy().astype(np.int16)
        sf.write(tmpfile.name, audio_array, sample_rate, format='WAV')  # Save audio to a .wav file
        # Now load the audio using librosa
        audio, sr = librosa.load(tmpfile.name, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCCs
        mfccs_mean = np.mean(mfccs, axis=1)  # Take mean across time frames for each MFCC
    return mfccs_mean

# Assuming the dataset has integer labels and you have a mapping of class indices to names:
labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', '_silence_', '_unknown_']

def label_data(dataset):
    X = []
    y = []
    # Print the class names
    print("Label info:", info.features['label'])
    print("Number of classes:", info.features['label'].num_classes)
    # Get all class labels from the dataset info
    labels = info.features['label'].names
    print("Class labels:", labels)

    # Find the index of 'yes'
    yes_index = labels.index('yes')
    print(f"Index of 'yes': {yes_index}")


    for audio, label in dataset:
        label_value = label.numpy()  # Convert tensor to integer
        #print(f"Label value (integer): {label_value}")

        # Map the integer label to its corresponding string class
        label_str = labels[label_value]  # Get the class name using the integer index
        #print(f"Mapped label: {label_str}")

        features = extract_features(audio)  # Extract features from the audio

        # Label the data based on the string label
        if label_str == 'yes':
            X.append(features)
            y.append(1)  # 'Yes' command is the positive class
        else:
            X.append(features)
            y.append(0)  # All other commands are noise (negative class)

    return np.array(X), np.array(y)


# Label and prepare training and testing data
X_train, y_train = label_data(train_data)
X_test, y_test = label_data(test_data)

# Ensure the feature vectors are of shape (490,) for each input
# You can pad or trim MFCC features if needed, but we'll assume 490 dimensions for this model
X_train = np.pad(X_train, ((0, 0), (0, 490 - X_train.shape[1])), mode='constant')
X_test = np.pad(X_test, ((0, 0), (0, 490 - X_test.shape[1])), mode='constant')

# Train the model
model.fit(X_train, y_train, epochs=75, batch_size=32, validation_data=(X_test, y_test))

# Get the weights from layers
conv_weights = model.layers[1].get_weights()[0]  # Convolution weights (depthwise)
conv_bias = model.layers[1].get_weights()[1]  # Convolution biases
fc_weights = model.layers[3].get_weights()[0]  # Fully connected layer weights
fc_bias = model.layers[3].get_weights()[1]  # Fully connected layer biases
fc_weights2 = model.layers[4].get_weights()[0]  # Fully connected layer weights
fc_bias2 = model.layers[4].get_weights()[1]  # Fully connected layer biases
# Optionally, quantize the weights and biases
def quantize_weights(weights, num_bits=8):
    scale = np.max(np.abs(weights))  # Scaling factor
    quantized_weights = np.round(weights * (2**(num_bits - 1)) / scale)
    return quantized_weights.astype(np.int8), scale

# Quantize weights
conv_weights_quantized, conv_scale = quantize_weights(conv_weights)
conv_bias_quantized, _ = quantize_weights(conv_bias)
fc_weights_quantized, fc_scale = quantize_weights(fc_weights)
fc_bias_quantized, _ = quantize_weights(fc_bias)
fc_weights_quantized2, fc_scale = quantize_weights(fc_weights2)
fc_bias_quantized2, _ = quantize_weights(fc_bias2)

# Print or save the quantized weights and biases
#print("Quantized Conv Weights:", conv_weights_quantized)
#print("Quantized Conv Bias:", conv_bias_quantized)
#print("Quantized FC Weights:", fc_weights_quantized)
#print("Quantized FC Bias:", fc_bias_quantized)

# Save the quantized FC weights to a text file
with open("fc_weights.txt", "w") as f:
    f.write("Quantized FC Weights: ")
    f.write(", ".join(map(str, fc_weights_quantized.flatten())))

with open("fc_bias.txt", "w") as f:
    f.write("Quantized FC Bias: ")
    f.write(", ".join(map(str, fc_bias_quantized.flatten())))

with open("Conv_weights.txt", "w") as f:
    f.write("Quantized Conv Weights: ")
    f.write(", ".join(map(str, conv_weights_quantized.flatten())))

with open("Conv_bias.txt", "w") as f:
    f.write("Quantized Conv Bias: ")
    f.write(", ".join(map(str, conv_bias_quantized.flatten())))

with open("fc_bias2.txt", "w") as f:
    f.write("Quantized FC Bias2: ")
    f.write(", ".join(map(str, fc_bias_quantized2.flatten())))

with open("fc_weights2.txt", "w") as f:
    f.write("Quantized FC Weights2: ")
    f.write(", ".join(map(str, fc_weights_quantized2.flatten())))


