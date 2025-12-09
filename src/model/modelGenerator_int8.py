import tensorflow as tf
import numpy as np
import tempfile
import soundfile as sf
import librosa
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
print(tf.__version__)

# Load the Speech Commands dataset
dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True)

# Extract the train and test sets (limited to 1000 samples each for simplicity)
train_data = dataset['train'].take(10000)
test_data = dataset['test'].take(10000)

# Define the model
model = models.Sequential([
    # Reshape input from [490] to [49, 10, 1]
    layers.Reshape((49, 10, 1), input_shape=(490,)),

    # Depthwise Convolutional Layer (1 input channel, 1 output channels, 5x4 kernel size)
    layers.DepthwiseConv2D(kernel_size=(5, 4), depth_multiplier=1, padding='valid', activation='relu',use_bias=False),

    # Flatten the output from 4D to 1D
    layers.Flatten(),

    # Fully connected layer with 4 units
    layers.Dense(4, activation='relu'),

    # Output layer with sigmoid (binary classification)
    layers.Dense(1, activation='sigmoid')  # 'Yes' vs 'Not Yes' classification
])

# import tensorflow_model_optimization as tfmot

# # Convert the model to a quantized model
# quantize_model = tfmot.quantization.keras.quantize_model
# model = quantize_model(model)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Function to extract MFCC features from the audio data
def extract_features(audio_data, sample_rate=16000):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        # Convert tensor to numpy array and save audio as a .wav file
        audio_array = audio_data.numpy().astype(np.int16)
        sf.write(tmpfile.name, audio_array, sample_rate, format='WAV')

        # Load the audio file and extract MFCCs
        audio, sr = librosa.load(tmpfile.name, sr=sample_rate, res_type='fft')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=50)
        mfccs_mean = np.mean(mfccs, axis=1)  # Average across time frames
    return mfccs_mean

# Label data preparation function
def label_data(dataset):
    X = []
    y = []
    labels = info.features['label'].names  # Get class labels from the dataset

    # Find the index of 'yes'
    yes_index = labels.index('yes')

    for audio, label in dataset:
        label_value = label.numpy()  # Convert tensor to integer
        label_str = labels[label_value]  # Map label index to class name

        features = extract_features(audio)  # Extract MFCC features from the audio

        # Classify as 1 for 'yes', 0 for everything else
        if label_str == 'yes':
            X.append(features)
            y.append(1)
        else:
            X.append(features)
            y.append(0)

    return np.array(X), np.array(y)

# Prepare the training and testing data
X_train, y_train = label_data(train_data)
X_test, y_test = label_data(test_data)

# Ensure feature vectors have consistent shape (490,)
X_train = np.pad(X_train, ((0, 0), (0, 490 - X_train.shape[1])), mode='constant')
X_test = np.pad(X_test, ((0, 0), (0, 490 - X_test.shape[1])), mode='constant')

# Calculate class weights
class_weights = {0: 1, 1: 15}  # Weight for 'yes' class is much higher (20x)

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=200, batch_size=50, validation_data=(X_test, y_test), class_weight=class_weights)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Quantize the weights for deployment (8-bit quantization)
def quantize_weights(weights, num_bits=8):
    scale = np.max(np.abs(weights))  # Scaling factor
    quantized_weights = np.round(weights * (2**(num_bits - 1)) / scale)
    return quantized_weights.astype(np.int8), scale

# Extract and quantize the model's weights
conv_weights = model.layers[1].get_weights()[0]
#conv_bias = model.layers[1].get_weights()[1]
fc_weights = model.layers[3].get_weights()[0]
fc_bias = model.layers[3].get_weights()[1]
fc_weights2 = model.layers[4].get_weights()[0]
fc_bias2 = model.layers[4].get_weights()[1]

conv_weights_quantized, conv_scale = quantize_weights(conv_weights)
#conv_bias_quantized, _ = quantize_weights(conv_bias)
fc_weights_quantized, fc_scale = quantize_weights(fc_weights)
fc_bias_quantized, _ = quantize_weights(fc_bias)
fc_weights_quantized2, fc_scale = quantize_weights(fc_weights2)
fc_bias_quantized2, _ = quantize_weights(fc_bias2)

# Save the quantized weights and biases to text files
def save_quantized_weights(filename, data):
    with open(filename, "w") as f:
        f.write(", ".join(map(str, data.flatten())))

# Save the quantized weights and biases to files
save_quantized_weights("fc_weights.txt", fc_weights_quantized)
save_quantized_weights("fc_bias.txt", fc_bias_quantized)
save_quantized_weights("conv_weights.txt", conv_weights_quantized)
#save_quantized_weights("conv_bias.txt", conv_bias_quantized) #not necessary
save_quantized_weights("fc_weights2.txt", fc_weights_quantized2)
save_quantized_weights("fc_bias2.txt", fc_bias_quantized2)
