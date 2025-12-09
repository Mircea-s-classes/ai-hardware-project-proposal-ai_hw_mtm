import tensorflow as tf
import numpy as np
import tempfile
import soundfile as sf
import librosa
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns


# Load the Speech Commands dataset
dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True)
print(dataset)
# Extract the train and test sets (limited to 10000 samples each for simplicity)
train_data = dataset['train']#.take(10000)
test_data = dataset['test']#.take(10000)


# Check the size of the train and test datasets
train_size = sum(1 for _ in train_data)
test_size = sum(1 for _ in test_data)


print(f"Train dataset size: {train_size}")
print(f"Test dataset size: {test_size}")


# Define the model
model = models.Sequential([
    layers.Reshape((1, 50, 1), input_shape=(50,)),


    # Depthwise Convolutional Layer (1 input channel, 1 output channels, 1x5 kernel size)
    layers.DepthwiseConv2D(kernel_size=(1, 5), depth_multiplier=1, padding='valid', activation='relu', use_bias=False),


    # Flatten the output from 4D to 1D
    layers.Flatten(),


    # Fully connected layer with 4 units
    layers.Dense(32, activation='relu'),


    # Output layer with sigmoid (binary classification)
    layers.Dense(1, activation='sigmoid')  # 'Yes' vs 'Not Yes' classification
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("model compiled")
model.summary()


import concurrent.futures


def extract_features_optimized(audio_data, sample_rate=16000):
    # Convert tensor to numpy array
    audio_array = audio_data.numpy().astype(np.float32) / 32768.0  # Normalize 16-bit PCM
    # Extract MFCC features directly in memory
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=50)
    mfccs_mean = np.mean(mfccs, axis=1)  # Average across time frames
    return mfccs_mean




def process_sample(audio, label, selected_labels, labels):
    label_value = label.numpy()
    label_str = labels[label_value]


    if label_str in selected_labels:
        features = extract_features_optimized(audio)  # Extract features
        if label_str == 'yes':
            return features, 1
        else:
            return features, 0
    return None


def label_data_filtered_optimized(dataset, selected_labels):
    labels = info.features['label'].names  # Get class labels from the dataset
    print(labels)
    # Parallelize feature extraction
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_sample, audio, label, selected_labels, labels)
            for audio, label in dataset
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)


    # Separate features and labels
    from collections import Counter
    train_labels = [labels[label.numpy()] for _, label in train_data]
    test_labels = [labels[label.numpy()] for _, label in test_data]
    print("Train set:", Counter(train_labels))
    print("Test set:", Counter(test_labels))
    X, y = zip(*results)
    return np.array(X), np.array(y)


# Filter the test dataset
selected_classes = ['yes', 'off']
X_train_filtered, y_train = label_data_filtered_optimized(train_data, selected_classes)
X_test_filtered, y_test = label_data_filtered_optimized(test_data, selected_classes)


 # Ensure feature vectors have consistent shape (50,)
# X_train = np.pad(X_train_filtered, ((0, 0), (0, 50 - X_train_filtered.shape[1])), mode='constant')
# X_test = np.pad(X_test_filtered, ((0, 0), (0, 50 - X_test_filtered.shape[1])), mode='constant')


# Train the model with class weights
from sklearn.utils import class_weight


# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


# Train the model with class weights
history = model.fit(X_train_filtered, y_train, epochs=50, batch_size=50, validation_data=(X_test_filtered, y_test), class_weight=class_weight_dict)




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


# Extract the model's weights and biases
conv_weights = model.layers[1].get_weights()[0]
#conv_bias = model.layers[1].get_weights()[1]
fc_weights = model.layers[3].get_weights()[0]
fc_bias = model.layers[3].get_weights()[1]
fc_weights2 = model.layers[4].get_weights()[0]
fc_bias2 = model.layers[4].get_weights()[1]


# Save weights and biases to text files (floating-point format)
def save_weights(filename, data):
    with open(filename, "w") as f:
        f.write(", ".join(map(str, data.flatten())))


save_weights("conv_weights_float.txt", conv_weights)
#save_weights("conv_bias_float.txt", conv_bias)
save_weights("fc_weights_float.txt", fc_weights)
save_weights("fc_bias_float.txt", fc_bias)
save_weights("fc_weights2_float.txt", fc_weights2)
save_weights("fc_bias2_float.txt", fc_bias2)


print("Floating-point weights and biases saved.")




#-------------------------------------------------#
#This is where I use the model to make predictions:
#-------------------------------------------------#




# Define the model with the same architecture
model = models.Sequential([
    layers.Reshape((1, 50, 1), input_shape=(50,)),


    # Depthwise Convolutional Layer (1 input channel, 1 output channel, 1x5 kernel size)
    layers.DepthwiseConv2D(kernel_size=(1, 5), depth_multiplier=1, padding='valid', activation='relu', use_bias=False),


    # Flatten the output from 4D to 1D
    layers.Flatten(),


    # Fully connected layer with 32 units
    layers.Dense(32, activation='relu'),


    # Output layer with sigmoid (binary classification)
    layers.Dense(1, activation='sigmoid')  # 'Yes' vs 'Not Yes' classification
])


# Load the weights from saved files
conv_weights = np.genfromtxt("conv_weights_float.txt", delimiter=',')
fc_weights = np.genfromtxt("fc_weights_float.txt", delimiter=',')
fc_bias = np.genfromtxt("fc_bias_float.txt", delimiter=',')
fc_weights2 = np.genfromtxt("fc_weights2_float.txt", delimiter=',')
fc_bias2 = np.genfromtxt("fc_bias2_float.txt", delimiter=',')


conv_weights=conv_weights.reshape(1,5,1,1)
fc_weights=fc_weights.reshape(46,32)
fc_weights2=fc_weights2.reshape(32,1)
fc_bias2=fc_bias2.reshape(1,)


# Set the weights in the model
model.layers[1].set_weights([conv_weights])  # Convolutional layer weights
model.layers[3].set_weights([fc_weights, fc_bias])  # Fully connected layer 1 weights and bias
model.layers[4].set_weights([fc_weights2, fc_bias2])  # Fully connected layer 2 weights and bias


# Validate the model using test data (ensure data is preprocessed properly)


# Get predictions for test set
y_pred = model.predict(X_test_filtered)


# Separate predictions for "yes" and "not yes"
yes_preds = y_pred[y_test == 1]
not_yes_preds = y_pred[y_test == 0]


# Plot histograms using matplotlib to have more control over colors
plt.hist(yes_preds, bins=50, color='green', alpha=0.6, label='Yes', density=True)
plt.hist(not_yes_preds, bins=50, color='red', alpha=0.6, label='Off', density=True)


# Draw the threshold line at 0.7
plt.axvline(x=0.7, color='black', linestyle='--', label='Threshold = 0.7')


# Display labels and title
plt.legend()
plt.title('Prediction Score Distribution')
plt.xlabel('Prediction Score')
plt.ylabel('Density')  # Show density instead of frequency
plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix


# Get predicted classes (0 or 1) based on the threshold (e.g., 0.5)
y_pred_class = (y_pred >= 0.5).astype(int)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy}")


# Print confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
print(f"Confusion Matrix:\n{cm}")

