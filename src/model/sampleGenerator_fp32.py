import tensorflow as tf
import librosa
import soundfile as sf
import numpy as np
import tempfile
import tensorflow_datasets as tfds

# Load the Speech Commands dataset
dataset, info = tfds.load("speech_commands", with_info=True, as_supervised=True)
train_data = dataset['train']  # Do not limit, just shuffle

# Shuffle the dataset and then take one random sample
#train_data = train_data.shuffle(20)  # Shuffle buffer size (1000 is reasonable for shuffling)

# Extract the MFCC features for that sample
def extract_features(audio_data, sample_rate=16000):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        # Convert tensor to numpy array and normalize to [-1.0, 1.0]
        audio_array = audio_data.numpy().astype(np.float32) / 32768.0  # Normalize 16-bit PCM
        sf.write(tmpfile.name, audio_array, sample_rate, format='WAV')

        # Load the audio file and extract MFCCs
        audio, sr = librosa.load(tmpfile.name, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=50)  # 50 MFCC coefficients
        mfccs_mean = np.mean(mfccs, axis=1)  # Average across time frames
    return mfccs_mean


# Get the label index for "yes"
label_index_yes = info.features['label'].names.index('yes')
label_index_off = info.features['label'].names.index('off')

# Filter the dataset to get only "yes" samples
for audio, label in train_data:
    if label == label_index_yes:
        features = extract_features(audio)  # Extract MFCC features

        # Map the label index to the class name
        label_name = info.features['label'].names[label.numpy()]
        print(f"Sample label: {label_name}")
        
        # Pad features to ensure consistent size (50 elements)
        features_padded = np.pad(features, (0, 50 - features.shape[0]), mode='constant')
        
        # Print the floating-point 50-element vector
        print(", ".join(map(str, features_padded)))
    if label == label_index_off:
        features = extract_features(audio)  # Extract MFCC features

        # Map the label index to the class name
        label_name = info.features['label'].names[label.numpy()]
        print(f"Sample label: {label_name}")
        
        # Pad features to ensure consistent size (50 elements)
        features_padded = np.pad(features, (0, 50 - features.shape[0]), mode='constant')
        
        # Print the floating-point 50-element vector
        print(", ".join(map(str, features_padded)))
