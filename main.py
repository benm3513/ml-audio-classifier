import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow_io as tfio
from itertools import groupby
import csv

# Paths to sample audio files for Capuchinbird and non-Capuchinbird sounds
CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')

def load_wav_16k_mono(filename):
    """Loads a wav file, resamples it to 16kHz, and returns it as a mono audio tensor."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # Remove the extra channel dimension
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)  # Resample to 16kHz
    return wav

# Load wav files for Capuchinbird and non-Capuchinbird samples
wave = load_wav_16k_mono(CAPUCHIN_FILE)
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)

# Define positive and negative datasets for Capuchinbird and non-Capuchinbird sounds
POS = os.path.join('data', 'Parsed_Capuchinbird_Clips', '*.wav')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', '*.wav')

pos = tf.data.Dataset.list_files(POS)
neg = tf.data.Dataset.list_files(NEG)

# Label the positive (Capuchinbird) and negative (non-Capuchinbird) samples
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

# Combine positive and negative datasets
data = positives.concatenate(negatives)

# Collect lengths of Capuchinbird wav files for analysis
lengths = []
capuchin_dir = os.path.join('data', 'Parsed_Capuchinbird_Clips')
for filename in os.listdir(capuchin_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(capuchin_dir, filename)
        tensor_wave = load_wav_16k_mono(filepath)
        lengths.append(len(tensor_wave))

def preprocess(file_path, label):
    """Preprocesses a file by loading, padding, and converting it into a spectrogram."""
    wav = load_wav_16k_mono(file_path)
    wav = wav[:48000]  # Limit the wav length to 48000 samples
    
    # Zero-pad to ensure the wav is exactly 48000 samples long
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    
    # Compute the spectrogram
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    
    # Add a channel dimension to the spectrogram
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    
    return spectrogram, label

# Shuffle and preprocess one sample for visualization or testing
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

# Map preprocessing function to the dataset and prepare for training
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# Split the data into training and testing sets
train = data.take(36)
test = data.skip(36).take(15)

# Display shape of training samples
samples, labels = train.as_numpy_iterator().next()
print(samples.shape)

# Define a Sequential CNN model for binary classification
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(1491, 257, 1)),  # 1491x257 is the size of spectrogram
    Conv2D(16, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output a single value for binary classification
])

# Compile the model with Adam optimizer, binary crossentropy loss, and precision/recall metrics
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

# Show the model summary
print(model.summary())

# Train the model on the training data and validate on the test data
hist = model.fit(train, epochs=3, validation_data=test)

# Save the trained model
model.save('my_model.h5')
print('Model Saved!')

# Load the saved model for future use
savedModel = load_model('my_model.h5')

# Test the model on the testing data
X_test, y_test = test.as_numpy_iterator().next()
yhat = savedModel.predict(X_test)

# Convert predictions to binary labels
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
print(yhat)
print(y_test)

def load_mp3_16k_mono(filename):
    """Loads an mp3 file, resamples to 16kHz, and returns it as a mono audio tensor."""
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2  # Convert stereo to mono
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

# Load an MP3 file and convert it into a waveform
mp3 = os.path.join('data', 'Forest Recordings', 'recording_00.mp3')
wav = load_mp3_16k_mono(mp3)

# Create batches of audio slices from the wav file
audio_slices = tf.keras.utils.timeseries_dataset_from_array(
    wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)

# Get one batch of audio slices
samples, index = audio_slices.as_numpy_iterator().next()

def preprocess_mp3(sample, index):
    """Preprocess MP3 audio samples by converting to spectrograms."""
    sample = sample[0]
    
    # Zero-pad the audio to ensure it's exactly 48000 samples long
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    
    # Compute the spectrogram
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    
    # Add a channel dimension to the spectrogram
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    
    return spectrogram

# Prepare MP3 recordings for classification
results = {}
for file in os.listdir(os.path.join('data', 'Forest Recordings')):
    FILEPATH = os.path.join('data', 'Forest Recordings', file)

    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(
        wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    # Predict class labels using the saved model
    yhat = savedModel.predict(audio_slices)
    results[file] = yhat

# Post-process the prediction results
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]

postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)])

# Output the final predictions and post-processed results
print(class_preds)
print(postprocessed)

# Save the prediction resutls to .csv file
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
