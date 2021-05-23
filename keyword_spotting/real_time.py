from queue import Queue
import sys
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time
import pyaudio
import numpy as np
data_c = None

# Use 1101 for 2sec input audio
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model

"""## Build the model"""

from keras.models import load_model
model = load_model('./keyword_spotting/tr_model_t.h5')
#model = load_model('./tr_model_t.h5')

def detect_triggerword_spectrum(x):
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)

def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False

"""# Record audio stream from mic"""
chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

def get_spectrogram(data):
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def plt_spectrogram(data):
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _, _ = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

"""### Audio stream"""

def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream

# Queue to communiate between the audio callback and main thread
q = Queue()
run = True
silence_threshold = 100
timeout = 0.5*60  # 0.5 minutes from now
# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold
    if time.time() > timeout:
        run = False
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('------------dddd-----')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.............dddd....')
    data = np.append(data,data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

def check_where():
    stream = get_audio_input_stream(callback)
    stream.start_stream()
    count=0
    global run, timeout
    try:
        while count<timeout:
            data = q.get()
            spectrum = get_spectrogram(data)
            preds = detect_triggerword_spectrum(spectrum)
            new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
            if new_trigger:
                print('I CAN HEAR TRIGGER##################')
            else:
                print('I CAN HEAR NOTHING$$$$$$$$$$$$$$$$$$')
            #time.sleep(1)
            count = count+1
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False
    stream.stop_stream()
    stream.close()