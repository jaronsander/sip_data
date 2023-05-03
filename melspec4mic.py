import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK_SIZE = 1024
WINDOW_SIZE = 1024
FREQ_RANGE = (0, 11025)  # Hz
FIG_SIZE = (10, 6)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE)

# Create figure and axes
fig, ax = plt.subplots(figsize=FIG_SIZE)

# Create arrays for time and frequency axes
time_axis = np.arange(WINDOW_SIZE) / float(RATE)
freq_axis = np.linspace(FREQ_RANGE[0], FREQ_RANGE[1], WINDOW_SIZE // 2 + 1)

# Create initial empty plot
img = ax.imshow(np.zeros((WINDOW_SIZE, WINDOW_SIZE // 2 + 1)),
                extent=[0, time_axis[-1], FREQ_RANGE[0], FREQ_RANGE[1]],
                aspect='auto', cmap='inferno', vmin=-60, vmax=0, origin='lower')

# Function to update spectrogram plot
def update(frame):
    # Read audio data
    audio_data = stream.read(CHUNK_SIZE, exception_on_overflow = False)
    audio_data = np.frombuffer(audio_data, dtype=np.int16)

    # Apply window function
    audio_data = audio_data * np.hanning(len(audio_data))

    # Compute power spectrum
    spectrum = np.abs(np.fft.rfft(audio_data, n=WINDOW_SIZE)) ** 2
    spectrum = 10 * np.log10(spectrum / CHUNK_SIZE ** 2)

    # Update spectrogram plot
    # print(img.get_array().shape)
    img.get_array()[:-1, :] = img.get_array()[1:, :]
    # print(img.get_array()[-1, :].shape)
    img.get_array()[-1, :] = spectrum.T

    return [img]

# Create animation object
ani = FuncAnimation(fig, update, interval=50, blit=True)

# Show plot
plt.show()

# Clean up
stream.stop_stream()
stream.close()
audio.terminate()