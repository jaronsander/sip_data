import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import numpy as np

# Load the WAV file and extract the audio data
waveform, sample_rate = torchaudio.load('Long Falls/shower-long_fall_4.wav')
NFFT = 1024

WINDOW_SIZE = 1.5  # window size in seconds
window_frames = int(WINDOW_SIZE * 2 * sample_rate / NFFT)

# Compute the spectrogram for each channel
# spectrograms = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=None, hop_length=None,
#                                                  power=None)(waveform)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=None,
    n_mels=64
)
spectrograms = []
for spec in mel(waveform):
    spectrograms.append(librosa.power_to_db(spec))
# spectrograms = torch.Tensor(spectrograms)
# spec = torch.cat((f1, spectrograms), 2)

# Create a new plot window
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

# Initialize the plot images
images = []

def init():
    for i in range(len(axs)):
        image = axs[i].imshow(np.zeros((64, window_frames)), aspect='auto')
        axs[i].invert_yaxis()
        images.append(image)
        fig.colorbar(image, ax=axs[i])
        image.set_clim(np.min(0, 0))
    return images

def col_generator():
    fr = 1
    print(int(len(spectrograms[0])))
    print(int(len(spectrograms[0][0])))
    for i in range(int(len(spectrograms[0][0])-fr)):
        frame = np.array(spectrograms).astype('float32')[:, :, i:i+fr]
        yield frame


# Define the update function
vmax = [0.1]
def update(frame):
    # Update the plot images for each channel
    for i in range(len(axs)):
        print(frame[i].shape)
        arr = images[i].get_array()[:, 1:]
        print(arr.shape)
        arr = np.c_[arr,frame[i]]
        print(arr.shape)

        # arr.aframe[i,:,0]
        # images[i].set_clim(0, np.max(images[i].get_array()))
        images[i].set(data=arr)
        fm = np.max(frame)
        # print(fm)
        if fm > vmax[-1]:
            vmax.append(fm)
            images[i].set_clim(0, fm)
    return images


# Start the animation
ani = FuncAnimation(fig, update, frames=col_generator(), init_func=init, blit=True,
                    interval=200,save_count=10)
plt.show()
