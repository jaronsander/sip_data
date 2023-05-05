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
    spectrograms.append(np.array(librosa.power_to_db(spec)))
f1 = torch.zeros((64, window_frames)).unsqueeze(0).repeat(3,1,1)
spectrograms = torch.Tensor(spectrograms)
spec = torch.cat((f1, spectrograms), 2)

# Create a new plot window
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

# Initialize the plot images
images = []

def col_generator():
    mul = 1
    for i in range(int((spec.shape[2]-window_frames)/mul)):
        frame = spec[:, :, mul*i:mul*i + window_frames].numpy().astype('float32')
        yield frame


# Define the update function

def update(frame):
    # Update the plot images for each channel
    for i in range(len(axs)):
        if (len(images) != 3):
            image = axs[i].imshow(frame[i], aspect='auto')
            axs[i].invert_yaxis()
            images.append(image)
            fig.colorbar(image, ax=axs[i])
            image.set_clim(np.min(frame), np.max(frame))
        else:
            images[i].set(data=frame[i])
            images[i].set_clim(np.min(frame), np.max(frame))
    return images


# Start the animation
ani = FuncAnimation(fig, update, frames=col_generator(), blit=True,
                    interval=200, cache_frame_data=False)
plt.show()
