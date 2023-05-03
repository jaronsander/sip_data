import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa

# Load the WAV file and extract the audio data
waveform, sample_rate = torchaudio.load('sweep.wav')
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
spectrograms = mel(waveform)
f1 = torch.zeros((64, window_frames)).unsqueeze(0)
spec = torch.cat((f1, spectrograms), 2)

# Create a new plot window
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

# Initialize the plot images
images = []

vmax = [1.0]


# for i in range(spectrograms.shape[0]):
#     image = axs[i].imshow(librosa.power_to_db(spectrograms[i, :, :window_frames]),
#                                               aspect='auto',
#                           origin='lower',
#                           cmap='inferno')
#     images.append(image)

def init():
    f1 = torch.zeros((window_frames, int(NFFT / 2) + 1))
    imgs = []
    for i in range(3):
        image = axs[i].imshow(f1,
                              cmap='inferno')
        imgs.append(image)
    return imgs


def col_generator():
    for i in range(spec.shape[2]):
        frame = spec[0, :, i:i + window_frames].numpy().astype('float32')
        yield frame


# Define the update function

def update(frame):
    # Update the plot images for each channel

    for i in range(len(axs)):
        if (len(images) != 3):
            image = axs[i].imshow(frame, cmap='inferno', vmin=0, aspect='auto')
            axs[i].invert_yaxis()
            images.append(image)
            fig.colorbar(image, ax=axs[i])
        max = torch.max(torch.Tensor(frame[-1]))
        if vmax[-1] < max:
            vmax.append(max)
        images[i].set_clim(0, vmax[-1])
        images[i].set(data=frame)
    return images


# Start the animation
ani = FuncAnimation(fig, update, frames=col_generator(), blit=True,
                    interval=2, cache_frame_data=False)
plt.show()
