import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import numpy as np
import torch.nn as nn
from torch import Tensor
# model
# from https://github.com/maciejbalawejder/Deep-Learning-Collection/blob/main/ConvNets/MobileNetV3/mobilenetv3_pytorch.py
class BCNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(
                num_features=16,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(
                num_features=32,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(
                num_features=64,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(
                num_features=128,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3200 * 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
device = torch.device("cpu")
cnn = BCNNNetwork()
cnn.load_state_dict(torch.load("cnn_3ch_mn3_f0_e9.pth",map_location=device))
# Load the WAV file and extract the audio data
# waveform, sample_rate = torchaudio.load('Long Falls/bath-long_fall_6-15.wav')
waveform, sample_rate = torchaudio.load('Long Falls/long_fall-2.wav')
# waveform, sample_rate = torchaudio.load('fold0/ambient-fall_30_1.wav')
NFFT = 1024

WINDOW_SIZE = 1.5  # window size in seconds
# window_frames = int(WINDOW_SIZE * 2 * sample_rate / NFFT)
window_frames = 130
map = ["No fall", "  Fall  "]
# Compute the spectrogram for each channel
# spectrograms = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=None, hop_length=None,
#                                                  power=None)(waveform)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
spectrograms = mel(waveform)
# spectrograms = []
# for spec in mel(waveform):
#     print(spec.shape)
#     spectrograms.append(np.array(librosa.power_to_db(spec)))
# spectrograms = torch.Tensor(spectrograms)
f1 = torch.zeros((64, window_frames)).unsqueeze(0).repeat(3,1,1)

spec = torch.cat((f1, spectrograms), 2)

# Create a new plot window
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(14, 8), sharex=True, sharey=True)

# Initialize the plot images
images = []
txt = []

def col_generator():
    mul = 3
    for i in range(int((spec.shape[2]-window_frames)/mul)):
        frame = spec[:, :, mul*i:mul*i + window_frames].numpy().astype('float32')
        yield frame


# Define the update function
fall_cache = []
run = [0]
mx = [0]
def update(frame):
    # Do prediction
    pred = None
    with torch.no_grad():
        pred = cnn(torch.Tensor(frame)[None,:])
    # Update the plot images for each channel
    val = torch.max(pred, 1)[1]

    if val == 1:
        run[0] += 1
        if run[0] > mx[0]:
            mx[0] = run[0]
    else:
        run[0] = 0
    print(mx)
    fall_cache.append(val)
    fall = np.sum(fall_cache[-25:]) == 25
    if fall:
        pred = map[1]
    else:
        pred = map[0]
    # print(pred)
    # fig.suptitle(map[torch.max(pred, 1)[1]])

    for i in range(len(axs)):
        if i == 0:
            if len(images) == 0:
                images.append(axs[i].text(0.5, 0.5, "%s" % pred,
                                          transform=axs[0].transAxes, ha="center",fontsize=30,
                                          bbox=dict(facecolor='green', edgecolor='green',
                                                    pad=40.0)))
                axs[i].axis('off')
                # axs[i].plot([1],[1])
                axs[i].set_frame_on(True)
                axs[i].set_facecolor("green")
            else:
                if not fall:
                    images[i].set_backgroundcolor("green")
                else:
                    # print(val)
                    images[i].set_backgroundcolor("red")
                images[i].set_text(pred)
        else:
            f = librosa.power_to_db(frame[i-1])
            if (len(images) != 4):
                image = axs[i].imshow(f, aspect='auto')
                axs[i].invert_yaxis()
                images.append(image)
                fig.colorbar(image, ax=axs[i])
                image.set_clim(np.min(f), np.max(f))
            else:
                images[i].set(data=f)
                images[i].set_clim(np.min(f), np.max(f))
    return images


# Start the animation
ani = FuncAnimation(fig, update, frames=col_generator(), blit=True,
                    interval=10, cache_frame_data=False, repeat=False)
plt.show()
