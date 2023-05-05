import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib import pyplot as plt, animation
import numpy as np
import torch
import torchaudio
import librosa

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

n_ch = 3

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

plt.axes(xlim=(0, 2), ylim=(-2, 2))
# fig = plt.Figure(dpi=100)
# ax = fig.add_subplot(xlim=(0, 2), ylim=(-1, 1))
# line, = ax.plot([], [], lw=2)
fig, ax = plt.subplots(nrows=n_ch, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()

toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button = tkinter.Button(master=root, text="Quit", command=root.quit)
button.pack(side=tkinter.BOTTOM)

sample_text = tkinter.Entry(root)
sample_text.pack()
def set_text_by_button():
    # Delete is going to erase anything
    # in the range of 0 and end of file,
    # The respective range given here
    sample_text.delete(0, "end")

    # Insert method inserts the text at
    # specified position, Here it is the
    # beginning
    sample_text.insert(0, "Text set by button")
ubutton = tkinter.Button(master=root, text="Update", command=set_text_by_button)
ubutton.pack(side=tkinter.RIGHT)

toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

# Do audio stuff


# images = []
# def init():
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x))
#     for i in range(n_ch):
#         images.append(ax[i].imshow)
#     # line.set_data([], [])
#     return ax,
#
# def animate(i):
#     print(i)
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     # if y[1] > 0.8:
#     #     sample_text.insert(0, "True")
#     # else:
#     #     sample_text.insert(0, "True")
#     #
#     # line.set_data(x, y)
#     # return line,
#     for j in range(n_ch):
#         ax[j].plot(x,y)
#     return ax
#
# anim = animation.FuncAnimation(fig, animate, init_func=init,frames=200, interval=20, blit=False)
# Load the WAV file and extract the audio data
waveform, sample_rate = torchaudio.load('Long Falls/shower-long_fall_4.wav')
NFFT = 1024
print(waveform.shape)
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
print(spec.shape)


# Create a new plot window
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

# Initialize the plot images
images = []

def col_generator():
    mul = 4
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
anim = animation.FuncAnimation(fig, update, frames=col_generator(), blit=False,
                    interval=20, cache_frame_data=False)

tkinter.mainloop()