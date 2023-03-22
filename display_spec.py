import librosa
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    files = librosa.util.find_files('falls3')
    audio = []
    sr = 44100 / 2
    for x in files:
        chs, sr = librosa.load(x, mono=False, sr=sr)
        print(len(chs))
        audio.append(chs[0:3])
    names = [x.split('/')[-1] for x in files]
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax = ax.flatten()
    file_index = 91
    ch_names = ['ch1', 'ch2', 'geophone']
    for i in range(3):
        y = audio[file_index][i]
        hop_length = 512
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)))
        img = librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
                                       x_axis='time', ax=ax[i], cmap='magma')
        ax[i].set(title=names[file_index] + '_' + ch_names[i])
        ax[i].label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    fig.set_size_inches(16, 8)
    plt.savefig(names[file_index]+'.png')