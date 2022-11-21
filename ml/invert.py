import librosa
from librosa import display
import glob
import numpy as np
import json
import soundfile as sf



recon = librosa.feature.inverse.mel_to_audio(M = mel, sr = sr, n_fft = 2048)

sf.write('stereo_file.wav', recon, sr)



















# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)

# librosa.display.waveshow(y, sr=frames['sr'][0], color='g', ax=ax[1])
# ax[0].set(title='Griffin-Lim reconstruction', xlabel=None)
# ax[0].label_outer()
