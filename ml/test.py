from dataload import channel_transpose

import torch
from torch import nn
import librosa
import glob
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset
import itertools
from matplotlib import pyplot as plt
from dataload import Singular
import network


def reconstruct_audio(frame_list, sr):
    
    full = frame_list[0].detach().numpy()
    for index, frame in enumerate(frame_list):
        if index != 0:
            full = np.concatenate((full,frame.detach().numpy()), axis = 2)

    # np.vstack((full,end_tensor))
    # print("RECON")
    # print(full[0].shape)
    # show_array(full, 5)

    res = librosa.feature.inverse.mel_to_audio(full[0], 
                                            sr=sr, 
                                            n_fft=2048, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0, 
                                            n_iter=32)


    sf.write('stereo_file.wav', res, sr)








dataset = Singular('tdis/3.wav')

batch_size = 5


# FOLDER_PATH = "dmusic"
# music_path = str(FOLDER_PATH) + '/*'
# music_files = sorted(glob.glob(music_path))

# mel, sr = full_spectrogram(music_files[0])

# frame_list, end_tensor = end_segment(mel)
# # print(end_tensor.shape)
PATH = "trained/trained.pt"
DEVICE = "cpu"
model = network.NeuralNetwork().to(DEVICE)

model.load_state_dict(torch.load(PATH))


frame_list = []
for X in dataset.frames:
    frame_list.append(model(X))






reconstruct_audio(frame_list, dataset.sr)


# song 
# full numpy array
# segment

# train model

# run each frame through model


    

# convert to audio file