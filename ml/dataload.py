
import glob
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
import itertools
from matplotlib import pyplot as plt
from melspec import full_spectrogram

SQUARE_ARRAY_LENGTH = 128

INPUT_PATH = 'music_data/*.npy'
OUTPUT_PATH = 'dmusic_data/*.npy'


def channel_transpose(x):
    x.unsqueeze_(-1)
    x = x.expand(x.shape[0],x.shape[1],1)
    x = x.transpose(2, 0)
    x = x.transpose(1, 2)
    return x


def segment(full_npy):
    cut_length = full_npy.shape[1] - (full_npy.shape[1] % SQUARE_ARRAY_LENGTH) 
    npy_cut = full_npy[:,:cut_length]
    split_list = np.array_split(npy_cut, cut_length / SQUARE_ARRAY_LENGTH, axis = 1)
    for index, _segment in enumerate(split_list):
        split_list[index] = channel_transpose(torch.from_numpy(split_list[index]))
        
        
        
    return split_list



def array_segment_zip(input_npy, output_npy):
    ziped_npy = zip(segment(input_npy), segment(output_npy))
    
    return ziped_npy


class MusicDistortionPair(Dataset):
    def __init__(self, music_path, distortion_path):
        music_data = sorted(glob.glob(music_path))
        dmusic_data = sorted(glob.glob(distortion_path))
        all_pairs = []
        for file_pair in zip(music_data, dmusic_data):
            
            music_array = np.load(file_pair[0])
            distorition_array = np.load(file_pair[1])
            all_pairs.append(array_segment_zip(music_array, distorition_array))
            
        self.spectrogram_pairs = list(itertools.chain(*all_pairs))
        
        
    def __len__(self):
        return len(self.spectrogram_pairs)
    
    def __getitem__(self,index):
        return (self.spectrogram_pairs[index][1], self.spectrogram_pairs[index][0])

# music_data = sorted(glob.glob('music_data/*.npy'))
# dmusic_data = sorted(glob.glob('dmusic_data/*.npy'))
# input_test = np.load(music_data[0])
# output_test = np.load(dmusic_data[0])


class Singular(Dataset):
    def __init__(self, music_path):
        
        music_array, sr = full_spectrogram(music_path)
        
        # music_data = sorted(glob.glob(music_path))
        
        # music_array = np.load(music_data[index])
            
        frames = segment(music_array)
        
        self.frames = frames
        self.sr = sr
        
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,index):
        return (self.frames[index][1], self.frames[index][0])




# def testload(input_path, output_path, index):
#     music_data = sorted(glob.glob(input_path))
#     dmusic_data = sorted(glob.glob(output_path))
#     input_test = np.load(music_data[index])
#     output_test = np.load(dmusic_data[index])
#     zipped = individual_zip(input_test, output_test)
#     return zipped




# test = testload(INPUT_PATH, OUTPUT_PATH, 0)

# for pair in test:
#     print(pair[0].shape)
#     print(pair[1].shape)


# x = torch.randn(128, 128)
# print(x)
# print("transposed")



# print(x)