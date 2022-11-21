




# Beat tracking example

import numpy as np
import glob
import os
from pedalboard import Pedalboard, Compressor, Delay, Distortion, Gain, PitchShift, Reverb, Mix
from pedalboard.io import AudioFile



def create_distortion(music_folder, distortion_folder):



    files = glob.glob(str(music_folder) + '/*')


    for number,audio_file in enumerate(files):
        with AudioFile(audio_file, 'r') as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        # Make a Pedalboard object, containing multiple plugins:
        board = Pedalboard([Distortion()])

        # Run the audio through this pedalboard!
        effected = board(audio, samplerate)

        # Write the audio back as a wav file:
        output_path = distortion_folder + '/' + str(number) + '.wav'
        with AudioFile(output_path, 'w', samplerate, effected.shape[0]) as f:
            f.write(effected)



# create_distortion('music', 'dmusic')
  

