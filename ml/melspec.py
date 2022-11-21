import librosa
from librosa import display
import glob
import numpy as np
import json
from json import JSONEncoder
import soundfile as sf



                

def full_spectrogram(file_path):
    y , sr = librosa.load(file_path, mono = True)
    
    spectrogram = librosa.feature.melspectrogram(y=y,
                                            sr=sr, 
                                            n_fft=2048, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                            n_mels=128)
    
    return spectrogram, sr
    

def create_spec_set(music_folder, npy_folder):
    music_path = str(music_folder) + '/*'
    music_files = glob.glob(music_path)
    
    for number, file in enumerate(music_files):
        spectro_path = str(npy_folder) + '/' + str(number) + '.npy'
        
        mel, sr = full_spectrogram(file)
        
        with open(spectro_path, 'wb') as f:
            np.save(f, mel)
        
        
        # with open( sr_path , "w" ) as j:
        #     json.dump( sr_json , j, cls = specEncoder )
            
        
# create_spec_set('dmusic', 'dmusic_data')
    



    
    
    
    
    
    
    
    
    
    
    


   






