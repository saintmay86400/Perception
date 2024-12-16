import audiofile
from .audio_features_extractor import AudioFeaturesExtractor
from .emovo import Emovo
import opensmile
import pandas as pd

class Opensperce(AudioFeaturesExtractor):
    def __init__(self, dataset):
        self.smile = opensmile.Smile(
            feature_set= opensmile.FeatureSet.eGeMAPSv02,
            feature_level= opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='logs/operce.log')
        self._csvpath =  "datasets/vocals/perception_speech.csv"
        self.dataset = dataset
        
    def get_csvpath(self):
        return self._csvpath
    
    def get_dataset(self):
        return self.dataset
    
    def read_audio(self, path):
        signal, sampling_rate = audiofile.read(path, always_2d=True)
        return signal, sampling_rate
    
    def extract_features(self, signal, sampling_rate):
        processed_signal = self.smile.process_signal(signal, sampling_rate)
        return processed_signal
    
    def run(self, audio_file_paths):
        first_read = True
        mode = 'w'
        for path in audio_file_paths:
            signal, sampling_rate = self.read_audio(path)
            processed_signal = self.extract_features(signal, sampling_rate)
            processed_signal['file_name'] = path.split('/')[-1]
            processed_signal.to_csv(self._csvpath, index=False, header=first_read, mode=mode)
            first_read = False
            mode = 'a'