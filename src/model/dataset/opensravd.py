import audiofile
from .audio_features_extractor import AudioFeaturesExtractor
from .ravdess import Ravdess
import opensmile
import pandas as pd
import random

class Opensravd(AudioFeaturesExtractor):
    def __init__(self, dataset: Ravdess):
        self.smile = opensmile.Smile(
            feature_set= opensmile.FeatureSet.eGeMAPSv02,
            feature_level= opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='logs/opravd.log')
        self._csvpath =  "datasets/opensravd_speech"+str(random.randint(1,100))+".csv"
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
    
      
    def get_emotion(self, path):
        emotions = self.dataset.get_emotions()
        intensities = self.dataset.get_intensities()
        track_name_parts = path.split('/')[-1].split('-')
        emo_code = emotions[track_name_parts[2]][0]
        intens_code = intensities[track_name_parts[3]][0]
        if emo_code in [1, 4]:
            label = emotions[track_name_parts[2]][1]
        elif intens_code == 2:
            label = intensities[track_name_parts[3]][1] + ' ' + emotions[track_name_parts[2]][1]
        else: label = emotions[track_name_parts[2]][1]
        
        return label, emo_code, intens_code
    
    @staticmethod       
    def get_speaker_gender(path):
        
        track_name_parts = path.split('/')[-1].split('-')        
        if(int(track_name_parts[-1].replace('.wav', '')) % 2 == 0):
            gender = 'female'
        else: 
            gender = 'male'
            
        return gender
    
    def run(self):
        first_read = True
        for path in self.dataset.get_file_paths():
            signal, sampling_rate = self.read_audio(path)
            processed_signal = self.extract_features(signal, sampling_rate)
            emotion_label, emotion_code, intensity_code = self.get_emotion(path)
            processed_signal['Emotion_Code'] = emotion_code
            processed_signal['Intensity_Code'] = intensity_code
            processed_signal['Emotion'] = emotion_label
            #processed_signal['Speaker_Gender'] = Opensmile.get_speaker_gender(path)
            processed_signal.to_csv(self._csvpath, index=False, header=first_read, mode='a')
            first_read = False