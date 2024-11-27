import audiofile
from .audio_features_extractor import AudioFeaturesExtractor
import opensmile
import pandas as pd

class Opensemov(AudioFeaturesExtractor):
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set= opensmile.FeatureSet.eGeMAPSv02,
            feature_level= opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='logs/opemov.log')
        self._csvpath =  "datasets/augmented_emovo.csv"
        
    def get_csvpath(self):
        return self._csvpath
    
    def read_audio(self, path):
        signal, sampling_rate = audiofile.read(path, always_2d=True)
        return signal, sampling_rate
    
    def extract_features(self, signal, sampling_rate):
        processed_signal = self.smile.process_signal(signal, sampling_rate)
        return processed_signal
    
    @staticmethod       
    def get_emotion(path, emotions):
        track_name_parts = path.split('/')[-1].split('-')
        emo_key = track_name_parts[0]
        code = emotions[emo_key][0]
        label = emotions[emo_key][1]
        
        return label, code
    
    @staticmethod       
    def get_speaker_gender(path):
        
        track_name_parts = path.split('/')[-1].split('-')        
        if track_name_parts[1].startswith("f"):
            gender = 'female'
        elif track_name_parts[1].startswith("m"): 
            gender = 'male'
        else:
            gender= "undefined"
            
        return gender
    
    def run(self, audio_file_paths, emotions):
        first_read = True
        for path in audio_file_paths:
            signal, sampling_rate = self.read_audio(path)
            processed_signal = self.extract_features(signal, sampling_rate)
            emotion_label, emotion_code = Opensemov.get_emotion(path, emotions)
            processed_signal['Emotion_Code'] = emotion_code
            processed_signal['Emotion'] = emotion_label
            #processed_signal['Speaker_Gender'] = Opensemov.get_speaker_gender(path)
            processed_signal.to_csv(self._csvpath, index=False, header=first_read, mode='a')
            first_read = False