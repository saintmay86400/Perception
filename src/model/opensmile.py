import audiofile
from .audio_features_extractor import AudioFeaturesExtractor
import opensmile
import pandas as pd

class Opensmile(AudioFeaturesExtractor):
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set= opensmile.FeatureSet.eGeMAPSv02,
            feature_level= opensmile.FeatureLevel.Functionals,
            loglevel=2,
            logfile='logs/wayne.log')
        self._csvpath =  "datasets/speech.csv"
    
    def read_audio(self, path):
        signal, sampling_rate = audiofile.read(path, always_2d=True)
        return signal, sampling_rate
    
    def extract_features(self, signal, sampling_rate):
        processed_signal = self.smile.process_signal(signal, sampling_rate)
        return processed_signal
    
    @staticmethod       
    def get_emotion(path, emotions, intensity):
        track_name_parts = path.split('/')[-1].split('-')
        emo_code = track_name_parts[2]
        intens_code = track_name_parts[3]
        if emo_code in ['01', '02']:
            label = emotions[emo_code]
        elif intens_code == '02':
            label = intensity[intens_code] + ' ' + emotions[emo_code]
        else: label = emotions[emo_code]
        
        return label, emo_code, intens_code
    
    @staticmethod       
    def get_speaker_gender(path):
        
        track_name_parts = path.split('/')[-1].split('-')        
        if(int(track_name_parts[-1].replace('.wav', '')) % 2 == 0):
            gender = 'female'
        else: 
            gender = 'male'
            
        return gender
    
    def run(self, audio_file_paths, emotions, intensities):
        for path in audio_file_paths:
            signal, sampling_rate = self.read_audio(path)
            processed_signal = self.extract_features(signal, sampling_rate)
            emotion_label, emotion_code, intensity_code = Opensmile.get_emotion(path, emotions, intensities)
            processed_signal['Emotion_Code'] = emotion_code
            processed_signal['Intensity_Code'] = intensity_code
            processed_signal['Emotion'] = emotion_label
            processed_signal['Speaker_Gender'] = Opensmile.get_speaker_gender(path)
            processed_signal.to_csv(self._csvpath, index=False, mode='a')