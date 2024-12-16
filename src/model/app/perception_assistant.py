from .user import User
from machine_learning import *
from dataset import *
import pandas as pd
import os
import whisper
import ssl
import config

# Disabilita la verifica del certificato SSL
ssl._create_default_https_context = ssl._create_unverified_context

class PerceptionAssistant(User):
    def __init__(self, username, user_id, model: BaseModel):
        super().__init__(username, user_id)
        
        self.model = model
        self.audio_translator = whisper.load_model("base")
        self.audio_features_extractor = Opensperce(Emovo())

    def transcribe(self, audio_path):
        result =  self.audio_translator.transcribe(audio_path)
        return result["text"]
    
    def get_audio_features(self, audio_path):
        self.audio_features_extractor.run(audio_path)
        features = pd.read_csv(os.path.join(config.root, self.audio_features_extractor.get_csvpath()))
        X =features.drop(['file_name'], axis=1)
        return X
    
    def analyze_audio(self, audio_path):
        X = self.get_audio_features(audio_path)
        return {"emotion": self.model.predict(X)[0], "content": self.transcribe(audio_path)}
        
    def give_adivice(self, audio_path):
        report = self.analyze_audio(audio_path)
        list_emotions = self.audio_features_extractor.get_dataset().get_emotions()
        audio_emotion = [value[1] for key, value in list_emotions.items() if value[0] == report['emotion']]
        return f"Il paziente ha detto: {report['content']}. Rispondi considerando di essere un medico che si rivolge ad un paziente il cui stato emotivo attuale è: {audio_emotion[0]}"
        
    
        