import numpy as np
import os
import pandas as pd
import pickle
from sklearn import svm
from model import *
from sklearn.model_selection import train_test_split
import pyaudio
import time


#root del progetto: /perception
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



####### WESAD + MY CUSTOM EXTRACTOR (Biometricfeat) #########
#############################################################


wesad_root = os.path.join(root, "public", "WESAD")
wesad = Wesad(wesad_root)
wesad.calculate_file_paths()

biometricfeat = Biometricfeat()
#biometricfeat.run(wesad.get_file_paths(), wesad.get_emotions())
'''
csv_biometrics = pd.read_csv(os.path.join(root, biometricfeat.get_csvpath()))
csv_biometrics.fillna(0, inplace=True)
csv_biometrics.replace([np.inf, -np.inf], 0, inplace=True)
z = csv_biometrics.pop('Emotion_Code')
T =csv_biometrics.drop(['Emotion'], axis=1)
T.replace(np.inf, T.max(), inplace=True)
T.replace(-np.inf, T.min(), inplace=True)
z.replace(np.inf, z.max(), inplace=True)
z.replace(-np.inf, z.min(), inplace=True)

T_train, T_test, z_train, z_test = train_test_split(T, z, test_size=0.3, stratify=z, random_state=42)

bio_svc = Svc()

bio_svc.train(T_train, z_train)
print(bio_svc.evaluate(T_test, z_test))
z_pred = bio_svc.predict(T_test)
print(bio_svc.confusion_matrix(z_test, z_pred))
print(bio_svc.cross_validation(T,z))
bio_svc.save_model(os.path.join(root, "models", "bio_svc.pkl"))
'''


####### RAVDES + OPENSRAVD #########
#####################################
'''
# path della cartella ravness (contiene-> cartelle "Attori" con file audio)

ravdess_root = os.path.join(root, "public", "Audio_Speech_Actors_01-24")
ravdess = Ravdess(ravdess_root)
ravdess.set_emotions({'01':[1,'neutral'], '03':[2,'happy'], '04':[3,'sad']})
ravdess.calculate_file_paths()

opensravd = Opensravd()
#opensravd.run(ravdess.get_file_paths(), ravdess.get_emotions(), ravdess.get_intensities())
'''


####### EMOVO + OPENSEMOV #########
#####################################

# path della cartella emovo (contiene-> cartelle "Attori" con file audio)


emovo_root = os.path.join(root, "public", "EMOVO", "actors")
emovo = Emovo(emovo_root)
emovo.calculate_file_paths()

opensemov = Opensemov()
#opensemov.run(emovo.get_file_paths(), emovo.get_emotions())


'''csv_emovo = pd.read_csv(os.path.join(root, opensemov.get_csvpath()))
y = csv_emovo.pop('Emotion_Code')
X =csv_emovo.drop(['Emotion'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

emo_svc = Svc()

emo_svc.train(X_train, y_train)
print(emo_svc.evaluate(X_test, y_test))
y_pred = emo_svc.predict(X_test)
print(emo_svc.confusion_matrix(y_test, y_pred))
print(emo_svc.cross_validation(X,y))
#emo_svc.save_model(os.path.join(root, "models", "emo_aug_svc.pkl"))'''


emo_aug_svc = Svc().load_model(os.path.join(root, "models", "emo_aug_svc.pkl"))
#bio_svc = Svc().load_model(os.path.join(root, "models", "bio_svc.pkl"))

print(emo_aug_svc)
#print(bio_svc)

'''recorder = AudioRecorder(pyaudio.paInt16, 1, 44100, 1024, os.path.join(root, "vocals", "inbox", "perception_vocal_"))
mic = Microphone(recorder)
mic.start_mic()

# Esecuzione di altri task nel thread principale
try:
    while True:
        time.sleep(3)  # Esegui altri task
except KeyboardInterrupt:
    mic.stop_mic()
    print("Script terminato.")'''
    
'''opensperce = Opensperce()
dir = os.path.join(root,"vocals", "inbox",)
paths = [os.path.join(dir, file) for file in os.listdir(dir)]

#opensperce.run(paths)

csv_vocals = pd.read_csv(os.path.join(root, opensperce.get_csvpath()))
X =csv_vocals.drop(['file_name'], axis=1)

print(emo_aug_svc.predict(X))'''

import whisper
import ssl
import urllib.request

# Disabilita la verifica del certificato SSL
ssl._create_default_https_context = ssl._create_unverified_context

'''# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")'''

model = whisper.load_model("base")
result = model.transcribe(os.path.join(root, "vocals", "inbox", "perception_vocal_20241127_225911.wav"))
print(result["text"])