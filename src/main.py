import config
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import svm
from model import *
from sklearn.model_selection import train_test_split
import pyaudio
import time



####### WESAD + MY CUSTOM EXTRACTOR (Biometricfeat) #########
#############################################################


'''wesad_root = os.path.join(config.root, "public", "WESAD")
wesad = Wesad(wesad_root)
wesad.calculate_file_paths()

biometricfeat = Biometricfeat(wesad)

csv_biometrics = pd.read_csv(os.path.join(config.root, biometricfeat.get_csvpath()))
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
print('Precisione media di predizione: ', bio_svc.evaluate(T_test, z_test))
z_pred = bio_svc.predict(T_test)
print('Matrice di confusione: \n', bio_svc.confusion_matrix(z_test, z_pred)'''



#print(bio_svc.cross_validation(T,z))
#bio_svc.save_model(os.path.join(root, "models", "bio_svc.pkl"))


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


'''emovo_root = os.path.join(config.root, "public", "EMOVO", "actors")
emovo = Emovo(emovo_root)
emovo.calculate_file_paths()

opensemov = Opensemov(emovo)
#opensemov.run()


csv_emovo = pd.read_csv(os.path.join(config.root, opensemov.get_csvpath()))
y = csv_emovo.pop('Emotion_Code')
X =csv_emovo.drop(['Emotion'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

emo_svc = Svc()

emo_svc.train(X_train, y_train)
print('Precisione media di predizione: ',emo_svc.evaluate(X_test, y_test))
y_pred = emo_svc.predict(X_test)
print('Matrice di confusione: \n', emo_svc.confusion_matrix(y_test, y_pred))


#print(emo_svc.cross_validation(X,y))
emo_svc.save_model(os.path.join(config.root, "models", "emovo_aug_svc.pkl"))'''


#emo_aug_svc = Svc().load_model(os.path.join(config.root, "models", "emovo_aug_svc.pkl"))
#bio_svc = Svc().load_model(os.path.join(root, "models", "bio_svc.pkl"))

#print(emo_aug_svc)
#print(bio_svc)


'''recorder = AudioRecorder(pyaudio.paInt16, 1, 44100, 1024, os.path.join(config.root, "vocals", "inbox", "perception_vocal_"))
mic = Microphone(recorder)
mic.start_mic()

# Esecuzione di altri task nel thread principale
try:
    while True:
        time.sleep(3)  # Esegui altri task
except KeyboardInterrupt:
    mic.stop_mic()
    print("Script terminato.")'''


'''emo_aug_svc = Svc().load_model(os.path.join(config.root, "models", "emovo_aug_svc.pkl"))

opensperce = Opensperce()
dir = os.path.join(config.root,"vocals", "inbox",)
paths = [os.path.join(dir, file) for file in os.listdir(dir)]

opensperce.run(paths)

csv_vocals = pd.read_csv(os.path.join(config.root, opensperce.get_csvpath()))
X =csv_vocals.drop(['file_name'], axis=1)

print(emo_aug_svc.predict(X))'''


'''model = whisper.load_model("base")
result = model.transcribe(os.path.join(root, "vocals", "inbox", "perception_vocal_20241127_225930.wav"))
print(result["text"])'''

'''from groq import Groq
client = Groq(
    api_key = os.environ.get("GROQ_API_KEY")
              )
chat_completion = client.chat.completions.create(
    messages = [
        {
            "role":"user",
            "content":"il gatto è morto",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)'''
