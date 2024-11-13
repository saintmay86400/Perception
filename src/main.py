import numpy as np
import os
from model import *


#root del progetto: /perception
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



####### WESAD + MY CUSTOM EXTRACTOR (Biometricfeat) #########
#############################################################

'''
wesad_root = os.path.join(root, "public", "WESAD")
wesad = Wesad(wesad_root)
wesad.calculate_file_paths()

biometricfeat = Biometricfeat()
biometricfeat.run(wesad.get_file_paths(), wesad.get_emotions())
'''

####### RAVEDES + OPENSMILE #########
#####################################

# path della cartella ravness (contiene-> cartelle "Attori" con file audio)
'''
ravdess_root = os.path.join(root, "public", "Audio_Speech_Actors_01-24")
ravdess = Ravdess(ravdess_root)
ravdess.calculate_file_paths()

opensmile = Opensmile()
#opensmile.run(ravdess.get_file_paths(), ravdess.get_emotions(), ravdess.get_intensities())
'''

print(os.path.dirname(root))