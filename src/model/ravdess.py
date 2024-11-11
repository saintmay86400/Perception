import os
from .dataset import Dataset

class Ravdess(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self._emotions = {'01':'neutral', '02':'calm', '03':'happy', '04':'sad'}
        self._intensities = {'01':'normal', '02':'very'}
    
    def calculate_file_paths(self):
        # list dei path di tutti i file audio
         for folder in os.listdir(self._root): 
            if folder.startswith('Actor_'):
                for file in os.listdir(os.path.join(self._root,folder)): 
                    if file.endswith('.wav') and file.split('-')[2] in self._emotions:
                                self._file_paths.append(os.path.join(self._root, folder, file))
                                
    def set_emotions(self, emotions):
        self._emotions = emotions
        
    def set_intensities(self, intensities):
        self._intensities = intensities
        
    def get_emotions(self):
        return self._emotions
    
    def get_intensities(self):
        return self._intensities
        