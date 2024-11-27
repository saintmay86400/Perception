import os
from .dataset import Dataset

class Emovo(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self._emotions = {'neu':[1,'neutral'], 'gio':[2,'happy'], 'tri':[3,'sad']}
    
    def calculate_file_paths(self):
        # list dei path di tutti i file audio
         for folder in os.listdir(self._root): 
            if (folder.startswith('f') or folder.startswith('m')) :
                for file in os.listdir(os.path.join(self._root,folder)): 
                    if file.endswith('.wav') and file.split('-')[0] in self._emotions:
                                self._file_paths.append(os.path.join(self._root, folder, file))
                                
    def set_emotions(self, emotions):
        self._emotions = emotions
        
    def get_emotions(self):
        return self._emotions
        