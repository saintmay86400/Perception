import os
from .dataset import Dataset

class Wesad(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self._emotions = {0:'not_defined/transient', 1:'baseline', 2:'stress', 3:'amusement', 4:'meditation'}
    
    def calculate_file_paths(self):
        # list dei path di tutti i file audio
         for folder in os.listdir(self._root): 
            if folder.startswith('S'):
                for file in os.listdir(os.path.join(self._root,folder)): 
                    if file.endswith('.pkl'):
                                self._file_paths.append(os.path.join(self._root, folder, file))
    
    def set_emotions(self, emotions):
        self._emotions = emotions
    
    def get_emotions(self):
        return self._emotions
        
