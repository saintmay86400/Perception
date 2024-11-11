class Dataset:
    def __init__(self, root):
        self._root = root
        self._file_paths = []
    
    def calculate_file_paths(self):
        
        raise NotImplementedError("Questo metodo deve essere implementato dalle sottoclassi")
    
    def set_file_paths(self, paths):
        if not isinstance(paths, list):
            raise ValueError("I percorsi devono essere in una lista.")
        self._file_paths = paths
    
    def get_file_paths(self):
        return self._file_paths
        