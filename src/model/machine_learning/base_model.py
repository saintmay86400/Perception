from abc import ABC, abstractmethod
import pickle

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass
        
    def save_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Modello salvato in {file_path}.")

    @staticmethod
    def load_model(file_path):
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        print(f"Modello caricato da {file_path}.")
        return model