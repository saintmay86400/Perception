from abc import ABC, abstractmethod

class AudioFeaturesExtractor(ABC):
    @abstractmethod
    def read_audio(self, path):
        pass
            
    @abstractmethod
    def extract_features(self, signal, sampling_rate):
        pass
        



