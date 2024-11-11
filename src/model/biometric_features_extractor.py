from abc import ABC, abstractmethod

class BiometricFeaturesExtractor(ABC):
    
    @abstractmethod
    def read_file(self, path):
        pass
    
    @abstractmethod
    def extract_features(self, signal, sampling_rate):
        pass