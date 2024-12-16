from abc import ABC, abstractmethod

# Interfaccia Message
class Message(ABC):
    
    @abstractmethod
    def get_content(self):
        
        pass