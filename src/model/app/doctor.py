from .user import User
from groq import Groq
from .perception_assistant import PerceptionAssistant
import os

class Doctor(User):
    def __init__(self, username, user_id):
        super().__init__(username, user_id)
        self.client = Groq(api_key = os.environ.get("GROQ_API_KEY"))
        self.model = "llama3-8b-8192"
        self.brain = None
        self.assistant = None
        
    def set_assistant(self, assistant: PerceptionAssistant):
        self.assistant = assistant
        
    def readMessage(self, content):
        self.brain = self.client.chat.completions.create(
            messages = [
                {
                    "role":"user",
                    "content":content,
                }
            ],
            model=self.model,)
    
    def makeAnsware(self):
        return self.brain.choices[0].message.content
        
        
    