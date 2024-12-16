from .message import Message

class MessageSender:
    
    def send(self, message: Message, recipient: str):
        
        print(f"Sending '{message.get_content()}' to {recipient}")