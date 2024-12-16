from .text_message import TextMessage
from .voice_message import VoiceMessage

class MessageFactory:
    @staticmethod
    def create_text_message(content):
        return TextMessage(content)

    @staticmethod
    def create_voice_message(content):
        return VoiceMessage(content)