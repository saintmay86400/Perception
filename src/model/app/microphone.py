from pynput import keyboard
from .audio_recorder import AudioRecorder

class Microphone():
    def __init__(self, audio_recorder: AudioRecorder):
        self.activation_key = keyboard.KeyCode(char='q')
        self.audio_recorder = audio_recorder
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
    
    def on_press(self, key):
        if key == self.activation_key:
            try:
                self.audio_recorder.record()
            
            except AttributeError:
                print("errore")
    
    def on_release(self, key):
        if key == self.activation_key:
            self.audio_recorder.stop_record()
        
    def start_mic(self):
        self.listener.start()
        print('sono pronto')
        #self.listener.join()
        
    def stop_mic(self):
        self.listener.stop()
            
    
            