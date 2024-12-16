import pyaudio
import wave
from datetime import datetime
import threading

class AudioRecorder():
    def __init__(self, format, channels, rate, chunk, filename):
        self.format = format # Formato audio (es. 16-bit)
        self.channels = channels # Canali (es. 1 = mono)
        self.rate = rate # Frequenza di campionamento
        self.chunk = chunk # Dimensione del blocco
        self.filename = filename # Nome del file audio (prima parte)
        # Inizializza PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frame = []
        self.is_recording = False
        self.recording_thread = None
               
    def record(self):
        if self.is_recording:
            print("Registrazione già in corso!")
            return

        print("Avvio registrazione...")
        self.is_recording = True
        self.frames = []  # Per salvare i dati audio
        
        self.recording_thread = threading.Thread(target=self._record_loop)
        self.recording_thread.start()
            
    def _record_loop(self):
        try:
            self.stream = self.audio.open(format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk)
            
            while self.is_recording:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            print(f"Errore durante la registrazione: {e}")

        
        
    def stop_record(self):
        if not self.is_recording:
            print("Nessuna registrazione in corso!")
            return

        print("Interruzione registrazione...")
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.join()  # Aspetta che il thread termini
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        #self.audio.terminate()
        self.save()   
            
    def save(self):
        if self.frames:
            timestamp = datetime.now()
            with wave.open(self.filename+timestamp.strftime("%Y%m%d_%H%M%S")+'.wav', 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
            print(f"File salvato come: {self.filename}")
        else:
            print("Nessun audio registrato.")