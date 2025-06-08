#!/usr/bin/env python3
"""
Collect "Jade" wake word samples for training
"""
import pyaudio
import wave
import os
import time
from pathlib import Path

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        
    def record_sample(self, duration=2, filename="sample.wav"):
        """Record a single sample"""
        print(f"Recording {filename} for {duration} seconds...")
        print("Say 'Jade' clearly after the beep!")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording now!")
        
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Save file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Saved {filename}")
    
    def collect_training_samples(self, num_positive=50, num_negative=30):
        """Collect training samples"""
        
        # Create directories
        Path("training_data/jade").mkdir(parents=True, exist_ok=True)
        Path("training_data/negative").mkdir(parents=True, exist_ok=True)
        
        print("=== Collecting POSITIVE samples (say 'Jade') ===")
        for i in range(num_positive):
            filename = f"training_data/jade/jade_{i:03d}.wav"
            self.record_sample(duration=2, filename=filename)
            time.sleep(1)
        
        print("\n=== Collecting NEGATIVE samples (say other words) ===")
        print("Say random words like: hello, computer, start, stop, etc.")
        for i in range(num_negative):
            filename = f"training_data/negative/negative_{i:03d}.wav"
            self.record_sample(duration=2, filename=filename)
            time.sleep(1)
        
        print("Data collection complete!")
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.audio.terminate()

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.collect_training_samples()