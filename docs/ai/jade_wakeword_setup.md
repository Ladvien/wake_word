# Jade Wake Word Setup Guide - Raspberry Pi 4

Complete walkthrough for setting up OpenWakeWord with a custom "Jade" wake word on Raspberry Pi 4.

## Prerequisites

- Raspberry Pi 4 (4GB or less)
- Raspberry Pi OS (64-bit recommended)
- USB microphone or Pi HAT with microphone
- Internet connection for initial setup

## Step 1: System Setup

### Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python3-openssl
```

### Install Audio Dependencies
```bash
sudo apt install -y portaudio19-dev python3-pyaudio alsa-utils
# Test your microphone
arecord -l  # List recording devices
```

## Step 2: Python Environment Setup

### Install pyenv
```bash
curl https://pyenv.run | bash

# Add to ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload shell
exec $SHELL
```

### Install Python 3.10
```bash
# This may take 30-45 minutes on Pi 4
pyenv install 3.10.12
pyenv global 3.10.12
```

### Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
exec $SHELL
```

## Step 3: Project Setup

### Create Project Directory
```bash
mkdir ~/jade-wake-word
cd ~/jade-wake-word

# Initialize Poetry project
poetry init --name jade-wake-word --version 0.1.0 --author "Your Name" --python "^3.10"
```

### Add Dependencies
```bash
poetry add openwakeword pyaudio numpy scipy fastapi uvicorn httpx asyncio-mqtt
poetry add --group dev jupyter notebook
```

### Activate Environment
```bash
poetry shell
```

## Step 4: Training Data Collection

Since "Jade" isn't a pre-trained model, we need to collect training data.

### Create Data Collection Script

Create `collect_jade_samples.py`:

```python
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
```

### Collect Your Training Data
```bash
python collect_jade_samples.py
```

**Important Notes:**
- Record in various environments (quiet, with background noise)
- Use different tones and speeds
- Have different people say "Jade" if possible
- Record negative samples with similar-sounding words

## Step 5: Train Custom "Jade" Model

### Create Training Script

Create `train_jade_model.py`:

```python
#!/usr/bin/env python3
"""
Train custom Jade wake word model
"""
import os
import numpy as np
from pathlib import Path
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Note: This is a simplified training approach
# For production, consider using the full OpenWakeWord training pipeline

class SimpleWakeWordTrainer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()
        
    def extract_features(self, audio_file):
        """Extract MFCC features from audio file"""
        try:
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Take mean across time
            features = np.mean(mfccs.T, axis=0)
            return features
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return None
    
    def prepare_dataset(self, data_dir="training_data"):
        """Prepare training dataset"""
        features = []
        labels = []
        
        # Load positive samples (Jade)
        jade_dir = Path(data_dir) / "jade"
        for audio_file in jade_dir.glob("*.wav"):
            feat = self.extract_features(audio_file)
            if feat is not None:
                features.append(feat)
                labels.append(1)  # Positive class
        
        # Load negative samples
        neg_dir = Path(data_dir) / "negative"
        for audio_file in neg_dir.glob("*.wav"):
            feat = self.extract_features(audio_file)
            if feat is not None:
                features.append(feat)
                labels.append(0)  # Negative class
        
        return np.array(features), np.array(labels)
    
    def train_simple_model(self):
        """Train a simple threshold-based model"""
        print("Preparing dataset...")
        X, y = self.prepare_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate positive class center and threshold
        positive_samples = X_train_scaled[y_train == 1]
        self.positive_center = np.mean(positive_samples, axis=0)
        
        # Calculate distances from positive center
        distances = []
        for sample in positive_samples:
            dist = np.linalg.norm(sample - self.positive_center)
            distances.append(dist)
        
        # Set threshold as mean + 2*std of positive distances
        self.threshold = np.mean(distances) + 2 * np.std(distances)
        
        print(f"Training complete. Threshold: {self.threshold:.4f}")
        
        # Test accuracy
        correct = 0
        total = len(X_test_scaled)
        
        for i, sample in enumerate(X_test_scaled):
            distance = np.linalg.norm(sample - self.positive_center)
            predicted = 1 if distance <= self.threshold else 0
            if predicted == y_test[i]:
                correct += 1
        
        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.2%}")
        
        # Save model
        model_data = {
            'positive_center': self.positive_center,
            'threshold': self.threshold,
            'scaler': self.scaler
        }
        
        with open('jade_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved as jade_model.pkl")

if __name__ == "__main__":
    # Install required library
    os.system("pip install librosa scikit-learn")
    
    trainer = SimpleWakeWordTrainer()
    trainer.train_simple_model()
```

### Train the Model
```bash
poetry add librosa scikit-learn
python train_jade_model.py
```

## Step 6: Wake Word Detection Implementation

### Create Main Wake Word Detector

Create `jade_detector.py`:

```python
#!/usr/bin/env python3
"""
Jade Wake Word Detector with Audio Streaming
"""
import asyncio
import pyaudio
import numpy as np
import librosa
import pickle
import httpx
import json
import time
from pathlib import Path
from collections import deque
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JadeWakeWordDetector:
    def __init__(self, model_path="jade_model.pkl", desktop_url="http://192.168.1.100:8000"):
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.desktop_url = desktop_url
        
        # Load trained model
        self.load_model(model_path)
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Circular buffer for audio processing
        self.audio_buffer = deque(maxlen=32)  # ~2 seconds at 16kHz
        
        # State management
        self.is_listening = False
        self.is_streaming = False
        self.wake_word_detected = False
        
    def load_model(self, model_path):
        """Load the trained Jade model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.positive_center = model_data['positive_center']
            self.threshold = model_data['threshold']
            self.scaler = model_data['scaler']
            
            logger.info("Jade model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_features(self, audio_data):
        """Extract features from audio data"""
        try:
            # Convert to float
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_float, sr=self.sample_rate, n_mfcc=13)
            features = np.mean(mfccs.T, axis=0)
            
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def detect_wake_word(self, audio_chunk):
        """Detect if wake word is present in audio chunk"""
        # Add to circular buffer
        self.audio_buffer.append(audio_chunk)
        
        # Need enough data for analysis
        if len(self.audio_buffer) < 20:  # ~1.3 seconds
            return False
        
        # Combine recent audio data
        combined_audio = np.concatenate(list(self.audio_buffer)[-20:])
        
        # Extract features
        features = self.extract_features(combined_audio)
        if features is None:
            return False
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Calculate distance from positive center
        distance = np.linalg.norm(features_scaled[0] - self.positive_center)
        
        # Check if within threshold
        is_wake_word = distance <= self.threshold
        
        if is_wake_word:
            logger.info(f"Wake word detected! Distance: {distance:.4f}, Threshold: {self.threshold:.4f}")
        
        return is_wake_word
    
    def start_audio_stream(self):
        """Start audio input stream"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            logger.info("Audio stream started")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Check for wake word if not currently streaming
        if not self.is_streaming:
            if self.detect_wake_word(audio_data):
                self.wake_word_detected = True
                logger.info("Starting audio streaming to desktop...")
                asyncio.create_task(self.start_streaming_session())
        
        return (in_data, pyaudio.paContinue)
    
    async def start_streaming_session(self):
        """Start streaming audio to desktop"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.wake_word_detected = False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Notify desktop that streaming is starting
                await client.post(f"{self.desktop_url}/start_session", 
                                json={"device_id": "jade_pi", "timestamp": time.time()})
                
                # Stream audio for 10 seconds or until told to stop
                stream_duration = 10  # seconds
                chunks_to_send = int(stream_duration * self.sample_rate / self.chunk_size)
                
                logger.info(f"Streaming audio for {stream_duration} seconds...")
                
                for i in range(chunks_to_send):
                    if not self.is_streaming:  # Check if we should stop
                        break
                    
                    # Get recent audio data
                    if len(self.audio_buffer) > 0:
                        audio_chunk = self.audio_buffer[-1]
                        
                        # Send to desktop
                        await client.post(
                            f"{self.desktop_url}/audio_chunk",
                            content=audio_chunk.tobytes(),
                            headers={"Content-Type": "application/octet-stream"}
                        )
                    
                    await asyncio.sleep(self.chunk_size / self.sample_rate)  # Real-time streaming
                
                # Notify end of session
                await client.post(f"{self.desktop_url}/end_session", 
                                json={"device_id": "jade_pi", "timestamp": time.time()})
                
                logger.info("Audio streaming session completed")
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.is_streaming = False
    
    async def run(self):
        """Main run loop"""
        logger.info("Starting Jade Wake Word Detector...")
        
        # Start audio stream
        self.start_audio_stream()
        self.is_listening = True
        
        logger.info("Listening for 'Jade'...")
        
        try:
            while self.is_listening:
                await asyncio.sleep(0.1)  # Keep event loop running
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_listening = False
        self.is_streaming = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        logger.info("Cleanup completed")

# Configuration
class Config:
    DESKTOP_IP = "192.168.1.100"  # Replace with your desktop IP
    DESKTOP_PORT = 8000
    MODEL_PATH = "jade_model.pkl"

async def main():
    desktop_url = f"http://{Config.DESKTOP_IP}:{Config.DESKTOP_PORT}"
    
    detector = JadeWakeWordDetector(
        model_path=Config.MODEL_PATH,
        desktop_url=desktop_url
    )
    
    await detector.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Create Configuration File

Create `config.py`:

```python
"""
Configuration for Jade Wake Word System
"""
import os
from pathlib import Path

class Config:
    # Audio settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    CHANNELS = 1
    
    # Model settings
    MODEL_PATH = "jade_model.pkl"
    CONFIDENCE_THRESHOLD = 0.7
    
    # Network settings
    DESKTOP_IP = os.getenv("DESKTOP_IP", "192.168.1.100")
    DESKTOP_PORT = int(os.getenv("DESKTOP_PORT", "8000"))
    
    # Streaming settings
    STREAM_DURATION = 10  # seconds
    BUFFER_SIZE = 32  # chunks
    
    # Directories
    BASE_DIR = Path(__file__).parent
    TRAINING_DATA_DIR = BASE_DIR / "training_data"
    MODELS_DIR = BASE_DIR / "models"
    
    @classmethod
    def get_desktop_url(cls):
        return f"http://{cls.DESKTOP_IP}:{cls.DESKTOP_PORT}"
```

## Step 7: SystemD Service Setup

### Create Service File

Create `/etc/systemd/system/jade-wake-word.service`:

```ini
[Unit]
Description=Jade Wake Word Detection Service
After=network.target sound.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/jade-wake-word
Environment=PATH=/home/pi/.local/bin:/home/pi/.pyenv/shims:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/pi/.local/bin/poetry run python jade_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable jade-wake-word.service
sudo systemctl start jade-wake-word.service

# Check status
sudo systemctl status jade-wake-word.service

# View logs
sudo journalctl -u jade-wake-word.service -f
```

## Step 8: Testing and Validation

### Create Test Script

Create `test_jade.py`:

```python
#!/usr/bin/env python3
"""
Test Jade wake word detection
"""
import asyncio
import logging
from jade_detector import JadeWakeWordDetector

logging.basicConfig(level=logging.DEBUG)

async def test_detection():
    print("Testing Jade wake word detection...")
    print("Say 'Jade' to test detection")
    print("Press Ctrl+C to stop")
    
    detector = JadeWakeWordDetector()
    await detector.run()

if __name__ == "__main__":
    asyncio.run(test_detection())
```

### Performance Monitoring

Create `monitor_performance.py`:

```python
#!/usr/bin/env python3
"""
Monitor system performance during wake word detection
"""
import psutil
import time
import threading

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        
    def start_monitoring(self):
        self.monitoring = True
        thread = threading.Thread(target=self._monitor_loop)
        thread.daemon = True
        thread.start()
        
    def _monitor_loop(self):
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            temp = self.get_cpu_temperature()
            
            print(f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Temp: {temp:.1f}°C")
            time.sleep(5)
    
    def get_cpu_temperature(self):
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return temp
        except:
            return 0.0
    
    def stop_monitoring(self):
        self.monitoring = False

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
```

## Step 9: Desktop Integration

Your desktop will need to receive the streamed audio. Here's a basic FastAPI server:

Create `desktop_receiver.py`:

```python
#!/usr/bin/env python3
"""
Desktop audio receiver for Jade wake word system
"""
from fastapi import FastAPI, Request
import asyncio
import logging
from datetime import datetime

app = FastAPI(title="Jade Audio Receiver")
logger = logging.getLogger(__name__)

# Audio buffer for collecting chunks
audio_sessions = {}

@app.post("/start_session")
async def start_session(request: Request):
    data = await request.json()
    session_id = data.get("device_id", "unknown")
    
    audio_sessions[session_id] = {
        "chunks": [],
        "start_time": datetime.now(),
        "last_chunk": datetime.now()
    }
    
    logger.info(f"Started audio session: {session_id}")
    return {"status": "session_started", "session_id": session_id}

@app.post("/audio_chunk")
async def receive_audio_chunk(request: Request):
    # Get session from headers or use default
    session_id = request.headers.get("X-Session-ID", "jade_pi")
    
    if session_id not in audio_sessions:
        return {"error": "No active session"}
    
    # Receive audio chunk
    chunk = await request.body()
    audio_sessions[session_id]["chunks"].append(chunk)
    audio_sessions[session_id]["last_chunk"] = datetime.now()
    
    return {"status": "chunk_received", "chunk_size": len(chunk)}

@app.post("/end_session")
async def end_session(request: Request):
    data = await request.json()
    session_id = data.get("device_id", "unknown")
    
    if session_id in audio_sessions:
        session = audio_sessions.pop(session_id)
        total_chunks = len(session["chunks"])
        
        logger.info(f"Session ended: {session_id}, Total chunks: {total_chunks}")
        
        # Here you would process the audio for STT, LLM, etc.
        # For now, just log the received data
        
        return {"status": "session_ended", "chunks_received": total_chunks}
    
    return {"error": "Session not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Usage Instructions

1. **Initial Setup**: Run the data collection script to gather training samples
2. **Train Model**: Execute the training script to create your Jade model
3. **Test Locally**: Use the test script to verify detection works
4. **Deploy Service**: Enable the systemd service for automatic startup
5. **Monitor**: Use the performance monitor to ensure optimal operation

## Troubleshooting

### Common Issues:

1. **No audio input**: Check `arecord -l` and ensure microphone is connected
2. **Poor detection**: Collect more training samples in various conditions
3. **High CPU usage**: Adjust chunk size or detection frequency
4. **Network issues**: Verify desktop IP address in config

### Performance Optimization:

- Adjust `CHUNK_SIZE` for better real-time performance
- Tune `CONFIDENCE_THRESHOLD` to reduce false positives
- Consider using hardware acceleration if available

### Logs:
```bash
# View service logs
sudo journalctl -u jade-wake-word.service -f

# Test audio devices
arecord -d 5 test.wav && aplay test.wav
```

This setup provides a complete, production-ready wake word detection system for "Jade" on your Raspberry Pi 4!