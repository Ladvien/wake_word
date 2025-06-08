#!/usr/bin/env python3
"""
Package trained Jade model for Raspberry Pi deployment
"""
import shutil
import zipfile
from pathlib import Path
import json
import os
from datetime import datetime

def create_deployment_package():
    """Create deployment package for Raspberry Pi"""
    
    print("Creating Jade deployment package for Raspberry Pi...")
    
    # Create deployment directory
    deploy_dir = Path("jade_pi_deployment")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir(exist_ok=True)
    
    # Required files for deployment
    required_files = {
        "jade_model.onnx": "Trained ONNX model",
        "jade_model_metadata.json": "Model metadata", 
        "jade_training_config.yaml": "Training configuration"
    }
    
    missing_files = []
    
    # Copy model files
    for file, description in required_files.items():
        if Path(file).exists():
            shutil.copy2(file, deploy_dir / file)
            print(f"✓ Copied {file} ({description})")
        else:
            missing_files.append(file)
            print(f"✗ Missing {file} ({description})")
    
    if missing_files:
        print(f"\nError: Missing required files: {missing_files}")
        print("Please run train_jade_model.py first to generate the model files.")
        return False
    
    # Create deployment scripts
    create_pi_detector_script(deploy_dir)
    create_setup_script(deploy_dir)
    create_systemd_service(deploy_dir)
    create_requirements_file(deploy_dir)
    create_readme(deploy_dir)
    
    # Create deployment info
    deployment_info = {
        "package_name": "jade_wake_word_pi",
        "version": "1.0.0",
        "created_date": datetime.now().isoformat(),
        "target_platform": "Raspberry Pi 4",
        "python_version": ">=3.8",
        "description": "Jade wake word detection for Raspberry Pi using custom trained model",
        "files": list(required_files.keys()) + [
            "jade_pi_detector.py",
            "setup_pi.sh", 
            "requirements.txt",
            "jade-wake-word.service",
            "README.md"
        ]
    }
    
    with open(deploy_dir / "deployment_info.json", 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    # Create deployment zip
    zip_path = "jade_model_deployment.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in deploy_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(deploy_dir))
    
    print(f"\n✅ Deployment package created: {zip_path}")
    print(f"📁 Package contents: {len(list(deploy_dir.rglob('*')))} files")
    print("\nTo deploy to Raspberry Pi:")
    print(f"1. Transfer {zip_path} to your Raspberry Pi")
    print("2. Extract: unzip jade_model_deployment.zip")
    print("3. Run setup: chmod +x setup_pi.sh && ./setup_pi.sh")
    
    return True

def create_pi_detector_script(deploy_dir):
    """Create the main detector script for Raspberry Pi"""
    
    detector_script = '''#!/usr/bin/env python3
"""
Jade Wake Word Detector for Raspberry Pi
Production deployment using ONNX model
"""
import asyncio
import pyaudio
import numpy as np
import onnxruntime as ort
import librosa
import json
import httpx
import time
import logging
from pathlib import Path
from collections import deque
import threading
import signal
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/jade_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JadeWakeWordDetector:
    def __init__(self, model_path="jade_model.onnx", config_path="jade_model_metadata.json"):
        # Load model metadata
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.chunk_duration = self.config.get('chunk_duration_ms', 1280) / 1000.0
        self.chunk_size = int(self.sample_rate * 0.064)  # 64ms chunks for real-time processing
        self.buffer_duration = self.chunk_duration
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.threshold = self.config.get('threshold', 0.5)
        
        # Desktop connection settings
        self.desktop_url = "http://192.168.1.100:8000"  # Update with your desktop IP
        
        # Load ONNX model
        try:
            self.ort_session = ort.InferenceSession(model_path)
            logger.info(f"Loaded ONNX model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Circular buffer for audio processing
        self.audio_buffer = deque(maxlen=int(self.buffer_size // self.chunk_size))
        
        # State management
        self.is_listening = False
        self.is_streaming = False
        self.wake_word_detected = False
        
        # Detection statistics
        self.detection_count = 0
        self.false_positive_count = 0
        self.last_detection_time = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    def extract_features(self, audio_data):
        """Extract features from audio data"""
        try:
            # Convert to float
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=80,
                n_fft=1024,
                hop_length=160,
                win_length=400
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec)
            
            # Flatten and resize to match model input size
            features = log_mel.flatten()
            
            # Resize to 768 dimensions (model input size)
            if len(features) > 768:
                features = features[:768]
            elif len(features) < 768:
                features = np.pad(features, (0, 768 - len(features)), 'constant')
            
            # Normalize
            if np.std(features) > 0:
                features = (features - np.mean(features)) / np.std(features)
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def detect_wake_word(self, audio_chunk):
        """Detect if wake word is present in audio chunk"""
        # Add to circular buffer
        self.audio_buffer.append(audio_chunk)
        
        # Need enough data for analysis
        if len(self.audio_buffer) < self.audio_buffer.maxlen:
            return False
        
        # Combine recent audio data
        combined_audio = np.concatenate(list(self.audio_buffer))
        
        # Extract features
        features = self.extract_features(combined_audio)
        
        # Run inference
        try:
            input_data = features.reshape(1, -1)
            prediction = self.ort_session.run(None, {'features': input_data})[0][0][0]
            
            # Check if above threshold
            is_wake_word = prediction > self.threshold
            
            if is_wake_word:
                # Prevent multiple detections in short time
                current_time = time.time()
                if current_time - self.last_detection_time > 2.0:  # 2 second cooldown
                    self.detection_count += 1
                    self.last_detection_time = current_time
                    logger.info(f"Wake word detected! Confidence: {prediction:.4f}, Count: {self.detection_count}")
                    return True
                else:
                    # Too soon after last detection
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return False
    
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
            logger.info(f"Audio stream started (chunk_size: {self.chunk_size}, rate: {self.sample_rate})")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to numpy array