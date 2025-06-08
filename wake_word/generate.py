#!/usr/bin/env python3
"""
Wake Word Synthetic Data Generator
Generates training data for wake word detection models using various TTS engines

Usage:
    python -m wake_word.generate
    python -m wake_word.generate --wake-word "computer" --samples 1000
    python wake_word/generate.py --config custom_config.yaml
"""
import os
import sys
import yaml
import random
import numpy as np
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm
import soundfile as sf
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse

# Get project root (parent of wake_word directory)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Audio processing with fallback
try:
    import librosa
    HAS_LIBROSA = True
    print("✓ Using librosa for audio processing")
except ImportError:
    print("⚠ librosa not available, using scipy fallback")
    import scipy.signal
    import scipy.io.wavfile as wav
    HAS_LIBROSA = False
    
    class LibrosaFallback:
        @staticmethod
        def load(file_path, sr=16000):
            try:
                import soundfile as sf
                audio, orig_sr = sf.read(file_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
            except ImportError:
                orig_sr, audio = wav.read(file_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32) / 32768.0
            
            if orig_sr != sr:
                num_samples = int(len(audio) * sr / orig_sr)
                audio = scipy.signal.resample(audio, num_samples)
            return audio.astype(np.float32), sr
        
        class effects:
            @staticmethod
            def time_stretch(y, rate):
                if rate == 1.0:
                    return y
                new_length = int(len(y) / rate)
                return scipy.signal.resample(y, new_length)
    
    librosa = LibrosaFallback()


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = CONFIG_PATH
        
        self.config_path = Path(config_path)
        self.project_root = PROJECT_ROOT
        self.config = self._load_and_validate_config()
        self._resolve_paths()
    
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please ensure config.yaml exists in the project root: {PROJECT_ROOT}"
            )
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        
        # Validate required sections
        required_sections = ['project', 'wake_word', 'audio', 'data_generation', 'paths']
        missing = [section for section in required_sections if section not in config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        return config
    
    def _resolve_paths(self):
        """Resolve all paths relative to project root"""
        paths_config = self.config.get('paths', {})
        
        # Resolve relative paths
        for key, path_str in paths_config.items():
            if isinstance(path_str, str) and not Path(path_str).is_absolute():
                paths_config[key] = str(self.project_root / path_str)
        
        # Ensure output directories exist
        for key in ['output_dir', 'models_dir', 'data_dir', 'logs_dir']:
            if key in paths_config:
                Path(paths_config[key]).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_path(self, key_path: str, default=None) -> Optional[Path]:
        """Get path from config and return as Path object"""
        path_str = self.get(key_path, default)
        if path_str:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return None
    
    def update(self, key_path: str, value):
        """Update configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


class TTSEngine(ABC):
    """Abstract base class for Text-to-Speech engines"""
    
    @abstractmethod
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        """Generate audio samples for given text variants"""
        pass


class PiperTTSEngine(TTSEngine):
    """Piper TTS engine implementation"""
    
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        try:
            # Setup Piper path
            piper_path = config.get_path('paths.piper_sample_generator')
            if not piper_path or not piper_path.exists():
                print(f"⚠ Piper not found at {piper_path}, trying fallback engine")
                return 0
            
            # Add to Python path
            sys.path.insert(0, str(piper_path))
            from generate_samples import generate_samples
            
            total_generated = 0
            n_samples = config.get('data_generation.n_samples', 1000)
            samples_per_variant = max(1, n_samples // len(text_variants))
            
            piper_config = config.get('data_generation.tts.piper', {})
            model_path = config.project_root / piper_config.get('model_path', '')
            
            if not model_path.exists():
                print(f"⚠ Piper model not found at {model_path}")
                return 0
            
            print(f"🎙️ Using Piper TTS with model: {model_path.name}")
            
            for variant in text_variants:
                variant_dir = output_dir / self._sanitize_filename(variant)
                variant_dir.mkdir(exist_ok=True)
                
                try:
                    generate_samples(
                        text=[variant],
                        max_samples=samples_per_variant,
                        output_dir=str(variant_dir),
                        model_path=str(model_path),
                        batch_size=piper_config.get('batch_size', 10),
                        max_speakers=piper_config.get('max_speakers', 200),
                        length_scales=piper_config.get('length_scales', [1.0]),
                        noise_scales=piper_config.get('noise_scales', [0.333])
                    )
                    total_generated += samples_per_variant
                    print(f"  ✓ Generated {samples_per_variant} samples for '{variant}'")
                    
                except Exception as e:
                    print(f"  ✗ Error generating samples for '{variant}': {e}")
                    continue
            
            return total_generated
            
        except ImportError as e:
            print(f"⚠ Piper TTS import failed: {e}")
            return 0
        except Exception as e:
            print(f"⚠ Piper TTS failed: {e}")
            return 0
    
    def _sanitize_filename(self, text: str) -> str:
        """Sanitize text for use as filename"""
        return text.replace(" ", "_").replace(",", "").replace(".", "").replace("?", "").replace("!", "")


class PyTTSX3Engine(TTSEngine):
    """PyTTSX3 TTS engine implementation"""
    
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if not voices:
                print("⚠ No system voices available for PyTTSX3")
                return 0
            
            pyttsx3_config = config.get('data_generation.tts.pyttsx3', {})
            voices_to_use = min(pyttsx3_config.get('voices_to_use', 3), len(voices))
            speech_rates = pyttsx3_config.get('speech_rates', [180])
            
            print(f"🎙️ Using PyTTSX3 with {voices_to_use} voices")
            
            total_generated = 0
            
            for i, phrase in enumerate(text_variants):
                for voice_idx in range(voices_to_use):
                    engine.setProperty('voice', voices[voice_idx].id)
                    
                    for rate_idx, rate in enumerate(speech_rates):
                        engine.setProperty('rate', rate)
                        filename = output_dir / f"pyttsx3_{i:03d}_{voice_idx}_{rate_idx}.wav"
                        
                        try:
                            engine.save_to_file(phrase, str(filename))
                            engine.runAndWait()
                            total_generated += 1
                        except Exception as e:
                            print(f"  ✗ Error with voice {voice_idx}, rate {rate}: {e}")
                            continue
            
            print(f"  ✓ Generated {total_generated} samples using PyTTSX3")
            return total_generated
            
        except ImportError:
            print("⚠ PyTTSX3 not available")
            return 0
        except Exception as e:
            print(f"⚠ PyTTSX3 failed: {e}")
            return 0


class KokoroTTSEngine(TTSEngine):
    """Kokoro TTS engine using Hugging Face Spaces API"""
    
    def __init__(self):
        self.api_url = "https://hexgrad-kokoro-tts.hf.space/api/predict"
        self.session = None
        
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        try:
            import requests
            import io
            import time
            
            print("🎙️ Using Kokoro TTS from Hugging Face Spaces")
            
            # Get configuration
            n_samples = config.get('data_generation.n_samples', 1000)
            samples_per_variant = max(1, n_samples // len(text_variants))
            
            # Kokoro TTS configuration
            kokoro_config = config.get('data_generation.tts.kokoro', {})
            voice_presets = kokoro_config.get('voice_presets', ['af_sarah', 'am_adam', 'bf_emma', 'bm_lewis'])
            speed_variations = kokoro_config.get('speed_variations', [0.9, 1.0, 1.1])
            
            total_generated = 0
            session = requests.Session()
            
            for variant_idx, variant in enumerate(text_variants):
                print(f"  📝 Generating samples for '{variant}'...")
                variant_samples = 0
                
                while variant_samples < samples_per_variant:
                    # Select voice and speed variation
                    voice = random.choice(voice_presets)
                    speed = random.choice(speed_variations)
                    
                    try:
                        # Prepare API request
                        payload = {
                            "data": [
                                variant,  # text
                                voice,    # voice preset
                                speed,    # speed
                                0,        # pitch (0 = default)
                                "mp3"     # format
                            ]
                        }
                        
                        # Make API request
                        response = session.post(
                            self.api_url, 
                            json=payload,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # The API returns a data structure with the audio file path
                            if 'data' in result and len(result['data']) > 0:
                                audio_data = result['data'][0]
                                
                                # Download the generated audio
                                if isinstance(audio_data, str) and audio_data.startswith('http'):
                                    audio_response = session.get(audio_data, timeout=30)
                                    
                                    if audio_response.status_code == 200:
                                        # Save the audio file
                                        filename = output_dir / f"kokoro_{variant_idx}_{voice}_{variant_samples:03d}.wav"
                                        
                                        # Convert MP3 to WAV if needed
                                        if audio_data.endswith('.mp3'):
                                            # Save as MP3 first, then convert
                                            mp3_filename = filename.with_suffix('.mp3')
                                            with open(mp3_filename, 'wb') as f:
                                                f.write(audio_response.content)
                                            
                                            # Convert to WAV using ffmpeg or pydub
                                            try:
                                                import subprocess
                                                subprocess.run([
                                                    'ffmpeg', '-y', '-i', str(mp3_filename), 
                                                    '-ar', '16000', '-ac', '1', str(filename)
                                                ], check=True, capture_output=True)
                                                mp3_filename.unlink()  # Remove MP3 file
                                            except (subprocess.CalledProcessError, FileNotFoundError):
                                                # Fallback: try with pydub
                                                try:
                                                    from pydub import AudioSegment
                                                    audio = AudioSegment.from_mp3(str(mp3_filename))
                                                    audio = audio.set_frame_rate(16000).set_channels(1)
                                                    audio.export(str(filename), format="wav")
                                                    mp3_filename.unlink()
                                                except ImportError:
                                                    print(f"  ⚠ Could not convert MP3 to WAV. Install ffmpeg or pydub.")
                                                    filename.with_suffix('.mp3').rename(filename.with_suffix('.mp3'))
                                        else:
                                            # Save directly as WAV
                                            with open(filename, 'wb') as f:
                                                f.write(audio_response.content)
                                        
                                        variant_samples += 1
                                        total_generated += 1
                                        
                                        if variant_samples % 10 == 0:
                                            print(f"    ✓ Generated {variant_samples}/{samples_per_variant} samples")
                                    else:
                                        print(f"  ⚠ Failed to download audio: {audio_response.status_code}")
                                else:
                                    print(f"  ⚠ Unexpected audio data format: {type(audio_data)}")
                            else:
                                print(f"  ⚠ No audio data in API response")
                        else:
                            print(f"  ⚠ API request failed: {response.status_code}")
                            print(f"    Response: {response.text[:200]}...")
                    
                    except requests.exceptions.RequestException as e:
                        print(f"  ⚠ Request error: {e}")
                    except Exception as e:
                        print(f"  ⚠ Error generating sample: {e}")
                    
                    # Rate limiting - be nice to the API
                    time.sleep(random.uniform(1, 3))
                    
                    # Safety break if we're not making progress
                    if variant_samples == 0 and total_generated == 0:
                        print(f"  ⚠ Could not generate any samples, trying fallback...")
                        break
                
                print(f"  ✓ Generated {variant_samples} samples for '{variant}' using voice '{voice}'")
            
            session.close()
            return total_generated
            
        except ImportError:
            print("⚠ requests library not available for Kokoro TTS")
            return 0
        except Exception as e:
            print(f"⚠ Kokoro TTS failed: {e}")
            return 0


class HuggingFaceTTSEngine(TTSEngine):
    """Generic Hugging Face TTS engine using transformers"""
    
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        try:
            from transformers import pipeline
            import torch
            
            print("🎙️ Using Hugging Face transformers TTS")
            
            # Initialize TTS pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts_config = config.get('data_generation.tts.huggingface', {})
            model_name = tts_config.get('model', 'microsoft/speecht5_tts')
            
            tts = pipeline("text-to-speech", model=model_name, device=device)
            
            n_samples = config.get('data_generation.n_samples', 1000)
            samples_per_variant = max(1, n_samples // len(text_variants))
            
            total_generated = 0
            
            for variant_idx, variant in enumerate(text_variants):
                print(f"  📝 Generating samples for '{variant}'...")
                
                for i in range(samples_per_variant):
                    try:
                        # Generate speech
                        result = tts(variant)
                        
                        # Save audio
                        filename = output_dir / f"hf_tts_{variant_idx}_{i:03d}.wav"
                        
                        # Extract audio data and sample rate
                        audio_data = result["audio"]
                        sample_rate = result["sampling_rate"]
                        
                        # Save using soundfile
                        sf.write(filename, audio_data, sample_rate)
                        total_generated += 1
                        
                        if (i + 1) % 10 == 0:
                            print(f"    ✓ Generated {i+1}/{samples_per_variant} samples")
                            
                    except Exception as e:
                        print(f"  ⚠ Error generating sample {i}: {e}")
                        continue
                
                print(f"  ✓ Generated {samples_per_variant} samples for '{variant}'")
            
            return total_generated
            
        except ImportError:
            print("⚠ transformers library not available for Hugging Face TTS")
            return 0
        except Exception as e:
            print(f"⚠ Hugging Face TTS failed: {e}")
            return 0
    """Fallback TTS engine that generates synthetic audio"""
    
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        print("🔊 Using synthetic audio generation (fallback)")
        
        wake_word = config.get('wake_word.primary', 'wake_word')
        fallback_config = config.get('data_generation.tts.fallback', {})
        n_samples = fallback_config.get('n_placeholder_samples', 100)
        sr = config.get('audio.sample_rate', 16000)
        
        total_generated = 0
        
        for variant in text_variants:
            samples_for_variant = n_samples // len(text_variants)
            
            for i in range(samples_for_variant):
                audio = self._create_synthetic_audio(variant, sr, 1.5)
                filename = output_dir / f"synthetic_{self._sanitize_text(variant)}_{i:03d}.wav"
                sf.write(filename, audio, sr)
                total_generated += 1
        
        print(f"  ✓ Generated {total_generated} synthetic samples")
        return total_generated
    
    def _create_synthetic_audio(self, text: str, sr: int, duration: float) -> np.ndarray:
        """Create synthetic audio that resembles speech patterns"""
        t = np.linspace(0, duration, int(sr * duration))
        
        # Improved phonetic mapping with more realistic formants
        phonetic_freqs = {
            'jade': [
                (300, 2200, 0.4, 0.3),   # /dʒ/ - voiced palato-alveolar affricate
                (500, 1800, 0.5, 0.4),   # /eɪ/ - diphthong
                (250, 1700, 0.3, 0.2)    # /d/ - voiced alveolar stop
            ],
            'computer': [
                (350, 1500, 0.3, 0.2),   # /k/
                (400, 1200, 0.4, 0.3),   # /ʌ/
                (500, 1800, 0.3, 0.3),   # /m/
                (450, 1600, 0.4, 0.3),   # /p/
                (550, 1900, 0.4, 0.3),   # /ju/
                (300, 1400, 0.3, 0.2),   # /t/
                (400, 1500, 0.3, 0.3)    # /ər/
            ],
            'hey': [
                (500, 1500, 0.3, 0.2),   # /h/
                (500, 1800, 0.5, 0.4)    # /eɪ/
            ],
        }
        
        # Extract primary word for phonetic mapping
        primary_word = text.split()[0].lower() if text else 'default'
        freqs = phonetic_freqs.get(primary_word, [(400, 1500, 0.3, 0.3), (500, 1200, 0.4, 0.3)])
        
        audio = np.zeros_like(t)
        segment_length = len(t) // len(freqs)
        
        # Create fundamental frequency (pitch)
        f0 = 120 + 30 * np.sin(2 * np.pi * 2 * t)  # Varying pitch around 120Hz
        
        for i, (f1, f2, intensity1, intensity2) in enumerate(freqs):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(t))
            segment_t = t[start_idx:end_idx]
            segment_f0 = f0[start_idx:end_idx]
            
            # Create voiced speech with harmonics
            segment_audio = np.zeros_like(segment_t)
            
            # Add harmonics (like a vocal tract)
            for harmonic in range(1, 6):  # First 5 harmonics
                harmonic_freq = segment_f0 * harmonic
                
                # Formant filtering - emphasize frequencies near F1 and F2
                f1_response = 1.0 / (1.0 + ((harmonic_freq - f1) / 100) ** 2)
                f2_response = 1.0 / (1.0 + ((harmonic_freq - f2) / 150) ** 2)
                
                amplitude = (intensity1 * f1_response + intensity2 * f2_response) / harmonic
                segment_audio += amplitude * np.sin(2 * np.pi * harmonic_freq * segment_t)
            
            # Add speech-like envelope (attack, sustain, release)
            envelope_length = len(segment_t)
            attack_length = int(0.1 * envelope_length)
            release_length = int(0.2 * envelope_length)
            sustain_length = envelope_length - attack_length - release_length
            
            envelope = np.ones(envelope_length)
            
            # Attack
            if attack_length > 0:
                envelope[:attack_length] = np.linspace(0, 1, attack_length)
            
            # Release
            if release_length > 0:
                envelope[-release_length:] = np.linspace(1, 0, release_length)
            
            # Add some amplitude modulation for natural speech
            mod_freq = 5 + 3 * np.random.random()  # 5-8 Hz modulation
            amplitude_mod = 1 + 0.2 * np.sin(2 * np.pi * mod_freq * segment_t)
            envelope *= amplitude_mod
            
            segment_audio *= envelope
            audio[start_idx:end_idx] = segment_audio
        
        # Add some breathiness/noise for realism
        noise = 0.05 * np.random.normal(0, 1, len(audio))
        audio += noise
        
        # Normalize to reasonable amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8  # Increase amplitude
        
        return audio
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for filename"""
        return "".join(c for c in text if c.isalnum() or c in (' ', '_')).replace(' ', '_')


class SyntheticDataGenerator:
    """Main class for generating synthetic wake word training data"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = ConfigManager(config_path)
        self._setup_logging()
        self._setup_directories()
        self._setup_tts_engines()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create logs directory
        log_file = self.config.get_path('paths.logs_dir') / 'data_generation.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler() if log_config.get('console', True) else logging.NullHandler(),
                logging.FileHandler(log_file) if log_config.get('file', True) else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized data generator with config: {self.config.config_path}")
    
    def _setup_directories(self):
        """Setup all required directories"""
        # Main output directories
        self.output_dir = self.config.get_path('paths.output_dir')
        self.data_dir = self.config.get_path('paths.data_dir')
        
        # Training data directories
        self.positive_dir = self.data_dir / "raw_positive"
        self.negative_dir = self.data_dir / "raw_negative"
        self.augmented_dir = self.data_dir / "augmented"
        self.background_dir = self.data_dir / "background"
        
        # Final training directories
        training_config = self.config.get('paths.training_data', {})
        self.training_positive = Path(training_config.get('positive', self.data_dir / "training/positive"))
        self.training_negative = Path(training_config.get('negative', self.data_dir / "training/negative"))
        self.training_validation = Path(training_config.get('validation', self.data_dir / "validation"))
        
        # Create all directories
        for directory in [
            self.positive_dir, self.negative_dir, self.augmented_dir, self.background_dir,
            self.training_positive, self.training_negative, self.training_validation
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Setup directories under: {self.data_dir}")
    
    def _setup_tts_engines(self):
        """Setup TTS engines in order of preference"""
        self.tts_engines = [
            KokoroTTSEngine(),
            HuggingFaceTTSEngine(),
            PiperTTSEngine(),
            PyTTSX3Engine(),
            FallbackTTSEngine()
        ]
        
        engine_name = self.config.get('data_generation.tts.engine', 'kokoro')
        self.logger.info(f"Preferred TTS engine: {engine_name}")
    
    def generate_positive_samples(self) -> int:
        """Generate positive wake word samples"""
        self.logger.info("🎯 Generating positive wake word samples...")
        
        wake_word_variants = self.config.get('wake_word.variants', [])
        if not wake_word_variants:
            primary = self.config.get('wake_word.primary', 'wake_word')
            wake_word_variants = [primary]
        
        print(f"📝 Wake word variants: {wake_word_variants}")
        
        total_generated = 0
        engine_name = self.config.get('data_generation.tts.engine', 'piper')
        
        # Try engines in order
        for engine in self.tts_engines:
            if engine_name.lower() in engine.__class__.__name__.lower() or engine_name == 'fallback':
                try:
                    generated = engine.generate_samples(wake_word_variants, self.positive_dir, self.config)
                    if generated > 0:
                        total_generated = generated
                        break
                except Exception as e:
                    self.logger.warning(f"Engine {engine.__class__.__name__} failed: {e}")
                    continue
        
        if total_generated == 0:
            self.logger.warning("All preferred engines failed, using fallback")
            fallback_engine = FallbackTTSEngine()
            total_generated = fallback_engine.generate_samples(wake_word_variants, self.positive_dir, self.config)
        
        # Debug: Check what files exist before consolidation
        all_files = list(self.positive_dir.rglob("*.wav"))
        self.logger.info(f"Found {len(all_files)} audio files before consolidation")
        for f in all_files[:5]:  # Log first 5 files
            self.logger.debug(f"Found file: {f}")
        
        # Consolidate samples
        consolidated = self._consolidate_samples()
        self.logger.info(f"✅ Generated {total_generated} samples, consolidated {consolidated} positive samples")
        return consolidated
    
    def _consolidate_samples(self) -> int:
        """Consolidate samples from subdirectories"""
        sample_count = 0
        wake_word = self.config.get('wake_word.primary', 'sample')
        
        # First, count direct samples in positive_dir (from fallback engine)
        direct_samples = list(self.positive_dir.glob("*.wav"))
        for wav_file in direct_samples:
            if not wav_file.name.startswith(f"{wake_word}_"):
                # Rename to standard format
                new_name = self.positive_dir / f"{wake_word}_{sample_count:06d}.wav"
                shutil.move(str(wav_file), str(new_name))
                sample_count += 1
            else:
                # Already in correct format
                sample_count += 1
        
        # Then, move files from subdirectories to main positive directory (from Piper/PyTTSX3)
        for subdir in self.positive_dir.iterdir():
            if subdir.is_dir():
                for wav_file in subdir.glob("*.wav"):
                    new_name = self.positive_dir / f"{wake_word}_{sample_count:06d}.wav"
                    shutil.move(str(wav_file), str(new_name))
                    sample_count += 1
                
                # Remove empty subdirectory
                try:
                    subdir.rmdir()
                except OSError:
                    pass
        
        return sample_count
    
    def generate_background_data(self):
        """Generate background audio for negative samples"""
        self.logger.info("🎵 Generating background audio data...")
        
        if not self.config.get('background_audio.enabled', True):
            self.logger.info("Background audio generation disabled")
            return
        
        self._generate_noise_samples()
        self._generate_speech_samples()
        self._generate_music_samples()
        
        self.logger.info("✅ Background data generation complete")
    
    def _generate_noise_samples(self):
        """Generate various noise types"""
        noise_config = self.config.get('background_audio.noise', {})
        noise_dir = self.background_dir / "noise"
        noise_dir.mkdir(exist_ok=True)
        
        sr = self.config.get('audio.sample_rate', 16000)
        duration = noise_config.get('duration', 10.0)
        n_samples = noise_config.get('n_samples', 100)
        noise_types = noise_config.get('types', ['white_noise'])
        
        print(f"🔊 Generating {n_samples} noise samples...")
        
        for i in tqdm(range(n_samples), desc="Noise samples"):
            noise_type = random.choice(noise_types)
            audio = self._create_noise_audio(noise_type, duration, sr)
            filename = noise_dir / f"{noise_type}_{i:03d}.wav"
            sf.write(filename, audio, sr)
    
    def _generate_speech_samples(self):
        """Generate synthetic speech-like audio"""
        speech_config = self.config.get('background_audio.speech', {})
        speech_dir = self.background_dir / "speech"
        speech_dir.mkdir(exist_ok=True)
        
        sr = self.config.get('audio.sample_rate', 16000)
        duration = speech_config.get('duration', 8.0)
        n_samples = speech_config.get('n_samples', 50)
        
        print(f"🗣️ Generating {n_samples} speech-like samples...")
        
        for i in tqdm(range(n_samples), desc="Speech samples"):
            audio = self._create_speech_audio(speech_config, sr, duration)
            filename = speech_dir / f"background_speech_{i:03d}.wav"
            sf.write(filename, audio, sr)
    
    def _generate_music_samples(self):
        """Generate synthetic music audio"""
        music_config = self.config.get('background_audio.music', {})
        music_dir = self.background_dir / "music"
        music_dir.mkdir(exist_ok=True)
        
        sr = self.config.get('audio.sample_rate', 16000)
        duration = music_config.get('duration', 12.0)
        n_samples = music_config.get('n_samples', 30)
        
        print(f"🎼 Generating {n_samples} music samples...")
        
        for i in tqdm(range(n_samples), desc="Music samples"):
            audio = self._create_music_audio(music_config, sr, duration)
            filename = music_dir / f"background_music_{i:03d}.wav"
            sf.write(filename, audio, sr)
    
    def _create_noise_audio(self, noise_type: str, duration: float, sr: int) -> np.ndarray:
        """Create specific type of noise audio"""
        samples = int(sr * duration)
        
        if noise_type == "white_noise":
            return np.random.normal(0, 0.1, samples)
        elif noise_type == "pink_noise":
            white = np.random.normal(0, 1, samples)
            freqs = np.fft.fftfreq(samples, 1/sr)
            freqs[0] = 1
            fft = np.fft.fft(white)
            pink_fft = fft / np.sqrt(np.abs(freqs))
            return np.real(np.fft.ifft(pink_fft)) * 0.1
        elif noise_type == "brown_noise":
            white = np.random.normal(0, 0.01, samples)
            brown = np.cumsum(white)
            return brown / np.max(np.abs(brown)) * 0.1 if np.max(np.abs(brown)) > 0 else brown
        elif noise_type == "fan_noise":
            t = np.linspace(0, duration, samples)
            return (0.1 * np.sin(2 * np.pi * 60 * t) + 
                   0.05 * np.random.normal(0, 1, samples))
        elif noise_type == "traffic_noise":
            t = np.linspace(0, duration, samples)
            audio = np.zeros(samples)
            for f in [30, 80, 150, 300]:
                amplitude = 0.02 * np.random.uniform(0.5, 1.5)
                audio += amplitude * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2*np.pi))
            return audio + 0.03 * np.random.normal(0, 1, samples)
        else:  # cafe_ambience or default
            audio = 0.02 * np.random.normal(0, 1, samples)
            for _ in range(random.randint(3, 8)):
                start = random.randint(0, samples - sr)
                burst_len = random.randint(sr//4, sr)
                burst = 0.05 * np.random.normal(0, 1, burst_len)
                end_idx = min(start + burst_len, samples)
                audio[start:end_idx] += burst[:end_idx-start]
            return audio
    
    def _create_speech_audio(self, speech_config: Dict, sr: int, duration: float) -> np.ndarray:
        """Create synthetic speech-like audio"""
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = np.zeros(samples)
        
        formants = speech_config.get('formants', [[400, 1500]])
        segment_duration = speech_config.get('segment_duration', 0.5)
        segments_per_sample = int(duration / segment_duration)
        
        for seg in range(segments_per_sample):
            start_idx = int(seg * segment_duration * sr)
            end_idx = min(int((seg + 1) * segment_duration * sr), samples)
            
            f1, f2 = random.choice(formants)
            f1 *= random.uniform(0.8, 1.2)
            f2 *= random.uniform(0.8, 1.2)
            
            seg_t = t[start_idx:end_idx]
            segment = (0.1 * np.sin(2 * np.pi * f1 * seg_t) + 
                      0.05 * np.sin(2 * np.pi * f2 * seg_t))
            
            mod_freq = random.uniform(3, 8)
            amplitude_mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * seg_t)
            segment *= amplitude_mod
            
            audio[start_idx:end_idx] = segment
        
        return audio + 0.01 * np.random.normal(0, 1, samples)
    
    def _create_music_audio(self, music_config: Dict, sr: int, duration: float) -> np.ndarray:
        """Create synthetic music audio"""
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = np.zeros(samples)
        
        notes = music_config.get('notes', [440.0])  # Default to A4
        chord_duration = music_config.get('chord_duration', 2.0)
        chords_per_sample = int(duration / chord_duration)
        
        for chord_idx in range(chords_per_sample):
            start_idx = int(chord_idx * chord_duration * sr)
            end_idx = min(int((chord_idx + 1) * chord_duration * sr), samples)
            
            chord_notes = random.sample(notes, min(3, len(notes)))
            chord_t = t[start_idx:end_idx]
            chord_audio = np.zeros(len(chord_t))
            
            for note_freq in chord_notes:
                note_audio = (0.3 * np.sin(2 * np.pi * note_freq * chord_t) +
                             0.15 * np.sin(2 * np.pi * note_freq * 2 * chord_t) +
                             0.075 * np.sin(2 * np.pi * note_freq * 3 * chord_t))
                
                envelope = np.exp(-chord_t * 0.5)
                note_audio *= envelope
                chord_audio += note_audio
            
            audio[start_idx:end_idx] = chord_audio
        
        # Add reverb
        reverb_delay = int(0.1 * sr)
        if len(audio) > reverb_delay:
            reverb_audio = np.zeros(len(audio))
            reverb_audio[reverb_delay:] = audio[:-reverb_delay] * 0.3
            audio += reverb_audio
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.2
        
        return audio
    
    def augment_samples(self) -> int:
        """Apply augmentation to positive samples"""
        self.logger.info("🔧 Applying data augmentation...")
        
        if not self.config.get('augmentation.enabled', True):
            self.logger.info("Augmentation disabled, copying original samples")
            return self._copy_original_samples()
        
        positive_files = list(self.positive_dir.glob("*.wav"))
        if not positive_files:
            self.logger.warning("No positive samples found for augmentation")
            return 0
        
        augmentation_types = self.config.get('augmentation.types', ['original'])
        total_augmented = 0
        
        print(f"🔄 Augmenting {len(positive_files)} samples with {len(augmentation_types)} techniques...")
        
        for audio_file in tqdm(positive_files, desc="Augmenting"):
            try:
                audio, sr = librosa.load(audio_file, sr=self.config.get('audio.sample_rate'))
                
                for aug_type in augmentation_types:
                    augmented_audio = self._apply_augmentation(audio, sr, aug_type)
                    
                    if augmented_audio is not None:
                        output_path = self.augmented_dir / f"{audio_file.stem}_{aug_type}.wav"
                        sf.write(output_path, augmented_audio, sr)
                        total_augmented += 1
                        
            except Exception as e:
                self.logger.warning(f"Error augmenting {audio_file}: {e}")
                continue
        
        self.logger.info(f"✅ Created {total_augmented} augmented samples")
        return total_augmented
    
    def _copy_original_samples(self) -> int:
        """Copy original samples without augmentation"""
        positive_files = list(self.positive_dir.glob("*.wav"))
        
        for audio_file in positive_files:
            shutil.copy2(audio_file, self.augmented_dir / f"{audio_file.stem}_original.wav")
        
        return len(positive_files)
    
    def _apply_augmentation(self, audio: np.ndarray, sr: int, aug_type: str) -> Optional[np.ndarray]:
        """Apply specific augmentation type"""
        try:
            params = self.config.get('augmentation.parameters', {})
            
            if aug_type == "original":
                return audio
            elif aug_type == "volume_low":
                vol_range = params.get('volume_low_range', [0.3, 0.6])
                return audio * random.uniform(*vol_range)
            elif aug_type == "volume_high":
                vol_range = params.get('volume_high_range', [1.2, 1.5])
                return audio * random.uniform(*vol_range)
            elif aug_type == "speed_slow":
                rate = params.get('speed_slow_rate', 0.9)
                return librosa.effects.time_stretch(audio, rate=rate)
            elif aug_type == "speed_fast":
                rate = params.get('speed_fast_rate', 1.1)
                return librosa.effects.time_stretch(audio, rate=rate)
            elif aug_type == "background_noise":
                noise_amp = params.get('noise_amplitude', 0.005)
                noise = np.random.normal(0, noise_amp, len(audio))
                return audio + noise
            elif aug_type == "room_reverb":
                reverb_delay_ms = params.get('reverb_delay_ms', 100)
                reverb_amp = params.get('reverb_amplitude', 0.3)
                reverb_delay = int(reverb_delay_ms * sr / 1000)
                
                reverb_audio = np.zeros(len(audio) + reverb_delay)
                reverb_audio[:len(audio)] = audio
                reverb_audio[reverb_delay:] += audio * reverb_amp
                return reverb_audio[:len(audio)]
            else:
                self.logger.warning(f"Unknown augmentation type: {aug_type}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error in augmentation {aug_type}: {e}")
            return None
    
    def prepare_training_data(self) -> Tuple[int, int]:
        """Prepare final training data structure"""
        self.logger.info("📁 Preparing training data structure...")
        
        # Copy augmented positive samples to training directory
        augmented_files = list(self.augmented_dir.glob("*.wav"))
        for i, wav_file in enumerate(augmented_files):
            target_path = self.training_positive / f"positive_{i:06d}.wav"
            shutil.copy2(wav_file, target_path)
        
        # Generate negative samples
        negative_count = self._create_negative_samples()
        positive_count = len(augmented_files)
        
        self.logger.info(f"✅ Training data prepared: {positive_count} positive, {negative_count} negative")
        return positive_count, negative_count
    
    def _create_negative_samples(self) -> int:
        """Create negative samples from background data"""
        print("🔄 Creating negative samples from background data...")
        
        # Collect background files
        background_files = list(self.background_dir.rglob("*.wav"))
        
        # Calculate number of negative samples needed
        n_positive = len(list(self.augmented_dir.glob("*.wav")))
        if n_positive == 0:
            n_positive = self.config.get('data_generation.n_samples', 1000)
        
        negative_multiplier = self.config.get('data_generation.negative_multiplier', 2)
        n_negative = n_positive * negative_multiplier
        
        # Audio parameters
        chunk_duration = self.config.get('audio.chunk_duration_ms', 1280) / 1000.0
        sr = self.config.get('audio.sample_rate', 16000)
        chunk_samples = int(chunk_duration * sr)
        
        negative_count = 0
        
        for i in tqdm(range(n_negative), desc="Creating negative samples"):
            if background_files:
                # Use background audio
                bg_file = random.choice(background_files)
                try:
                    bg_audio, _ = librosa.load(bg_file, sr=sr)
                    
                    if len(bg_audio) > chunk_samples:
                        start_idx = random.randint(0, len(bg_audio) - chunk_samples)
                        chunk = bg_audio[start_idx:start_idx + chunk_samples]
                    else:
                        chunk = np.tile(bg_audio, (chunk_samples // len(bg_audio)) + 1)[:chunk_samples]
                    
                    if np.max(np.abs(chunk)) > 0:
                        chunk = chunk / np.max(np.abs(chunk)) * 0.7
                    
                except Exception as e:
                    self.logger.warning(f"Error loading {bg_file}: {e}")
                    chunk = self._create_synthetic_negative_chunk(chunk_samples, sr)
            else:
                chunk = self._create_synthetic_negative_chunk(chunk_samples, sr)
            
            output_path = self.training_negative / f"negative_{negative_count:06d}.wav"
            sf.write(output_path, chunk, sr)
            negative_count += 1
        
        return negative_count
    
    def _create_synthetic_negative_chunk(self, chunk_samples: int, sr: int) -> np.ndarray:
        """Create synthetic negative audio chunk"""
        chunk = np.random.normal(0, 0.1, chunk_samples)
        
        duration = chunk_samples / sr
        t = np.linspace(0, duration, chunk_samples)
        
        # Add speech-like frequencies
        freqs = [150, 350, 800, 1500]
        for freq in freqs:
            amplitude = random.uniform(0.02, 0.08)
            phase = random.uniform(0, 2 * np.pi)
            chunk += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        envelope = np.random.uniform(0.3, 1.0, chunk_samples)
        chunk *= envelope
        
        if np.max(np.abs(chunk)) > 0:
            chunk = chunk / np.max(np.abs(chunk)) * 0.5
        
        return chunk
    
    def run_full_pipeline(self) -> bool:
        """Run the complete synthetic data generation pipeline"""
        self.logger.info("🚀 Starting synthetic data generation pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Generate positive samples
            print("\n📝 Step 1: Generating positive samples...")
            positive_count = self.generate_positive_samples()
            
            if positive_count == 0:
                self.logger.error("❌ Failed to generate positive samples")
                return False
            
            # Step 2: Generate background data
            print("\n🎵 Step 2: Generating background data...")
            self.generate_background_data()
            
            # Step 3: Augment positive samples
            print("\n🔧 Step 3: Augmenting positive samples...")
            augmented_count = self.augment_samples()
            
            # Step 4: Prepare final training data
            print("\n📁 Step 4: Preparing training data...")
            pos_count, neg_count = self.prepare_training_data()
            
            # Step 5: Generate summary
            self._generate_summary(pos_count, neg_count)
            
            print("\n" + "=" * 60)
            print("✅ Synthetic data generation pipeline completed successfully!")
            print(f"📊 Dataset Summary:")
            print(f"   • Positive samples: {pos_count:,}")
            print(f"   • Negative samples: {neg_count:,}")
            print(f"   • Total samples: {pos_count + neg_count:,}")
            print(f"   • Ratio (pos:neg): 1:{neg_count//pos_count if pos_count > 0 else 0}")
            print(f"📁 Output directory: {self.output_dir}")
            print(f"🎯 Ready for training!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_summary(self, pos_count: int, neg_count: int):
        """Generate summary of data generation process"""
        summary = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'config_file': str(self.config.config_path),
                'wake_word': self.config.get('wake_word.primary'),
                'variants': self.config.get('wake_word.variants'),
            },
            'dataset_statistics': {
                'positive_samples': pos_count,
                'negative_samples': neg_count,
                'total_samples': pos_count + neg_count,
                'positive_negative_ratio': f"1:{neg_count//pos_count if pos_count > 0 else 0}",
                'augmentation_enabled': self.config.get('augmentation.enabled', True),
                'tts_engine_used': self.config.get('data_generation.tts.engine', 'unknown')
            },
            'audio_settings': self.config.get('audio', {}),
            'paths': {
                'training_positive': str(self.training_positive),
                'training_negative': str(self.training_negative),
                'validation': str(self.training_validation)
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "data_generation_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"📄 Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic wake word training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wake_word.generate
  python -m wake_word.generate --wake-word "computer" --samples 1000
  python wake_word/generate.py --config custom_config.yaml --output-dir ./my_model
        """
    )
    
    parser.add_argument("--config", type=Path, help="Path to configuration file (default: config.yaml in project root)")
    parser.add_argument("--wake-word", help="Override wake word from config")
    parser.add_argument("--samples", type=int, help="Override number of samples to generate")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--engine", choices=['kokoro', 'huggingface', 'piper', 'pyttsx3', 'fallback'], help="Override TTS engine")
    parser.add_argument("--no-augment", action='store_true', help="Disable data augmentation")
    parser.add_argument("--test-mode", action='store_true', help="Generate small dataset for testing")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        print("🎙️ Wake Word Synthetic Data Generator")
        print("=" * 50)
        print(f"📁 Project root: {PROJECT_ROOT}")
        print(f"⚙️ Config file: {args.config or CONFIG_PATH}")
        print()
        
        generator = SyntheticDataGenerator(args.config)
        
        # Apply command line overrides
        if args.wake_word:
            print(f"🔄 Overriding wake word: {args.wake_word}")
            generator.config.update('wake_word.primary', args.wake_word)
            # Update variants
            variants = [args.wake_word, f"hey {args.wake_word}", f"{args.wake_word} please"]
            generator.config.update('wake_word.variants', variants)
        
        if args.samples:
            print(f"🔄 Overriding sample count: {args.samples}")
            generator.config.update('data_generation.n_samples', args.samples)
        
        if args.output_dir:
            print(f"🔄 Overriding output directory: {args.output_dir}")
            generator.config.update('paths.output_dir', str(args.output_dir))
            generator._setup_directories()
        
        if args.engine:
            print(f"🔄 Overriding TTS engine: {args.engine}")
            generator.config.update('data_generation.tts.engine', args.engine)
        
        if args.no_augment:
            print("🔄 Disabling data augmentation")
            generator.config.update('augmentation.enabled', False)
        
        if args.test_mode:
            print("🧪 Test mode: generating small dataset")
            generator.config.update('data_generation.n_samples', 50)
            generator.config.update('background_audio.noise.n_samples', 20)
            generator.config.update('background_audio.speech.n_samples', 10)
            generator.config.update('background_audio.music.n_samples', 10)
        
        # Run the pipeline
        success = generator.run_full_pipeline()
        
        if success:
            print("\n🎉 Success! Next steps:")
            print("  1. Review generated data in the output directory")
            print("  2. Run the training script: python -m wake_word.train")
            print("  3. Test the trained model: python -m wake_word.test")
            return 0
        else:
            print("\n❌ Data generation failed. Check logs for details.")
            return 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())