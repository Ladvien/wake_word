# wake_word/generate.py

import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List
import time

from wake_word.tts_base import TTSEngine

class KokoroONNXTTSEngine(TTSEngine):
    """Kokoro TTS engine using local ONNX model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.voice_models = {}
        self.sample_rate = 24000
        
    def _load_model(self):
        """Load Kokoro ONNX model and tokenizer"""
        try:
            import onnxruntime as ort
            
            print("🎙️ Loading Kokoro ONNX model...")
            
            # Try to import kokoro-onnx package first
            try:
                from kokoro_onnx import KokoroONNX
                self.model = KokoroONNX()
                print("✓ Using kokoro-onnx package")
            except ImportError:
                # Fallback: try to load ONNX model directly
                print("⚠ kokoro-onnx package not found, trying direct ONNX loading...")
                return self._load_direct_onnx()
            
            # Available voices in Kokoro
            self.available_voices = [
                'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
                'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 
                'bm_george', 'bm_lewis'
            ]
            
            print(f"✓ Kokoro ONNX loaded with {len(self.available_voices)} voices")
            return True
            
        except ImportError as e:
            print(f"⚠ Kokoro ONNX dependencies missing: {e}")
            print("  Install with: pip install onnxruntime")
            print("  For kokoro-onnx: pip install kokoro-onnx")
            return False
        except Exception as e:
            print(f"⚠ Failed to load Kokoro ONNX: {e}")
            return False
    
    def _load_direct_onnx(self):
        """Load ONNX model directly if kokoro-onnx package is not available"""
        try:
            import onnxruntime as ort
            
            # Look for ONNX model files in common locations
            possible_paths = [
                "kokoro-v0_19.onnx",
                "models/kokoro-v0_19.onnx",
                "./kokoro-v0_19.onnx",
                "../kokoro-v0_19.onnx"
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                print("⚠ No Kokoro ONNX model found. Please download kokoro-v0_19.onnx")
                print("  Expected locations: kokoro-v0_19.onnx, models/kokoro-v0_19.onnx")
                return False
            
            print(f"🎙️ Loading Kokoro ONNX model from {model_path}...")
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(model_path)
            
            # Get input/output info
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            print(f"✓ Direct ONNX model loaded - Inputs: {self.input_names}")
            
            # Limited voices for direct ONNX (would need specific implementation)
            self.available_voices = ['af_sarah', 'am_adam', 'bf_emma', 'bm_lewis']
            
            return True
            
        except Exception as e:
            print(f"⚠ Failed to load direct ONNX model: {e}")
            return False
    
    def generate_samples(self, text_variants: List[str], output_dir: Path, config: ConfigManager) -> int:
        """Generate audio samples using Kokoro ONNX"""
        
        # Load model if not already loaded
        if self.model is None and not hasattr(self, 'session'):
            if not self._load_model():
                print("⚠ Could not load Kokoro ONNX model, falling back to next engine")
                return 0
        
        # Get configuration
        n_samples = config.get('data_generation.n_samples', 1000)
        samples_per_variant = max(1, n_samples // len(text_variants))
        
        # Kokoro TTS configuration
        kokoro_config = config.get('data_generation.tts.kokoro', {})
        voice_presets = kokoro_config.get('voice_presets', self.available_voices[:4])
        
        # Validate voice presets
        valid_voices = [v for v in voice_presets if v in self.available_voices]
        if not valid_voices:
            print(f"⚠ No valid voices found, using defaults")
            valid_voices = self.available_voices[:4]
        
        print(f"🎙️ Using Kokoro ONNX with voices: {valid_voices}")
        
        total_generated = 0
        target_sample_rate = config.get('audio.sample_rate', 16000)
        
        try:
            for variant_idx, variant in enumerate(text_variants):
                print(f"  📝 Generating samples for '{variant}'...")
                variant_samples = 0
                
                while variant_samples < samples_per_variant:
                    # Select voice variation
                    voice = random.choice(valid_voices)
                    
                    try:
                        # Generate audio with Kokoro ONNX
                        audio = self._generate_audio(variant, voice, kokoro_config)
                        
                        if audio is not None:
                            # Resample if needed
                            if self.sample_rate != target_sample_rate:
                                audio = self._resample_audio(audio, self.sample_rate, target_sample_rate)
                            
                            # Save audio file
                            filename = output_dir / f"kokoro_{variant_idx}_{voice}_{variant_samples:03d}.wav"
                            
                            # Save as WAV file
                            sf.write(
                                filename,
                                audio,
                                target_sample_rate,
                                subtype='PCM_16'
                            )
                            
                            variant_samples += 1
                            total_generated += 1
                            
                            if variant_samples % 10 == 0:
                                print(f"    ✓ Generated {variant_samples}/{samples_per_variant} samples")
                        else:
                            print(f"  ⚠ Failed to generate audio for '{variant}' with voice {voice}")
                    
                    except Exception as e:
                        print(f"  ⚠ Error generating sample with voice {voice}: {e}")
                        # Continue with next attempt
                        continue
                    
                    # Small delay to prevent overheating
                    time.sleep(0.1)
                
                print(f"  ✓ Generated {variant_samples} samples for '{variant}'")
            
            return total_generated
            
        except Exception as e:
            print(f"⚠ Kokoro ONNX TTS failed: {e}")
            return total_generated
    
    def _generate_audio(self, text: str, voice: str, config: dict) -> np.ndarray:
        """Generate audio for given text and voice"""
        try:
            # Get generation parameters
            speed = config.get('speed', 1.0)
            temperature = config.get('temperature', 0.7)
            
            # Add some variation
            speed_variations = config.get('speed_variations', [1.0])
            if speed_variations:
                speed = random.choice(speed_variations)
            
            temperature_variations = config.get('temperature_variations', [temperature])
            if temperature_variations:
                temperature = random.choice(temperature_variations)
            
            # Clean text
            text = self._clean_text(text)
            
            # Generate audio using Kokoro ONNX
            if self.model is not None:
                # Using kokoro-onnx package
                audio_data = self.model.generate(
                    text=text,
                    voice=voice,
                    speed=speed,
                    temperature=temperature
                )
            elif hasattr(self, 'session'):
                # Using direct ONNX - this would need specific implementation
                # based on the actual model inputs/outputs
                print(f"⚠ Direct ONNX inference not implemented yet")
                return None
            else:
                return None
            
            # Normalize audio to prevent clipping
            if audio_data is not None and len(audio_data) > 0:
                # Convert to float32 and normalize
                audio_data = np.array(audio_data, dtype=np.float32)
                
                # Normalize to [-0.95, 0.95] to prevent clipping
                max_val = np.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data * (0.95 / max_val)
                
                return audio_data
            
            return None
            
        except Exception as e:
            print(f"  ⚠ Audio generation error: {e}")
            return None
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio
        
        try:
            # Try using librosa for resampling
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: simple resampling using scipy
            try:
                import scipy.signal
                num_samples = int(len(audio) * target_sr / orig_sr)
                return scipy.signal.resample(audio, num_samples).astype(np.float32)
            except ImportError:
                print("⚠ Cannot resample audio: install librosa or scipy")
                return audio
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for TTS"""
        # Basic text cleaning
        text = text.strip()
        
        # Remove or replace problematic characters
        replacements = {
            '"': '',
            '"': '',
            '"': '',
            ''': "'",
            ''': "'",
            '—': '-',
            '–': '-',
            '…': '...',
            '\n': ' ',
            '\t': ' ',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure text ends with punctuation for better prosody
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text


# Also update the _setup_tts_engines method in SyntheticDataGenerator class:
def _setup_tts_engines(self):
    """Setup TTS engines in order of preference"""
    self.tts_engines = [
        KokoroONNXTTSEngine(),  # Use the ONNX version instead
        PiperTTSEngine(),
        PyTTSX3Engine(),
        FallbackTTSEngine()
    ]
    
    engine_name = self.config.get('data_generation.tts.engine', 'kokoro')
    self.logger.info(f"Preferred TTS engine: {engine_name}")