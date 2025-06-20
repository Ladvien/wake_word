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
from datetime import datetime

from wake_word.config import ConfigManager
from wake_word.engines import KokoroTTSEngine


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


class SyntheticDataGenerator:
    """Main class for generating synthetic wake word training data"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = ConfigManager(config_path)
        self._setup_logging()
        self._setup_directories()
        self._setup_tts_engines()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))

        # Create logs directory
        log_file = self.config.get_path("paths.logs_dir") / "data_generation.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=log_level,
            format=log_config.get(
                "format", "%(asctime)s - %(levelname)s - %(message)s"
            ),
            handlers=[
                (
                    logging.StreamHandler()
                    if log_config.get("console", True)
                    else logging.NullHandler()
                ),
                (
                    logging.FileHandler(log_file)
                    if log_config.get("file", True)
                    else logging.NullHandler()
                ),
            ],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized data generator with config: {self.config.config_path}"
        )

    def _setup_directories(self):
        """Setup all required directories"""
        # Main output directories
        self.output_dir = self.config.get_path("paths.output_dir")
        self.data_dir = self.config.get_path("paths.data_dir")

        # Training data directories
        self.positive_dir = self.data_dir / "raw_positive"
        self.negative_dir = self.data_dir / "raw_negative"
        self.augmented_dir = self.data_dir / "augmented"
        self.background_dir = self.data_dir / "background"

        # Final training directories
        training_config = self.config.get("paths.training_data", {})
        self.training_positive = Path(
            training_config.get("positive", self.data_dir / "training/positive")
        )
        self.training_negative = Path(
            training_config.get("negative", self.data_dir / "training/negative")
        )
        self.training_validation = Path(
            training_config.get("validation", self.data_dir / "validation")
        )

        # Create all directories
        for directory in [
            self.positive_dir,
            self.negative_dir,
            self.augmented_dir,
            self.background_dir,
            self.training_positive,
            self.training_negative,
            self.training_validation,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Setup directories under: {self.data_dir}")

    def _setup_tts_engines(self):
        """Setup TTS engines in order of preference"""
        self.tts_engines = [
            KokoroTTSEngine(),  # Add this line
        ]

        engine_name = self.config.get("data_generation.tts.engine", "kokoro")
        self.logger.info(f"Preferred TTS engine: {engine_name}")

    def generate_positive_samples(self) -> int:
        """Generate positive wake word samples"""
        self.logger.info("🎯 Generating positive wake word samples...")

        wake_word_variants = self.config.get("wake_word.variants", [])
        if not wake_word_variants:
            primary = self.config.get("wake_word.primary", "wake_word")
            wake_word_variants = [primary]

        print(f"📝 Wake word variants: {wake_word_variants}")

        total_generated = 0
        engine_name = self.config.get("data_generation.tts.engine", "piper")

        # Try engines in order
        for engine in self.tts_engines:
            if (
                engine_name.lower() in engine.__class__.__name__.lower()
                or engine_name == "fallback"
            ):
                try:
                    generated = engine.generate_samples(
                        wake_word_variants, self.positive_dir, self.config
                    )
                    if generated > 0:
                        total_generated = generated
                        break
                except Exception as e:
                    self.logger.warning(
                        f"Engine {engine.__class__.__name__} failed: {e}"
                    )
                    continue

        if total_generated == 0:
            self.logger.warning("All preferred engines failed, using fallback")
            fallback_engine = FallbackTTSEngine()
            total_generated = fallback_engine.generate_samples(
                wake_word_variants, self.positive_dir, self.config
            )

        # Debug: Check what files exist before consolidation
        all_files = list(self.positive_dir.rglob("*.wav"))
        self.logger.info(f"Found {len(all_files)} audio files before consolidation")
        for f in all_files[:5]:  # Log first 5 files
            self.logger.debug(f"Found file: {f}")

        # Consolidate samples
        consolidated = self._consolidate_samples()
        self.logger.info(
            f"✅ Generated {total_generated} samples, consolidated {consolidated} positive samples"
        )
        return consolidated

    def _consolidate_samples(self) -> int:
        """Consolidate samples from subdirectories"""
        sample_count = 0
        wake_word = self.config.get("wake_word.primary", "sample")

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

        if not self.config.get("background_audio.enabled", True):
            self.logger.info("Background audio generation disabled")
            return

        self._generate_noise_samples()
        self._generate_speech_samples()
        self._generate_music_samples()

        self.logger.info("✅ Background data generation complete")

    def _generate_noise_samples(self):
        """Generate various noise types"""
        noise_config = self.config.get("background_audio.noise", {})
        noise_dir = self.background_dir / "noise"
        noise_dir.mkdir(exist_ok=True)

        sr = self.config.get("audio.sample_rate", 16000)
        duration = noise_config.get("duration", 10.0)
        n_samples = noise_config.get("n_samples", 100)
        noise_types = noise_config.get("types", ["white_noise"])

        print(f"🔊 Generating {n_samples} noise samples...")

        for i in tqdm(range(n_samples), desc="Noise samples"):
            noise_type = random.choice(noise_types)
            audio = self._create_noise_audio(noise_type, duration, sr)
            filename = noise_dir / f"{noise_type}_{i:03d}.wav"
            sf.write(filename, audio, sr)

    def _generate_speech_samples(self):
        """Generate synthetic speech-like audio"""
        speech_config = self.config.get("background_audio.speech", {})
        speech_dir = self.background_dir / "speech"
        speech_dir.mkdir(exist_ok=True)

        sr = self.config.get("audio.sample_rate", 16000)
        duration = speech_config.get("duration", 8.0)
        n_samples = speech_config.get("n_samples", 50)

        print(f"🗣️ Generating {n_samples} speech-like samples...")

        for i in tqdm(range(n_samples), desc="Speech samples"):
            audio = self._create_speech_audio(speech_config, sr, duration)
            filename = speech_dir / f"background_speech_{i:03d}.wav"
            sf.write(filename, audio, sr)

    def _generate_music_samples(self):
        """Generate synthetic music audio"""
        music_config = self.config.get("background_audio.music", {})
        music_dir = self.background_dir / "music"
        music_dir.mkdir(exist_ok=True)

        sr = self.config.get("audio.sample_rate", 16000)
        duration = music_config.get("duration", 12.0)
        n_samples = music_config.get("n_samples", 30)

        print(f"🎼 Generating {n_samples} music samples...")

        for i in tqdm(range(n_samples), desc="Music samples"):
            audio = self._create_music_audio(music_config, sr, duration)
            filename = music_dir / f"background_music_{i:03d}.wav"
            sf.write(filename, audio, sr)

    def _create_noise_audio(
        self, noise_type: str, duration: float, sr: int
    ) -> np.ndarray:
        """Create specific type of noise audio"""
        samples = int(sr * duration)

        if noise_type == "white_noise":
            return np.random.normal(0, 0.1, samples)
        elif noise_type == "pink_noise":
            white = np.random.normal(0, 1, samples)
            freqs = np.fft.fftfreq(samples, 1 / sr)
            freqs[0] = 1
            fft = np.fft.fft(white)
            pink_fft = fft / np.sqrt(np.abs(freqs))
            return np.real(np.fft.ifft(pink_fft)) * 0.1
        elif noise_type == "brown_noise":
            white = np.random.normal(0, 0.01, samples)
            brown = np.cumsum(white)
            return (
                brown / np.max(np.abs(brown)) * 0.1
                if np.max(np.abs(brown)) > 0
                else brown
            )
        elif noise_type == "fan_noise":
            t = np.linspace(0, duration, samples)
            return 0.1 * np.sin(2 * np.pi * 60 * t) + 0.05 * np.random.normal(
                0, 1, samples
            )
        elif noise_type == "traffic_noise":
            t = np.linspace(0, duration, samples)
            audio = np.zeros(samples)
            for f in [30, 80, 150, 300]:
                amplitude = 0.02 * np.random.uniform(0.5, 1.5)
                audio += amplitude * np.sin(
                    2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi)
                )
            return audio + 0.03 * np.random.normal(0, 1, samples)
        else:  # cafe_ambience or default
            audio = 0.02 * np.random.normal(0, 1, samples)
            for _ in range(random.randint(3, 8)):
                start = random.randint(0, samples - sr)
                burst_len = random.randint(sr // 4, sr)
                burst = 0.05 * np.random.normal(0, 1, burst_len)
                end_idx = min(start + burst_len, samples)
                audio[start:end_idx] += burst[: end_idx - start]
            return audio

    def _create_speech_audio(
        self, speech_config: Dict, sr: int, duration: float
    ) -> np.ndarray:
        """Create synthetic speech-like audio"""
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = np.zeros(samples)

        formants = speech_config.get("formants", [[400, 1500]])
        segment_duration = speech_config.get("segment_duration", 0.5)
        segments_per_sample = int(duration / segment_duration)

        for seg in range(segments_per_sample):
            start_idx = int(seg * segment_duration * sr)
            end_idx = min(int((seg + 1) * segment_duration * sr), samples)

            f1, f2 = random.choice(formants)
            f1 *= random.uniform(0.8, 1.2)
            f2 *= random.uniform(0.8, 1.2)

            seg_t = t[start_idx:end_idx]
            segment = 0.1 * np.sin(2 * np.pi * f1 * seg_t) + 0.05 * np.sin(
                2 * np.pi * f2 * seg_t
            )

            mod_freq = random.uniform(3, 8)
            amplitude_mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * seg_t)
            segment *= amplitude_mod

            audio[start_idx:end_idx] = segment

        return audio + 0.01 * np.random.normal(0, 1, samples)

    def _create_music_audio(
        self, music_config: Dict, sr: int, duration: float
    ) -> np.ndarray:
        """Create synthetic music audio"""
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)
        audio = np.zeros(samples)

        notes = music_config.get("notes", [440.0])  # Default to A4
        chord_duration = music_config.get("chord_duration", 2.0)
        chords_per_sample = int(duration / chord_duration)

        for chord_idx in range(chords_per_sample):
            start_idx = int(chord_idx * chord_duration * sr)
            end_idx = min(int((chord_idx + 1) * chord_duration * sr), samples)

            chord_notes = random.sample(notes, min(3, len(notes)))
            chord_t = t[start_idx:end_idx]
            chord_audio = np.zeros(len(chord_t))

            for note_freq in chord_notes:
                note_audio = (
                    0.3 * np.sin(2 * np.pi * note_freq * chord_t)
                    + 0.15 * np.sin(2 * np.pi * note_freq * 2 * chord_t)
                    + 0.075 * np.sin(2 * np.pi * note_freq * 3 * chord_t)
                )

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

        if not self.config.get("augmentation.enabled", True):
            self.logger.info("Augmentation disabled, copying original samples")
            return self._copy_original_samples()

        positive_files = list(self.positive_dir.glob("*.wav"))
        if not positive_files:
            self.logger.warning("No positive samples found for augmentation")
            return 0

        augmentation_types = self.config.get("augmentation.types", ["original"])
        total_augmented = 0

        print(
            f"🔄 Augmenting {len(positive_files)} samples with {len(augmentation_types)} techniques..."
        )

        for audio_file in tqdm(positive_files, desc="Augmenting"):
            try:
                audio, sr = librosa.load(
                    audio_file, sr=self.config.get("audio.sample_rate")
                )

                for aug_type in augmentation_types:
                    augmented_audio = self._apply_augmentation(audio, sr, aug_type)

                    if augmented_audio is not None:
                        output_path = (
                            self.augmented_dir / f"{audio_file.stem}_{aug_type}.wav"
                        )
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
            shutil.copy2(
                audio_file, self.augmented_dir / f"{audio_file.stem}_original.wav"
            )

        return len(positive_files)

    def _apply_augmentation(
        self, audio: np.ndarray, sr: int, aug_type: str
    ) -> Optional[np.ndarray]:
        """Apply specific augmentation type"""
        try:
            params = self.config.get("augmentation.parameters", {})

            if aug_type == "original":
                return audio
            elif aug_type == "volume_low":
                vol_range = params.get("volume_low_range", [0.3, 0.6])
                return audio * random.uniform(*vol_range)
            elif aug_type == "volume_high":
                vol_range = params.get("volume_high_range", [1.2, 1.5])
                return audio * random.uniform(*vol_range)
            elif aug_type == "speed_slow":
                rate = params.get("speed_slow_rate", 0.9)
                return librosa.effects.time_stretch(audio, rate=rate)
            elif aug_type == "speed_fast":
                rate = params.get("speed_fast_rate", 1.1)
                return librosa.effects.time_stretch(audio, rate=rate)
            elif aug_type == "background_noise":
                noise_amp = params.get("noise_amplitude", 0.005)
                noise = np.random.normal(0, noise_amp, len(audio))
                return audio + noise
            elif aug_type == "room_reverb":
                reverb_delay_ms = params.get("reverb_delay_ms", 100)
                reverb_amp = params.get("reverb_amplitude", 0.3)
                reverb_delay = int(reverb_delay_ms * sr / 1000)

                reverb_audio = np.zeros(len(audio) + reverb_delay)
                reverb_audio[: len(audio)] = audio
                reverb_audio[reverb_delay:] += audio * reverb_amp
                return reverb_audio[: len(audio)]
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

        self.logger.info(
            f"✅ Training data prepared: {positive_count} positive, {negative_count} negative"
        )
        return positive_count, negative_count

    def _create_negative_samples(self) -> int:
        """Create negative samples from background data"""
        print("🔄 Creating negative samples from background data...")

        # Collect background files
        background_files = list(self.background_dir.rglob("*.wav"))

        # Calculate number of negative samples needed
        n_positive = len(list(self.augmented_dir.glob("*.wav")))
        if n_positive == 0:
            n_positive = self.config.get("data_generation.n_samples", 1000)

        negative_multiplier = self.config.get("data_generation.negative_multiplier", 2)
        n_negative = n_positive * negative_multiplier

        # Audio parameters
        chunk_duration = self.config.get("audio.chunk_duration_ms", 1280) / 1000.0
        sr = self.config.get("audio.sample_rate", 16000)
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
                        chunk = bg_audio[start_idx : start_idx + chunk_samples]
                    else:
                        chunk = np.tile(bg_audio, (chunk_samples // len(bg_audio)) + 1)[
                            :chunk_samples
                        ]

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

    def _create_synthetic_negative_chunk(
        self, chunk_samples: int, sr: int
    ) -> np.ndarray:
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
            print(
                f"   • Ratio (pos:neg): 1:{neg_count//pos_count if pos_count > 0 else 0}"
            )
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
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "config_file": str(self.config.config_path),
                "wake_word": self.config.get("wake_word.primary"),
                "variants": self.config.get("wake_word.variants"),
            },
            "dataset_statistics": {
                "positive_samples": pos_count,
                "negative_samples": neg_count,
                "total_samples": pos_count + neg_count,
                "positive_negative_ratio": f"1:{neg_count//pos_count if pos_count > 0 else 0}",
                "augmentation_enabled": self.config.get("augmentation.enabled", True),
                "tts_engine_used": self.config.get(
                    "data_generation.tts.engine", "unknown"
                ),
            },
            "audio_settings": self.config.get("audio", {}),
            "paths": {
                "training_positive": str(self.training_positive),
                "training_negative": str(self.training_negative),
                "validation": str(self.training_validation),
            },
        }

        # Save summary
        summary_path = self.output_dir / "data_generation_summary.yaml"
        with open(summary_path, "w") as f:
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
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: config.yaml in project root)",
    )
    parser.add_argument("--wake-word", help="Override wake word from config")
    parser.add_argument(
        "--samples", type=int, help="Override number of samples to generate"
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument(
        "--engine", choices=["piper", "pyttsx3", "fallback"], help="Override TTS engine"
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Generate small dataset for testing"
    )

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
            generator.config.update("wake_word.primary", args.wake_word)
            # Update variants
            variants = [
                args.wake_word,
                f"hey {args.wake_word}",
                f"{args.wake_word} please",
            ]
            generator.config.update("wake_word.variants", variants)

        if args.samples:
            print(f"🔄 Overriding sample count: {args.samples}")
            generator.config.update("data_generation.n_samples", args.samples)

        if args.output_dir:
            print(f"🔄 Overriding output directory: {args.output_dir}")
            generator.config.update("paths.output_dir", str(args.output_dir))
            generator._setup_directories()

        if args.engine:
            print(f"🔄 Overriding TTS engine: {args.engine}")
            generator.config.update("data_generation.tts.engine", args.engine)

        if args.no_augment:
            print("🔄 Disabling data augmentation")
            generator.config.update("augmentation.enabled", False)

        if args.test_mode:
            print("🧪 Test mode: generating small dataset")
            generator.config.update("data_generation.n_samples", 50)
            generator.config.update("background_audio.noise.n_samples", 20)
            generator.config.update("background_audio.speech.n_samples", 10)
            generator.config.update("background_audio.music.n_samples", 10)

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
