#!/usr/bin/env python3
"""
Train wake word model using OpenWakeWord
"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
import json
import shutil
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wake_word.generate import ConfigManager

try:
    import openwakeword
    from openwakeword import train_model
    from openwakeword.model import Model
    HAS_OPENWAKEWORD = True
except ImportError:
    HAS_OPENWAKEWORD = False
    print("⚠️  OpenWakeWord not found. Install with: pip install openwakeword")

class OpenWakeWordTrainer:
    """Train wake word models using OpenWakeWord"""
    
    def __init__(self, config_path=None):
        self.config = ConfigManager(config_path)
        self._setup_logging()
        self._setup_directories()
        
    def _setup_logging(self):
        """Setup logging"""
        log_file = self.config.get_path('paths.logs_dir') / 'openwakeword_training.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_directories(self):
        """Setup required directories"""
        self.data_dir = self.config.get_path('paths.data_dir')
        self.models_dir = self.config.get_path('paths.models_dir')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Training data paths
        training_config = self.config.get('paths.training_data', {})
        self.positive_dir = Path(training_config.get('positive', self.data_dir / "training/positive"))
        self.negative_dir = Path(training_config.get('negative', self.data_dir / "training/negative"))
        
    def prepare_openwakeword_dataset(self):
        """Prepare dataset in OpenWakeWord format"""
        self.logger.info("Preparing dataset for OpenWakeWord...")
        
        # OpenWakeWord expects a specific directory structure
        oww_data_dir = self.data_dir / "openwakeword_format"
        oww_data_dir.mkdir(exist_ok=True)
        
        # Create positive and negative directories
        oww_positive = oww_data_dir / "positive"
        oww_negative = oww_data_dir / "negative"
        oww_positive.mkdir(exist_ok=True)
        oww_negative.mkdir(exist_ok=True)
        
        # Copy positive samples
        positive_files = list(self.positive_dir.glob("*.wav"))
        self.logger.info(f"Found {len(positive_files)} positive samples")
        
        for i, src_file in enumerate(tqdm(positive_files, desc="Copying positive samples")):
            dst_file = oww_positive / f"positive_{i:06d}.wav"
            shutil.copy2(src_file, dst_file)
            
        # Copy negative samples
        negative_files = list(self.negative_dir.glob("*.wav"))
        self.logger.info(f"Found {len(negative_files)} negative samples")
        
        for i, src_file in enumerate(tqdm(negative_files, desc="Copying negative samples")):
            dst_file = oww_negative / f"negative_{i:06d}.wav"
            shutil.copy2(src_file, dst_file)
            
        # Create dataset metadata
        metadata = {
            "wake_word": self.config.get('wake_word.primary'),
            "positive_samples": len(positive_files),
            "negative_samples": len(negative_files),
            "sample_rate": self.config.get('audio.sample_rate', 16000),
            "created": datetime.now().isoformat()
        }
        
        with open(oww_data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return oww_data_dir
        
    def train_with_openwakeword(self, data_dir):
        """Train model using OpenWakeWord"""
        if not HAS_OPENWAKEWORD:
            raise ImportError("OpenWakeWord is required for training. Install with: pip install openwakeword")
            
        self.logger.info("Starting OpenWakeWord training...")
        
        wake_word = self.config.get('wake_word.primary', 'wake_word')
        
        # Training configuration
        training_config = {
            "model_name": f"{wake_word}_model",
            "positive_audio_dir": str(data_dir / "positive"),
            "negative_audio_dir": str(data_dir / "negative"),
            "output_dir": str(self.models_dir),
            
            # Model parameters
            "epochs": self.config.get('training.epochs', 50),
            "batch_size": self.config.get('training.batch_size', 64),
            "learning_rate": self.config.get('training.learning_rate', 0.001),
            
            # Audio parameters
            "sample_rate": self.config.get('audio.sample_rate', 16000),
            "feature_type": "melspectrogram",  # OpenWakeWord uses mel spectrograms
            
            # Model architecture
            "model_type": "dnn",  # Deep Neural Network
            "hidden_units": [128, 64, 32],  # Layer sizes
            "dropout": self.config.get('training.model.dropout', 0.2),
            
            # Training behavior
            "patience": self.config.get('training.early_stopping.patience', 10),
            "validation_split": 0.2,
            
            # Export settings
            "export_onnx": True,
            "export_tflite": False  # Disable TFLite due to compatibility issues
        }
        
        try:
            # Train the model
            self.logger.info("Training configuration:")
            for key, value in training_config.items():
                self.logger.info(f"  {key}: {value}")
            
            # Create and train model
            model = train_model(**training_config)
            
            # Save model paths
            model_paths = {
                "onnx": self.models_dir / f"{wake_word}_model.onnx",
                "tflite": self.models_dir / f"{wake_word}_model.tflite",
                "metadata": self.models_dir / f"{wake_word}_model_metadata.json"
            }
            
            # Save metadata
            model_metadata = {
                "model_name": f"{wake_word}_model",
                "wake_word": wake_word,
                "training_config": training_config,
                "training_date": datetime.now().isoformat(),
                "framework": "openwakeword",
                "performance": {
                    "accuracy": model.history.get('accuracy', [])[-1] if hasattr(model, 'history') else None,
                    "val_accuracy": model.history.get('val_accuracy', [])[-1] if hasattr(model, 'history') else None,
                }
            }
            
            with open(model_paths["metadata"], 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.logger.info(f"✅ Model training completed!")
            self.logger.info(f"📁 Models saved to: {self.models_dir}")
            
            return model_paths
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
    def evaluate_model(self, model_path):
        """Evaluate the trained model"""
        if not HAS_OPENWAKEWORD:
            return
            
        self.logger.info("Evaluating model...")
        
        try:
            # Load model
            model = Model(
                wakeword_models=[str(model_path)],
                inference_framework="onnx"
            )
            
            # Test on a few samples
            test_positive = list(self.positive_dir.glob("*.wav"))[:10]
            test_negative = list(self.negative_dir.glob("*.wav"))[:10]
            
            positive_detections = 0
            for audio_file in test_positive:
                # Process audio file
                predictions = model.predict_from_file(str(audio_file))
                if predictions and max(predictions.values()) > 0.5:
                    positive_detections += 1
                    
            negative_rejections = 0
            for audio_file in test_negative:
                predictions = model.predict_from_file(str(audio_file))
                if not predictions or max(predictions.values()) < 0.5:
                    negative_rejections += 1
                    
            self.logger.info(f"Quick evaluation:")
            self.logger.info(f"  Positive samples detected: {positive_detections}/10")
            self.logger.info(f"  Negative samples rejected: {negative_rejections}/10")
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        self.logger.info("🚀 Starting OpenWakeWord training pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Check if we have training data
            if not self.positive_dir.exists() or not any(self.positive_dir.glob("*.wav")):
                self.logger.error("No training data found! Run data generation first:")
                self.logger.error("  python -m wake_word.generate")
                return False
                
            # Step 1: Prepare dataset
            print("\n📁 Step 1: Preparing dataset for OpenWakeWord...")
            oww_data_dir = self.prepare_openwakeword_dataset()
            
            # Step 2: Train model
            print("\n🧠 Step 2: Training model with OpenWakeWord...")
            model_paths = self.train_with_openwakeword(oww_data_dir)
            
            # Step 3: Evaluate model
            print("\n📊 Step 3: Evaluating model...")
            if model_paths.get("onnx") and model_paths["onnx"].exists():
                self.evaluate_model(model_paths["onnx"])
            
            # Step 4: Generate summary
            self._generate_summary(model_paths)
            
            print("\n" + "=" * 60)
            print("✅ Training completed successfully!")
            print(f"🎯 Wake word: {self.config.get('wake_word.primary')}")
            print(f"📁 Models saved to: {self.models_dir}")
            print("\nNext steps:")
            print("  1. Test the model: python -m wake_word.test --model openwakeword")
            print("  2. Deploy to Raspberry Pi: python -m wake_word.package")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _generate_summary(self, model_paths):
        """Generate training summary"""
        summary = {
            "training_info": {
                "timestamp": datetime.now().isoformat(),
                "wake_word": self.config.get('wake_word.primary'),
                "framework": "openwakeword",
            },
            "model_files": {
                "onnx": str(model_paths.get("onnx", "")),
                "tflite": str(model_paths.get("tflite", "")),
                "metadata": str(model_paths.get("metadata", ""))
            },
            "dataset_info": {
                "positive_samples": len(list(self.positive_dir.glob("*.wav"))),
                "negative_samples": len(list(self.negative_dir.glob("*.wav"))),
            },
            "configuration": {
                "sample_rate": self.config.get('audio.sample_rate'),
                "epochs": self.config.get('training.epochs'),
                "batch_size": self.config.get('training.batch_size'),
            }
        }
        
        summary_path = self.models_dir / "training_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
            
        self.logger.info(f"📄 Summary saved to: {summary_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train wake word model using OpenWakeWord"
    )
    
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--wake-word", help="Override wake word from config")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    
    args = parser.parse_args()
    
    try:
        print("🧠 OpenWakeWord Model Trainer")
        print("=" * 50)
        
        # Check if OpenWakeWord is installed
        if not HAS_OPENWAKEWORD:
            print("\n❌ OpenWakeWord is not installed!")
            print("Install it with: pip install openwakeword")
            print("\nAlternatively, you can use the built-in trainer:")
            print("  python -m wake_word.train")
            return 1
        
        trainer = OpenWakeWordTrainer(args.config)
        
        # Apply overrides
        if args.wake_word:
            print(f"🔄 Overriding wake word: {args.wake_word}")
            trainer.config.update('wake_word.primary', args.wake_word)
            
        if args.epochs:
            print(f"🔄 Overriding epochs: {args.epochs}")
            trainer.config.update('training.epochs', args.epochs)
        
        # Run training
        success = trainer.run_training_pipeline()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Training cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())