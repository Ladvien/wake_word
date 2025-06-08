#!/usr/bin/env python3
"""
Wake Word Model Training
Trains a wake word detection model using generated synthetic data

Usage:
    python -m wake_word.train
    python -m wake_word.train --config custom_config.yaml
    python wake_word/train.py --data-dir ./output/data/training
"""
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from tqdm import tqdm
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# Get project root and add to path
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

# Audio processing with fallback
try:
    import librosa
    HAS_LIBROSA = True
    print("✓ Using librosa for feature extraction")
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
        
        class feature:
            @staticmethod
            def melspectrogram(y, sr=16000, n_mels=80, n_fft=1024, hop_length=160, win_length=400):
                # Simple mel spectrogram using scipy
                f, t, stft = scipy.signal.stft(y, fs=sr, nperseg=win_length, 
                                              noverlap=win_length-hop_length, nfft=n_fft)
                power_spec = np.abs(stft) ** 2
                
                # Simplified mel scaling
                mel_freqs = np.linspace(0, sr//2, n_mels)
                mel_spec = np.zeros((n_mels, power_spec.shape[1]))
                
                for i in range(n_mels):
                    freq_idx = int(mel_freqs[i] * n_fft / sr)
                    freq_idx = min(freq_idx, power_spec.shape[0] - 1)
                    mel_spec[i] = power_spec[freq_idx]
                
                return mel_spec
        
        @staticmethod
        def power_to_db(S):
            return 10.0 * np.log10(np.maximum(S, 1e-10))
    
    librosa = LibrosaFallback()


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = CONFIG_PATH
        
        self.config_path = Path(config_path)
        self.project_root = PROJECT_ROOT
        self.config = self._load_config()
        self._resolve_paths()
    
    def _load_config(self):
        """Load configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _resolve_paths(self):
        """Resolve paths relative to project root"""
        paths_config = self.config.get('paths', {})
        for key, path_str in paths_config.items():
            if isinstance(path_str, str) and not Path(path_str).is_absolute():
                paths_config[key] = str(self.project_root / path_str)
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_path(self, key_path: str, default=None) -> Path:
        """Get path from config"""
        path_str = self.get(key_path, default)
        if path_str:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.project_root / path
            return path
        return None


class WakeWordDataset(Dataset):
    """Dataset for wake word training data"""
    
    def __init__(self, positive_dir: Path, negative_dir: Path, config: ConfigManager):
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir)
        self.config = config
        
        # Audio settings
        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.chunk_duration = config.get('audio.chunk_duration_ms', 1280) / 1000.0
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Feature settings
        self.feature_type = config.get('features.type', 'mel_spectrogram')
        self.input_size = config.get('training.model.input_size', 768)
        
        # Load file lists
        self.positive_files = list(self.positive_dir.glob("*.wav"))
        self.negative_files = list(self.negative_dir.glob("*.wav"))
        
        if not self.positive_files:
            raise ValueError(f"No positive samples found in {self.positive_dir}")
        if not self.negative_files:
            raise ValueError(f"No negative samples found in {self.negative_dir}")
        
        # Balance dataset
        self._balance_dataset()
        
        print(f"📊 Dataset loaded:")
        print(f"   • Positive samples: {len(self.positive_files):,}")
        print(f"   • Negative samples: {len(self.negative_files):,}")
        print(f"   • Total samples: {len(self):,}")
    
    def _balance_dataset(self):
        """Balance positive and negative samples"""
        n_positive = len(self.positive_files)
        n_negative = len(self.negative_files)
        
        if n_negative < n_positive:
            # Duplicate negative files
            multiplier = (n_positive // n_negative) + 1
            self.negative_files = self.negative_files * multiplier
        
        # Limit to reasonable ratio (2:1 negative to positive)
        max_negative = n_positive * 2
        self.negative_files = self.negative_files[:max_negative]
    
    def __len__(self):
        return len(self.positive_files) + len(self.negative_files)
    
    def __getitem__(self, idx):
        # Determine if positive or negative sample
        if idx < len(self.positive_files):
            audio_file = self.positive_files[idx]
            label = 1.0
        else:
            neg_idx = idx - len(self.positive_files)
            audio_file = self.negative_files[neg_idx]
            label = 0.0
        
        # Load and process audio
        try:
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # Ensure consistent length
            audio = self._normalize_audio_length(audio)
            
            # Extract features
            features = self._extract_features(audio)
            
        except Exception as e:
            print(f"⚠ Error loading {audio_file}: {e}")
            # Return dummy features
            features = np.zeros(self.input_size)
        
        return torch.FloatTensor(features), torch.FloatTensor([label])
    
    def _normalize_audio_length(self, audio):
        """Normalize audio to consistent length"""
        if len(audio) > self.chunk_samples:
            # Random crop for training diversity
            if len(audio) > self.chunk_samples:
                start_idx = np.random.randint(0, len(audio) - self.chunk_samples + 1)
                audio = audio[start_idx:start_idx + self.chunk_samples]
        elif len(audio) < self.chunk_samples:
            # Pad with zeros
            padding = self.chunk_samples - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        return audio
    
    def _extract_features(self, audio):
        """Extract features from audio"""
        if self.feature_type == 'mel_spectrogram':
            return self._extract_mel_features(audio)
        elif self.feature_type == 'mfcc':
            return self._extract_mfcc_features(audio)
        else:
            return self._extract_raw_features(audio)
    
    def _extract_mel_features(self, audio):
        """Extract mel spectrogram features"""
        mel_config = self.config.get('features.mel_spectrogram', {})
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=mel_config.get('n_mels', 80),
            n_fft=mel_config.get('n_fft', 1024),
            hop_length=mel_config.get('hop_length', 160),
            win_length=mel_config.get('win_length', 400)
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec)
        
        # Flatten and resize
        features = log_mel.flatten()
        
        # Resize to target size
        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features = np.pad(features, (0, self.input_size - len(features)), 'constant')
        
        # Normalize
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        return features
    
    def _extract_mfcc_features(self, audio):
        """Extract MFCC features"""
        mfcc_config = self.config.get('features.mfcc', {})
        
        # Compute MFCCs
        if HAS_LIBROSA:
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=mfcc_config.get('n_mfcc', 13),
                n_fft=mfcc_config.get('n_fft', 1024),
                hop_length=mfcc_config.get('hop_length', 160)
            )
        else:
            # Simplified MFCC using mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
            log_mel = librosa.power_to_db(mel_spec)
            # Take first 13 mel coefficients as MFCC approximation
            mfccs = log_mel[:13]
        
        # Flatten and process
        features = mfccs.flatten()
        
        # Resize to target size
        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features = np.pad(features, (0, self.input_size - len(features)), 'constant')
        
        # Normalize
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        return features
    
    def _extract_raw_features(self, audio):
        """Extract raw audio features"""
        # Simple raw audio processing
        features = audio[:self.input_size] if len(audio) >= self.input_size else np.pad(audio, (0, self.input_size - len(audio)), 'constant')
        
        # Normalize
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        return features


class WakeWordModel(nn.Module):
    """Configurable wake word detection model"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        
        model_config = config.get('training.model', {})
        self.input_size = model_config.get('input_size', 768)
        self.hidden_size = model_config.get('hidden_size', 128)
        self.num_layers = model_config.get('num_layers', 2)
        self.dropout = model_config.get('dropout', 0.2)
        model_type = model_config.get('type', 'simple_classifier')
        
        if model_type == 'simple_classifier':
            self._build_simple_classifier()
        elif model_type == 'cnn':
            self._build_cnn()
        elif model_type == 'rnn':
            self._build_rnn()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_simple_classifier(self):
        """Build simple fully connected classifier"""
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ])
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
        
        # Output layer
        layers.extend([
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ])
        
        self.classifier = nn.Sequential(*layers)
    
    def _build_cnn(self):
        """Build CNN model (placeholder for future implementation)"""
        # For now, fall back to simple classifier
        self._build_simple_classifier()
    
    def _build_rnn(self):
        """Build RNN model (placeholder for future implementation)"""
        # For now, fall back to simple classifier
        self._build_simple_classifier()
    
    def forward(self, x):
        return self.classifier(x)


class WakeWordTrainer:
    """Main training class"""
    
    def __init__(self, config_path: Path = None, data_dir: Path = None):
        self.config = ConfigManager(config_path)
        self.data_dir = data_dir or self.config.get_path('paths.data_dir')
        
        # Setup logging
        self._setup_logging()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🖥️ Using device: {self.device}")
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1_scores = []
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create logs directory
        log_dir = self.config.get_path('paths.logs_dir')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'training.log'
        
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
        self.logger.info(f"🎯 Initialized wake word trainer")
        self.logger.info(f"📁 Data directory: {self.data_dir}")
        self.logger.info(f"⚙️ Config file: {self.config.config_path}")
    
    def create_datasets(self):
        """Create training and validation datasets"""
        self.logger.info("📊 Creating datasets...")
        
        # Find training data directories
        training_dir = self.data_dir / "training"
        positive_dir = training_dir / "positive"
        negative_dir = training_dir / "negative"
        
        if not positive_dir.exists():
            raise FileNotFoundError(
                f"Positive training data not found: {positive_dir}\n"
                f"Please run: python -m wake_word.generate first"
            )
        
        if not negative_dir.exists():
            raise FileNotFoundError(
                f"Negative training data not found: {negative_dir}\n"
                f"Please run: python -m wake_word.generate first"
            )
        
        # Create dataset
        full_dataset = WakeWordDataset(positive_dir, negative_dir, self.config)
        
        # Split into train/validation
        val_config = self.config.get('development.validation', {})
        val_split = val_config.get('split_ratio', 0.2)
        
        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        dataloader_config = self.config.get('training.dataloader', {})
        batch_size = self.config.get('training.batch_size', 64)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=dataloader_config.get('shuffle', True),
            num_workers=dataloader_config.get('num_workers', 2),
            pin_memory=dataloader_config.get('pin_memory', True)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_config.get('num_workers', 2),
            pin_memory=dataloader_config.get('pin_memory', True)
        )
        
        self.logger.info(f"✅ Datasets created:")
        self.logger.info(f"   • Training samples: {train_size:,}")
        self.logger.info(f"   • Validation samples: {val_size:,}")
        self.logger.info(f"   • Batch size: {batch_size}")
        self.logger.info(f"   • Training batches: {len(self.train_loader)}")
        self.logger.info(f"   • Validation batches: {len(self.val_loader)}")
    
    def train_model(self):
        """Train the wake word model"""
        self.logger.info("🚀 Starting model training...")
        
        # Initialize model
        model = WakeWordModel(self.config).to(self.device)
        self.logger.info(f"🧠 Model architecture: {self.config.get('training.model.type', 'simple_classifier')}")
        
        # Training configuration
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 50)
        learning_rate = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 0.0001)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        # Early stopping configuration
        early_stopping_config = training_config.get('early_stopping', {})
        early_stopping_enabled = early_stopping_config.get('enabled', True)
        patience = early_stopping_config.get('patience', 10)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        
        # Performance targets
        target_accuracy = training_config.get('target_accuracy', 0.85)
        target_recall = training_config.get('target_recall', 0.7)
        target_precision = training_config.get('target_precision', 0.8)
        
        # Training state
        best_accuracy = 0.0
        best_model_path = None
        patience_counter = 0
        
        self.logger.info(f"📈 Training configuration:")
        self.logger.info(f"   • Epochs: {epochs}")
        self.logger.info(f"   • Learning rate: {learning_rate}")
        self.logger.info(f"   • Target accuracy: {target_accuracy}")
        self.logger.info(f"   • Target recall: {target_recall}")
        self.logger.info(f"   • Early stopping: {early_stopping_enabled}")
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_features, batch_labels in train_pbar:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)
                
                # Update progress bar
                current_acc = correct_predictions / total_predictions
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })
            
            # Validation phase
            val_metrics = self._evaluate_model(model)
            scheduler.step(val_metrics['accuracy'])
            
            # Calculate epoch metrics
            train_accuracy = correct_predictions / total_predictions
            avg_loss = total_loss / len(self.train_loader)
            
            # Store metrics
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_recalls.append(val_metrics['recall'])
            self.val_precisions.append(val_metrics['precision'])
            self.val_f1_scores.append(val_metrics['f1'])
            
            # Logging
            self.logger.info(f"📊 Epoch {epoch+1}/{epochs}:")
            self.logger.info(f"   • Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            self.logger.info(f"   • Val Acc: {val_metrics['accuracy']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            self.logger.info(f"   • Val Precision: {val_metrics['precision']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_accuracy + min_delta:
                best_accuracy = val_metrics['accuracy']
                wake_word = self.config.get('wake_word.primary', 'wake_word')
                best_model_path = self._save_model(model, optimizer, epoch, val_metrics, wake_word)
                self.logger.info(f"💾 New best model saved: {best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check performance targets
            if (val_metrics['accuracy'] >= target_accuracy and 
                val_metrics['recall'] >= target_recall and 
                val_metrics['precision'] >= target_precision):
                self.logger.info("🎯 Target performance reached!")
                break
            
            # Early stopping
            if early_stopping_enabled and patience_counter >= patience:
                self.logger.info("⏰ Early stopping triggered")
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self._save_checkpoint(model, optimizer, epoch, val_metrics)
                self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        return best_model_path, best_accuracy
    
    def _evaluate_model(self, model):
        """Evaluate model on validation set"""
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_features)
                predicted = (outputs > 0.5).float()
                
                correct_predictions += (predicted == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)
                
                # Calculate confusion matrix components
                for i in range(len(predicted)):
                    if batch_labels[i] == 1 and predicted[i] == 1:
                        true_positives += 1
                    elif batch_labels[i] == 0 and predicted[i] == 1:
                        false_positives += 1
                    elif batch_labels[i] == 1 and predicted[i] == 0:
                        false_negatives += 1
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _save_model(self, model, optimizer, epoch, metrics, wake_word):
        """Save the best model"""
        models_dir = self.config.get_path('paths.models_dir')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / f"{wake_word}_model_epoch_{epoch+1}_acc_{metrics['accuracy']:.4f}.pth"
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.config,
            'wake_word': wake_word
        }, model_path)
        
        return model_path
    
    def _save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save training checkpoint"""
        models_dir = self.config.get_path('paths.models_dir')
        checkpoint_path = models_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_recalls': self.val_recalls
        }, checkpoint_path)
        
        return checkpoint_path
    
    def plot_training_metrics(self):
        """Plot and save training metrics"""
        if not self.train_losses:
            self.logger.warning("No training metrics to plot")
            return
        
        output_dir = self.config.get_path('paths.output_dir')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.set_title('Training Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Accuracy comparison
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Accuracy Comparison', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Validation metrics
        ax3.plot(self.val_recalls, label='Recall', color='green')
        ax3.plot(self.val_precisions, label='Precision', color='orange')
        ax3.plot(self.val_f1_scores, label='F1 Score', color='purple')
        ax3.set_title('Validation Metrics', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning curves
        ax4.plot(self.train_accuracies, label='Train Accuracy', color='blue', alpha=0.7)
        ax4.plot(self.val_accuracies, label='Val Accuracy', color='red', alpha=0.7)
        ax4.fill_between(range(len(self.train_accuracies)), self.train_accuracies, alpha=0.2, color='blue')
        ax4.fill_between(range(len(self.val_accuracies)), self.val_accuracies, alpha=0.2, color='red')
        ax4.set_title('Learning Curves', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Training metrics plot saved: {plot_path}")
        return plot_path
    
    def export_model(self, model_path: Path):
        """Export trained model to deployment formats"""
        self.logger.info("📦 Exporting model for deployment...")
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        model = WakeWordModel(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        export_config = self.config.get('export', {})
        formats = export_config.get('formats', ['onnx'])
        
        exported_models = {}
        
        # Export to ONNX
        if 'onnx' in formats:
            onnx_path = self._export_onnx(model, checkpoint)
            if onnx_path:
                exported_models['onnx'] = onnx_path
        
        # Export PyTorch model
        if 'pytorch' in formats:
            pytorch_path = self._export_pytorch(model, checkpoint)
            if pytorch_path:
                exported_models['pytorch'] = pytorch_path
        
        # Create model metadata
        metadata_path = self._create_model_metadata(checkpoint, exported_models)
        
        self.logger.info(f"✅ Model export complete:")
        for format_name, path in exported_models.items():
            self.logger.info(f"   • {format_name.upper()}: {path}")
        self.logger.info(f"   • Metadata: {metadata_path}")
        
        return exported_models
    
    def _export_onnx(self, model, checkpoint):
        """Export model to ONNX format"""
        try:
            models_dir = self.config.get_path('paths.models_dir')
            wake_word = checkpoint.get('wake_word', 'wake_word')
            onnx_path = models_dir / f"{wake_word}_model.onnx"
            
            # Create dummy input
            input_size = self.config.get('training.model.input_size', 768)
            dummy_input = torch.randn(1, input_size)
            
            # ONNX export settings
            onnx_config = self.config.get('export.onnx', {})
            
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                input_names=['features'],
                output_names=['prediction'],
                dynamic_axes={'features': {0: 'batch_size'}} if onnx_config.get('dynamic_axes', True) else None,
                export_params=onnx_config.get('export_params', True),
                opset_version=onnx_config.get('opset_version', 11)
            )
            
            self.logger.info(f"✅ ONNX model exported: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"❌ ONNX export failed: {e}")
            return None
    
    def _export_pytorch(self, model, checkpoint):
        """Export model in PyTorch format"""
        try:
            models_dir = self.config.get_path('paths.models_dir')
            wake_word = checkpoint.get('wake_word', 'wake_word')
            pytorch_path = models_dir / f"{wake_word}_model.pt"
            
            # Save just the model state dict for deployment
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.config.get('training.model', {}),
                'audio_config': self.config.get('audio', {}),
                'features_config': self.config.get('features', {}),
                'wake_word': wake_word,
                'metrics': checkpoint.get('metrics', {})
            }, pytorch_path)
            
            self.logger.info(f"✅ PyTorch model exported: {pytorch_path}")
            return pytorch_path
            
        except Exception as e:
            self.logger.error(f"❌ PyTorch export failed: {e}")
            return None
    
    def _create_model_metadata(self, checkpoint, exported_models):
        """Create model metadata file"""
        models_dir = self.config.get_path('paths.models_dir')
        wake_word = checkpoint.get('wake_word', 'wake_word')
        metadata_path = models_dir / f"{wake_word}_model_metadata.json"
        
        # Collect metadata
        metadata = {
            'model_info': {
                'name': wake_word,
                'version': '1.0',
                'created_date': datetime.now().isoformat(),
                'framework': 'pytorch',
                'description': self.config.get('export.metadata.description', 'Custom wake word detection model')
            },
            'performance': {
                'accuracy': checkpoint.get('metrics', {}).get('accuracy', 0.0),
                'precision': checkpoint.get('metrics', {}).get('precision', 0.0),
                'recall': checkpoint.get('metrics', {}).get('recall', 0.0),
                'f1_score': checkpoint.get('metrics', {}).get('f1', 0.0)
            },
            'model_architecture': self.config.get('training.model', {}),
            'audio_settings': self.config.get('audio', {}),
            'feature_settings': self.config.get('features', {}),
            'deployment': {
                'threshold': self.config.get('export.metadata.threshold', 0.5),
                'confidence_threshold': self.config.get('export.metadata.confidence_threshold', 0.7),
                'input_size': self.config.get('training.model.input_size', 768),
                'sample_rate': self.config.get('audio.sample_rate', 16000),
                'chunk_duration_ms': self.config.get('audio.chunk_duration_ms', 1280)
            },
            'files': {
                format_name: str(path.name) for format_name, path in exported_models.items()
            },
            'training_info': {
                'epoch': checkpoint.get('epoch', 0),
                'dataset_info': {
                    'positive_samples': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0,
                    'validation_samples': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0
                }
            }
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        self.logger.info("🚀 Starting wake word training pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Create datasets
            print("\n📊 Step 1: Loading training data...")
            self.create_datasets()
            
            # Step 2: Train model
            print("\n🧠 Step 2: Training model...")
            best_model_path, best_accuracy = self.train_model()
            
            if best_model_path is None:
                self.logger.error("❌ Training failed - no model was saved")
                return False
            
            # Step 3: Plot metrics
            print("\n📈 Step 3: Generating training plots...")
            plot_path = self.plot_training_metrics()
            
            # Step 4: Export model
            print("\n📦 Step 4: Exporting model...")
            exported_models = self.export_model(best_model_path)
            
            # Success summary
            print("\n" + "=" * 60)
            print("✅ Training pipeline completed successfully!")
            print(f"🎯 Best validation accuracy: {best_accuracy:.4f}")
            print(f"💾 Best model: {best_model_path.name}")
            print(f"📈 Training plot: {plot_path.name}")
            print(f"🚀 Ready for deployment!")
            
            if 'onnx' in exported_models:
                print(f"\n🔗 Next steps:")
                print(f"   1. Test model: python -m wake_word.test")
                print(f"   2. Package for Pi: python -m wake_word.package")
                print(f"   3. Deploy: Copy to Raspberry Pi and run detector")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train wake word detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m wake_word.train
  python -m wake_word.train --config custom_config.yaml
  python wake_word/train.py --data-dir ./my_data/training
        """
    )
    
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--data-dir", type=Path, help="Path to training data directory")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--no-cuda", action='store_true', help="Disable CUDA even if available")
    parser.add_argument("--resume", type=Path, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    try:
        print("🧠 Wake Word Model Training")
        print("=" * 50)
        print(f"📁 Project root: {PROJECT_ROOT}")
        print(f"⚙️ Config file: {args.config or CONFIG_PATH}")
        print()
        
        # Initialize trainer
        trainer = WakeWordTrainer(args.config, args.data_dir)
        
        # Apply command line overrides
        if args.epochs:
            print(f"🔄 Overriding epochs: {args.epochs}")
            trainer.config.config['training']['epochs'] = args.epochs
        
        if args.batch_size:
            print(f"🔄 Overriding batch size: {args.batch_size}")
            trainer.config.config['training']['batch_size'] = args.batch_size
        
        if args.learning_rate:
            print(f"🔄 Overriding learning rate: {args.learning_rate}")
            trainer.config.config['training']['learning_rate'] = args.learning_rate
        
        if args.no_cuda:
            print("🔄 Disabling CUDA")
            trainer.device = torch.device('cpu')
        
        # Run training
        success = trainer.run_training_pipeline()
        
        if success:
            print("\n🎉 Training completed successfully!")
            return 0
        else:
            print("\n❌ Training failed. Check logs for details.")
            return 1
        
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