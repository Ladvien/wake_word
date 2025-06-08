# neuron_template
A template for the Neuron projects


### Manjaro Setup
```sh
# Install LLVM and development tools
sudo pacman -S llvm clang base-devel

# Set the LLVM_CONFIG environment variable
export LLVM_CONFIG=/usr/bin/llvm-config
echo 'export LLVM_CONFIG=/usr/bin/llvm-config' >> ~/.bashrc

# Also install audio dependencies
sudo pacman -S portaudio alsa-lib

# Now try poetry install again
poetry install
```


# AI Instructions
# Wake Word Data Generator - Command Line Usage

## Prerequisites

Ensure you have the `config.yaml` file at your project root with your desired settings.

## Basic Commands

### 1. Generate with default config.yaml
```bash
# Run from project root
python -m wake_word.generate

# Or run directly
python wake_word/generate.py
```

### 2. Generate with custom wake word
```bash
python -m wake_word.generate --wake-word "computer"
```

### 3. Generate specific number of samples
```bash
python -m wake_word.generate --samples 500
```

### 4. Use specific TTS engine
```bash
python -m wake_word.generate --engine piper
python -m wake_word.generate --engine pyttsx3
python -m wake_word.generate --engine fallback
```

### 5. Custom output directory
```bash
python -m wake_word.generate --output-dir ./my_custom_model
```

### 6. Test mode (small dataset)
```bash
python -m wake_word.generate --test-mode
```

### 7. Disable augmentation
```bash
python -m wake_word.generate --no-augment
```

## Combined Examples

### Train "Alexa" with 1000 samples using Piper TTS
```bash
python -m wake_word.generate \
  --wake-word "alexa" \
  --samples 1000 \
  --engine piper \
  --output-dir ./alexa_model
```

### Quick test with "jade" (small dataset)
```bash
python -m wake_word.generate \
  --wake-word "jade" \
  --test-mode \
  --engine fallback
```

### Generate large dataset without augmentation
```bash
python -m wake_word.generate \
  --wake-word "computer" \
  --samples 2000 \
  --no-augment \
  --output-dir ./computer_model_large
```

### Use custom config file
```bash
python -m wake_word.generate --config ./configs/my_config.yaml
```

## Full Workflow Example

```bash
# 1. Edit config.yaml with your settings
nano config.yaml

# 2. Generate data for "jade" wake word
python -m wake_word.generate --wake-word "jade" --samples 1000

# 3. Check the output
ls ./output/data/training/
# Should show positive/ and negative/ directories

# 4. Train the model (if you have training script)
python -m wake_word.train

# 5. Test the model
python -m wake_word.test
```

## Help and Options

### See all available options
```bash
python -m wake_word.generate --help
```

Output:
```
usage: generate.py [-h] [--config CONFIG] [--wake-word WAKE_WORD] [--samples SAMPLES] 
                   [--output-dir OUTPUT_DIR] [--engine {piper,pyttsx3,fallback}] 
                   [--no-augment] [--test-mode]

Generate synthetic wake word training data

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file (default: config.yaml in project root)
  --wake-word WAKE_WORD Override wake word from config
  --samples SAMPLES     Override number of samples to generate
  --output-dir OUTPUT_DIR Override output directory
  --engine {piper,pyttsx3,fallback} Override TTS engine
  --no-augment          Disable data augmentation
  --test-mode           Generate small dataset for testing

Examples:
  python -m wake_word.generate
  python -m wake_word.generate --wake-word "computer" --samples 1000
  python wake_word/generate.py --config custom_config.yaml --output-dir ./my_model
```

## Project Structure After Generation

```
your_project/
├── config.yaml                    # Your configuration
├── output/                        # Generated data
│   ├── data/
│   │   ├── raw_positive/         # Original TTS samples
│   │   ├── augmented/            # Augmented positive samples
│   │   ├── background/           # Background audio
│   │   │   ├── noise/
│   │   │   ├── speech/
│   │   │   └── music/
│   │   └── training/             # Final training data
│   │       ├── positive/         # Ready for training
│   │       └── negative/         # Ready for training
│   ├── logs/
│   │   └── data_generation.log   # Generation logs
│   └── data_generation_summary.yaml  # Summary report
└── wake_word/
    ├── generate.py               # This script
    ├── train.py                  # Training script
    └── ...
```

## Common Use Cases

### For Development/Testing
```bash
# Quick test with synthetic audio
python -m wake_word.generate --test-mode --engine fallback

# Test different wake words quickly
python -m wake_word.generate --wake-word "test" --samples 100 --test-mode
```

### For Production Training
```bash
# High-quality dataset with Piper TTS
python -m wake_word.generate \
  --wake-word "jade" \
  --samples 2000 \
  --engine piper \
  --output-dir ./production_jade_model

# Large dataset with all augmentations
python -m wake_word.generate \
  --wake-word "computer" \
  --samples 5000 \
  --engine piper
```

### For Different Languages/Accents
```bash
# Configure in config.yaml first, then:
python -m wake_word.generate \
  --wake-word "bonjour" \
  --engine pyttsx3 \
  --samples 1500
```

### For Resource-Constrained Environments
```bash
# Use fallback engine when Piper/PyTTSX3 unavailable
python -m wake_word.generate \
  --wake-word "jade" \
  --engine fallback \
  --samples 500 \
  --no-augment
```

## Troubleshooting

### Common Issues and Solutions

**1. Config file not found:**
```bash
# Error: Configuration file not found
# Solution: Ensure config.yaml is in project root
ls config.yaml  # Should exist
python -m wake_word.generate --config ./path/to/your/config.yaml
```

**2. Piper TTS not available:**
```bash
# Error: Piper not found
# Solution: Use fallback or install Piper
python -m wake_word.generate --engine fallback
# Or setup Piper according to the installation guide
```

**3. Audio dependencies missing:**
```bash
# Error: librosa not available
# Solution: Install audio dependencies
pip install librosa soundfile
# Or the script will automatically use scipy fallback
```

**4. No samples generated:**
```bash
# Check logs for detailed error information
cat ./output/logs/data_generation.log

# Try test mode first
python -m wake_word.generate --test-mode --engine fallback
```

**5. Permission errors:**
```bash
# Error: Permission denied writing to output directory
# Solution: Check permissions or use different output directory
python -m wake_word.generate --output-dir ~/my_wake_word_data
```

## Configuration Tips

### Quick Config Changes via Command Line
Instead of editing config.yaml every time, use command line overrides:

```bash
# Change wake word without editing config
python -m wake_word.generate --wake-word "alexa"

# Use different sample counts for testing vs production
python -m wake_word.generate --samples 100  # testing
python -m wake_word.generate --samples 2000 # production

# Switch engines easily
python -m wake_word.generate --engine piper      # best quality
python -m wake_word.generate --engine pyttsx3    # system voices
python -m wake_word.generate --engine fallback   # no dependencies
```

### Environment Variables
You can also set some options via environment variables:

```bash
export WAKE_WORD_CONFIG="./custom_config.yaml"
export WAKE_WORD_OUTPUT_DIR="./models"
python -m wake_word.generate
```

## Performance Optimization

### For Faster Generation
```bash
# Disable augmentation for speed
python -m wake_word.generate --no-augment

# Use test mode for quick iterations
python -m wake_word.generate --test-mode

# Use fallback engine (fastest)
python -m wake_word.generate --engine fallback
```

### For Better Quality
```bash
# Use Piper TTS with more samples
python -m wake_word.generate --engine piper --samples 3000

# Enable all augmentations (default)
python -m wake_word.generate  # augmentation enabled by default
```

## Integration with Training Pipeline

### Complete Workflow
```bash
# 1. Generate data
python -m wake_word.generate --wake-word "jade" --samples 1000

# 2. Train model (assuming you have train.py)
python -m wake_word.train --config ./output/data_generation_summary.yaml

# 3. Package for deployment (assuming you have package.py)
python -m wake_word.package --model ./output/models/jade_model.onnx

# 4. Test on Raspberry Pi
python -m wake_word.test --model ./output/models/jade_model.onnx
```

### Batch Processing Multiple Wake Words
```bash
# Generate multiple wake words
for word in "jade" "computer" "alexa" "assistant"; do
  echo "Generating data for: $word"
  python -m wake_word.generate \
    --wake-word "$word" \
    --samples 1000 \
    --output-dir "./models/${word}_model"
done
```

## Monitoring and Logging

### View Generation Progress
```bash
# Run with verbose logging (edit config.yaml to set logging.level: DEBUG)
python -m wake_word.generate

# Monitor log file in real-time
tail -f ./output/logs/data_generation.log
```

### Check Generated Data Quality
```bash
# After generation, inspect the output
ls -la ./output/data/training/positive/  # Should have positive samples
ls -la ./output/data/training/negative/  # Should have negative samples

# Check summary
cat ./output/data_generation_summary.yaml
```

## Advanced Usage

### Custom Config for Specific Use Cases
```bash
# Create custom config for different scenarios
cp config.yaml config_raspberry_pi.yaml
# Edit config_raspberry_pi.yaml for Pi-specific settings

# Use custom config
python -m wake_word.generate --config config_raspberry_pi.yaml
```

### Programmatic Usage
You can also import and use the generator in your own Python scripts:

```python
from wake_word.generate import SyntheticDataGenerator

# Initialize with custom config
generator = SyntheticDataGenerator("my_config.yaml")

# Override settings programmatically
generator.config.update('wake_word.primary', 'my_word')
generator.config.update('data_generation.n_samples', 500)

# Run pipeline
success = generator.run_full_pipeline()
```

This gives you maximum flexibility for integrating wake word generation into larger ML pipelines or automation scripts.