# YAMNet Audio Classification

A fast and optimized audio classification system using Google's YAMNet model, designed for M1 MacBooks.

## Features

- **Fast Audio Classification**: Classify audio files into 521 different sound categories
- **Local Model Caching**: Automatically caches the model locally for faster subsequent runs
- **Optimized Performance**: 1.5x speedup with caching, 27x speedup for multiple files
- **M1 Mac Optimized**: Designed to work efficiently on Apple Silicon Macs
- **Multi-label Classification**: Detects multiple sounds simultaneously using sigmoid activation

## Files

### Core Files
- `yamnet_classifier.py` - Main YAMNet classifier with local caching
- `download_yamnet_model.py` - Utility to download and cache the YAMNet model
- `requirements.txt` - Python dependencies

### Performance Testing
- `simple_performance_test.py` - Simple performance comparison (first run vs cached)
- `comprehensive_performance_test.py` - Detailed performance analysis

### Audio Samples
- `great_dane_bark8.wav` - Dog bark audio sample
- `male-yelling-out-a-war-cry-4-without-reverb.wav` - Human vocal audio sample

### Directories
- `my_model_cache/` - Local cache for the YAMNet model
- `venv/` - Python virtual environment

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Classify an audio file:**
   ```bash
   python yamnet_classifier.py great_dane_bark8.wav
   ```

3. **Test performance:**
   ```bash
   python simple_performance_test.py great_dane_bark8.wav
   ```

## Performance

| Scenario | Time | Speedup |
|----------|------|---------|
| First run (loading + inference) | ~3.0s | 1.0x |
| Cached run | ~2.0s | **1.5x** |
| Multiple files (inference only) | ~0.11s | **27x** |

## Example Output

```
============================================================
Audio Classification Results for: great_dane_bark8.wav
============================================================
 1. Dog                            (0.648)
 2. Animal                         (0.642)
 3. Domestic animals, pets         (0.636)
 4. Bark                           (0.609)
 5. Bow-wow                        (0.572)
============================================================
Top prediction: Dog (0.648)
```

## Usage

### Basic Classification
```bash
python yamnet_classifier.py <audio_file.wav>
```

### With Custom Cache Directory
```bash
python yamnet_classifier.py <audio_file.wav> <cache_dir>
```

### Performance Testing
```bash
python simple_performance_test.py <audio_file.wav>
```

## Dependencies

- TensorFlow 2.x
- TensorFlow Hub
- Librosa
- NumPy
- Pathlib

## Notes

- The model automatically caches locally in `my_model_cache/` for faster subsequent runs
- Supports various audio formats (WAV, MP3, etc.)
- Uses sigmoid activation for multi-label classification (multiple sounds can be detected simultaneously)
- Optimized for M1 Mac with Metal GPU acceleration 