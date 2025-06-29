#!/usr/bin/env python3
"""
Download and cache YAMNet model locally for faster subsequent runs
"""

import os
import time
import tensorflow_hub as hub
from pathlib import Path

def download_yamnet_model(cache_dir="./yamnet_cache"):
    """
    Download and cache the YAMNet model locally
    
    Args:
        cache_dir (str): Directory to cache the model
    """
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(exist_ok=True)
    
    # Set TensorFlow Hub cache directory
    os.environ['TFHUB_CACHE_DIR'] = cache_dir
    
    print(f"Downloading YAMNet model to: {os.path.abspath(cache_dir)}")
    print("This may take a few minutes on first download...")
    
    start_time = time.time()
    
    try:
        # Load the model (this will download and cache it)
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        download_time = time.time() - start_time
        print(f"✅ Model downloaded and cached successfully!")
        print(f"Download time: {download_time:.2f} seconds")
        
        # Test the model
        print("Testing model...")
        import numpy as np
        test_audio = np.random.random(16000).astype(np.float32)
        scores, embeddings, spectrogram = model(test_audio)
        print(f"✅ Model test successful! Output shape: {scores.shape}")
        
        # Show cache size
        cache_size = get_directory_size(cache_dir)
        print(f"Cache directory size: {cache_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        return False

def get_directory_size(path):
    """Get the total size of a directory in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def check_cached_model(cache_dir="./yamnet_cache"):
    """Check if the model is already cached"""
    cache_path = Path(cache_dir)
    if cache_path.exists() and any(cache_path.iterdir()):
        size = get_directory_size(cache_dir)
        print(f"✅ Model already cached in {cache_dir} ({size:.1f} MB)")
        return True
    else:
        print(f"❌ No cached model found in {cache_dir}")
        return False

def main():
    import sys
    
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "./yamnet_cache"
    
    print("YAMNet Model Downloader")
    print("=" * 40)
    
    # Check if model is already cached
    if check_cached_model(cache_dir):
        response = input("Model is already cached. Re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Download the model
    success = download_yamnet_model(cache_dir)
    
    if success:
        print("\n" + "=" * 40)
        print("Setup complete! You can now run the classifier with:")
        print(f"python yamnet_classifier.py your_audio.wav {cache_dir}")
        print("\nOr set the environment variable permanently:")
        print(f"export TFHUB_CACHE_DIR={os.path.abspath(cache_dir)}")
    else:
        print("❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()