#!/usr/bin/env python3
"""
YAMNet Audio Classification Script for M1 MacBook
Classifies audio files using Google's YAMNet model
"""

import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv
from pathlib import Path

class YAMNetClassifier:
    def __init__(self, cache_dir=None):
        """Initialize YAMNet model and class labels"""
        # Set up caching directory for TensorFlow Hub
        if cache_dir:
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
        
        cache_location = os.environ.get('TFHUB_CACHE_DIR', '~/.cache/tfhub_modules')
        print(f"Using cache directory: {cache_location}")
        
        print("Loading YAMNet model...")
        # Load the YAMNet model from TensorFlow Hub (will cache automatically)
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class labels
        self.class_names = self._load_class_names()
        print(f"Model loaded successfully with {len(self.class_names)} classes")
    
    def _load_class_names(self):
        """Load YAMNet class names from the model"""
        # YAMNet uses AudioSet class names
        # We'll get them directly from the model
        class_map_path = self.model.class_map_path().numpy()
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            class_names = [row['display_name'] for row in reader]
        return class_names
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Load and preprocess audio file for YAMNet
        
        Args:
            audio_path (str): Path to the audio file
            target_sr (int): Target sample rate (YAMNet expects 16kHz)
            
        Returns:
            np.ndarray: Preprocessed audio waveform
        """
        try:
            # Load audio file
            waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            
            # Ensure we have a reasonable length (YAMNet can handle variable lengths)
            if len(waveform) == 0:
                raise ValueError("Audio file is empty or could not be loaded")
            
            print(f"Loaded audio: {len(waveform)} samples at {target_sr}Hz")
            print(f"Duration: {len(waveform)/target_sr:.2f} seconds")
            
            return waveform
            
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def classify_audio(self, audio_path, top_k=5):
        """
        Classify audio file using YAMNet
        
        Args:
            audio_path (str): Path to the audio file
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (class_name, confidence_score)
        """
        # Preprocess audio
        waveform = self.preprocess_audio(audio_path)
        
        # Convert to tensor
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        
        # Run inference
        print("Running inference...")
        scores, embeddings, spectrogram = self.model(waveform_tensor)
        
        # Use sigmoid instead of softmax for multi-label classification
        prediction = tf.nn.sigmoid(scores)
        
        # Average predictions across all time frames
        mean_prediction = tf.reduce_mean(prediction, axis=0)
        
        # Get top k predictions
        top_indices = tf.nn.top_k(mean_prediction, k=top_k).indices.numpy()
        top_scores = tf.nn.top_k(mean_prediction, k=top_k).values.numpy()
        
        # Format results
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            class_name = self.class_names[idx]
            results.append((class_name, float(score)))
        
        return results
    
    def print_results(self, results, audio_path):
        """Print classification results in a formatted way"""
        print(f"\n{'='*60}")
        print(f"Audio Classification Results for: {Path(audio_path).name}")
        print(f"{'='*60}")
        
        for i, (class_name, confidence) in enumerate(results, 1):
            print(f"{i:2d}. {class_name:<30} ({confidence:.3f})")
        
        print(f"{'='*60}")
        print(f"Top prediction: {results[0][0]} ({results[0][1]:.3f})")

def main():
    """Main function to run the classifier"""
    if len(sys.argv) not in [2, 3]:
        print("Usage: python yamnet_classifier.py <path_to_wav_file> [cache_dir]")
        print("Example: python yamnet_classifier.py sample_audio.wav")
        print("Example: python yamnet_classifier.py sample_audio.wav ./model_cache")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) == 3 else None
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    try:
        # Initialize classifier with optional cache directory
        classifier = YAMNetClassifier(cache_dir=cache_dir)
        
        # Classify audio
        results = classifier.classify_audio(audio_file, top_k=5)
        
        # Print results
        classifier.print_results(results, audio_file)
        
        # Return the top classification
        top_class = results[0][0]
        return top_class
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()