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
import pickle
import time

class YAMNetClassifier:
    def __init__(self, cache_dir=None, model_path=None):
        """Initialize YAMNet model and class labels"""
        self.cache_dir = cache_dir
        self.model_path = model_path
        
        # Set up caching directory for TensorFlow Hub
        if cache_dir:
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
        
        cache_location = os.environ.get('TFHUB_CACHE_DIR', '~/.cache/tfhub_modules')
        print(f"Using cache directory: {cache_location}")
        
        # Load model and class names
        self.model, self.class_names = self._load_model_and_classes()
        print(f"Model loaded successfully with {len(self.class_names)} classes")
    
    def _load_model_and_classes(self):
        """Load model and class names with caching"""
        # Try to load from local cache first
        if self.cache_dir:
            model, class_names = self._load_from_local_cache()
            if model is not None and class_names is not None:
                print("Loaded model from local cache")
                return model, class_names
        
        # Try to load from local path if provided
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading model from local path: {self.model_path}")
            model = tf.saved_model.load(self.model_path)
            class_names = self._load_class_names_from_model(model)
            return model, class_names
        
        # Try to load from TensorFlow Hub cache
        print("Loading YAMNet model from TensorFlow Hub...")
        start_time = time.time()
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        load_time = time.time() - start_time
        print(f"Model loaded from TensorFlow Hub in {load_time:.2f} seconds")
        
        class_names = self._load_class_names_from_model(model)
        
        # Cache the model locally for future use
        if self.cache_dir:
            self._cache_model_locally(model, class_names)
        
        return model, class_names
    
    def _load_class_names_from_model(self, model):
        """Load YAMNet class names from the model"""
        class_map_path = model.class_map_path().numpy()
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            class_names = [row['display_name'] for row in reader]
        return class_names
    
    def _cache_model_locally(self, model, class_names):
        """Cache model and class names locally for faster loading"""
        cache_path = Path(self.cache_dir) / "yamnet_local"
        cache_path.mkdir(exist_ok=True)
        
        # Save the model
        model_save_path = cache_path / "model"
        tf.saved_model.save(model, str(model_save_path))
        
        # Save class names
        class_names_path = cache_path / "class_names.pkl"
        with open(class_names_path, 'wb') as f:
            pickle.dump(class_names, f)
        
        print(f"Model cached locally at: {cache_path}")
    
    def _load_from_local_cache(self):
        """Load model and class names from local cache"""
        cache_path = Path(self.cache_dir) / "yamnet_local"
        
        if not cache_path.exists():
            return None, None
        
        # Load model
        model_path = cache_path / "model"
        if not model_path.exists():
            return None, None
        
        model = tf.saved_model.load(str(model_path))
        
        # Load class names
        class_names_path = cache_path / "class_names.pkl"
        if not class_names_path.exists():
            return None, None
        
        with open(class_names_path, 'rb') as f:
            class_names = pickle.load(f)
        
        return model, class_names
    
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
        start_time = time.time()
        scores, embeddings, spectrogram = self.model(waveform_tensor)
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.3f} seconds")
        
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
    
    def classify_audio_data(self, audio_data, sample_rate=16000, top_k=5):
        """
        Classify audio data directly using YAMNet
        
        Args:
            audio_data (np.ndarray): Audio waveform data
            sample_rate (int): Sample rate of the audio data
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of tuples (class_name, confidence_score)
        """
        # Ensure audio data is the right format
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
        
        # Convert to tensor
        waveform_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        scores, embeddings, spectrogram = self.model(waveform_tensor)
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.3f} seconds")
        
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