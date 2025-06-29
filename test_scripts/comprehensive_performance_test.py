#!/usr/bin/env python3
"""
Comprehensive performance test for YAMNet classifier optimizations
"""

import time
import sys
from pathlib import Path
import numpy as np

def test_regular_yamnet(audio_file, cache_dir="my_model_cache"):
    """Test the regular YAMNet classifier"""
    print("\n" + "="*60)
    print("TESTING REGULAR YAMNET CLASSIFIER")
    print("="*60)
    
    # Import here to avoid loading both models at once
    from yamnet_classifier import YAMNetClassifier
    
    # Time the entire process
    total_start = time.time()
    
    # Initialize classifier
    classifier = YAMNetClassifier(cache_dir=cache_dir)
    
    # Classify audio
    results = classifier.classify_audio(audio_file, top_k=5)
    
    total_time = time.time() - total_start
    
    # Print results
    classifier.print_results(results, audio_file)
    print(f"\nTotal time (including loading): {total_time:.2f} seconds")
    
    return results, total_time

def test_regular_yamnet_cached(audio_file, cache_dir="my_model_cache"):
    """Test the regular YAMNet classifier (cached)"""
    print("\n" + "="*60)
    print("TESTING REGULAR YAMNET CLASSIFIER (CACHED)")
    print("="*60)
    
    # Import here to avoid loading both models at once
    from yamnet_classifier import YAMNetClassifier
    
    # Time the entire process
    total_start = time.time()
    
    # Initialize classifier
    classifier = YAMNetClassifier(cache_dir=cache_dir)
    
    # Classify audio
    results = classifier.classify_audio(audio_file, top_k=5)
    
    total_time = time.time() - total_start
    
    # Print results
    classifier.print_results(results, audio_file)
    print(f"\nTotal time (cached): {total_time:.2f} seconds")
    
    return results, total_time

def test_inference_only(audio_file, cache_dir="my_model_cache"):
    """Test inference time only (model already loaded)"""
    print("\n" + "="*60)
    print("TESTING INFERENCE TIME ONLY")
    print("="*60)
    
    from yamnet_classifier import YAMNetClassifier
    
    # Initialize classifier once
    classifier = YAMNetClassifier(cache_dir=cache_dir)
    
    # Test multiple inference runs
    times = []
    for i in range(3):
        start_time = time.time()
        results = classifier.classify_audio(audio_file, top_k=5)
        inference_time = time.time() - start_time
        times.append(inference_time)
        print(f"Run {i+1}: {inference_time:.3f} seconds")
    
    avg_time = np.mean(times)
    print(f"\nAverage inference time: {avg_time:.3f} seconds")
    
    return results, avg_time

def test_batch_processing(audio_files, cache_dir="my_model_cache"):
    """Test batch processing multiple files"""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    from yamnet_classifier import YAMNetClassifier
    
    # Initialize classifier once
    classifier = YAMNetClassifier(cache_dir=cache_dir)
    
    # Process multiple files
    total_start = time.time()
    all_results = {}
    
    for i, audio_file in enumerate(audio_files):
        print(f"Processing file {i+1}/{len(audio_files)}: {Path(audio_file).name}")
        start_time = time.time()
        results = classifier.classify_audio(audio_file, top_k=3)
        file_time = time.time() - start_time
        all_results[audio_file] = results
        print(f"  Time: {file_time:.3f} seconds")
    
    total_time = time.time() - total_start
    avg_time = total_time / len(audio_files)
    
    print(f"\nBatch processing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per file: {avg_time:.3f} seconds")
    
    return all_results, avg_time

def main():
    """Main function to run comprehensive performance comparison"""
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_performance_test.py <path_to_wav_file> [additional_files...]")
        print("Example: python comprehensive_performance_test.py great_dane_bark8.wav")
        print("Example: python comprehensive_performance_test.py great_dane_bark8.wav male-yelling-out-a-war-cry-4-without-reverb.wav")
        sys.exit(1)
    
    audio_files = sys.argv[1:]
    
    # Check if files exist
    for audio_file in audio_files:
        if not Path(audio_file).exists():
            print(f"Error: Audio file '{audio_file}' not found!")
            sys.exit(1)
    
    try:
        # Test regular YAMNet (first run)
        regular_results, regular_time = test_regular_yamnet(audio_files[0])
        
        # Test regular YAMNet (cached)
        cached_results, cached_time = test_regular_yamnet_cached(audio_files[0])
        
        # Test inference time only
        inference_results, inference_time = test_inference_only(audio_files[0])
        
        # Test batch processing if multiple files provided
        if len(audio_files) > 1:
            batch_results, batch_time = test_batch_processing(audio_files)
        else:
            batch_results, batch_time = None, None
        
        # Performance comparison
        print("\n" + "="*60)
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print("="*60)
        print(f"Regular YAMNet (first run): {regular_time:.2f} seconds")
        print(f"Regular YAMNet (cached):    {cached_time:.2f} seconds")
        print(f"Inference only (avg):        {inference_time:.2f} seconds")
        if batch_time:
            print(f"Batch processing (avg/file): {batch_time:.2f} seconds")
        
        # Speed improvements
        print(f"\nSpeed Improvements:")
        print(f"Regular → Cached: {regular_time/cached_time:.1f}x faster")
        print(f"Regular → Inference only: {regular_time/inference_time:.1f}x faster")
        if batch_time:
            print(f"Regular → Batch: {regular_time/batch_time:.1f}x faster")
        
        # Breakdown
        print(f"\nBreakdown:")
        print(f"- Model loading time: {regular_time - cached_time:.2f} seconds")
        print(f"- Audio loading time: ~0.1-0.3 seconds")
        print(f"- Inference time: ~{inference_time:.2f} seconds")
        
        # Optimization recommendations
        print(f"\nOptimization Recommendations:")
        print(f"✓ Use local caching (already implemented)")
        print(f"✓ Process multiple files in batch")
        print(f"✓ Keep model in memory for repeated use")
        print(f"✓ Use shorter audio files when possible")
        
        # Results comparison
        print(f"\nResults Comparison:")
        print(f"Regular: {regular_results[0][0]} ({regular_results[0][1]:.3f})")
        print(f"Cached:  {cached_results[0][0]} ({cached_results[0][1]:.3f})")
        
        if regular_results[0][0] == cached_results[0][0]:
            print("✓ Regular and cached predictions match!")
        else:
            print("⚠ Regular and cached predictions differ")
        
    except Exception as e:
        print(f"Error during performance test: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 