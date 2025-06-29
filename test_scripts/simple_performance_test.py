#!/usr/bin/env python3
"""
Simple performance test for YAMNet classifier optimizations
"""

import time
import sys
from pathlib import Path

def test_regular_yamnet(audio_file, cache_dir="my_model_cache"):
    """Test the regular YAMNet classifier with timing"""
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

def test_second_run(audio_file, cache_dir="my_model_cache"):
    """Test second run to see if caching helps"""
    print("\n" + "="*60)
    print("TESTING SECOND RUN (CACHED)")
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

def main():
    """Main function to run performance comparison"""
    if len(sys.argv) != 2:
        print("Usage: python simple_performance_test.py <path_to_wav_file>")
        print("Example: python simple_performance_test.py great_dane_bark8.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    try:
        # Test regular YAMNet (first run)
        regular_results, regular_time = test_regular_yamnet(audio_file)
        
        # Test second run (should be faster due to caching)
        cached_results, cached_time = test_second_run(audio_file)
        
        # Performance comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"First run (loading + inference): {regular_time:.2f} seconds")
        print(f"Second run (cached):             {cached_time:.2f} seconds")
        print(f"Speed improvement: {regular_time/cached_time:.1f}x faster")
        
        # Results comparison
        print(f"\nTop prediction comparison:")
        print(f"First run:  {regular_results[0][0]} ({regular_results[0][1]:.3f})")
        print(f"Second run: {cached_results[0][0]} ({cached_results[0][1]:.3f})")
        
        if regular_results[0][0] == cached_results[0][0]:
            print("✓ Predictions match!")
        else:
            print("⚠ Predictions differ")
        
        # Breakdown of improvements
        print(f"\nBreakdown:")
        print(f"- Model loading time saved: {regular_time - cached_time:.2f} seconds")
        print(f"- Inference time remains: ~{cached_time:.2f} seconds")
        print(f"- Total speedup: {regular_time/cached_time:.1f}x")
        
    except Exception as e:
        print(f"Error during performance test: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 