"""
This script orchestrates the audio stream processor, yamnet classifier, and the logger.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
import yaml  # pip package: pyyaml

from audio_stream_processor import capture_audio, save_audio
from yamnet_classifier import YAMNetClassifier
from logger import Logger


# Load configuration
def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

"""
This is the main function that orchestrates the audio stream processor, yamnet classifier, and the logger.

It runs in a loop until the user stops it. It will:
1. Capture audio in 5 second chunks. Audio will be recorded continuously so there are no gaps between chunks.
2. Pass the audio to the yamnet classifier.
3. If the classifier detects a dog bark, it will log the event to the logger.
4. If the classifier does not detect a dog bark, it will wait for 5 seconds and then continue the loop.
5. If the classifier does not detect a dog bark, the audio will not be saved.
6. The loop will continue until the user stops it.
"""
def main():
    config = load_config()
    
    # Initialize components
    print("Initializing bark detection system...")
    
    # Initialize logger
    logger = Logger(log_dir=config['logs'].get('debug_log', 'logs').rsplit('/', 1)[0])
    
    # Initialize YAMNet classifier
    classifier = YAMNetClassifier(cache_dir="my_model_cache")
    
    # Get configuration values
    sample_rate = config['audio']['sample_rate']
    chunk_duration = 5  # 5 second chunks as specified
    threshold = config['detection']['threshold']
    device_index = config['audio']['device_index']
    
    print(f"Starting bark detection with:")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Chunk duration: {chunk_duration} seconds")
    print(f"  - Detection threshold: {threshold}")
    print(f"  - Audio device: {device_index}")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            print(f"\n--- Capturing {chunk_duration}s audio chunk ---")
            
            # Capture audio chunk
            audio_data, filename = capture_audio(
                duration=chunk_duration,
                samplerate=sample_rate,
                device=device_index,
                save_file=False,  # Don't save unless we detect a bark
                play_back=False   # Don't play back during detection
            )
            
            if audio_data is None:
                print("Failed to capture audio, retrying...")
                time.sleep(1)
                continue
            
            # Classify the audio
            print("Classifying audio...")
            try:
                results = classifier.classify_audio_data(audio_data, sample_rate=sample_rate, top_k=3)
                
                # Check if any of the top results contain "bark" or "dog"
                bark_detected = False
                bark_confidence = 0.0
                
                for class_name, confidence in results:
                    if any(keyword in class_name.lower() for keyword in ['bark', 'dog', 'woof']):
                        if confidence > threshold:
                            bark_detected = True
                            bark_confidence = confidence
                            break
                
                if bark_detected:
                    print(f"🐕 BARK DETECTED! Confidence: {bark_confidence:.3f}")
                    print(f"Top classifications: {results}")
                    
                    # Save the audio file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"bark_detected_{timestamp}.wav"
                    audio_path = save_audio(audio_data, filename, sample_rate)
                    
                    # Log the event
                    event_data = f"bark_detected,confidence:{bark_confidence:.3f},file:{filename}"
                    logger.log_event("bark_detection", event_data)
                    
                    print(f"Audio saved to: {audio_path}")
                    print("Event logged successfully")
                    
                else:
                    print(f"No bark detected. Top classifications: {results}")
                    
            except Exception as e:
                print(f"Error during classification: {e}")
                continue
            
            # Small delay before next capture to prevent overwhelming the system
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nBark detection stopped by user.")
        print("Thank you for using the bark detection system!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logging.error(f"Main loop error: {e}")


if __name__ == "__main__":
    main()

