"""
This script captures audio from the microphone and prints it to the console.
"""

import sounddevice as sd
import sys
import numpy as np
from datetime import datetime
import os


def list_audio_devices():
    """List all available audio devices with their indices and names."""
    devices = sd.query_devices()
    print("Available audio devices:")
    print("-" * 50)
    
    for i, device in enumerate(devices):
        device_type = []
        # Check if device supports input
        if 'max_inputs' in device and device['max_inputs'] > 0:
            device_type.append("INPUT")
        elif 'input_channels' in device and device['input_channels'] > 0:
            device_type.append("INPUT")
        
        # Check if device supports output
        if 'max_outputs' in device and device['max_outputs'] > 0:
            device_type.append("OUTPUT")
        elif 'output_channels' in device and device['output_channels'] > 0:
            device_type.append("OUTPUT")
        
        device_type_str = "/".join(device_type)
        print(f"Device {i}: {device['name']} ({device_type_str})")
        
        # Show input details if available
        if 'max_inputs' in device and device['max_inputs'] > 0:
            print(f"  - Input channels: {device['max_inputs']}")
        elif 'input_channels' in device and device['input_channels'] > 0:
            print(f"  - Input channels: {device['input_channels']}")
        
        if 'default_samplerate' in device:
            print(f"  - Default sample rate: {device['default_samplerate']}")
        print()
    
    return devices


def find_device_by_name(device_name, devices=None):
    """Find device index by name (case-insensitive partial match)."""
    if devices is None:
        devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        # Check if device supports input
        has_input = False
        if 'max_inputs' in device and device['max_inputs'] > 0:
            has_input = True
        elif 'input_channels' in device and device['input_channels'] > 0:
            has_input = True
            
        if device_name.lower() in device['name'].lower() and has_input:
            return i
    return None


def save_audio(audio, filename=None, samplerate=16000):
    """
    Save audio to a file in the audio_files directory.
    
    Args:
        audio: Audio data as numpy array
        filename: Output filename (optional, will generate timestamped name)
        samplerate: Sample rate in Hz
    """
    # Create audio_files directory if it doesn't exist
    audio_dir = "audio_files"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created directory: {audio_dir}")
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_recording_{timestamp}.wav"
    
    # Ensure the filename has .wav extension
    if not filename.endswith('.wav'):
        filename += '.wav'
    
    # Create full path in audio_files directory
    filepath = os.path.join(audio_dir, filename)
    
    try:
        # Try to use soundfile first (better quality)
        import soundfile as sf
        sf.write(filepath, audio, samplerate)
        print(f"Audio saved to: {filepath}")
        return filepath
    except ImportError:
        # Fallback to scipy if soundfile is not available
        try:
            from scipy.io import wavfile
            wavfile.write(filepath, samplerate, audio.astype(np.float32))
            print(f"Audio saved to: {filepath}")
            return filepath
        except ImportError:
            print("Warning: Neither soundfile nor scipy available. Audio not saved.")
            return None


def play_audio(audio, samplerate=16000):
    """
    Play audio through the default output device.
    
    Args:
        audio: Audio data as numpy array
        samplerate: Sample rate in Hz
    """
    try:
        print("Playing audio...")
        sd.play(audio, samplerate)
        sd.wait()  # Wait until audio is finished playing
        print("Playback finished.")
    except Exception as e:
        print(f"Error playing audio: {e}")


def load_and_play_audio(filename):
    """
    Load and play an audio file.
    
    Args:
        filename: Path to the audio file (can be relative to audio_files directory)
    """
    # If filename doesn't contain a path separator, assume it's in audio_files directory
    if not os.path.sep in filename and not filename.startswith('./') and not filename.startswith('../'):
        filename = os.path.join("audio_files", filename)
    
    try:
        # Try to use soundfile first
        import soundfile as sf
        audio, samplerate = sf.read(filename)
        print(f"Loaded audio file: {filename}")
        print(f"Sample rate: {samplerate} Hz")
        print(f"Duration: {len(audio) / samplerate:.2f} seconds")
        play_audio(audio, samplerate)
    except ImportError:
        # Fallback to scipy
        try:
            from scipy.io import wavfile
            samplerate, audio = wavfile.read(filename)
            print(f"Loaded audio file: {filename}")
            print(f"Sample rate: {samplerate} Hz")
            print(f"Duration: {len(audio) / samplerate:.2f} seconds")
            play_audio(audio, samplerate)
        except Exception as e:
            print(f"Error loading audio file: {e}")


def capture_audio(duration=2, samplerate=16000, device=None, save_file=True, play_back=True):
    """
    Capture audio from the specified device.
    
    Args:
        duration: Recording duration in seconds
        samplerate: Sample rate in Hz
        device: Device index, device name, or None for default
        save_file: Whether to save the audio to a file
        play_back: Whether to play the audio back after recording
    """
    if device is not None:
        if isinstance(device, str):
            # Try to find device by name
            device_idx = find_device_by_name(device)
            if device_idx is None:
                print(f"Error: No input device found matching '{device}'")
                return None
            device = device_idx
        
        print(f"Using device {device}: {sd.query_devices(device)['name']}")
    
    print(f"Recording {duration}s of audio…")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32", device=device)
    sd.wait()
    audio = audio.flatten()
    
    print(f"Audio captured: {len(audio)} samples")
    print(f"First 10 samples: {audio[:10]}")
    
    # Save to file if requested
    filename = None
    if save_file:
        filename = save_audio(audio, samplerate=samplerate)
    
    # Play back if requested
    if play_back:
        play_audio(audio, samplerate=samplerate)
    
    return audio, filename


def main():
    """Main function to run capture_audio"""
    
    # Check if device selection is requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_audio_devices()
            return
        elif sys.argv[1] == "--device":
            if len(sys.argv) < 3:
                print("Usage: python audio_stream_processor.py --device <device_name_or_index>")
                print("Or: python audio_stream_processor.py --list")
                return
            device = sys.argv[2]
            # Try to convert to int if it's a number
            try:
                device = int(device)
            except ValueError:
                pass  # Keep as string
        elif sys.argv[1] == "--play":
            if len(sys.argv) < 3:
                print("Usage: python audio_stream_processor.py --play <filename>")
                return
            load_and_play_audio(sys.argv[2])
            return
        else:
            print("Usage: python audio_stream_processor.py [--list] [--device <device_name_or_index>] [--play <filename>]")
            return
    else:
        device = None
    
    try:
        audio, filename = capture_audio(device=device)
        if audio is not None and filename:
            # Extract just the filename for the playback command
            playback_filename = os.path.basename(filename)
            print(f"\nRecording complete! File saved as: {filename}")
            print(f"You can play it back with: python audio_stream_processor.py --play {playback_filename}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()