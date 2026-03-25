"""
VAD High-Performance Module
- Fast Start: Opens microphone stream immediately.
- Auto-Folder: Saves all recordings in a folder named 'recordings'.
- Optimized: Minimal processing to ensure instant response.
"""

import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time

# ==========================================
# 1. Initialization (Fast Setup)
# ==========================================
# Create a dedicated folder for recordings if it doesn't exist
OUTPUT_DIR = "recordings"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading AI Model (Silero VAD)...")
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000 # Smaller chunks (0.25s) for faster detection

def is_speech(audio_chunk):
    audio_tensor = torch.from_numpy(audio_chunk).float()
    segments = get_speech_timestamps(audio_tensor, model, sampling_rate=SAMPLE_RATE, threshold=0.4)
    return len(segments) > 0

# ==========================================
# 2. Optimized Main Loop
# ==========================================
def run_fast_vad_system():
    print(f"\n[SYSTEM ACTIVE] Recordings will be saved in: /{OUTPUT_DIR}")
    
    while True:
        print("\nReady. Type '1' (Robo) to record instantly or 'exit' to quit:")
        trigger = input(">> ").strip().lower()

        if trigger == 'exit': break
        if trigger != '1': continue

        # --- INSTANT START ---
        print(">>> [RECORDING] Listening now...")
        
        recorded_audio = []
        silent_chunks = 0
        max_silence = 4 # Stop after ~1 second of silence

        try:
            # Using 'with' ensures the microphone is managed efficiently
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
                while True:
                    # Direct reading from buffer
                    chunk, _ = stream.read(CHUNK_SIZE)
                    chunk = chunk.flatten()
                    recorded_audio.append(chunk)

                    # Fast VAD check
                    if not is_speech(chunk):
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    if silent_chunks >= max_silence:
                        break
            
            # --- ASYNC SAVING (Fast File Naming) ---
            filename = os.path.join(OUTPUT_DIR, f"rec_{int(time.time())}.wav")
            final_audio = np.concatenate(recorded_audio)
            sf.write(filename, final_audio, SAMPLE_RATE)
            
            print(f">>> [DONE] Saved to: {filename}")

        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    run_fast_vad_system()
    
