import librosa
import numpy as np
import time
from pathlib import Path

# === Banner ===
print("="*60)
print("         WELCOME TO BIRD SOUND FREQUENCY ANALYZER")
print("="*60)

# === Load audio files ===
folder_path = Path(r"C:\Users\user\Desktop\AI_ML\Andean Guan_sound")
audio_paths = list(folder_path.glob("*.mp3"))

print(f"\nFound {len(audio_paths)} audio files. Starting analysis...\n")

# === Timer start ===
start_time = time.time()

# === Main loop ===
dominant_frequencies = []

for i, path in enumerate(audio_paths):
    print(f"\nProcessing sample {i+1}: {path.name}")
    try:
        y, sr = librosa.load(path, sr=None)

        if np.max(np.abs(y)) < 0.01:
            print("Audio too silent — skipping...")
            dominant_frequencies.append(0.0)
            continue

        y = y / np.max(np.abs(y))
        Y = np.fft.fft(y)
        freq = np.fft.fftfreq(len(y), d=1/sr)
        magnitude = np.abs(Y)

        valid = (freq > 100) & (freq < sr / 2)
        peak = np.argmax(magnitude[valid])
        dominant_freq = freq[valid][peak]

        print(f"Dominant frequency: {dominant_freq:.2f} Hz")
        dominant_frequencies.append(dominant_freq)

    except Exception as e:
        print(f"Error: {e}")
        dominant_frequencies.append(0.0)

# === Timer end ===
print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")
print("\nThank you for using the Bird Sound Frequency Analyzer!")
print("="*60)

