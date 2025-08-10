"""
===========================================================
        BIRD SOUND FREQUENCY ANALYZER - MINI PROJECT
===========================================================

Author      : Mohd Osaid
Date        : June 1, 2025
Description : This script analyzes bird audio files to find
              their dominant frequency using FFT.
===========================================================
"""

# ===== IMPORTS =====
import time
import numpy as np
import librosa
from pathlib import Path

# ===== BANNER =====
print("=" * 60)
print("         WELCOME TO BIRD SOUND FREQUENCY ANALYZER")
print("=" * 60)

# ===== LOAD AUDIO FILES =====
folder_path = Path(r"C:\Users\user\Desktop\AI_ML\Voice of Birds\Andean Guan_sound")
audio_paths = list(folder_path.glob("*.mp3"))

print(f"\nFound {len(audio_paths)} audio files for analysis...\n")

# ===== START TIMER =====
start_time = time.time()

# ===== MAIN ANALYSIS =====
dominant_frequencies = []

for i, path in enumerate(audio_paths):
    print(f"\nProcessing sample {i + 1}: {path.name}")

    try:
        y, sr = librosa.load(path, sr=None)

        # Skip silent files
        if np.max(np.abs(y)) < 0.01:
            print("Audio too silent — skipping...")
            dominant_frequencies.append(0.0)
            continue

        # Normalize
        y = y / np.max(np.abs(y))

        # FFT
        n = len(y)
        Y = np.fft.fft(y)
        magnitude = np.abs(Y)
        freq = np.fft.fftfreq(n, d=1 / sr)

        # Filter positive frequencies above 100 Hz
        valid = (freq > 100) & (freq < sr / 2)
        if not np.any(valid):
            print("No valid frequency content — skipping...")
            dominant_frequencies.append(0.0)
            continue

        peak_idx = np.argmax(magnitude[valid])
        dominant_freq = freq[valid][peak_idx]

        print(f"Dominant frequency: {dominant_freq:.2f} Hz")
        dominant_frequencies.append(dominant_freq)

    except Exception as e:
        print(f"Error processing file: {e}")
        dominant_frequencies.append(0.0)

# ===== PRINT SUMMARY =====
print("\n================= DOMINANT FREQUENCY SUMMARY =================")
print("{:<10} {:<50} {:>15}".format("Sample", "Filename", "Frequency (Hz)"))
print("-" * 80)
for idx, freq in enumerate(dominant_frequencies):
    filename = audio_paths[idx].name  # Cleaner than .split("\\")
    print("{:<10} {:<50} {:>15.2f}".format(f"Sample {idx+1}", filename, freq))

# ===== EXECUTION TIME =====
end_time = time.time()
print("\nTotal Execution Time: {:.2f} seconds".format(end_time - start_time))

# ===== END =====
print("\nThank you for using the Bird Sound Frequency Analyzer!")
print("=" * 60)
