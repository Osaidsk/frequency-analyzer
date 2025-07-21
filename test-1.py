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
import librosa
import numpy as np
import time

# ===== PROJECT BANNER =====
print("="*60)
print("         WELCOME TO BIRD SOUND FREQUENCY ANALYZER")
print("="*60)

# ===== AUDIO FILE LIST =====
audio_paths = [
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan2.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan3.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan4.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan5.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan6.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan7.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan8.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan9.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan10.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan11.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan12.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan13.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan14.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan15.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan16.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan17.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan18.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan19.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan20.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan21.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan22.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan23.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan24.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan25.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan26.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan27.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan28.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan29.mp3',
    r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound\Andean Guan30.mp3'
]

# ===== TIMER START =====
start_time = time.time()

# ===== ANALYSIS BEGINS =====
print("\nStarting frequency analysis of bird audio samples...\n")

dominant_frequencies = []

for i, path in enumerate(audio_paths):
    print(f"\nProcessing sample {i+1}: {path}")

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
        freq = np.fft.fftfreq(n, d=1/sr)

        # Filter positive frequencies above 100 Hz
        valid_indices = (freq > 100) & (freq < sr / 2)
        positive_freqs = freq[valid_indices]
        positive_magnitude = magnitude[valid_indices]

        if len(positive_magnitude) == 0:
            print("No valid frequency content — skipping...")
            dominant_frequencies.append(0.0)
            continue

        peak_idx = np.argmax(positive_magnitude)
        dominant_freq = positive_freqs[peak_idx]

        print(f"Dominant frequency: {dominant_freq:.2f} Hz")
        dominant_frequencies.append(dominant_freq)

    except Exception as e:
        print(f"Error processing file: {e}")
        dominant_frequencies.append(0.0)

# ===== SUMMARY =====
print("\n================= DOMINANT FREQUENCY SUMMARY =================")
print("{:<10} {:<50} {:>15}".format("Sample", "Filename", "Frequency (Hz)"))
print("-" * 80)
for idx, freq in enumerate(dominant_frequencies):
    filename = audio_paths[idx].split('\\')[-1]
    print("{:<10} {:<50} {:>15.2f}".format(f"Sample {idx+1}", filename, freq))

# ===== EXECUTION TIME =====
end_time = time.time()
print("\nTotal Execution Time: {:.2f} seconds".format(end_time - start_time))

# ===== CLOSING MESSAGE =====
print("\nThank you for using the Bird Sound Frequency Analyzer!")
print("="*60)
