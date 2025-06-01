"""
===========================================================
        BIRD SOUND FREQUENCY ANALYZER - MINI PROJECT
===========================================================

Author      : Your Name
Date        : June 1, 2025
Description : This script analyzes bird audio files to find
              their dominant frequency using FFT.
===========================================================
"""

# ===== IMPORTS =====
import librosa
import numpy as np
import time

# Optional: Uncomment if you want stylish terminal output
# from colorama import Fore, Style, init
# init()

# ===== PROJECT BANNER =====
print("="*60)
print("         WELCOME TO BIRD SOUND FREQUENCY ANALYZER")
print("="*60)

# ===== AUDIO FILE LIST =====
audio_paths = [
    r'C:\Users\user\Downloads\the-cry-of-a-beautiful-bird.mp3',
    r'C:\Users\user\Downloads\the-cry-of-an-eagle.mp3',
    r'C:\Users\user\Downloads\the-long-chirping-of-a-sparrow.mp3',
    r'C:\Users\user\Downloads\the-sound-of-a-crow-croaking.mp3'
]

# ===== TIMER START =====
start_time = time.time()

# ===== ANALYSIS BEGINS =====
print("\nStarting frequency analysis of bird audio samples...\n")

dominant_frequencies = []
i = 0
while i < 4:
    print(f"\nProcessing sample {i+1}: {audio_paths[i]}")

    # Load the audio
    y, sr = librosa.load(audio_paths[i])

    # Apply FFT
    n = len(y)
    Y = np.fft.fft(y)
    magnitude = np.abs(Y)
    freq = np.fft.fftfreq(n, d=1/sr)

    # Positive frequencies only
    half_n = n // 2
    positive_freqs = freq[:half_n]
    positive_magnitude = magnitude[:half_n]

    # Find dominant frequency
    peak_idx = np.argmax(positive_magnitude)
    dominant_freq = positive_freqs[peak_idx]

    print(f"Dominant frequency: {dominant_freq:.2f} Hz")
    dominant_frequencies.append(dominant_freq)
    i += 1

# ===== RESULTS SUMMARY =====
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
