import librosa
import numpy as np
import glob
import os

# ===== AUDIO FILE LIST (AUTO-DETECTED) =====
folder_path = r'C:\Users\user\Desktop\AI_ML\Andean Guan_sound'
audio_paths = sorted(glob.glob(os.path.join(folder_path, '*.mp3')))

print(f"\nFound {len(audio_paths)} audio files.\n")
print("Starting frequency analysis of bird audio samples...\n")

dominant_frequencies = []

for i, path in enumerate(audio_paths):
    print(f"\nProcessing sample {i+1}: {path}")

    try:
        y, sr = librosa.load(path, sr=None)

        if np.max(np.abs(y)) < 0.01:
            print("Audio too silent — skipping...")
            dominant_frequencies.append(0.0)
            continue

        y = y / np.max(np.abs(y))

        n = len(y)
        Y = np.fft.fft(y)
        magnitude = np.abs(Y)
        freq = np.fft.fftfreq(n, d=1/sr)

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
    filename = os.path.basename(audio_paths[idx])
    print("{:<10} {:<50} {:>15.2f}".format(f"Sample {idx+1}", filename, freq))
