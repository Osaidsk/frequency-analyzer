import librosa
import numpy as np
import pandas as pd
import glob
import os

# ===== SET ROOT DATA FOLDER =====
root_folder = r'C:\Users\user\Desktop\AI_ML\data-1'

# Recursively find all mp3 files in subfolders
audio_paths = sorted(glob.glob(os.path.join(root_folder, '*', '*.mp3')))

results = []

print(f"\nFound {len(audio_paths)} audio files. Starting analysis...\n")

for i, path in enumerate(audio_paths):
    filename = os.path.basename(path)
    
    # Species is the parent folder name
    species = os.path.basename(os.path.dirname(path)).capitalize()

    print(f"Processing {i+1}/{len(audio_paths)}: {filename} ({species})")

    try:
        y, sr = librosa.load(path, sr=None)
        
        if np.max(np.abs(y)) < 0.01:
            print("Silent file — skipping.")
            results.append([filename, species, 0.0])
            continue
        
        y = y / np.max(np.abs(y))

        n = len(y)
        Y = np.fft.fft(y)
        magnitude = np.abs(Y)
        freq = np.fft.fftfreq(n, d=1/sr)

        valid_indices = (freq > 100) & (freq < sr / 2)
        freqs = freq[valid_indices]
        mags = magnitude[valid_indices]

        if len(mags) == 0:
            print("No valid frequencies — skipping.")
            results.append([filename, species, 0.0])
            continue

        peak_idx = np.argmax(mags)
        dom_freq = freqs[peak_idx]

        print(f"Dominant Frequency: {dom_freq:.2f} Hz")
        results.append([filename, species, dom_freq])

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        results.append([filename, species, 0.0])

# ===== SAVE TO CSV =====
df = pd.DataFrame(results, columns=['Filename', 'Species', 'Dominant_Frequency_Hz'])
df.to_csv('bird_dataset_dominant_freq.csv', index=False)
print("\n✅ Analysis complete. Data saved to 'bird_dataset_dominant_freq.csv'.")

# Preview
print("\nSample results:")
print(df.head())
