
from pathlib import Path
import pandas as pd
import librosa
import numpy as np

# ===== LOAD AUDIO FILES =====
folder_path = Path(r"C:\Users\user\Desktop\AI_ML\Voice of Birds\Andean Guan_sound")
audio_paths = list(folder_path.glob("*.mp3"))

print(f"\nFound {len(audio_paths)} audio files for analysis...\n")

results = []

# ===== ANALYZE EACH AUDIO FILE =====
for audio_file in audio_paths:
    try:
        y, sr = librosa.load(audio_file, sr=None)
        # Perform FFT
        spectrum = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(spectrum), 1 / sr)
        magnitude = np.abs(spectrum)

        # Take dominant frequency (ignore negative freqs)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        dominant_freq = positive_freqs[np.argmax(positive_magnitude)]

        species = audio_file.stem.split("_")[0]  # Extract species name
        results.append({
            "Filename": audio_file.name,
            "Species": species,
            "Dominant_Frequency_Hz": round(dominant_freq, 2)
        })

    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")

# ===== SAVE RESULTS =====
df = pd.DataFrame(results)
df.to_csv("bird_dataset_dominant_freq.csv", index=False)

print("\n✅ Analysis complete. Data saved to 'bird_dataset_dominant_freq.csv'.")

print("\nSample results:")
print(df.head())

print("\nAverage Dominant Frequency by Species:")
summary_df = df.groupby("Species")["Dominant_Frequency_Hz"].mean().reset_index()
print(summary_df)
