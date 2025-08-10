Frequency Analyzer
Detect the dominant frequency in bird voice recordings.

Overview

This project analyzes audio recordings of bird sounds to determine their dominant frequencies. It’s designed for ornithologists, bioacoustics researchers, and hobbyists interested in bird vocalization analysis.

Features

Extracts the dominant frequency from audio files
Focuses on bird voice recordings
Written entirely in Python
Simple and easy to use

Getting Started

Prerequisites
Python 3.x

Required packages (install with pip):
bash
pip install  librosa pandas numpy

Usage

Place your audio file (.mp3) in the project directory.

Run the main script:
bash
python "Frequency_analyzer.py"

Follow the on-screen prompts to analyze your file.

Example Output:
    
    Found 30 audio files for analysis...


  ✅ Analysis complete. Data saved to 'bird_dataset_dominant_freq.csv'.

  Sample results:
    
            Filename             Species            Dominant_Frequency_Hz

    0  Andean Guan10.mp3      Andean Guan10                  28.97

    1  Andean Guan11.mp3      Andean Guan11                1274.80

    2  Andean Guan12.mp3      Andean Guan12                1416.15

    3  Andean Guan13.mp3      Andean Guan13                1504.31

    4  Andean Guan14.mp3      Andean Guan14                1239.44

Printed dominant frequency value

Files

Frequency_analyzer.py – Main script for frequency analysis

Author
Osaidsk
