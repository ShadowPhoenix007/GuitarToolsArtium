import aubio
import time
import sys
import librosa
from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd

#these values have to be the same across the note checker and note extractor
BUFFER_SIZE = 4096
HOP_SIZE = 512
SAMPLE_RATE = 44100

pitchWee = aubio.pitch("default", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitchWee.set_unit("Hz")
pitchWee.set_silence(-40)
pitchWee.set_tolerance(0.8)

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=50.0, fs=44100, order=4):
    b, a = butter_highpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=80.0, highcut=1300.0, fs=44100, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

songName = "Seven Nation Army"
audioData, sr = librosa.load(f"{songName}.wav", sr=44100, mono=True) #this has to be linked to the mp3 song file
harmonic, percussive = librosa.effects.hpss(audioData)
finProcessed = bandpass_filter(harmonic)
finProcessed = highpass_filter(finProcessed)

refPitches = []

print(len(finProcessed))

# Frame-based iteration
numFrames = int((len(finProcessed) - BUFFER_SIZE) / HOP_SIZE)

for i in range(numFrames):
    start = i * HOP_SIZE
    frame = finProcessed[start:start + HOP_SIZE]

    if len(frame) < HOP_SIZE:
        break

    frame = frame.astype(np.float32)
    freq = pitchWee(frame)[0]

    if not 60 < freq < 1319:
        continue


    timestamp = i * HOP_SIZE / SAMPLE_RATE

    print(f"{timestamp:.3f}s: freq = {freq:.2f}, and note = {aubio.freq2note(freq)}") 

    refPitches.append((timestamp, freq, aubio.freq2note(freq)))


pitchDf = pd.DataFrame(refPitches, columns=["timestamp", "frequency", "note"])

pitchDf.to_excel(f"{songName}Notes.xlsx", index=False)