import aubio
import time
import sys
import librosa
from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd
import sounddevice as sd
import signal
from collections import deque

#these values have to be the same across the note checker and note extractor
BUFFER_SIZE = 4096
HOP_SIZE = 512
SAMPLE_RATE = 44100

pitchWee = aubio.pitch("yinfft", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
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

metronomeBpm = 120 #bpm that will be heard by the student
originalBpm = 120 #bpm of the sample audio

songName = "Hotel California"
excelpath = f"{songName}Notes.xlsx"

songDf = pd.read_excel(excelpath)
songDf.head()

def handle_sigint(signum, frame):
    print("\nExiting")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# new: buffer to smooth frequencies and detect octave flips
recent_freqs = deque(maxlen=5)


iTimestamp = time.time()
print(iTimestamp)
fTimestamp = 0
accuracy = 0 
nReadings = 0 

metronomeBpm = 120 #bpm that will be heard by the student
originalBpm = 120 #bpm of the sample audio
metronomeCoeff = metronomeBpm/originalBpm

def callback(indata, frames, time, status):
    
    fTimestamp = time.time() 
    
    timelapsed = (fTimestamp - iTimestamp) * metronomeCoeff
    if status:
        print(status)

    samples = np.mean(indata, axis=1).astype(np.float32)

    freq = pitchWee(samples)[0]
    conf = pitchWee.get_confidence()

    filtered_freq = freq

    if filtered_freq > 100:
        half = filtered_freq / 2
        if any(abs(filtered_freq - (f*2)) < 5 for f in recent_freqs):
            filtered_freq = half

    recent_freqs.append(filtered_freq)

    # isCorrect = 0
    overallFreq = np.mean(recent_freqs)
    closest_idx = (songDf["timestamp"] - timelapsed).abs().idxmin()
    songDf.at[closest_idx, "played freq"] = filtered_freq
    songDf.at[closest_idx, "played note"] = aubio.freq2note(filtered_freq)
    
    # nReadings+=1
    # accuracy = (accuracy + isCorrect) * ((nReadings)/(nReadings+1)
    suggestedEnd = timelapsed>songDf.loc["timestamp", -1]

    if (timelapsed>suggestedEnd):
        print("Song over")


with sd.InputStream(channels=1, callback=callback,
                    blocksize=HOP_SIZE,
                    samplerate=SAMPLE_RATE):
    print("Listening\n")
    while True:
        sd.sleep(1000)