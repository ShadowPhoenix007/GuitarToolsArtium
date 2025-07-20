import sounddevice as sd
import numpy as np
import aubio
import signal
import sys
import math
from scipy.signal import butter, lfilter  # added for filtering
from collections import deque  # added for smoothing

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
notesMidi = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
notesToTune = ["E2", "A2", "D3", "G3", "B3", "E4"]

tuningTolerance = 3
tuningCoeff = 3.1415

def note_midi_freq(note):
    midiStem = notesMidi[note[:-1]]
    octave = int(note[-1])
    midiNum = (octave+1) * 12 + midiStem
    frequencyHz = 440 * (2**((midiNum-69)/12))
    print(f"the frequency of {note} is {frequencyHz:.2f}")
    return frequencyHz

desFreq = [note_midi_freq(note) for note in notesToTune]  

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

def handle_sigint(signum, frame):
    print("\nExiting")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# new: buffer to smooth frequencies and detect octave flips
recent_freqs = deque(maxlen=5)

def callback(indata, frames, time, status):
    if status:
        print(status)

    samples = np.mean(indata, axis=1).astype(np.float32)

    freq = pitchWee(samples)[0]
    conf = pitchWee.get_confidence()

    filtered_freq = freq

    # Octave-flip correction: if freq is ~2x a previous freq, halve it
    if filtered_freq > 100:
        half = filtered_freq / 2
        if any(abs(filtered_freq - (f*2)) < 5 for f in recent_freqs):
            filtered_freq = half

    # Add current frequency to smoothing buffer
    recent_freqs.append(filtered_freq)

    # Filter: must be in reasonable guitar range and confidence okay
    if 60 < filtered_freq < 500 and conf >= 0:
        difFreq = [abs(filtered_freq - desired) for desired in desFreq]
        minDif = min(difFreq)
        index = difFreq.index(minDif)
        tuningCoeff = (aubio.note2freq(notesToTune[index]) / 440) * 3.1415
        if minDif/tuningCoeff > tuningTolerance:
            if desFreq[index] > filtered_freq:
                print(f"String {index+1}: {notesToTune[index]}: Tune up +{(minDif/tuningCoeff):.2f}")
            else:
                print(f"String {index+1}: {notesToTune[index]}: Tune down -{(minDif/tuningCoeff):.2f}")
        else:
            print(f"String {index+1}: {notesToTune[index]} is in tune")

with sd.InputStream(channels=1, callback=callback,
                    blocksize=HOP_SIZE,
                    samplerate=SAMPLE_RATE):
    print("Listening\n")
    while True:
        sd.sleep(1000)
