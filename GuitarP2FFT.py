import sounddevice as sd
import numpy as np
import signal
import sys

notesToTune = ["E2", "A2", "D3", "G3", "B3", "E4"]
notesMidi = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
             "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
TUNING_TOLERANCE = 3
TUNING_COEFF = 3.1415
CONFIDENCE_THRESHOLD = 0.75


def note_midi_freq(note):
    midiStem = notesMidi[note[:-1]]
    octave = int(note[-1])
    midiNum = (octave + 1) * 12 + midiStem
    return 440 * (2 ** ((midiNum - 69) / 12))


desFreq = [note_midi_freq(note) for note in notesToTune]


def dominant_freq_with_confidence(samples, sample_rate):
    # Apply a window function
    windowed = samples * np.hanning(len(samples))

    # Perform FFT and take magnitude
    fft = np.fft.rfft(windowed)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)

    # Ignore DC and sub-50Hz
    min_index = np.searchsorted(freqs, 50)
    magnitude = magnitude[min_index:]
    freqs = freqs[min_index:]

    # Find peak
    peak_index = np.argmax(magnitude)
    peak_freq = freqs[peak_index]
    peak_mag = magnitude[peak_index]

    # Estimate confidence: ratio of peak to mean of others
    surrounding = np.concatenate((magnitude[:peak_index], magnitude[peak_index+1:]))
    noise_floor = np.mean(surrounding) + 1e-6  # avoid division by 0
    confidence = np.clip(peak_mag / noise_floor, 0, 10) / 10  # scale to [0, 1]

    return peak_freq, confidence


def handle_sigint(signum, frame):
    print("\nExiting")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


def callback(indata, frames, time, status):
    if status:
        print(status)

    samples = indata[:, 0]
    freq, confidence = dominant_freq_with_confidence(samples, SAMPLE_RATE)

    if 60 < freq < 500 and confidence > CONFIDENCE_THRESHOLD:
        difFreq = [abs(freq - desired) for desired in desFreq]
        minDif = min(difFreq)
        index = difFreq.index(minDif)

        if minDif > TUNING_TOLERANCE:
            if desFreq[index] > freq:
                print(f"String {index+1}: {notesToTune[index]}: Tune up +{(minDif/TUNING_COEFF):.2f} Hz | Confidence: {confidence:.2f}")
            else:
                print(f"String {index+1}: {notesToTune[index]}: Tune down -{(minDif/TUNING_COEFF):.2f} Hz | Confidence: {confidence:.2f}")
        else:
            print(f"String {index+1}: {notesToTune[index]} is in tune | Confidence: {confidence:.2f}")


print("Listening...")
with sd.InputStream(callback=callback, blocksize=BUFFER_SIZE,
                    samplerate=SAMPLE_RATE, channels=1):
    while True:
        sd.sleep(1000)
