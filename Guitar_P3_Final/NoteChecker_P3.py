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

#Config
iTimestamp = 0 
lastAccNoteTimestamp = 0

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

# Song setup

metronomeBpm = 120
originalBpm = 120
metronomeCoeff = metronomeBpm / originalBpm

songName = "Seven Nation Army"
excelpath = f"{songName}Notes.xlsx"
songDf = pd.read_excel(excelpath)
if "frequency" not in songDf.columns:
    songDf["frequency"] = np.nan 

recent_freqs = deque(maxlen=5)
nCorrect = 0
nTotal = 0
timing_errors = []
note_tolerance = 0.5  # semitones
time_tolerance = 0.2  # seconds
started = False
# Smooth Exit
def handle_sigint(signum, frame):
    print("\nExiting")
    sys.exit(0)
signal.signal(signal.SIGINT, handle_sigint)

def freq_to_midi(freq):
    return 69 + 12 * np.log2(freq / 440.0) if freq > 0 else None

def match_note_and_timing(played_freq, played_time):
    """Return True if pitch and timing match within tolerances."""
    global songDf

    # Find closest note by timestamp
    closest_idx = (songDf["timestamp"] - played_time).abs().idxmin()
    expected_freq = songDf.at[closest_idx, "frequency"]

    played_midi = freq_to_midi(played_freq)
    expected_midi = freq_to_midi(expected_freq)
    if expected_midi is None or played_midi is None:
        return False, 0

    pitch_match = abs(played_midi - expected_midi) <= note_tolerance
    expected_time = played_time
    closest_note_idx = closest_idx
    time_window = 3.0  # seconds
    recent_notes = songDf[(songDf["timestamp"] >= played_time - time_window) &
                        (songDf["timestamp"] <= played_time + time_window)]

    # Check if the played note exists in the recent window
    matching_indices = recent_notes[recent_notes["note"] == aubio.freq2note(np.clip(played_freq, 50, 1400))].index

    if len(matching_indices) > 0:
        # Pick the one closest in time to current_time
        closest_note_idx = min(matching_indices, key=lambda idx: abs(songDf.at[idx, "timestamp"] - played_time))
        pitch_match = True
        expected_time = songDf.at[closest_note_idx, "timestamp"]

    time_error = played_time - expected_time
    time_match = abs(time_error) <= time_tolerance

    return pitch_match, (0 if time_match else time_error)

# Callback 
def callback(indata, frames, time_info, status):
    global recent_freqs, nCorrect, nTotal, timing_errors, started, iTimestamp, lastAccNoteTimestamp

    if status:
        print(status)

    # Process audio
    samples = np.mean(indata, axis=1).astype(np.float32)
    samples = highpass_filter(samples, cutoff=50.0, fs=SAMPLE_RATE)
    freq = pitchWee(samples.astype(np.float32))[0]

    # Smooth octave jumps
    filtered_freq = freq
    # if filtered_freq > 100:
    #     if any(abs(filtered_freq - (f * 2)) < 5 for f in recent_freqs):
    #         filtered_freq /= 2
    #     if any(abs(filtered_freq*2 - f) < 5 for f in recent_freqs):
    #         filtered_freq *= 2

        
    recent_freqs.append(filtered_freq)
    overallFreq = np.mean(recent_freqs)

    # Start alignment logic
    if not started:
        # Only start if a note is played
        if (overallFreq != 0):
            started = True
            iTimestamp = time.time()
            print("Start detected! Beginning accuracy tracking.")

    else:
        # Current time in song (adjusted for metronome BPM)
        current_time = (time.time() - iTimestamp) * metronomeCoeff

        #  Accuracy tracking 
        if (current_time-lastAccNoteTimestamp > 0.3) and overallFreq != 0:
            match, time_error = match_note_and_timing(overallFreq, current_time)
            nTotal += 1
            if match:
                nCorrect += 1
            timing_errors.append(time_error)
            # Accuracy & timing feedback 
            accuracy = (nCorrect / nTotal) * 100
            if len(timing_errors) > 5:
                avg_error = np.mean(timing_errors[-min(10, len(timing_errors)):])
                if avg_error > 0.05:
                    timing_msg = f"Dragging by {avg_error:.2f} sec"
                elif avg_error < -0.05:
                    timing_msg = f"Rushing by {abs(avg_error):.2f} sec"
                else:
                    timing_msg = "Good timing!"
            else:
                timing_msg = "Collecting timing data..."
            lastAccNoteTimestamp = current_time

            #print commands
            printcorrect = "correct note" if match else "incorrect note"
            print(f"{printcorrect}| Accuracy: {accuracy:.1f}% | {timing_msg}")
            print("Detected:", overallFreq)
            print("current time: ", current_time)
            closest_idx = (songDf["timestamp"] - current_time).abs().idxmin()
            expected_freq = songDf.at[closest_idx, "frequency"]
            print("expectedfreq: ", expected_freq)

            lastTimestamp = songDf["timestamp"].iloc[-1]
            if (current_time>lastTimestamp):
                print('song over, well done')
                raise sd.CallbackStop #THIS is the end of the song, the code can end

# Main
with sd.InputStream(channels=1, callback=callback,
                    blocksize=HOP_SIZE,
                    samplerate=SAMPLE_RATE):
    print("Listening...\n")
    while True:
        sd.sleep(1000)
