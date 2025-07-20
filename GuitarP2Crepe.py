import numpy as np
import sounddevice as sd
import crepe
import threading
import queue
import traceback
import time

SAMPLE_RATE = 16000  # CREPE works best at 16 kHz
BUFFER_DURATION = 1  # Duration of the buffer in seconds
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

notesToTune = ["E2", "A2", "D3", "G3", "B3", "E4"]
notesMidi = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
             "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

def note_midi_freq(note):
    midiStem = notesMidi[note[:-1]]
    octave = int(note[-1])
    midiNum = (octave + 1) * 12 + midiStem
    return 440 * (2 ** ((midiNum - 69) / 12))

desFreq = [note_midi_freq(note) for note in notesToTune]
tuningTolerance = 3
tuningCoeff = 3.1415

q = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio = np.mean(indata, axis=1)
    q.put(audio.copy())

def process_audio():
    try:
        while not stop_event.is_set():
            if q.empty():
                time.sleep(0.01)
                continue

            audio_block = q.get()
            if len(audio_block) < BUFFER_SIZE:
                audio_block = np.pad(audio_block, (0, BUFFER_SIZE - len(audio_block)))
            audio_block = audio_block.astype(np.float32)
            time_, frequency, confidence, activation = crepe.predict(audio_block, SAMPLE_RATE, viterbi=True)

            idx = np.argmax(confidence)
            pitch = frequency[idx]
            conf = confidence[idx]

            if conf > 0.85:
                difFreq = [abs(pitch - f) for f in desFreq]
                minDif = min(difFreq)
                index = difFreq.index(minDif)
                if minDif > tuningTolerance:
                    if pitch < desFreq[index]:
                        print(f"String {index+1}: {notesToTune[index]} - Tune up +{minDif/tuningCoeff:.2f}")
                    else:
                        print(f"String {index+1}: {notesToTune[index]} - Tune down -{minDif/tuningCoeff:.2f}")
                else:
                    print(f"String {index+1}: {notesToTune[index]} is in tune")
    except Exception as e:
        print("Audio thread crashed:")
        traceback.print_exc()

# Start input stream
stream = sd.InputStream(callback=audio_callback, channels=1,
                        samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
stream.start()

# Start thread
thread = threading.Thread(target=process_audio, daemon=True)
thread.start()

print("Listening with CREPE... Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    stop_event.set()
    thread.join()
    stream.stop()
    stream.close()
