import sounddevice as sd
import soundfile as sf
import numpy as np
import aubio
import time
from datetime import datetime

# Parameters
samplerate = int(sd.query_devices(sd.default.device[0], 'input')['default_samplerate'])
channels = 1
buffer_size = 1024
silence_timeout = 4.0  # seconds of silence before stopping
pitch_threshold = 30.0  # minimum Hz considered a note
max_pitch = 2000  # ignore unrealistic pitch values

# Aubio pitch detector
pitch_o = aubio.pitch("default", buffer_size * 4, buffer_size, samplerate)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)

recording = []
is_recording = False
silence_start_time = None
pitches = []  # NEW: store detected pitches

songname = "Seven Nation Army" #This will be an input in the UI 

# NO save_wav function needed

def audio_callback(indata, frames, time_info, status):
    global is_recording, recording, silence_start_time, pitches

    samples = indata[:, 0]
    pitch = pitch_o(samples)[0]
    print(pitch)
    # NOTE DETECTED
    if pitch > pitch_threshold and pitch < max_pitch:
        if not is_recording:
            print(f"Note detected: Hz — starting recording")
            is_recording = True
            recording = []
            pitches = []  # reset pitch list
            silence_start_time = None

    if is_recording:
        recording.extend(samples)
        if (silence_start_time is None) | (60 < pitch < 500):
            silence_start_time = time.time()
            print(f"Silence Start Time Updated to {silence_start_time} ")
        elif ((time.time() - silence_start_time)> silence_timeout):
            # Stop recording
            wav_filename = f"{songname}.wav"
            sf.write(wav_filename, np.array(recording, dtype=np.float32), samplerate)  # <--- the only changed line
            print(f"Saved recording to {wav_filename}")
            is_recording = False
            recording = []

# Continuous listening
with sd.InputStream(channels=channels, samplerate=samplerate,
                    blocksize=buffer_size, callback=audio_callback):
    print("Listening for notes...")
    while True:
        time.sleep(0.1)
