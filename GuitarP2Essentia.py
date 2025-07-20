notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
notesMidi = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
def note_midi_freq(note):
    midiStem = notesMidi[note[:-1]]
    octave = int(note[-1])
    midiNum = (octave+1) * 12 + midiStem
    frequencyHz = 440 * (2**((midiNum-69)/12))
    print(f"the frequency of {note} is {frequencyHz:.2f}")

import sounddevice as sd
import numpy as np
from essentia.standard import MonoLoader, PitchYinFFT

def audio_freq(num_notes=1, sample_rate=44100, block_duration=0.2, silence_threshold=0.01):
    pitch_extractor = PitchYinFFT(frameSize=2048, sampleRate=sample_rate)
    detected_pitches = []

    print(f"Listening... Will detect {num_notes} distinct notes.")

    def callback(indata, frames, time, status):
        nonlocal detected_pitches

        if status:
            print("Sounddevice warning:", status)

        audio_block = indata[:, 0]  # mono
        volume = np.linalg.norm(audio_block)

        if volume < silence_threshold:
            return  # ignore silence

        pitch, _ = pitch_extractor(audio_block.astype(np.float32))

        # Round to 2 decimal places and avoid repeated near-identical pitches
        rounded_pitch = round(pitch, 2)
        if rounded_pitch > 0 and all(abs(rounded_pitch - p) > 1 for p in detected_pitches):
            detected_pitches.append(rounded_pitch)
            print(f"Detected pitch: {rounded_pitch} Hz")

        if len(detected_pitches) >= num_notes:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(channels=1,
                            callback=callback,
                            samplerate=sample_rate,
                            blocksize=int(sample_rate * block_duration)):
            sd.sleep(1000 * 60)  # run for up to 60 seconds or until stopped
    except sd.CallbackStop:
        pass

    return detected_pitches


note_midi_freq("C2")
audio_freq()