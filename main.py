from fastapi import FastAPI
from vosk import Model, KaldiRecognizer
import json
import sounddevice as sd
import queue

app = FastAPI(title="speech_to_text")

model_path = r"C:\Users\bille\Downloads\sp_to_text_vosk\vosk-model-en-in-0.5"
model = Model(model_path)

@app.post("/speech_to_text")
def speech_to_text():
    q = queue.Queue()
    samplerate = 16000
    device = None  # default microphone

    def callback(indata, frames, time, status): 
        if status:
            print(status)
        q.put(bytes(indata))  # convert samples to bytes and enqueue them

    print("Start speaking...")

    rec = KaldiRecognizer(model, samplerate)
    full_text = ""
    slice_count = 0

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback, device=device):
        # callback=callback sends audio chunks into queue

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                full_text += result.get("text", "") + " "
                slice_count = 0
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if not partial.strip():
                    slice_count += 1
                else:
                    slice_count = 0
                # Stop recording if silence is detected for too long
                if slice_count > 8:
                    break

        final_result = json.loads(rec.FinalResult())
        full_text += final_result.get("text", "")

    print("Recording Done")
    return {"transcription": full_text.strip()}
