from fastapi import FastAPI
from vosk import Model, KaldiRecognizer
import json
import sounddevice as sd
import queue
import os
from groq import Groq

app = FastAPI(title="Speech_to_Text_with_Qwen")

# ------------------------------
# 1. Load Vosk Model
# ------------------------------
model_path = r"C:\Users\bille\Downloads\sp_to_text\vosk-model-en-in-0.5"
model = Model(model_path)

# ------------------------------
# 2. Groq Client (Alibaba Qwen via Groq)
# ------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# 3. Speech → Text Endpoint
# ------------------------------
@app.post("/speech_to_text")
def speech_to_text():
    q = queue.Queue()
    samplerate = 16000
    device = None  # default mic

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    print("Listening... Start speaking!")

    rec = KaldiRecognizer(model, samplerate)
    full_text = ""
    slice_count = 0

    with sd.RawInputStream(
        samplerate=samplerate, blocksize=8000,
        dtype='int16', channels=1,
        callback=callback, device=device
    ):
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
                if slice_count > 8:
                    break

        final_result = json.loads(rec.FinalResult())
        full_text += final_result.get("text", "")

    print("Recording complete")
    return {"transcription": full_text.strip()}


# ------------------------------
# 4. Speech → Text → Qwen Model (Groq)
# ------------------------------
@app.post("/speech_to_qwen")
def speech_to_qwen():
    # 1. Get speech transcription
    q = queue.Queue()
    samplerate = 16000
    device = None

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    print("Speak now...")

    rec = KaldiRecognizer(model, samplerate)
    text = ""
    silent = 0

    with sd.RawInputStream(
        samplerate=samplerate, blocksize=8000,
        dtype="int16", channels=1,
        callback=callback, device=device
    ):
        while True:
            data = q.get()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text += result.get("text", "") + " "
                silent = 0
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if not partial.strip():
                    silent += 1
                if silent > 8:
                    break

        final = json.loads(rec.FinalResult())
        text += final.get("text", "")

    transcription = text.strip()

    # 2. Send to Qwen model through Groq API
    response = client.chat.completions.create(
        model="qwen-2.5-7b-instruct",   # Alibaba model on Groq
        messages=[
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": transcription}
        ]
    )

    qwen_output = response.choices[0].message["content"]

    return {
        "transcription": transcription,
        "qwen_response": qwen_output
    }
