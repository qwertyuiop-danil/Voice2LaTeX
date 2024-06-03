import vosk
import sys
import sounddevice as sd
import queue
import json

model = vosk.Model("model")
sample_rate = 16000
device = 0

q = queue.Queue()

def callback(indata, frames, time, status):
    #if status:
    #    print(status, file=sys.stderr)
    q.put(bytes(indata))


def recognize():
    with sd.RawInputStream(samplerate=sample_rate, blocksize = 500, device=device, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, sample_rate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result())['text']
                if text!='':
                    return text
