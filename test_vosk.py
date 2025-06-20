import os
import queue
import json
import sounddevice as sd
import vosk

MODEL_PATH = r"vosk-model-small-fr-0.22"

print("Vérification du dossier modèle...")
print("Contenu :", os.listdir(MODEL_PATH))

print("Tentative de chargement du modèle...")
model = vosk.Model(MODEL_PATH)
print("Modèle chargé avec succès !")

q = queue.Queue()
def callback(indata, frames, time, status):
    q.put(bytes(indata))

rec = vosk.KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("Parle, je détecte les mots-clés (Ctrl+C pour arrêter)")
    try:
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                print("Reconnu :", text)
                # Keyword spotting simple
                for kw in ["pizza", "livraison", "quatre fromages", "adresse"]:
                    if kw in text.lower():
                        print(f"Mot-clé détecté : {kw}")
    except KeyboardInterrupt:
        print("\nArrêté par l'utilisateur.") 