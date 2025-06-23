import torch
import torchaudio
from transformers import pipeline
import requests

AUDIO_PATH = "atester.wav"
device = 0 if torch.cuda.is_available() else -1

try:
    asr = pipeline(
        "automatic-speech-recognition",
        model="bofenghuang/whisper-large-v3-french",
        device=device,
    )
except Exception as e:
    print("Erreur lors du chargement du modèle :", e)
    exit(1)

# Charger l'audio (mono, 16kHz recommandé)
speech_array, sampling_rate = torchaudio.load(AUDIO_PATH)
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_array = resampler(speech_array)
input_audio = speech_array.squeeze().numpy()

# Transcription
result = asr(input_audio, chunk_length_s=30, stride_length_s=5)
transcription = result["text"]
print("Transcription :", transcription)

# Clé API Groq directement dans le code
GROQ_API_KEY = "BOC"
if not GROQ_API_KEY:
    print("Erreur : la clé GROQ_API_KEY est vide !")
    exit(1)

def improve_transcription_llama(transcription, context="Tu travailles pour une pizzeria. Reformule et améliore le texte pour la prise de commande :"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": transcription}
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Prompt contextuel exemple
prompt = "Tu es un assistant qui travaille pour une pizzeria. Reformule la transcription pour qu'elle soit claire, polie, et adaptée à la prise de commande."
try:
    texte_ameliore = improve_transcription_llama(transcription, context=prompt)
    print("\nTexte amélioré par Llama 3 (Groq) :\n", texte_ameliore)
except Exception as e:
    print("Erreur lors de l'appel à l'API Groq :", e)