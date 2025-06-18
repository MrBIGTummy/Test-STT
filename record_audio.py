import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import librosa

def record_audio(duration=5, sample_rate=16000):
    """
    Enregistre l'audio depuis le microphone
    :param duration: Durée de l'enregistrement en secondes
    :param sample_rate: Fréquence d'échantillonnage (16kHz par défaut pour wav2vec2)
    :return: Le nom du fichier enregistré
    """
    print(f"Enregistrement de {duration} secondes...")
    
    # Enregistrer l'audio
    recording = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1,
                      dtype='float32')
    
    # Attendre que l'enregistrement soit terminé
    sd.wait()
    
    # Créer le nom du fichier avec la date et l'heure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enregistrement_{timestamp}.wav"
    
    # Sauvegarder le fichier
    sf.write(filename, recording, sample_rate)
    print(f"Enregistrement sauvegardé dans : {filename}")
    
    return filename

def load_model():
    # Charger le modèle et le processeur
    model_name = "LeBenchmark/wav2vec2-FR-14K-xlarge"
    
    # Charger le tokenizer et le processeur
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("LeBenchmark/wav2vec2-FR-14K-xlarge")
    processor = Wav2Vec2Processor.from_pretrained("LeBenchmark/wav2vec2-FR-14K-xlarge")
    model = Wav2Vec2ForCTC.from_pretrained("LeBenchmark/wav2vec2-FR-14K-xlarge")
    
    return processor, model

def transcribe_audio(audio_path, processor, model):
    # Charger l'audio
    speech, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # Prétraiter l'audio
    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    
    # Faire la prédiction
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # Décoder la prédiction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def main():
    try:
        # Demander la durée de l'enregistrement
        duration = float(input("Entrez la durée de l'enregistrement en secondes (par défaut 5) : ") or "5")
        
        # Enregistrer l'audio
        filename = record_audio(duration=duration)
        
        print("\nVoulez-vous transcrire cet enregistrement ? (o/n)")
        if input().lower() == 'o':
            print("Chargement du modèle...")
            processor, model = load_model()
            print("Transcription en cours...")
            transcription = transcribe_audio(filename, processor, model)
            print("\nTranscription:", transcription)
            
    except KeyboardInterrupt:
        print("\nEnregistrement annulé.")
    except Exception as e:
        print(f"Une erreur est survenue : {str(e)}")

if __name__ == "__main__":
    main() 