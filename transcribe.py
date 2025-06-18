import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf

def load_model():
    """
    Charge le modèle français via Hugging Face
    """
    # Configuration du modèle - modèle Facebook spécialement entraîné pour le français
    model_name = "facebook/wav2vec2-large-xlsr-53-french"
    
    # Charger le processeur et le modèle
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    return processor, model

def transcribe_audio(audio_path, processor, model):
    """
    Transcrit un fichier audio en utilisant le modèle
    """
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
        # Chemin vers votre fichier audio
        audio_path = "enregistrement_20250618_160027.wav"
        
        # Charger le modèle
        print("Chargement du modèle français Facebook...")
        processor, model = load_model()
        
        # Transcrire l'audio
        print("Transcription en cours...")
        transcription = transcribe_audio(audio_path, processor, model)
        
        print("\nTranscription :")
        print(transcription)
        
    except Exception as e:
        print(f"Une erreur est survenue : {str(e)}")

if __name__ == "__main__":
    main() 