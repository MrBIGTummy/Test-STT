import nemo.collections.asr as nemo_asr

def main():
    try:
        # Chemin vers votre fichier audio
        audio_path = "enregistrement_20250618_160027.wav"

        # Charger le modèle NVIDIA Conformer CTC Large (fr)
        print("Chargement du modèle NVIDIA Conformer CTC Large (fr)...")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_fr_conformer_ctc_large")

        # Transcrire l'audio
        print("Transcription en cours...")
        output = asr_model.transcribe([audio_path])
        print("\nTranscription :")
        print(output[0].text)

    except Exception as e:
        print(f"Une erreur est survenue : {str(e)}")

if __name__ == "__main__":
    main() 