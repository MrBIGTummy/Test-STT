# French Speech Recognition

Ce projet contient plusieurs scripts pour la reconnaissance vocale en français, utilisant différents modèles et approches.

## Scripts disponibles

### 1. transcribe.py
Utilise le modèle Whisper pour la transcription en français.

### 2. transcribe_speechbrain.py
Utilise SpeechBrain avec le modèle wav2vec2 pour la transcription en français.

### 3. transcribe_nvidia.py
Utilise le modèle NVIDIA Conformer CTC Large pour la transcription en français (nécessite NeMo).

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-username/french-speech-recognition.git
cd french-speech-recognition
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Avec Whisper (transcribe.py)
```bash
python transcribe.py
```

### Avec SpeechBrain (transcribe_speechbrain.py)
```bash
python transcribe_speechbrain.py
```

### Avec NVIDIA NeMo (transcribe_nvidia.py)
```bash
python transcribe_nvidia.py
```

## Notes importantes

- Le script `transcribe_nvidia.py` nécessite l'installation de NeMo, qui est plus facile à installer sous Linux/WSL que sous Windows.
- Les fichiers audio doivent être au format WAV.
- Les modèles sont téléchargés automatiquement lors de la première utilisation.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails. 