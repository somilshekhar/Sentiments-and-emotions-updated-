import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_storage")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Audio parameters
SAMPLING_RATE = 16000
N_MELS = 64
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# Emotion classes
EMO_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised"
]
LABEL2ID = {label: idx for idx, label in enumerate(EMO_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(EMO_LABELS)}

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
TRAIN_TEST_SPLIT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model checkpoints
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
WHISPER_MODEL_NAME = "small"
DIALOGUE_MODEL_NAME = "xlm-roberta-base"
