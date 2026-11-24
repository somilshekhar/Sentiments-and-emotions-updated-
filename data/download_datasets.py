import os
import urllib.request
import zipfile
import config

RAVDESS_URL = "https://zenodo.org/record/1188976/files/audio_speech_actors_01-24.zip"

RAVDESS_EMO_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "angry",
    "08": "surprised"
}

IEMOCAP_EMO_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
    "fea": "fearful",
    "sur": "surprised"
}

def download_ravdess_dataset(destination_dir=config.DATA_DIR):
    ravdess_dir = os.path.join(destination_dir, "ravdess")
    os.makedirs(ravdess_dir, exist_ok=True)
    zip_path = os.path.join(destination_dir, "ravdess.zip")
    
    if not os.path.exists(zip_path) and len(os.listdir(ravdess_dir)) == 0:
        print(f"Downloading RAVDESS dataset...")
        urllib.request.urlretrieve(RAVDESS_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ravdess_dir)
        print("RAVDESS extraction complete.")
    return ravdess_dir

def parse_ravdess_files(ravdess_dir):
    file_paths = []
    labels = []
    for root, _, files in os.walk(ravdess_dir):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    mapped_emo = RAVDESS_EMO_MAP.get(emo_code)
                    if mapped_emo in config.LABEL2ID:
                        file_paths.append(os.path.join(root, file))
                        labels.append(config.LABEL2ID[mapped_emo])
    return file_paths, labels

def parse_iemocap_files(iemocap_dir):
    """
    Parses IEMOCAP dataset files if provided locally.
    Assumes standard IEMOCAP session structure.
    """
    file_paths = []
    labels = []
    if not os.path.exists(iemocap_dir):
        return file_paths, labels
        
    for root, _, files in os.walk(iemocap_dir):
        for file in files:
            if file.endswith(".wav"):
                lower_name = file.lower()
                for key, mapped_emo in IEMOCAP_EMO_MAP.items():
                    if key in lower_name and mapped_emo in config.LABEL2ID:
                        file_paths.append(os.path.join(root, file))
                        labels.append(config.LABEL2ID[mapped_emo])
                        break
    return file_paths, labels

def parse_custom_dataset(directory_path):
    """
    Loads any custom audio dataset organized in subfolders named after emotions:
    e.g. directory_path/neutral/*.wav, directory_path/sad/*.wav (supports multilingual/call center audio)
    """
    file_paths = []
    labels = []
    if not os.path.exists(directory_path):
        return file_paths, labels

    for emo in config.EMO_LABELS:
        emo_dir = os.path.join(directory_path, emo)
        if os.path.isdir(emo_dir):
            for file in os.listdir(emo_dir):
                if file.endswith((".wav", ".mp3", ".flac")):
                    file_paths.append(os.path.join(emo_dir, file))
                    labels.append(config.LABEL2ID[emo])
    return file_paths, labels
