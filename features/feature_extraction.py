import numpy as np
import librosa
import torch

def logmel(wav, sr=16000, n_mels=64, hop_length=512, n_fft=2048):
    """
    Compute log-Mel filterbank energies from audio waveform.
    """
    mels = librosa.feature.melspectrogram(
        y=wav, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    return librosa.power_to_db(mels, ref=np.max).astype(np.float32)

def extract_mfcc(wav, sr=16000, n_mfcc=40):
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs).
    """
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)

def extract_prosodic_features(wav, sr=16000):
    """
    Extract prosodic acoustic features: energy, zero-crossing rate, and fundamental frequency.
    """
    energy = librosa.feature.rms(y=wav)
    zcr = librosa.feature.zero_crossing_rate(y=wav)
    pitches, _ = librosa.piptrack(y=wav, sr=sr)
    pitch_values = pitches[pitches > 0]
    mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    
    return {
        "mean_energy": float(np.mean(energy)),
        "mean_zcr": float(np.mean(zcr)),
        "mean_pitch": mean_pitch
    }

def process_audio_file(audio_path, sr=16000, n_mels=64, max_duration=3.0):
    """
    Load audio, normalize length, and return log-Mel spectrogram tensor.
    """
    wav, _ = librosa.load(audio_path, sr=sr)
    target_len = int(sr * max_duration)
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)), mode='constant')
    else:
        wav = wav[:target_len]
        
    mel = logmel(wav, sr=sr, n_mels=n_mels)
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
