import numpy as np
import librosa

def spectral_subtraction_noise_reduction(wav, noise_reduce_factor=1.5):
    """
    Apply spectral subtraction to suppress background noise.
    """
    n_frames = min(len(wav), 1600)
    noise_profile = wav[:n_frames]
    
    stft = librosa.stft(wav)
    magnitude, phase = np.abs(stft), np.angle(stft)
    
    noise_stft = librosa.stft(noise_profile, n_fft=stft.shape[0] * 2 - 2)
    noise_mean = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
    
    subtracted_mag = np.maximum(magnitude - noise_reduce_factor * noise_mean[:magnitude.shape[0]], 0.05 * magnitude)
    clean_stft = subtracted_mag * np.exp(1j * phase)
    clean_wav = librosa.istft(clean_stft, length=len(wav))
    return clean_wav.astype(np.float32)

def remove_silence(wav, top_db=25):
    """
    Trim silent intervals from audio.
    """
    trimmed_wav, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed_wav if len(trimmed_wav) > 0 else wav

def normalize_amplitude(wav):
    """
    Normalize waveform amplitude to [-1.0, 1.0].
    """
    peak = np.max(np.abs(wav))
    if peak > 0:
        return (wav / peak).astype(np.float32)
    return wav.astype(np.float32)

def preprocess_audio(wav, sr=16000):
    """
    Standard preprocessing chain: normalize, noise reduce, trim silence.
    """
    wav = normalize_amplitude(wav)
    wav = spectral_subtraction_noise_reduction(wav)
    wav = remove_silence(wav)
    wav = normalize_amplitude(wav)
    return wav
