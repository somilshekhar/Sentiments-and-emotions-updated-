import os
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from features.feature_extraction import logmel
from data.preprocessing import preprocess_audio
import config

class SERDataset(Dataset):
    """
    PyTorch Dataset for speech emotion recognition from log-Mel spectrograms.
    """
    def __init__(self, file_paths, labels, target_duration=3.0, sr=config.SAMPLING_RATE, n_mels=config.N_MELS):
        self.file_paths = file_paths
        self.labels = labels
        self.target_duration = target_duration
        self.sr = sr
        self.n_mels = n_mels
        self.target_len = int(self.sr * self.target_duration)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            wav, _ = librosa.load(file_path, sr=self.sr)
            wav = preprocess_audio(wav, sr=self.sr)
            
            if len(wav) < self.target_len:
                wav = np.pad(wav, (0, self.target_len - len(wav)), mode='constant')
            else:
                wav = wav[:self.target_len]
                
            mel = logmel(wav, sr=self.sr, n_mels=self.n_mels)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
            return mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception:
            mel_tensor = torch.zeros((1, self.n_mels, 94), dtype=torch.float32)
            return mel_tensor, torch.tensor(label, dtype=torch.long)

def create_dataloaders(file_paths, labels, batch_size=config.BATCH_SIZE, train_split=config.TRAIN_TEST_SPLIT):
    n_total = len(file_paths)
    indices = np.random.permutation(n_total)
    split_point = int(n_total * train_split)
    
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    
    train_paths = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    test_paths = [file_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    train_dataset = SERDataset(train_paths, train_labels)
    test_dataset = SERDataset(test_paths, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
