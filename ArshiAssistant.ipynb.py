import os
import json
import time
import random
import sqlite3
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from textblob import TextBlob
import whisper

SAMPLING_RATE = 16000
N_MELS = 64
N_CLASSES = 6
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMO_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]
LABEL2ID = {label: i for i, label in enumerate(EMO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMO_LABELS)}

def logmel(wav, sr=16000, n_mels=64, hop_length=512, n_fft=2048):
    mels = librosa.feature.melspectrogram(
        y=wav, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    return librosa.power_to_db(mels, ref=np.max).astype(np.float32)

def preprocess_audio_signal(wav, sr=16000, target_duration=3.0):
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak
    trimmed, _ = librosa.effects.trim(wav, top_db=25)
    if len(trimmed) > 0:
        wav = trimmed
    target_samples = int(sr * target_duration)
    if len(wav) < target_samples:
        wav = np.pad(wav, (0, target_samples - len(wav)), mode='constant')
    else:
        wav = wav[:target_samples]
    return wav.astype(np.float32)

class CNNBiLSTM(nn.Module):
    def __init__(self, n_mels=64, n_classes=len(EMO_LABELS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d((2, 2)),
        )
        self.lstm = nn.LSTM(
            input_size=(n_mels // 4) * 32,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        conv_out = self.conv(x)
        b, c, h, w = conv_out.shape
        lstm_in = conv_out.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        lstm_out, _ = self.lstm(lstm_in)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits

class SERDataset(Dataset):
    def __init__(self, audio_paths, labels, target_duration=3.0):
        self.audio_paths = audio_paths
        self.labels = labels
        self.target_duration = target_duration

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        try:
            wav, _ = librosa.load(path, sr=SAMPLING_RATE)
            wav = preprocess_audio_signal(wav, sr=SAMPLING_RATE, target_duration=self.target_duration)
            mel = logmel(wav, sr=SAMPLING_RATE, n_mels=N_MELS)
            return torch.tensor(mel, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        except Exception:
            mel = np.zeros((N_MELS, 94), dtype=np.float32)
            return torch.tensor(mel, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

def train_ser_model(train_loader, test_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    model = CNNBiLSTM(n_mels=N_MELS, n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds = torch.argmax(model(x), dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        acc = accuracy_score(all_targets, all_preds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {running_loss/max(1, len(train_loader)):.4f} - Accuracy: {acc*100:.2f}%")

    return model

class ProfessionalDialogueManager:
    def __init__(self, db_path="assistant_records.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                transcript TEXT,
                emotion TEXT,
                sentiment_score REAL,
                response TEXT
            )
        """)
        self.conn.commit()

        self.responses = {
            "sad": [
                "I acknowledge that this situation may feel overwhelming. Please take your time; I am here to assist and support you.",
                "It sounds like you are facing some difficulties right now. How can I best assist with your current priorities?",
                "I understand things are challenging at the moment. Would you like to review immediate action items together?"
            ],
            "fearful": [
                "I understand this may be causing anxiety. Let us address each item methodically and step-by-step.",
                "Your well-being is important. Let us break down the tasks to make them manageable and clear."
            ],
            "angry": [
                "I understand your frustration regarding this matter and am committed to resolving it promptly and thoroughly.",
                "Thank you for bringing this to attention. Let us examine what occurred and execute the corrective measures."
            ],
            "happy": [
                "It is great to note this positive momentum. How can we proceed to achieve the next objective?",
                "Excellent progress. I am prepared to assist with your next scheduled milestone."
            ],
            "surprised": [
                "That is an unexpected development. Shall we examine the updated data records to verify the details?",
                "An unexpected update may require adjusting plans. Let us review the parameters together."
            ],
            "neutral": [
                "Greetings. I am ready to assist you. How may I be of service today?",
                "System standing by. Please state the analysis, inquiry, or task you would like to proceed with."
            ]
        }

    def generate_response(self, transcript, emotion, sentiment_polarity):
        category = emotion.lower() if emotion.lower() in self.responses else "neutral"
        reply = random.choice(self.responses[category])

        if "time" in transcript.lower():
            reply += f" System time is {time.strftime('%I:%M %p')}."
        elif "status" in transcript.lower():
            reply += " All speech emotion recognition and processing modules are operational."

        self.cursor.execute(
            "INSERT INTO interactions (timestamp, transcript, emotion, sentiment_score, response) VALUES (?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), transcript, emotion, sentiment_polarity, reply)
        )
        self.conn.commit()
        return reply

class MultimodalSpeechAssistant:
    def __init__(self, ser_model=None):
        self.whisper_model = whisper.load_model("small")
        self.ser_model = ser_model if ser_model else CNNBiLSTM().to(DEVICE)
        self.ser_model.eval()
        self.dialogue_manager = ProfessionalDialogueManager()

    def process_audio_interaction(self, audio_path, visual_emotion=None):
        asr_result = self.whisper_model.transcribe(audio_path)
        transcript = asr_result.get("text", "").strip()
        if not transcript:
            transcript = "[Audio input captured]"

        wav, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
        wav = preprocess_audio_signal(wav, sr=SAMPLING_RATE)
        mel = logmel(wav, sr=SAMPLING_RATE, n_mels=N_MELS)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.ser_model(mel_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))

        acoustic_emotion = ID2LABEL[pred_id]
        emotion_confidence = float(probs[pred_id])
        sentiment_polarity = float(TextBlob(transcript).sentiment.polarity)

        final_emotion = visual_emotion if visual_emotion and visual_emotion in EMO_LABELS and acoustic_emotion == "neutral" else acoustic_emotion

        response = self.dialogue_manager.generate_response(
            transcript=transcript,
            emotion=final_emotion,
            sentiment_polarity=sentiment_polarity
        )

        return {
            "transcript": transcript,
            "detected_acoustic_emotion": acoustic_emotion,
            "visual_emotion": visual_emotion if visual_emotion else "Not Provided",
            "fused_emotion": final_emotion,
            "emotion_confidence": round(emotion_confidence, 4),
            "text_sentiment_polarity": round(sentiment_polarity, 4),
            "assistant_response": response
        }

if __name__ == "__main__":
    demo_audio = "demo_test_audio.wav"
    t = np.linspace(0, 3.0, int(SAMPLING_RATE * 3.0), endpoint=False)
    synth_wave = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(demo_audio, synth_wave, SAMPLING_RATE)

    assistant = MultimodalSpeechAssistant()
    results = assistant.process_audio_interaction(demo_audio)

    for key, val in results.items():
        print(f"{key}: {val}")
