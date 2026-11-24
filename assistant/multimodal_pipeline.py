import os
import torch
import numpy as np
import librosa
from textblob import TextBlob
import whisper

import config
from features.feature_extraction import logmel
from data.preprocessing import preprocess_audio
from models.cnn_bilstm import CNNBiLSTM
from assistant.dialogue_manager import ProfessionalDialogueManager
from assistant.tts_engine import EmotionAdaptiveTTSEngine

class MultimodalSpeechEmotionAssistant:
    def __init__(self, ser_model_path=None, device=config.DEVICE):
        self.device = device
        self.whisper_model = whisper.load_model(config.WHISPER_MODEL_NAME, device=self.device)
        self.ser_model = CNNBiLSTM().to(self.device)
        if ser_model_path and os.path.exists(ser_model_path):
            self.ser_model.load_state_dict(torch.load(ser_model_path, map_location=self.device))
        self.ser_model.eval()
        
        self.dialogue_manager = ProfessionalDialogueManager()
        self.tts = EmotionAdaptiveTTSEngine()

    def transcribe_audio(self, audio_path):
        result = self.whisper_model.transcribe(audio_path)
        return result.get("text", "").strip()

    def predict_speech_emotion(self, audio_path):
        wav, sr = librosa.load(audio_path, sr=config.SAMPLING_RATE)
        wav = preprocess_audio(wav, sr=config.SAMPLING_RATE)
        
        target_len = int(config.SAMPLING_RATE * 3.0)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)), mode='constant')
        else:
            wav = wav[:target_len]
            
        mel = logmel(wav, sr=config.SAMPLING_RATE, n_mels=config.N_MELS)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.ser_model(mel_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            
        detected_emotion = config.ID2LABEL[pred_id]
        confidence = float(probs[pred_id])
        return detected_emotion, confidence, probs

    def analyze_text_sentiment(self, text):
        blob = TextBlob(text)
        return float(blob.sentiment.polarity)

    def process_audio_interaction(self, audio_path, visual_emotion=None, speak_output=True, output_wav=None):
        """
        Multimodal interaction cycle combining speech transcription, acoustic emotion,
        text sentiment, and optional visual facial emotion.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        transcript = self.transcribe_audio(audio_path)
        if not transcript:
            transcript = "[Audio input captured]"

        acoustic_emotion, emo_conf, emo_probs = self.predict_speech_emotion(audio_path)
        sentiment_score = self.analyze_text_sentiment(transcript)

        # Fused emotion consideration: if visual facial cue is provided, incorporate it
        final_emotion = acoustic_emotion
        if visual_emotion and visual_emotion in config.EMO_LABELS:
            # When acoustic and visual align, or visual overrides neutral speech
            final_emotion = visual_emotion if acoustic_emotion == "neutral" else acoustic_emotion

        assessment = {
            "transcript": transcript,
            "detected_acoustic_emotion": acoustic_emotion,
            "visual_facial_emotion": visual_emotion if visual_emotion else "Not Provided",
            "fused_emotion": final_emotion,
            "emotion_confidence": round(emo_conf, 4),
            "text_sentiment_polarity": round(sentiment_score, 4),
        }

        response = self.dialogue_manager.generate_response(
            user_transcript=transcript,
            detected_emotion=final_emotion,
            sentiment_polarity=sentiment_score
        )
        assessment["assistant_response"] = response

        if output_wav:
            self.tts.save_to_file(response, output_wav, emotion=final_emotion)
        elif speak_output:
            self.tts.speak(response, emotion=final_emotion)

        return assessment
