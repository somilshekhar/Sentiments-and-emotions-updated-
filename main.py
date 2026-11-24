import argparse
import os
import sys
import numpy as np
import soundfile as sf
import torch

import config
from data.download_datasets import download_ravdess_dataset, parse_ravdess_files, parse_iemocap_files, parse_custom_dataset
from data.dataset_loader import create_dataloaders
from models.cnn_bilstm import CNNBiLSTM
from training.train import train_model
from training.evaluate import evaluate_model
from assistant.multimodal_pipeline import MultimodalSpeechEmotionAssistant

def generate_synthetic_audio(output_path, duration=3.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 440 * t)
    audio = (signal * np.exp(-t / 1.5)).astype(np.float32)
    sf.write(output_path, audio, sr)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition & Multimodal Sentiment Assistant")
    parser.add_argument("--download-data", action="store_true", help="Download RAVDESS dataset")
    parser.add_argument("--train", action="store_true", help="Train CNN-BiLSTM SER model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate SER model checkpoint")
    parser.add_argument("--demo", action="store_true", help="Run multimodal speech assistant demo")
    parser.add_argument("--audio-file", type=str, default=None, help="Input WAV file for inference")
    parser.add_argument("--visual-emotion", type=str, default=None, choices=config.EMO_LABELS, help="Optional visual facial emotion label")
    parser.add_argument("--custom-data-dir", type=str, default=None, help="Path to custom emotion audio directory (e.g. multilingual or call center)")
    parser.add_argument("--iemocap-dir", type=str, default=None, help="Path to local IEMOCAP dataset directory if available")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--model-path", type=str, default=os.path.join(config.MODELS_DIR, "best_cnn_bilstm.pt"))

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.download_data:
        download_ravdess_dataset()

    if args.train:
        file_paths, labels = [], []
        
        # Priority 1: Custom dataset (multilingual / call center) if provided
        if args.custom_data_dir:
            file_paths, labels = parse_custom_dataset(args.custom_data_dir)
            print(f"Loaded {len(file_paths)} samples from custom dataset: {args.custom_data_dir}")
        # Priority 2: IEMOCAP dataset if directory provided
        elif args.iemocap_dir:
            file_paths, labels = parse_iemocap_files(args.iemocap_dir)
            print(f"Loaded {len(file_paths)} samples from IEMOCAP dataset: {args.iemocap_dir}")
        # Priority 3: RAVDESS benchmark
        else:
            ravdess_dir = os.path.join(config.DATA_DIR, "ravdess")
            if not os.path.exists(ravdess_dir) or len(os.listdir(ravdess_dir)) == 0:
                download_ravdess_dataset()
            file_paths, labels = parse_ravdess_files(ravdess_dir)

        if len(file_paths) == 0:
            mock_dir = os.path.join(config.DATA_DIR, "mock_samples")
            os.makedirs(mock_dir, exist_ok=True)
            for i in range(12):
                mock_file = os.path.join(mock_dir, f"sample_{i}.wav")
                generate_synthetic_audio(mock_file, duration=2.0)
                file_paths.append(mock_file)
                labels.append(i % len(config.EMO_LABELS))

        train_loader, test_loader = create_dataloaders(file_paths, labels)
        model = CNNBiLSTM(n_mels=config.N_MELS, n_classes=len(config.EMO_LABELS))
        
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.model_path
        )

    if args.evaluate:
        ravdess_dir = os.path.join(config.DATA_DIR, "ravdess")
        file_paths, labels = parse_ravdess_files(ravdess_dir)
        if len(file_paths) == 0:
            print("No audio samples found for evaluation.")
            return

        _, test_loader = create_dataloaders(file_paths, labels)
        model = CNNBiLSTM(n_mels=config.N_MELS, n_classes=len(config.EMO_LABELS))
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
        
        metrics = evaluate_model(model, test_loader, device=config.DEVICE)
        print("\n" + "="*50)
        print("          EVALUATION RESULTS")
        print("="*50)
        print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
        print(f"  Precision : {metrics['precision']*100:.2f}%")
        print(f"  Recall    : {metrics['recall']*100:.2f}%")
        print(f"  F1-Score  : {metrics['f1_score']*100:.2f}%")
        print("="*50)
        print("Confusion Matrix:")
        print(np.array(metrics['confusion_matrix']))

    if args.demo:
        target_audio = args.audio_file
        if not target_audio or not os.path.exists(target_audio):
            demo_dir = os.path.join(config.BASE_DIR, "demo_samples")
            os.makedirs(demo_dir, exist_ok=True)
            target_audio = os.path.join(demo_dir, "test_input.wav")
            generate_synthetic_audio(target_audio, duration=3.0)

        assistant = MultimodalSpeechEmotionAssistant(
            ser_model_path=args.model_path if os.path.exists(args.model_path) else None
        )
        
        result = assistant.process_audio_interaction(
            audio_path=target_audio,
            visual_emotion=args.visual_emotion,
            speak_output=False,
            output_wav=os.path.join(config.RESULTS_DIR, "assistant_reply.wav")
        )
        
        print("\n" + "="*60)
        print("       MULTIMODAL SPEECH EMOTION & SENTIMENT ASSESSMENT")
        print("="*60)
        for k, v in result.items():
            print(f"  {k:28s}: {v}")
        print("="*60)

if __name__ == "__main__":
    main()
