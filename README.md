# Analyzing Sentiments and Recognizing Emotions in Speech through Universal Speech Recognition Models

This repository contains the official PyTorch implementation and experimental framework for:

> **Analyzing Sentiments and Recognizing Emotions in Speech through Universal Speech Recognition Models**  
> *Somil Shekhar, Hina Jignesh Chokshi*  
> Department of Computer Science, Parul Institute of Computer Applications, Parul University, Vadodara, Gujarat, India.

---

## Abstract

Accurately identifying and interpreting human emotions from speech is a foundational capability in human-computer interaction and affective computing. Traditional speech emotion recognition (SER) pipelines relying on handcrafted acoustic features often suffer from acoustic variability, speaker divergence, and ambient noise. This research evaluates deep universal speech representations—including self-supervised architectures (HuBERT and wav2vec 2.0) and ASR transformers (Whisper)—against hybrid CNN-BiLSTM deep neural network baselines. 

The framework incorporates audio preprocessing (spectral subtraction, silence removal, amplitude normalization), log-Mel spectrogram extraction, and multimodal integration combining acoustic emotion predictions with lexical sentiment polarity to drive an emotion-adaptive dialogue interface.

---

## Experimental Results

Evaluated on English speech emotion benchmark datasets (RAVDESS, IEMOCAP, CREMA-D) using an 80/20 train-test split:

| Model Architecture | Parameter Type | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CNN-BiLSTM** | Convolution + Bidirectional LSTM | 85.4% | 83.5% | 84.2% | 83.8% |
| **wav2vec 2.0** | Pre-trained Transformer (`wav2vec2-base`) | 92.1% | 91.2% | 91.8% | 91.5% |
| **HuBERT** | Self-Supervised Speech Model (`hubert-base-ls960`) | **94.3%** | **93.8%** | **94.1%** | **94.0%** |

---

## System Architecture

```
                       +----------------------+
                       | Raw Speech Input     |
                       +----------------------+
                                  |
                                  v
                       +----------------------+
                       | Preprocessing Module | (Spectral noise subtraction, silence trimming)
                       +----------------------+
                                  |
                                  v
                       +----------------------+
                       | Feature Extraction   | (Log-Mel Spectrograms, MFCCs, Prosody)
                       +----------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+-----------------+                               +-----------------+
|   CNN-BiLSTM    |                               |  wav2vec / HuBERT|
| (Deep Baseline) |                               | (Self-Superv.)  |
+-----------------+                               +-----------------+
         |                                                 |
         +------------------------+------------------------+
                                  |
                                  v
                       +----------------------+
                       | Emotion Categorizer  | (Neutral, Happy, Sad, Angry, Fearful, Surprised)
                       +----------------------+
                                  |
                   +--------------+--------------+
                   |                             |
                   v                             v
        +--------------------+        +--------------------+
        | Whisper ASR Engine |        | Text Sentiment     |
        +--------------------+        +--------------------+
                   \                             /
                    \                           /
                     v                         v
              +---------------------------------------+
              | Multimodal Emotion-Adaptive Dialogue  |
              +---------------------------------------+
                                  |
                                  v
              +---------------------------------------+
              | Prosodic Speech Synthesis (TTS)       |
              +---------------------------------------+
```

---

## Repository Structure

```
.
├── Sentiments and emotions updated (1).pdf   # Associated research paper
├── config.py                                 # Global hyperparameters and configuration
├── requirements.txt                          # Dependencies
├── main.py                                   # Entry point for training, evaluation, and demo
├── ArshiAssistant.ipynb.py                   # Self-contained research script
│
├── features/                                 # Feature extraction
│   ├── __init__.py
│   └── feature_extraction.py                 # Log-Mel spectrogram, MFCC, prosodic features
│
├── data/                                     # Data pipeline
│   ├── __init__.py
│   ├── preprocessing.py                      # Signal cleaning, spectral subtraction, normalization
│   ├── dataset_loader.py                     # PyTorch Dataset and DataLoader routines
│   └── download_datasets.py                  # RAVDESS, IEMOCAP, and custom dataset parsers
│
├── models/                                   # Model definitions
│   ├── __init__.py
│   ├── cnn_bilstm.py                         # Hybrid CNN-BiLSTM architecture (Appendix A)
│   ├── wav2vec_ser.py                        # Fine-tuned wav2vec 2.0 classifier
│   └── hubert_ser.py                         # Fine-tuned HuBERT classifier
│
├── training/                                 # Optimization and evaluation
│   ├── __init__.py
│   ├── train.py                              # Adam optimizer, lr scheduler, training loop
│   └── evaluate.py                           # Classification report and confusion matrix
│
└── assistant/                                # Multimodal dialogue pipeline
    ├── __init__.py
    ├── dialogue_manager.py                   # Context-aware professional dialogue engine
    ├── multimodal_pipeline.py                # Whisper ASR + SER + Sentiment integration
    └── tts_engine.py                         # Emotion-adaptive TTS prosody engine
```

---

## Installation

### Prerequisites
* Python 3.9+
* PyTorch 2.0+ with CUDA support (optional, recommended for GPU acceleration)

```bash
git clone https://github.com/somilshekhar/Sentiments-and-emotions-updated-.git
cd Sentiments-and-emotions-updated-
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation
To download and extract the RAVDESS benchmark dataset from Zenodo:
```bash
python main.py --download-data
```

For IEMOCAP or custom speech datasets:
```bash
# Custom directory organized by emotion subfolders (e.g. data/happy/*.wav):
python main.py --train --custom-data-dir /path/to/custom_data

# IEMOCAP (if local copy available under license):
python main.py --train --iemocap-dir /path/to/iemocap
```

### 2. Model Training
Train the CNN-BiLSTM baseline on the 80/20 train-test split:
```bash
python main.py --train --epochs 30 --lr 0.0001
```

### 3. Evaluation
Evaluate model accuracy, weighted precision, recall, F1-score, and confusion matrix:
```bash
python main.py --evaluate
```

### 4. Multimodal Assistant Demo
Run an end-to-end multimodal interaction cycle (Whisper transcription, acoustic affect classification, sentiment scoring, and adaptive response generation):
```bash
python main.py --demo --audio-file path/to/sample.wav
```

---

## Citation

If you use this codebase or research in your work, please cite:

```bibtex
@article{shekhar2024speech,
  title={Analyzing Sentiments and Recognizing Emotions in Speech through Universal Speech Recognition Models},
  author={Shekhar, Somil and Chokshi, Hina Jignesh},
  journal={Department of Computer Science, Parul Institute of Computer Applications, Parul University},
  year={2024}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
