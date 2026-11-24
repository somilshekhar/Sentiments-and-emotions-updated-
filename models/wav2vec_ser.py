import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import config

class Wav2Vec2ForSER(nn.Module):
    def __init__(self, pretrained_model_name=config.WAV2VEC2_MODEL_NAME, num_labels=len(config.EMO_LABELS)):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        hidden_size = self.wav2vec2.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled)
        return logits
