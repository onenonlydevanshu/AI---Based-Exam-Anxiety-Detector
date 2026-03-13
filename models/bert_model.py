"""
BERT-based Anxiety Classification Model.
Fine-tunes bert-base-uncased for 3-class anxiety level prediction.
"""
import os
import sys

import torch
import torch.nn as nn
from transformers import BertModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, NUM_LABELS


class BertAnxietyClassifier(nn.Module):
    """BERT encoder + dropout + linear head for anxiety classification."""

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] token representation
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
