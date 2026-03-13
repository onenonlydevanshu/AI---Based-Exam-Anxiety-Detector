"""
Inference utilities for the Anxiety Detection model.
Loads a trained BERT model and provides prediction functions.
"""
import os
import sys

import torch
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME,
    MAX_LENGTH,
    NUM_LABELS,
    LABEL_MAP,
    MODEL_PATH,
    MODEL_DOWNLOAD_URL,
    MODEL_AUTO_DOWNLOAD,
    MODEL_DOWNLOAD_TIMEOUT,
)
from data.preprocessing import clean_text
from download_model import download_model
from models.bert_model import BertAnxietyClassifier


class AnxietyPredictor:
    """Loads the trained BERT model and predicts anxiety levels from text."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        self.model = BertAnxietyClassifier(MODEL_NAME, NUM_LABELS)

        if not os.path.exists(model_path) and MODEL_AUTO_DOWNLOAD and MODEL_DOWNLOAD_URL:
            print("Model file missing. Attempting download from MODEL_DOWNLOAD_URL...")
            try:
                download_model(
                    url=MODEL_DOWNLOAD_URL,
                    output_path=model_path,
                    timeout=MODEL_DOWNLOAD_TIMEOUT,
                )
            except Exception as exc:
                print(f"WARNING: Auto-download failed: {exc}")

        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            print(f"Model loaded from {model_path}")
        else:
            print(f"WARNING: No trained model found at {model_path}. Using untrained model.")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Predict the anxiety level for a given text.
        Returns dict with label, confidence, and all class probabilities.
        """
        cleaned = clean_text(text)

        encoding = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() # pyright: ignore[reportArgumentType] # type: ignore

        return {
            "label": LABEL_MAP[pred_idx], # type: ignore
            "confidence": round(confidence, 4),
            "probabilities": {
                LABEL_MAP[i]: round(probs[i].item(), 4) for i in range(NUM_LABELS)
            },
        }
