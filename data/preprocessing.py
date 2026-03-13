"""
Text preprocessing utilities for the Anxiety Detection system.
Handles cleaning, tokenization, and encoding using BERT tokenizer.
"""
import re
import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, MAX_LENGTH, REVERSE_LABEL_MAP, DATASET_PATH, TRAIN_SPLIT

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs, special chars, extra spaces."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s'.!?,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AnxietyDataset(Dataset):
    """PyTorch Dataset for anxiety text classification."""

    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_and_preprocess(dataset_path: str = DATASET_PATH):
    """Load CSV, clean texts, encode labels, split into train/val DataLoaders."""
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label_id"] = df["label"].map(REVERSE_LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_SPLIT)
    train_texts = df["text"][:split_idx].tolist()
    train_labels = df["label_id"][:split_idx].tolist()
    val_texts = df["text"][split_idx:].tolist()
    val_labels = df["label_id"][split_idx:].tolist()

    train_dataset = AnxietyDataset(train_texts, train_labels)
    val_dataset = AnxietyDataset(val_texts, val_labels)

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_ds, val_ds = load_and_preprocess()
    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    sample = train_ds[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample label: {sample['labels'].item()}")
