"""
Training script for the BERT Anxiety Classifier.
Handles training loop, validation, and model saving.
"""
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODEL_PATH, LABEL_MAP, MODEL_DIR,
)
from data.preprocessing import load_and_preprocess
from models.bert_model import BertAnxietyClassifier


def train():
    """Fine-tune BERT for anxiety classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading and preprocessing dataset...")
    train_dataset, val_dataset = load_and_preprocess()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize model
    model = BertAnxietyClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # ── Training phase ────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        elapsed = time.time() - start

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (Val Acc: {val_acc:.4f})\n")

    # Final classification report
    target_names = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    print("\n=== Final Validation Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=target_names)) # pyright: ignore[reportPossiblyUnboundVariable]
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
