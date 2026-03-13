"""
Central configuration for the AI-Based Exam Anxiety Detection System.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_PATH = os.path.join(DATA_DIR, "anxiety_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "bert_anxiety_model.pt")

# Model settings
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
NUM_LABELS = 3
LABEL_MAP = {0: "Low Anxiety", 1: "Moderate Anxiety", 2: "High Anxiety"}
REVERSE_LABEL_MAP = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
TRAIN_SPLIT = 0.8

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
