# AI-Based Exam Anxiety Detection System
## Disclamer:This is the compressed code due to uploading issues (Sorry for that).
An intelligent mental-wellness support system that identifies and categorizes exam-related anxiety from student text inputs using **BERT-based NLP**.

## Architecture

```
Presentation Layer  →  Streamlit (app.py)
Application Layer   →  Preprocessing → BERT Model → FastAPI Backend
Data Layer          →  anxiety_dataset.csv · bert_anxiety_model.pt · Preprocessing Scripts
```

## Project Structure

```
├── backend/                  # FastAPI backend service
│   └── main.py
├── frontend/                 # Streamlit UI application
│   └── app.py
├── data/
│   ├── generate_dataset.py   # Synthetic dataset generator
│   ├── preprocessing.py      # Text cleaning & BERT tokenization
│   └── anxiety_dataset.csv   # Generated dataset (after running generator)
├── notebooks/                # EDA and experimentation notebooks
├── model/                    # Local trained model artifacts (git-ignored)
│   └── bert_anxiety_model.pt
├── models/                   # Training + inference model code
│   ├── bert_model.py
│   ├── train.py
│   └── predict.py
├── config.py                 # Central configuration
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup & Installation

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the dataset
```bash
python data/generate_dataset.py
```

### 4. Train the BERT model
```bash
python models/train.py
```

### Optional: Download model on demand (lightweight repo mode)
If the weights file is not in `model/bert_anxiety_model.pt`, you can download it when needed.

Set a direct URL to the `.pt` file:

```bash
# Windows PowerShell
$env:MODEL_DOWNLOAD_URL="https://your-host/path/bert_anxiety_model.pt"
python download_model.py
```

The backend will also auto-attempt download at startup when all are true:
- `MODEL_AUTO_DOWNLOAD` is not `0/false/no`
- `MODEL_DOWNLOAD_URL` is set
- local file `model/bert_anxiety_model.pt` is missing

### 5. Start the FastAPI backend
```bash
python backend/main.py
```
The API will be available at `http://127.0.0.1:8000`.

### 6. Launch the Streamlit frontend (in a new terminal)
```bash
streamlit run frontend/app.py
```

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | API status and available endpoints |
| GET    | `/health`  | Health check                       |
| POST   | `/predict` | Predict anxiety level from text    |

### Example `/predict` request
```json
POST /predict
{
  "text": "I can't stop worrying about the exam. I feel overwhelmed."
}
```

### Example response
```json
{
  "anxiety_level": "High Anxiety",
  "confidence": 0.9234,
  "probabilities": {
    "Low Anxiety": 0.0312,
    "Moderate Anxiety": 0.0454,
    "High Anxiety": 0.9234
  },
  "tips": [
    "Please consider reaching out to a counselor or trusted adult.",
    "Practice grounding: name 5 things you see, 4 you hear, 3 you touch.",
    "Remember: one exam does not define your worth or your future."
  ],
  "disclaimer": "This tool is for supportive purposes only..."
}
```

## Anxiety Categories

| Level    | Description                                              |
|----------|----------------------------------------------------------|
| 🟢 Low   | Student feels calm, confident, and well-prepared         |
| 🟡 Moderate | Some nervousness or uncertainty but manageable        |
| 🔴 High  | Significant distress, panic, or inability to cope        |

## Ethical Considerations

- **Non-diagnostic:** This is a supportive tool, NOT a clinical instrument.
- **Anonymity:** No personally identifiable information is collected or stored.
- **Transparency:** Users are informed about the tool's purpose and limitations.
- **Support-oriented:** High anxiety results include referrals to professional help.

## Tech Stack

- **Model:** BERT (bert-base-uncased) fine-tuned for 3-class classification
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **ML Framework:** PyTorch + HuggingFace Transformers
- **Data Processing:** Pandas, scikit-learn
