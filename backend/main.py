"""
FastAPI backend for the AI-Based Exam Anxiety Detection System.
Exposes a /predict endpoint for anxiety level inference.
"""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# Add project root so imports like config and models work from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL_MAP
from models.predict import AnxietyPredictor

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Exam Anxiety Detection API",
    description="AI-powered anxiety level classification from student text input.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
predictor = AnxietyPredictor()

# ── Tips per anxiety level ────────────────────────────────────────────────────

TIPS = {
    "Low Anxiety": [
        "Great job staying calm! Keep up your balanced study routine.",
        "Continue with your preparation strategy — it's clearly working.",
        "Maintain your healthy mindset and take breaks when needed.",
    ],
    "Moderate Anxiety": [
        "Try deep breathing exercises: inhale for 4 seconds, hold for 4, exhale for 4.",
        "Break your study material into smaller, manageable chunks.",
        "Take short walks between study sessions to refresh your mind.",
        "Talk to a friend or classmate about how you're feeling.",
        "Practice positive self-talk: remind yourself of past successes.",
    ],
    "High Anxiety": [
        "Please consider reaching out to a counselor or trusted adult.",
        "Practice grounding: name 5 things you see, 4 you hear, 3 you touch.",
        "Step away from studying for a short while — your mental health comes first.",
        "Try progressive muscle relaxation to release physical tension.",
        "Remember: one exam does not define your worth or your future.",
        "Consider speaking with your teacher about your stress levels.",
    ],
}

# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text input must not be empty.")
        return v.strip()


class PredictResponse(BaseModel):
    anxiety_level: str
    confidence: float
    probabilities: dict
    tips: list[str]
    disclaimer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Exam Anxiety Detection API is running.",
        "endpoints": {"/predict": "POST — Predict anxiety level from text"},
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict_anxiety(req: PredictRequest):
    """Predict exam anxiety level from student text."""
    try:
        result = predictor.predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    level = result["label"]
    import random
    tips = random.sample(TIPS[level], min(3, len(TIPS[level])))

    return PredictResponse(
        anxiety_level=level,
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        tips=tips,
        disclaimer=(
            "This tool is for supportive purposes only and is NOT a clinical "
            "diagnostic instrument. If you are experiencing severe anxiety, "
            "please consult a qualified mental health professional."
        ),
    )


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
