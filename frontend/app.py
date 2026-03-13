"""
Streamlit frontend for the AI-Based Exam Anxiety Detection System.
Provides an intuitive UI for students and educators.
"""
import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000"

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .anxiety-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    .low-anxiety {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .moderate-anxiety {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border: 2px solid #ffc107;
    }
    .high-anxiety {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .disclaimer-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #555;
    }
    .tip-item {
        background: #f8f9fa;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧠 AI-Based Exam Anxiety Detector")
st.markdown(
    "**Identify and understand exam-related anxiety through AI-powered text analysis.**"
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This system uses a **BERT-based NLP model** to classify exam anxiety 
        from student text into three levels:
        
        - 🟢 **Low Anxiety** — Calm and prepared
        - 🟡 **Moderate Anxiety** — Some nervousness
        - 🔴 **High Anxiety** — Significant distress
        
        ---
        
        **How to use:**
        1. Type your thoughts about an upcoming exam
        2. Click **Analyze Anxiety Level**
        3. View your results and tips
        
        ---
        
        ⚠️ **Disclaimer:** This is a supportive tool, 
        NOT a clinical diagnostic instrument.
        """
    )
    st.markdown("---")
    st.markdown("**Built with:** Streamlit · FastAPI · BERT · PyTorch")

# ── Main input ────────────────────────────────────────────────────────────────

st.subheader("📝 Share Your Thoughts")
st.markdown(
    "Express how you're feeling about your upcoming exam. "
    "Write freely — your input is anonymous and not stored."
)

user_text = st.text_area(
    "Your thoughts or feelings about the exam:",
    height=150,
    placeholder="Example: I've been studying but I still feel nervous about the math section...",
)

# Example prompts
with st.expander("💡 Need inspiration? Try these examples"):
    examples = [
        "I feel confident and well-prepared for tomorrow's exam.",
        "I'm a bit nervous about the essay questions but I think I'll manage.",
        "I can't sleep at night because of exam stress. I feel like I'm going to fail.",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            user_text = ex

# ── Analysis ──────────────────────────────────────────────────────────────────

EMOJI_MAP = {
    "Low Anxiety": "🟢",
    "Moderate Anxiety": "🟡",
    "High Anxiety": "🔴",
}

CSS_MAP = {
    "Low Anxiety": "low-anxiety",
    "Moderate Anxiety": "moderate-anxiety",
    "High Anxiety": "high-anxiety",
}

EMOJI_FACE = {
    "Low Anxiety": "😊",
    "Moderate Anxiety": "😟",
    "High Anxiety": "😰",
}

analyze_clicked = st.button("🔍 Analyze Anxiety Level", type="primary", use_container_width=True)

if analyze_clicked:
    if not user_text or not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing your text with AI..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the API server. "
                    "Please make sure the FastAPI backend is running on port 8000.\n\n"
                    "Run: `python backend/main.py`"
                )
                st.stop()
            except requests.HTTPError as e:
                st.error(f"API error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        level = result["anxiety_level"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        tips = result["tips"]
        disclaimer = result["disclaimer"]

        # ── Result display ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        emoji = EMOJI_MAP.get(level, "⚪")
        css_class = CSS_MAP.get(level, "")
        face = EMOJI_FACE.get(level, "")

        st.markdown(
            f"""
            <div class="anxiety-card {css_class}">
                <h2>{face} {emoji} {level}</h2>
                <p style="font-size: 1.1rem;">Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability breakdown
        st.markdown("#### Probability Breakdown")
        for label_name, prob in probs.items():
            emoji_label = EMOJI_MAP.get(label_name, "⚪")
            st.progress(prob, text=f"{emoji_label} {label_name}: {prob:.1%}")

        # Tips
        st.markdown("#### 💡 Recommendations")
        for tip in tips:
            st.markdown(
                f'<div class="tip-item">✅ {tip}</div>',
                unsafe_allow_html=True,
            )

        # Disclaimer
        st.markdown(
            f'<div class="disclaimer-box">⚠️ {disclaimer}</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "AI-Based Exam Anxiety Detection System · Built for student well-being · "
    "Not a substitute for professional mental health advice"
    "</div>",
    unsafe_allow_html=True,
)
