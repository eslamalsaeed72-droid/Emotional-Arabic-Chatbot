import streamlit as st
import os
import pickle
import gdown
import tempfile
from pathlib import Path
import numpy as np

# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(
    page_title="Emotional Arabic Chatbot - Model v2",
    page_icon="💬",
    layout="wide",
)

# ============================================================
# Google Drive model download configuration
# ============================================================
DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

# Cache directory for model files
CACHE_DIR = Path(tempfile.gettempdir()) / "emotional_arabic_chatbot_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ZIP_PATH = CACHE_DIR / "models_v2.zip"
MODEL_EXTRACT_PATH = CACHE_DIR / "models_v2"


# ============================================================
# Download and extract model from Google Drive
# ============================================================
def download_and_extract_model():
    """
    Download models_v2 ZIP from Google Drive and extract.
    
    Returns the path to the extracted model directory.
    """
    # Reuse cached model if it exists
    if MODEL_EXTRACT_PATH.exists() and (MODEL_EXTRACT_PATH / "label_encoder.pkl").exists():
        return str(MODEL_EXTRACT_PATH)

    st.info("📥 Downloading model_v2 from Google Drive (~513 MB)...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

    gdown.download(url, output=str(MODEL_ZIP_PATH), quiet=False)

    if not MODEL_ZIP_PATH.exists():
        raise FileNotFoundError("Failed to download from Google Drive.")

    st.info("📦 Extracting model files...")
    import zipfile
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(CACHE_DIR)

    if not (MODEL_EXTRACT_PATH / "label_encoder.pkl").exists():
        raise FileNotFoundError("Extracted model missing label_encoder.pkl.")

    st.success("✅ Model downloaded and extracted!")
    return str(MODEL_EXTRACT_PATH)


# ============================================================
# Load pickled models (sklearn-style)
# ============================================================
@st.cache_resource
def load_sklearn_models():
    """
    Load emotion detection model and vectorizer from pickle files.
    
    This function loads:
    - emotion_model.pkl (trained classifier)
    - emotion_tfidf.pkl (TF-IDF vectorizer)
    - label_encoder.pkl (label encoder)
    """
    model_dir = Path(download_and_extract_model())

    # Try different possible file names (in case naming varies)
    model_candidates = [
        "emotion_model.pkl",
        "emotion_model_v1.pkl",
        "emotionmodel.pkl",
    ]
    tfidf_candidates = [
        "emotion_tfidf.pkl",
        "emotion_tfidf_v1.pkl",
        "emotiontfidf.pkl",
    ]
    encoder_candidates = [
        "label_encoder.pkl",
        "emotion_encoder.pkl",
        "emotion_encoder_v1.pkl",
    ]

    # Find actual file names in directory
    model_file = None
    tfidf_file = None
    encoder_file = None

    for candidate in model_candidates:
        if (model_dir / candidate).exists():
            model_file = model_dir / candidate
            break

    for candidate in tfidf_candidates:
        if (model_dir / candidate).exists():
            tfidf_file = model_dir / candidate
            break

    for candidate in encoder_candidates:
        if (model_dir / candidate).exists():
            encoder_file = model_dir / candidate
            break

    if not model_file or not tfidf_file or not encoder_file:
        available = list(model_dir.glob("*.pkl"))
        raise FileNotFoundError(
            f"Could not find model files.\n"
            f"Looking for: model .pkl, tfidf .pkl, encoder .pkl\n"
            f"Available: {available}"
        )

    # Load model, vectorizer, and encoder
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    with open(tfidf_file, "rb") as f:
        vectorizer = pickle.load(f)

    with open(encoder_file, "rb") as f:
        encoder = pickle.load(f)

    return model, vectorizer, encoder


# ============================================================
# Inference helper
# ============================================================
def predict_emotion_sklearn(text, model, vectorizer, encoder):
    """
    Predict emotion using sklearn-style model.
    
    Returns:
    - label (str): predicted emotion
    - confidence (float): confidence score
    - prob_dict (dict): label -> probability for all classes
    """
    # Vectorize input text
    X = vectorizer.transform([text])

    # Predict
    pred_idx = model.predict(X)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    # Get confidence
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        prob_dict = {encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}
    else:
        confidence = 1.0
        prob_dict = {pred_label: 1.0}

    return pred_label, confidence, prob_dict


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("💬 Emotional Arabic Chatbot")
    st.markdown("**Module 1 – Emotion Model (models_v2)**")
    st.markdown("---")
    st.markdown(
        "This app downloads the emotion detection model from Google Drive "
        "and tests it on Arabic text."
    )
    st.markdown("- First run: ~513 MB download\n- Later runs: cached\n- Local inference only")
    st.markdown("---")
    st.caption("Tip: Try different Arabic sentences!")


# ============================================================
# Main header
# ============================================================
st.markdown(
    """
    <h1 style='text-align:right; direction:rtl;'>🌙 Emotional Arabic Chatbot</h1>
    <p style='text-align:right; direction:rtl; color:gray;'>
    نموذج التعرف على المشاعر من النص العربي (models_v2)
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================
# Text input
# ============================================================
st.markdown(
    "<div style='text-align:right; direction:rtl;'>اكتب جملة بالعربية:</div>",
    unsafe_allow_html=True,
)

user_text = st.text_area(
    "",
    placeholder="مثال: أنا مبسوط جدًا لأن المشروع اشتغل!",
    height=120,
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_btn = st.button("🔍 Analyze")

with col_info:
    st.metric("Model", "sklearn-based")
    st.metric("Source", "Google Drive")


# ============================================================
# Run prediction
# ============================================================
if run_btn:
    if not user_text.strip():
        st.warning("Please enter Arabic text first.")
    else:
        try:
            with st.spinner("Loading and predicting..."):
                model, vectorizer, encoder = load_sklearn_models()
                pred_label, confidence, prob_dict = predict_emotion_sklearn(
                    user_text, model, vectorizer, encoder
                )

            col_res, col_chart = st.columns([1.2, 1])

            with col_res:
                st.subheader("💖 Result")
                st.markdown(
                    f"<h2 style='color:#ff4b4b; text-align:center;'>{pred_label}</h2>",
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                st.caption(f"Confidence: {confidence * 100:.2f}%")

            with col_chart:
                st.subheader("📊 Probabilities")
                st.bar_chart(prob_dict)

            with st.expander("Debug info"):
                st.write("Predicted probabilities:", prob_dict)

        except Exception as e:
            st.error("Error loading model or running prediction.")
            st.exception(e)
else:
    st.info("Enter Arabic text and click **Analyze** to test.")
