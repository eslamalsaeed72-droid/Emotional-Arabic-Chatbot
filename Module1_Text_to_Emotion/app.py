import streamlit as st
import os
import torch
import pickle
import gdown
import tempfile
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(
    page_title="Emotional Arabic Chatbot - Transformer Model",
    page_icon="💬",
    layout="wide",
)

# ============================================================
# Google Drive model download configuration
# ============================================================
# The model_v2 file/folder on Google Drive (513 MB)
# ID extracted from: https://drive.google.com/file/d/12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v/view
DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

# Simple persistent cache directory using the system temp folder
CACHE_DIR = Path(tempfile.gettempdir()) / "emotional_arabic_chatbot_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ZIP_PATH = CACHE_DIR / "models_v2.zip"
MODEL_EXTRACT_PATH = CACHE_DIR / "models_v2"  # this must match the folder name inside the zip


# ============================================================
# Helper: Download and extract model from Google Drive
# ============================================================
def download_and_extract_model():
    """
    Download the models_v2 ZIP file from Google Drive and extract it.

    Workflow:
    - If the extracted folder already exists and contains config.json,
      reuse it (no download).
    - Otherwise, download the ZIP file from Google Drive using gdown.
    - Extract the ZIP into CACHE_DIR.
    - Return the path to the extracted model directory.
    """
    # Reuse existing extracted model if available
    if MODEL_EXTRACT_PATH.exists() and (MODEL_EXTRACT_PATH / "config.json").exists():
        return str(MODEL_EXTRACT_PATH)

    # Download ZIP file from Google Drive
    st.info("📥 Downloading model_v2 from Google Drive (~513 MB)... This may take a few minutes on first run.")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

    gdown.download(
        url,
        output=str(MODEL_ZIP_PATH),
        quiet=False,
    )

    if not MODEL_ZIP_PATH.exists():
        raise FileNotFoundError(f"Failed to download model ZIP from Google Drive (expected at {MODEL_ZIP_PATH})")

    # Extract ZIP file
    st.info("📦 Extracting model files...")
    import zipfile
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(CACHE_DIR)

    # Verify extraction
    if not (MODEL_EXTRACT_PATH / "config.json").exists():
        raise FileNotFoundError(
            "Extracted model folder does not contain config.json. "
            f"Check the ZIP structure under: {CACHE_DIR}"
        )

    st.success("✅ Model downloaded and extracted successfully!")
    return str(MODEL_EXTRACT_PATH)


# ============================================================
# Helper: Load label encoder mapping
# ============================================================
def load_label_mapping(pickle_path: Path):
    """
    Load a scikit-learn style LabelEncoder from pickle and
    build an index-to-label mapping for the Transformer config.
    """
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"label_encoder.pkl not found at: {pickle_path}\n"
            f"Parent directory contents: {list(pickle_path.parent.iterdir()) if pickle_path.parent.exists() else 'N/A'}"
        )

    with open(pickle_path, "rb") as f:
        encoder = pickle.load(f)

    if not hasattr(encoder, "classes_"):
        raise ValueError("Loaded label encoder does not expose a `classes_` attribute.")

    idx2label = {int(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}
    label2idx = {v: k for k, v in idx2label.items()}
    return idx2label, label2idx


# ============================================================
# Cached loaders for config, tokenizer, and model
# ============================================================
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the Transformer model, tokenizer and label mappings from Google Drive.

    Steps:
    - Download and extract models_v2 into CACHE_DIR (if not already present).
    - Load label_encoder.pkl to obtain human-readable labels.
    - Load config.json and inject id2label / label2id mappings.
    - Load tokenizer and model weights from the extracted directory.
    - Move model to the available device (CPU / GPU).
    """
    # Download / extract model if needed
    model_dir = Path(download_and_extract_model())

    # Paths inside extracted folder
    label_encoder_path = model_dir / "label_encoder.pkl"
    config_path = model_dir / "config.json"
    tokenizer_path = str(model_dir)
    model_path = str(model_dir)

    # Load label encoder and build mappings
    idx2label, label2idx = load_label_mapping(label_encoder_path)

    # Load configuration and inject label mappings
    config = AutoConfig.from_pretrained(str(config_path))
    config.id2label = idx2label
    config.label2id = label2idx
    config.num_labels = len(idx2label)

    # Load tokenizer and model from extracted directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    # Select device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


# ============================================================
# Inference helper
# ============================================================
def predict_emotion(text: str, model, tokenizer, config, device):
    """
    Run an emotion classification pass on Arabic text.

    Returns:
    - pred_label (str): predicted emotion label.
    - confidence (float): confidence score in [0, 1].
    - prob_dict (dict): mapping label -> probability for all classes.
    """
    # Tokenize input
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Move tensors to device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass (no gradients needed)
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Choose the most probable label
    pred_idx = int(probs.argmax())
    pred_label = config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

    # Build probability dictionary for all labels
    prob_dict = {config.id2label[i]: float(p) for i, p in enumerate(probs)}

    return pred_label, confidence, prob_dict


# ============================================================
# Sidebar content
# ============================================================
with st.sidebar:
    st.title("💬 Emotional Arabic Chatbot")
    st.markdown("**Module 1 – Transformer Emotion Model (models_v2)**")
    st.markdown("---")
    st.markdown(
        "This demo downloads the fine‑tuned Arabic Transformer model from "
        "Google Drive the first time it runs, then uses a cached copy."
    )
    st.markdown(
        "- First run: downloads ~513 MB (can take a few minutes).\n"
        "- Later runs: reuse cached model (much faster).\n"
        "- All inference executed locally on your machine.\n"
        "- Intended for research and educational purposes only."
    )
    st.markdown("---")
    st.caption("Tip: Try different dialects and emotional tones to test the model.")


# ============================================================
# Main header (RTL styling for Arabic text)
# ============================================================
st.markdown(
    """
    <h1 style='text-align:right; direction:rtl;'>🌙 Emotional Arabic Chatbot – Transformer Version</h1>
    <p style='text-align:right; direction:rtl; color:gray;'>
    تجربة نموذج التعرف على المشاعر من النص العربي باستخدام <b>models_v2</b> المحمَّل من جوجل درايف.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================
# Text input section
# ============================================================
st.markdown(
    "<div style='text-align:right; direction:rtl;'>اكتب جملة أو فقرة بالعربية لاختبار النموذج:</div>",
    unsafe_allow_html=True,
)

default_example = "أنا مبسوط جدًا النهارده لأن مشروع الشات بوت اشتغل أخيرًا! 😊"

user_text = st.text_area(
    "",
    value="",
    height=150,
    placeholder=default_example,
)

col_btn, col_meta = st.columns([1, 3])
with col_btn:
    run_inference = st.button("🔍 Analyze Emotion")

with col_meta:
    st.metric("Model Source", "Google Drive cache")
    st.metric("Task", "Emotion Classification")


# ============================================================
# Run prediction
# ============================================================
if run_inference:
    if not user_text.strip():
        st.warning("⚠️ Please enter an Arabic sentence first.")
    else:
        try:
            with st.spinner("⏳ Loading model and running inference..."):
                model, tokenizer, config, device = load_model_and_tokenizer()
                pred_label, confidence, prob_dict = predict_emotion(
                    user_text, model, tokenizer, config, device
                )

            col_main, col_chart = st.columns([1.2, 1])

            with col_main:
                st.subheader("💖 Predicted Emotion")
                st.markdown(
                    f"""
                    <div style='text-align:center; direction:rtl;'>
                        <h2 style='color:#ff4b4b;'>{pred_label}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                st.caption(f"Confidence: {confidence * 100:.2f}%")

                st.markdown("**ℹ️ Model Information**")
                st.markdown(
                    "- Checkpoint: Transformer model stored in `models_v2` (downloaded from Google Drive).\n"
                    "- Labels: Loaded from `label_encoder.pkl`.\n"
                    "- Probabilities: Softmax over logits.\n"
                    "- Max sequence length: 128 tokens."
                )

            with col_chart:
                st.subheader("📊 Class Probabilities")
                st.bar_chart(prob_dict)

            with st.expander("🔧 Advanced details (for debugging)"):
                st.write("Config id2label mapping:", config.id2label)
                st.write("Raw probabilities:", prob_dict)
                st.write("Device used:", device)

        except Exception as e:
            st.error(
                "❌ An error occurred while downloading/loading the model or running inference.\n\n"
                "Please check the details below. Common issues:\n"
                "- Google Drive link restricted or invalid.\n"
                "- Not enough disk space in the temp directory.\n"
                "- Internet connectivity problems during download."
            )
            st.exception(e)
else:
    st.info("👇 Enter some Arabic text above and click **Analyze Emotion** to test the model.")
