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
# The model_v2 folder on Google Drive (513 MB)
# Extract the ID from: https://drive.google.com/file/d/12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v/view?usp=drive_link
DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

# Create a persistent cache directory in Streamlit's cache folder
CACHE_DIR = Path(st.config.get_option("client.caching_related_query_params_enabled"))
if CACHE_DIR is None or str(CACHE_DIR) == "None":
    CACHE_DIR = Path(tempfile.gettempdir()) / "streamlit_model_cache"
else:
    CACHE_DIR = Path(tempfile.gettempdir()) / "streamlit_model_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ZIP_PATH = CACHE_DIR / "models_v2.zip"
MODEL_EXTRACT_PATH = CACHE_DIR / "models_v2"


# ============================================================
# Helper: Download and extract model from Google Drive
# ============================================================
def download_and_extract_model():
    """
    Download the models_v2 ZIP file from Google Drive.
    
    This function:
    - Checks if the model is already extracted locally.
    - If not, downloads the ZIP file from Google Drive.
    - Extracts the ZIP to the cache directory.
    - Returns the path to the extracted model directory.
    """
    # If model is already extracted, return immediately
    if MODEL_EXTRACT_PATH.exists() and (MODEL_EXTRACT_PATH / "config.json").exists():
        return str(MODEL_EXTRACT_PATH)

    # Download phase
    st.info("📥 Downloading model_v2 from Google Drive (~513 MB)... This may take 2-3 minutes on first run.")
    
    try:
        # Download ZIP file from Google Drive
        gdown.download(
            f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
            output=str(MODEL_ZIP_PATH),
            quiet=False,
        )

        if not MODEL_ZIP_PATH.exists():
            raise FileNotFoundError(f"Failed to download from Google Drive. File not found at {MODEL_ZIP_PATH}")

        # Extract phase
        st.info("📦 Extracting model files...")
        import zipfile
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(CACHE_DIR)

        # Verify extraction
        if not (MODEL_EXTRACT_PATH / "config.json").exists():
            raise FileNotFoundError("Extracted model missing config.json. Check the ZIP structure.")

        st.success("✅ Model downloaded and extracted successfully!")
        return str(MODEL_EXTRACT_PATH)

    except Exception as e:
        st.error(f"❌ Failed to download/extract model from Google Drive: {e}")
        raise


# ============================================================
# Helper: Load label encoder mapping
# ============================================================
def load_label_mapping(pickle_path):
    """
    Load a scikit-learn style LabelEncoder from pickle and
    build an index-to-label mapping for the Transformer config.
    """
    pickle_path = Path(pickle_path)
    
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"label_encoder.pkl not found at: {pickle_path}\n"
            f"Parent directory contents: {list(pickle_path.parent.iterdir()) if pickle_path.parent.exists() else 'N/A'}"
        )

    with open(pickle_path, "rb") as f:
        encoder = pickle.load(f)

    if not hasattr(encoder, "classes_"):
        raise ValueError("Loaded object does not have `classes_` attribute.")

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
    - Download and extract model_v2 from Google Drive if not cached.
    - Load label encoder to get human-readable emotion labels.
    - Load configuration and inject id2label / label2id mappings.
    - Load tokenizer and model weights from the extracted directory.
    - Move the model to the available device (CPU / GPU).
    """
    # Download/extract model from Google Drive
    model_dir = download_and_extract_model()

    # Define paths to model files within the extracted directory
    label_encoder_path = Path(model_dir) / "label_encoder.pkl"
    config_path = Path(model_dir) / "config.json"
    tokenizer_path = model_dir
    model_path = model_dir

    # Load label encoder and build mappings
    idx2label, label2idx = load_label_mapping(label_encoder_path)

    # Load configuration from config.json
    config = AutoConfig.from_pretrained(str(config_path))
    config.id2label = idx2label
    config.label2id = label2idx
    config.num_labels = len(idx2label)

    # Load tokenizer from extracted directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load model weights from extracted directory
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    # Select device (CPU / GPU) and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


# ============================================================
# Inference helper
# ============================================================
def predict_emotion(text, model, tokenizer, config, device):
    """
    Run an emotion classification forward pass on Arabic text.
    
    Returns:
    - predicted_label (str): The emotion class as a string.
    - confidence (float): Confidence score in [0, 1].
    - prob_dict (dict): Label -> probability mapping for all classes.
    """
    # Tokenize input text
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Move tokenized tensors to the selected device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass (no gradients for inference)
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    # Convert logits to probabilities via softmax
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Get the predicted emotion class
    pred_idx = int(probs.argmax())
    pred_label = config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

    # Build a dictionary of all class probabilities
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
        "This demo loads the fine‑tuned Arabic Transformer model from "
        "Google Drive and performs emotion detection on Arabic text."
    )
    st.markdown(
        "📥 **First run:** Downloads model_v2 (~513 MB) from Google Drive.\n\n"
        "⚡ **Subsequent runs:** Uses cached model (much faster).\n\n"
        "🏠 **Local inference:** All processing happens on your machine.\n\n"
        "📚 **Purpose:** Research and educational use only."
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
    تجربة نموذج التعرف على المشاعر من النص العربي باستخدام <b>models_v2</b> من جوجل درايف.
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
    st.metric("Model Source", "Google Drive")
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

            # Main prediction card
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
                    "- Model checkpoint: Transformer (models_v2)\n"
                    "- Labels: Loaded from `label_encoder.pkl`\n"
                    "- Probabilities: Softmax over logits\n"
                    "- Max sequence length: 128 tokens"
                )

            # Probability distribution chart
            with col_chart:
                st.subheader("📊 Class Probabilities")
                st.bar_chart(prob_dict)

            # Advanced debugging information
            with st.expander("🔧 Advanced details (for debugging)"):
                st.write("**Config id2label mapping:**", config.id2label)
                st.write("**Raw probabilities:**", prob_dict)
                st.write("**Device used:**", device)

        except Exception as e:
            st.error(
                "❌ An error occurred while loading the model or running inference.\n\n"
                "Please check the error details below and verify that:\n"
                "- Google Drive link is accessible (not restricted)\n"
                "- Internet connection is stable\n"
                "- You have sufficient disk space for the model (~513 MB)"
            )
            st.exception(e)
else:
    st.info("👇 Enter some Arabic text above and click **Analyze Emotion** to test the model.")
