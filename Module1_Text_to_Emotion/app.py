import streamlit as st
import os
import torch
import pickle
import tempfile
from pathlib import Path
import gdown
import subprocess
import json

# ============================================================
# Streamlit page configuration
# ============================================================
st.set_page_config(
    page_title="Emotional Arabic Chatbot - Transformer Model",
    page_icon="💬",
    layout="wide",
)

# ============================================================
# Clone repo and get models_v2 path
# ============================================================
REPO_URL = "https://github.com/eslamalsaeed72-droid/Emotional-Arabic-Chatbot.git"
REPO_PATH = Path(tempfile.gettempdir()) / "emotional_arabic_repo"
MODELS_DIR = REPO_PATH / "Module1_Text_to_Emotion" / "models_v2"

# Google Drive model weights
DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"


def clone_and_get_models_dir():
    """
    Clone GitHub repo if not already cloned.
    """
    if MODELS_DIR.exists():
        return str(MODELS_DIR)

    st.info("📥 Cloning repository from GitHub (first time only)...")
    
    try:
        if REPO_PATH.exists():
            import shutil
            shutil.rmtree(REPO_PATH)
        
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(REPO_PATH)],
            check=True,
            capture_output=True,
        )

        if not MODELS_DIR.exists():
            raise FileNotFoundError(f"models_v2 not found at {MODELS_DIR}")

        st.success("✅ Repository cloned successfully!")
        return str(MODELS_DIR)

    except Exception as e:
        st.error(f"Failed to clone repository: {e}")
        raise


# ============================================================
# Download pytorch_model.bin from Google Drive
# ============================================================
def download_pytorch_model(model_dir):
    """
    Download pytorch_model.bin from Google Drive if not present.
    """
    model_path = Path(model_dir) / "pytorch_model.bin"
    
    if model_path.exists():
        return str(model_path)

    st.info("📥 Downloading pytorch_model.bin from Google Drive (~513 MB)...")
    
    try:
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, output=str(model_path), quiet=False)

        if not model_path.exists():
            raise FileNotFoundError("Failed to download pytorch_model.bin from Google Drive.")

        st.success("✅ pytorch_model.bin downloaded successfully!")
        return str(model_path)

    except Exception as e:
        st.error(f"Failed to download pytorch_model.bin: {e}")
        raise


# ============================================================
# Load label encoder
# ============================================================
def load_label_mapping(pickle_path):
    """
    Load label encoder from pickle file.
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"label_encoder.pkl not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        encoder = pickle.load(f)

    if not hasattr(encoder, "classes_"):
        raise ValueError("Encoder missing `classes_` attribute.")

    idx2label = {int(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}
    label2idx = {v: k for k, v in idx2label.items()}
    return idx2label, label2idx


# ============================================================
# Load model WITHOUT using AutoConfig
# ============================================================
@st.cache_resource
def load_transformer_model():
    """
    Load Transformer model, tokenizer, and label mappings.
    
    Does NOT rely on config.json being complete.
    Loads the model directly from pytorch_model.bin.
    """
    from transformers import AutoTokenizer, BertForSequenceClassification
    
    # Step 1: Clone repo
    model_dir = Path(clone_and_get_models_dir())

    # Step 2: Download pytorch_model.bin from Google Drive
    model_weights_path = download_pytorch_model(str(model_dir))

    # Step 3: Load label encoder
    label_encoder_path = model_dir / "label_encoder.pkl"
    idx2label, label2idx = load_label_mapping(str(label_encoder_path))
    num_labels = len(idx2label)

    # Step 4: Try to load config.json (for reference only)
    config_path = model_dir / "config.json"
    config_data = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)

    st.info("⏳ Loading model weights...")
    
    # Step 5: Load model weights directly
    try:
        # Try BertForSequenceClassification first (most common for emotion detection)
        model = BertForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=num_labels,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        # Fallback: load raw state dict and apply to model
        st.warning(f"BertForSequenceClassification failed, trying raw load: {str(e)[:100]}")
        state_dict = torch.load(model_weights_path, map_location="cpu", weights_only=False)
        
        # Create a simple model and load weights
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-uncased",  # Use a base model as template
            num_labels=num_labels,
        )
        model.load_state_dict(state_dict, strict=False)

    # Step 6: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    st.success("✅ Model loaded successfully!")
    
    return model, tokenizer, idx2label, device


# ============================================================
# Inference
# ============================================================
def predict_emotion(text, model, tokenizer, idx2label, device):
    """
    Predict emotion from Arabic text.
    """
    # Tokenize
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Move to device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    # Softmax
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Get prediction
    pred_idx = int(probs.argmax())
    pred_label = idx2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

    # Probability dictionary
    prob_dict = {idx2label[i]: float(p) for i, p in enumerate(probs)}

    return pred_label, confidence, prob_dict


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("💬 Emotional Arabic Chatbot")
    st.markdown("**Module 1 – Transformer Model (models_v2)**")
    st.markdown("---")
    st.markdown(
        "Fine-tuned Transformer model for Arabic emotion classification."
    )
    st.markdown(
        "**Data Sources:**\n"
        "- Tokenizer: GitHub repo\n"
        "- pytorch_model.bin: Google Drive (~513 MB)\n"
        "- Local inference only\n"
        "- **Note:** pytorch_model.bin loaded directly without config dependencies"
    )
    st.markdown("---")
    st.caption("Tip: Try different Arabic sentences!")


# ============================================================
# Main header
# ============================================================
st.markdown(
    """
    <h1 style='text-align:right; direction:rtl;'>🌙 Emotional Arabic Chatbot</h1>
    <p style='text-align:right; direction:rtl; color:gray;'>
    نموذج التعرف على المشاعر من النص العربي (Transformer models_v2)
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
    label="Arabic text input",
    placeholder="مثال: أنا مبسوط جدًا لأن مشروع الشات بوت اشتغل أخيرًا! 😊",
    height=120,
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_btn = st.button("🔍 Analyze Emotion")

with col_info:
    st.metric("Model Type", "BERT-based")
    st.metric("Data Source", "GitHub + Drive")


# ============================================================
# Run prediction
# ============================================================
if run_btn:
    if not user_text.strip():
        st.warning("⚠️ Please enter Arabic text first.")
    else:
        try:
            with st.spinner("⏳ Loading model and running inference..."):
                model, tokenizer, idx2label, device = load_transformer_model()
                pred_label, confidence, prob_dict = predict_emotion(
                    user_text, model, tokenizer, idx2label, device
                )

            col_res, col_chart = st.columns([1.2, 1])

            with col_res:
                st.subheader("💖 Predicted Emotion")
                st.markdown(
                    f"<h2 style='color:#ff4b4b; text-align:center; direction:rtl;'>{pred_label}</h2>",
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                st.caption(f"Confidence: {confidence * 100:.2f}%")
                
                st.markdown("**ℹ️ Model Details**")
                st.markdown(
                    f"- Architecture: BertForSequenceClassification\n"
                    f"- Num emotions: {len(idx2label)}\n"
                    f"- Max tokens: 128\n"
                    f"- Device: {device}"
                )

            with col_chart:
                st.subheader("📊 All Emotions Probabilities")
                st.bar_chart(prob_dict)

            with st.expander("🔧 Debug Information"):
                st.write("**Emotion labels:**", idx2label)
                st.write("**Raw probabilities:**", prob_dict)

        except Exception as e:
            st.error("❌ Error loading model or running prediction.")
            st.exception(e)
else:
    st.info("👇 Enter Arabic text and click **Analyze Emotion** to test the model.")
