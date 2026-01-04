import streamlit as st
import os
import torch
import pickle
import tempfile
from pathlib import Path
import numpy as np
from transformers import (
    AutoConfig, AutoTokenizer, PreTrainedModel,
    SequenceClassificationMixin, BertModel, PreTrainedTokenizer
)
import subprocess

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


def clone_and_get_models_dir():
    """
    Clone GitHub repo if not already cloned.
    Returns path to models_v2 directory.
    """
    if MODELS_DIR.exists() and (MODELS_DIR / "pytorch_model.bin").exists():
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
# Fix config and load model
# ============================================================
@st.cache_resource
def load_transformer_model():
    """
    Load Transformer model, tokenizer, config from models_v2.
    
    Handles missing model_type by inferring from available clues.
    """
    model_dir = Path(clone_and_get_models_dir())

    # Load label encoder
    label_encoder_path = model_dir / "label_encoder.pkl"
    idx2label, label2idx = load_label_mapping(str(label_encoder_path))
    num_labels = len(idx2label)

    # Load config
    config_path = model_dir / "config.json"
    config = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)

    # Fix missing model_type
    if not hasattr(config, "model_type") or config.model_type is None:
        # Try to infer from config attributes
        if hasattr(config, "architectures"):
            arch = config.architectures[0].lower()
            if "bert" in arch:
                config.model_type = "bert"
            elif "roberta" in arch:
                config.model_type = "roberta"
            elif "arabert" in arch or "arab" in arch:
                config.model_type = "bert"  # AraBERT is BERT-based
        else:
            # Default to BERT if unclear
            config.model_type = "bert"
    
    # Set label mappings
    config.id2label = idx2label
    config.label2id = label2idx
    config.num_labels = num_labels

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    # Load model with the fixed config
    model = torch.load(
        str(model_dir / "pytorch_model.bin"),
        map_location="cpu",
        weights_only=False,
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


# ============================================================
# Inference
# ============================================================
def predict_emotion(text, model, tokenizer, config, device):
    """
    Predict emotion from Arabic text using Transformer model.
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
        
        # Handle different output types
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("last_hidden_state"))
        else:
            logits = outputs[0] if hasattr(outputs, "__getitem__") else outputs

    # Ensure logits are the right shape
    if logits.dim() == 3:
        logits = logits[:, 0, :]  # Take [CLS] token

    # Softmax
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Get prediction
    pred_idx = int(probs.argmax())
    pred_label = config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

    # Probability dict
    prob_dict = {config.id2label[i]: float(p) for i, p in enumerate(probs)}

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
        "- First run: clones repo (~556 MB pytorch_model.bin)\n"
        "- Later runs: uses cached model\n"
        "- Local inference only"
    )
    st.markdown("---")
    st.caption("Tip: Try different Arabic sentences and dialects!")


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
    label="Arabic text",
    placeholder="مثال: أنا مبسوط جدًا لأن مشروع الشات بوت اشتغل أخيرًا! 😊",
    height=120,
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_btn = st.button("🔍 Analyze Emotion")

with col_info:
    st.metric("Model Type", "Transformer")
    st.metric("Source", "GitHub repo")


# ============================================================
# Run prediction
# ============================================================
if run_btn:
    if not user_text.strip():
        st.warning("⚠️ Please enter Arabic text first.")
    else:
        try:
            with st.spinner("⏳ Loading model and running inference..."):
                model, tokenizer, config, device = load_transformer_model()
                pred_label, confidence, prob_dict = predict_emotion(
                    user_text, model, tokenizer, config, device
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
                    "- Architecture: Transformer (pytorch_model.bin)\n"
                    "- Labels: Loaded from label_encoder.pkl\n"
                    "- Max tokens: 128\n"
                    f"- Device: {device}"
                )

            with col_chart:
                st.subheader("📊 All Emotions Probabilities")
                st.bar_chart(prob_dict)

            with st.expander("🔧 Debug Information"):
                st.write("**Config model_type:**", config.model_type)
                st.write("**Config id2label:**", config.id2label)
                st.write("**Raw probabilities:**", prob_dict)

        except Exception as e:
            st.error("❌ Error loading model or running prediction.")
            st.exception(e)
else:
    st.info("👇 Enter Arabic text and click **Analyze Emotion**")
