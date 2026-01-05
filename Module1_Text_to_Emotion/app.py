# =============================================================================
# STREAMLIT FIX
# =============================================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# =============================================================================
# IMPORTS
# =============================================================================
import json
import pickle
import torch
import streamlit as st
import torch.nn.functional as F
import gdown

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Arabic Emotion Analyzer",
    page_icon="🤖",
    layout="centered"
)

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = "Module1_Text_to_Emotion/models_v2"
MODEL_PATH = os.path.join(BASE_DIR, "pytorch_model.bin")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

os.makedirs(BASE_DIR, exist_ok=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model():

    # ---------------- Download model ----------------
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model weights..."):
            gdown.download(
                id=DRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    try:
        # ---------------- Load custom config ----------------
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            custom_cfg = json.load(f)

        num_labels = custom_cfg["num_labels"]

        # ---------------- Tokenizer ----------------
        tokenizer = BertTokenizer.from_pretrained(
            "aubmindlab/bert-base-arabertv02"
        )

        # ---------------- Build HF config manually ----------------
        config = BertConfig.from_pretrained(
            "aubmindlab/bert-base-arabertv02",
            num_labels=num_labels,
            hidden_dropout_prob=custom_cfg.get("dropout", 0.3),
            attention_probs_dropout_prob=custom_cfg.get("dropout", 0.3)
        )

        # ---------------- Model ----------------
        model = BertForSequenceClassification(config)

        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # ---------------- Label encoder ----------------
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        return tokenizer, model, label_encoder

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None


tokenizer, model, label_encoder = load_model()

# =============================================================================
# PREDICTION
# =============================================================================
def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    idx = torch.argmax(probs).item()
    return label_encoder.inverse_transform([idx])[0], probs[idx].item()

# =============================================================================
# UI
# =============================================================================
st.title("🤖 محلل المشاعر العربي الذكي")

text = st.text_area(
    "أدخل النص:",
    placeholder="مثال: أنا سعيد جدًا اليوم"
)

if st.button("تحليل المشاعر 🔍"):
    if not text.strip():
        st.warning("من فضلك أدخل نصًا")
    else:
        with st.spinner("جاري التحليل..."):
            emotion, conf = predict_emotion(text)
            st.success(f"المشاعر المتوقعة: **{emotion}** ({conf:.1%})")
