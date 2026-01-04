# =============================================================================
# FIX: Disable Streamlit file watcher (inotify limit)
# =============================================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import torch
import pickle
import gdown
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification
)

from safetensors.torch import load_file

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Arabic Emotion Analyzer",
    page_icon="🤖",
    layout="centered"
)

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = "Module1_Text_to_Emotion/models_v2"
MODEL_PATH = os.path.join(BASE_DIR, "model.safetensors")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

# =============================================================================
# LOAD MODEL (FINAL – GUARANTEED)
# =============================================================================
@st.cache_resource
def load_prediction_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights (≈500MB)..."):
            os.makedirs(BASE_DIR, exist_ok=True)
            gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)

    try:
        # ✅ tokenizer (slow but guaranteed)
        tokenizer = BertTokenizer.from_pretrained(BASE_DIR)

        # config
        config = BertConfig.from_pretrained(BASE_DIR)

        # model
        model = BertForSequenceClassification(config)
        state_dict = load_file(MODEL_PATH)
        model.load_state_dict(state_dict)
        model.eval()

        # label encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        return tokenizer, model, label_encoder

    except Exception as e:
        st.error(f"❌ Critical error loading model: {e}")
        return None, None, None


tokenizer, model, label_encoder = load_prediction_model()

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
        probs = F.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    idx = probs.argmax()
    return (
        label_encoder.inverse_transform([idx])[0],
        probs[idx],
        probs
    )

# =============================================================================
# UI
# =============================================================================
st.title("🤖 محلل المشاعر العربي الذكي")

text_input = st.text_area(
    "أدخل النص:",
    placeholder="مثال: أنا سعيد جدًا اليوم"
)

if st.button("تحليل المشاعر 🔍"):
    if not text_input.strip():
        st.warning("من فضلك اكتب نصًا أولًا")
    else:
        with st.spinner("جاري التحليل..."):
            emotion, conf, _ =_
