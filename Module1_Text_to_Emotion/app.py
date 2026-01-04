# =============================================================================
# STREAMLIT FIX
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
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

os.makedirs(BASE_DIR, exist_ok=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model():

    # -------- Download model if not exists --------
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model weights..."):
            gdown.download(
                id=DRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    try:
        # tokenizer (أي BERT عربي شغال)
        tokenizer = BertTokenizer.from_pretrained(
            "asafaya/bert-base-arabic"
        )

        # load label encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        num_labels = len(label_encoder.classes_)

        # config
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_labels
        )

        # model
        model = BertForSequenceClassification(config)

        # 🔥 load weights
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        return tokenizer, model, label_encoder

    except Exception as e:
        st.error(f"❌ Critical error loading model: {e}")
        return None, None, None


tokenizer, model, label_encoder = load_model()

# =============================================================================
# PREDICTION
# =============================================================================
def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
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
        st.warning("من فضلك اكتب نص")
    else:
        with st.spinner("جاري التحليل..."):
            emotion, conf = predict_emotion(text)
            st.success(f"المشاعر المتوقعة: **{emotion}** ({conf:.1%})")
