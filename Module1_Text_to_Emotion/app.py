# =============================================================================
# FIX 1: Disable Streamlit file watcher (inotify watch limit)
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
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification
)

from safetensors.torch import load_file

# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Arabic Emotion Analyzer",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
}

.stTextArea textarea {
    background-color: #f0f2f6;
    border-radius: 10px;
    font-size: 18px;
}

.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-size: 20px;
    border-radius: 10px;
    height: 50px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. PATHS
# =============================================================================
BASE_DIR = "Module1_Text_to_Emotion/models_v2"
MODEL_PATH = os.path.join(BASE_DIR, "model.safetensors")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

DRIVE_FILE_ID = "12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v"

# =============================================================================
# 3. LOAD MODEL (FINAL FIX)
# =============================================================================
@st.cache_resource
def load_prediction_model():

    # Download model if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights (≈500MB)... ⏳"):
            os.makedirs(BASE_DIR, exist_ok=True)
            gdown.download(
                id=DRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

    try:
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)

        # Config (BERT)
        config = BertConfig.from_pretrained(BASE_DIR)

        # Create empty model
        model = BertForSequenceClassification(config)

        # 🔥 Load safetensors manually
        state_dict = load_file(MODEL_PATH)
        model.load_state_dict(state_dict)

        model.eval()

        # Label Encoder
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        return tokenizer, model, label_encoder

    except Exception as e:
        st.error(f"❌ Critical error loading model: {e}")
        return None, None, None


tokenizer, model, label_encoder = load_prediction_model()

# =============================================================================
# 4. PREDICTION
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

    pred_idx = probs.argmax()
    confidence = probs[pred_idx]
    emotion = label_encoder.inverse_transform([pred_idx])[0]

    return emotion, confidence, probs

# =============================================================================
# 5. UI
# =============================================================================
st.title("🤖 محلل المشاعر العربي الذكي")
st.markdown("اكتب نصًا باللغة العربية وسيتم تحليل المشاعر باستخدام الذكاء الاصطناعي")

text_input = st.text_area(
    "أدخل النص:",
    height=150,
    placeholder="مثال: أنا سعيد جدًا بما حققته اليوم"
)

if st.button("تحليل المشاعر 🔍"):

    if not tokenizer or not model:
        st.error("فشل تحميل الموديل")
    elif not text_input.strip():
        st.warning("من فضلك أدخل نصًا أولًا")
    else:
        with st.spinner("جاري التحليل..."):
            emotion, confidence, probs = predict_emotion(text_input)

            emoji_map = {
                "joy": "😊",
                "sadness": "😢",
                "anger": "😡",
                "fear": "😨",
                "love": "❤️",
                "surprise": "😲",
                "neutral": "😐"
            }

            emoji = emoji_map.get(emotion, "🤔")

            st.markdown(f"""
            <div style="
                background-color:#f3f4f6;
                padding:20px;
                border-radius:15px;
                text-align:center;">
                <h1>{emoji}</h1>
                <h2>{emotion}</h2>
                <h4>دقة التوقع: {confidence:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 تفاصيل الاحتمالات")
            for cls, p in zip(label_encoder.classes_, probs):
                if p > 0.01:
                    st.write(f"**{cls}**: {p:.1%}")
                    st.progress(float(p))

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;'>تم التطوير باستخدام AraBERT و Streamlit</div>",
    unsafe_allow_html=True
)
