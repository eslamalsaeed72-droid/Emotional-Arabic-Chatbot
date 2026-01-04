import streamlit as st
import os
import torch
import pickle
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
# Paths for models_v2
# ============================================================
# Place this app.py inside: Module1_Text_to_Emotion/
# That directory must contain the folder: models_v2/
BASE_MODEL_DIR = "models_v2"

CONFIG_PATH = os.path.join(BASE_MODEL_DIR, "config.json")
TOKENIZER_PATH = BASE_MODEL_DIR
MODEL_PATH = BASE_MODEL_DIR
LABEL_ENCODER_PATH = os.path.join(BASE_MODEL_DIR, "label_encoder.pkl")


# ============================================================
# Helper: load label encoder mapping
# ============================================================
def load_label_mapping(pickle_path):
    """
    Load a scikit-learn style LabelEncoder from pickle and
    build an index-to-label mapping for the Transformer config.
    """
    if not os.path.exists(pickle_path):
        # Extra debug information if the file is not visible
        cwd = os.getcwd()
        available = os.listdir(os.path.dirname(pickle_path) or ".")
        raise FileNotFoundError(
            f"label_encoder.pkl not found at: {pickle_path}\n"
            f"Current working directory: {cwd}\n"
            f"Contents of `{os.path.dirname(pickle_path) or '.'}`: {available}"
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
    Load the Transformer model, tokenizer and label mappings
    from the local `models_v2` directory.
    """
    idx2label, label2idx = load_label_mapping(LABEL_ENCODER_PATH)

    config = AutoConfig.from_pretrained(CONFIG_PATH)
    config.id2label = idx2label
    config.label2id = label2idx
    config.num_labels = len(idx2label)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


# ============================================================
# Inference helper
# ============================================================
def predict_emotion(text, model, tokenizer, config, device):
    """
    Run an emotion classification pass on Arabic text and return:
    - predicted label (string)
    - confidence score (float in [0, 1])
    - dictionary mapping label -> probability for all classes.
    """
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

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
        "`models_v2` and performs emotion detection on Arabic text."
    )
    st.markdown(
        "- All inference runs locally on your machine.\n"
        "- The app is intended for research and educational purposes only."
    )
    st.markdown("---")
    st.caption("Tip: Try different dialects and emotional tones to stress‑test the model.")


# ============================================================
# Main header (RTL styling for Arabic text)
# ============================================================
st.markdown(
    """
    <h1 style='text-align:right; direction:rtl;'>🌙 Emotional Arabic Chatbot – Transformer Version</h1>
    <p style='text-align:right; direction:rtl; color:gray;'>
    تجربة نموذج التعرف على المشاعر من النص العربي باستخدام النسخة المحفوظة في <b>models_v2</b>.
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
    st.metric("Model Source", BASE_MODEL_DIR)
    st.metric("Task", "Emotion Classification")


# ============================================================
# Run prediction
# ============================================================
if run_inference:
    if not user_text.strip():
        st.warning("Please enter an Arabic sentence first.")
    else:
        try:
            with st.spinner("Loading models_v2 and running inference..."):
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

                st.markdown("**Model Notes**")
                st.markdown(
                    "- The prediction comes from the Transformer checkpoint stored under `models_v2`.\n"
                    "- Human‑readable labels are loaded from `label_encoder.pkl`.\n"
                    "- Probabilities are computed using softmax over the model logits."
                )

            with col_chart:
                st.subheader("📊 Class Probabilities")
                st.bar_chart(prob_dict)

            with st.expander("Advanced details (for debugging)"):
                st.write("Config label mapping:", config.id2label)
                st.write("Raw probability dictionary:", prob_dict)

        except Exception as e:
            st.error(
                "An error occurred while loading `models_v2` or running inference.\n"
                f"BASE_MODEL_DIR: `{BASE_MODEL_DIR}`.\n"
                "Make sure you run `streamlit run app.py` from the `Module1_Text_to_Emotion` folder."
            )
            st.exception(e)
else:
    st.info("Enter some Arabic text above and click **Analyze Emotion** to test the `models_v2` checkpoint.")
