import streamlit as st
import os
import json
import torch
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
# Paths for model_v2 (single Transformer model)
# ============================================================
# NOTE:
# Place app.py in the same directory that contains `model_v2/`
# or update BASE_MODEL_DIR accordingly.
BASE_MODEL_DIR = "model_v2"

CONFIG_PATH = os.path.join(BASE_MODEL_DIR, "config.json")
TOKENIZER_PATH = BASE_MODEL_DIR          # tokenizer.json, vocab.txt, special_tokens_map.json, tokenizer_config.json
LABEL_ENCODER_PATH = os.path.join(BASE_MODEL_DIR, "label_encoder.pkl")

# If the actual PyTorch weights file name is different (e.g. pytorch_model.bin),
# it will be loaded automatically via AutoModelForSequenceClassification.from_pretrained.
MODEL_PATH = BASE_MODEL_DIR


# ============================================================
# Utility: Load label mapping from pickle encoder
# ============================================================
def load_label_mapping(pickle_path):
    """
    Load label encoder saved with pickle and build an index-to-label mapping.
    This assumes a scikit-learn LabelEncoder-like object with `classes_` attribute.
    """
    import pickle

    with open(pickle_path, "rb") as f:
        enc = pickle.load(f)

    # Build index -> label mapping (as used by Transformers `id2label`)
    idx2label = {int(i): str(label) for i, label in enumerate(enc.classes_)}
    return idx2label


# ============================================================
# Cached loaders for config, tokenizer, and model
# ============================================================
@st.cache_resource
def load_model_and_tokenizer():
    """
    Load Transformer model and tokenizer from the local `model_v2` directory.
    The function:
    - Loads configuration and injects id2label/label2id from label_encoder.
    - Loads tokenizer (BERT-style) from local JSON / vocab files.
    - Loads sequence classification model with the updated configuration.
    """
    # Load label mapping from label_encoder.pkl
    idx2label = load_label_mapping(LABEL_ENCODER_PATH)
    label2idx = {v: k for k, v in idx2label.items()}

    # Load base configuration
    config = AutoConfig.from_pretrained(CONFIG_PATH)

    # Inject label dictionaries for nicer output
    config.id2label = idx2label
    config.label2id = label2idx

    # Load tokenizer files (tokenizer.json, vocab.txt, etc.)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Load model weights using the updated configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=config,
    )

    # Put model in evaluation mode
    model.eval()

    # Select device (CPU / GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, config, device


# ============================================================
# Inference helper
# ============================================================
def predict_emotion(text, model, tokenizer, config, device):
    """
    Run a single forward pass on the input Arabic text and return:
    - predicted label (string)
    - confidence score (float)
    - dictionary of label -> probability for all classes
    """
    # Tokenize input sentence
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Move tensors to the selected device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    # Convert logits to probabilities via softmax
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Get index of the most probable class
    pred_idx = int(probs.argmax())
    pred_label = config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])

    # Build probability dictionary: label -> prob
    prob_dict = {config.id2label[i]: float(p) for i, p in enumerate(probs)}

    return pred_label, confidence, prob_dict


# ============================================================
# Sidebar content
# ============================================================
with st.sidebar:
    st.title("💬 Emotional Arabic Chatbot")
    st.markdown("**Module 1 – Transformer Emotion Model (model_v2)**")
    st.markdown("---")
    st.markdown(
        "This demo loads the fine‑tuned Arabic Transformer model from "
        "`model_v2` and performs emotion detection on Arabic text."
    )
    st.markdown(
        "- Local, on‑device inference (no data is sent outside your machine).\n"
        "- Intended for research and educational purposes only."
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
    تجربة نموذج التعرف على المشاعر من النص العربي باستخدام نسخة النموذج المحفوظة في <b>model_v2</b>.
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
    st.metric("Model Source", "model_v2")
    st.metric("Task", "Emotion Classification")


# ============================================================
# Run prediction
# ============================================================
if run_inference:
    if not user_text.strip():
        st.warning("Please enter an Arabic sentence first.")
    else:
        try:
            # Load model and tokenizer once (cached)
            with st.spinner("Loading model_v2 and running inference..."):
                model, tokenizer, config, device = load_model_and_tokenizer()
                pred_label, confidence, prob_dict = predict_emotion(
                    user_text, model, tokenizer, config, device
                )

            # =======================
            # Results layout
            # =======================
            col_main, col_chart = st.columns([1.2, 1])

            # --- Main prediction card ---
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
                    "- This result is produced by the Transformer model stored under `model_v2`.\n"
                    "- Labels and mapping are loaded from `label_encoder.pkl`.\n"
                    "- Probabilities are computed using softmax over the model logits."
                )

            # --- Probability distribution chart ---
            with col_chart:
                st.subheader("📊 Class Probabilities")
                st.bar_chart(prob_dict)

            # --- Debug / advanced info ---
            with st.expander("Advanced details (for debugging)"):
                st.write("Config label mapping:", config.id2label)
                st.write("Raw probability dictionary:", prob_dict)

        except Exception as e:
            st.error(
                "An error occurred while loading `model_v2` or running inference. "
                "Please verify that all required files (config.json, tokenizer.json, vocab.txt, "
                "label_encoder.pkl, model weights) exist under the `model_v2` directory."
            )
            st.exception(e)
else:
    st.info("Enter some Arabic text above and click **Analyze Emotion** to test the `model_v2` checkpoint.")
