import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pickle
import os
import gdown
import torch.nn.functional as F

# ============================================================================
# 1. PAGE CONFIGURATION & UI STYLING
# ============================================================================
st.set_page_config(
    page_title="AI Arabic Emotion Analyzer",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS to support Arabic RTL (Right-to-Left) direction and enhance typography
st.markdown("""
<style>
    /* Import Cairo font for better Arabic readability */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stTextArea textarea {
        background-color: #f0f2f6;
        border-radius: 10px;
        border: 1px solid #d1d1d1;
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        height: 50px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. CONFIGURATION & FILE PATH SETUP
# ============================================================================

# Define local directory paths relative to the script location
BASE_DIR = "Module1_Text_to_Emotion/models_v2"
MODEL_FILENAME = "model.safetensors"  # Standard filename for Safetensors weights
MODEL_PATH_FULL = os.path.join(BASE_DIR, MODEL_FILENAME)
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# Google Drive File ID for the heavy model weights (~500MB)
DRIVE_FILE_ID = '12TtvlA3365gKRV0jCtKhCeN9oSk8fK1v'

# ============================================================================
# 3. MODEL LOADING LOGIC (CACHED & FIXED)
# ============================================================================

@st.cache_resource
def load_prediction_model():
    """
    Downloads model weights from Google Drive if missing, then loads 
    the Tokenizer, Model, and Label Encoder.
    
    Includes a CRITICAL FIX for 'Unrecognized model' error by forcing 
    architecture type to 'bert'.
    """
    # A) Verify existence of model weights; download from Cloud if missing
    if not os.path.exists(MODEL_PATH_FULL):
        with st.spinner('Downloading model weights from Cloud (approx. 500MB)... Please wait ⏳'):
            try:
                # Ensure the directory exists
                os.makedirs(BASE_DIR, exist_ok=True)
                # Download using gdown
                gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH_FULL, quiet=False)
                st.success("Model weights downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download files from Google Drive: {e}")
                return None, None, None

    # B) Load Model Architecture, Tokenizer, and Encoders
    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
        
        # 2. Load Configuration and Force Architecture Type
        # CRITICAL FIX: The config.json is missing the "model_type" key.
        # We explicitly set it to "bert" to resolve the "Unrecognized model" error.
        config = AutoConfig.from_pretrained(BASE_DIR)
        
        # Force model type if undefined
        if not hasattr(config, 'model_type') or config.model_type is None:
            config.model_type = 'bert'
            
        # 3. Load Model with the corrected configuration
        model = AutoModelForSequenceClassification.from_pretrained(BASE_DIR, config=config)
        
        # 4. Load Label Encoder to map indices back to emotion names
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
            
        return tokenizer, model, label_encoder

    except Exception as e:
        st.error(f"Critical error loading model components: {e}")
        return None, None, None

# Initialize resources on app startup
tokenizer, model, label_encoder = load_prediction_model()

# ============================================================================
# 4. INFERENCE LOGIC
# ============================================================================

def predict_emotion(text):
    """
    Performs preprocessing, tokenization, and inference on the input text.
    Returns: Predicted Label, Confidence Score, and Probability Distribution.
    """
    if not text:
        return None
    
    # 1. Preprocess and tokenize input text
    # Max length set to 128 to optimize performance/memory
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # 2. Perform Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply Softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=1).detach().numpy()[0]
    
    # 3. Decode Predictions
    pred_idx = probs.argmax()
    confidence = probs[pred_idx]
    emotion_name = label_encoder.inverse_transform([pred_idx])[0]
    
    return emotion_name, confidence, probs

# ============================================================================
# 5. MAIN USER INTERFACE (UI)
# ============================================================================

st.title("🤖 محلل المشاعر العربي الذكي")
st.markdown("قم بكتابة جملة باللغة العربية وسيقوم الذكاء الاصطناعي بتحليل المشاعر بدقة.")

# Text Input Area
text_input = st.text_area("أدخل النص هنا:", height=150, placeholder="مثال: أنا سعيد جداً اليوم لأنني أنجزت الكثير من العمل...")

# Analyze Button
if st.button("تحليل المشاعر 🔍"):
    if not tokenizer or not model:
        st.error("Model failed to load. Please check logs.")
    elif not text_input.strip():
        st.warning("الرجاء كتابة نص أولاً.")
    else:
        with st.spinner('جاري التحليل...'):
            emotion, confidence, all_probs = predict_emotion(text_input)
            
            # Emotion-to-Emoji Mapping
            emoji_map = {
                'joy': '😊', 'happy': '😊', 'happiness': '😊',
                'sadness': '😢', 'sad': '😢',
                'anger': '😡', 'angry': '😡',
                'fear': '😨',
                'love': '❤️',
                'surprise': '😲',
                'neutral': '😐'
            }
            # Note: Ensure keys match the classes in your LabelEncoder
            
            # Dynamic Background Color based on Emotion Category
            bg_color = "#e8f5e9" if emotion in ['joy', 'love', 'happy'] else "#ffebee" if emotion in ['anger', 'fear', 'sadness'] else "#f3f4f6"
            current_emoji = emoji_map.get(emotion, '🤔')

            # Display Primary Result Card
            st.markdown(f"""
            <div class="result-card" style="background-color: {bg_color};">
                <h1 style="font-size: 60px; margin: 0;">{current_emoji}</h1>
                <h2 style="color: #333; margin: 10px 0;">{emotion}</h2>
                <h4 style="color: #555;">دقة التوقع: {confidence:.1%}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Display Detailed Probabilities
            st.markdown("### 📊 تفاصيل التحليل:")
            
            # Sort and display classes by probability
            class_names = label_encoder.classes_
            sorted_indices = all_probs.argsort()[::-1] # Descending order
            
            for i in sorted_indices:
                cls_name = class_names[i]
                prob = all_probs[i]
                # Filter out very low probabilities (< 1%) for cleaner UI
                if prob > 0.01: 
                    st.write(f"**{cls_name}**: {prob:.1%}")
                    st.progress(float(prob))

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>تم التطوير باستخدام AraBERT و Streamlit</div>", unsafe_allow_html=True)
