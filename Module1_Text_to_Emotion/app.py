import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
import nltk

warnings.filterwarnings('ignore')

# Download NLTK resources explicitly to ensure they exist in cloud environment
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🧠 Emotional Chatbot - Module 1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Emotional Chatbot Module 1 - Arabic NLP System\nVersion 1.0.0"
    }
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
    }
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# PREPROCESSING FUNCTION (MUST BE ADDED)
# ============================================================================

def preprocess_arabic_text(text):
    """
    Preprocess Arabic text same as training phase
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove URLs/Emails
    text = re.sub(r'http\S+|www\S+|email', '', text)
    # Remove mentions/hashtags
    text = re.sub(r'@\w+|#', '', text)
    # Keep only Arabic letters and spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F]', '', text)
    # Remove repeated chars
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    return text

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and vectorizers"""
    try:
        # Define base path (works for both local and cloud)
        base_path = Path(__file__).parent / "models" if Path(__file__).parent.name == "Module1_Text_to_Emotion" else Path("Module1_Text_to_Emotion/models")
        
        # Fallback if paths are different in deployment structure
        if not base_path.exists():
            base_path = Path("models")

        models = {
            'emotion_model': pickle.load(open(base_path / 'emotions_models/emotion_model_v1.pkl', 'rb')),
            'emotion_vectorizer': pickle.load(open(base_path / 'emotions_models/emotion_tfidf_v1.pkl', 'rb')),
            'emotion_encoder': pickle.load(open(base_path / 'emotions_models/emotion_encoder_v1.pkl', 'rb')),
            'dialect_model': pickle.load(open(base_path / 'dialect_models/dialect_model_v1.pkl', 'rb')),
            'dialect_vectorizer': pickle.load(open(base_path / 'dialect_models/dialect_tfidf_v1.pkl', 'rb')),
            'dialect_encoder': pickle.load(open(base_path / 'dialect_models/dialect_encoder_v1.pkl', 'rb'))
        }
        return models, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, False

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_emotion(text, models):
    try:
        # CRITICAL: Clean text before prediction
        cleaned_text = preprocess_arabic_text(text)
        
        features = models['emotion_vectorizer'].transform([cleaned_text])
        pred = models['emotion_model'].predict(features)[0]
        proba = models['emotion_model'].predict_proba(features)[0]
        emotion_label = models['emotion_encoder'].inverse_transform([pred])[0]
        confidence = proba.max()
        
        top_3_indices = np.argsort(proba)[-3:][::-1]
        top_3_emotions = models['emotion_encoder'].inverse_transform(top_3_indices)
        top_3_proba = proba[top_3_indices]
        
        return {
            'emotion': emotion_label,
            'confidence': confidence,
            'top_3': list(zip(top_3_emotions, top_3_proba)),
            'all_proba': dict(zip(models['emotion_encoder'].classes_, proba)),
            'cleaned_text': cleaned_text
        }
    except Exception as e:
        st.error(f"Error in emotion prediction: {str(e)}")
        return None

def predict_dialect(text, models):
    try:
        # CRITICAL: Clean text before prediction
        cleaned_text = preprocess_arabic_text(text)
        
        features = models['dialect_vectorizer'].transform([cleaned_text])
        pred = models['dialect_model'].predict(features)[0]
        decision = models['dialect_model'].decision_function(features)[0]
        dialect_label = models['dialect_encoder'].inverse_transform([pred])[0]
        
        confidence = min(1.0, 0.5 + 0.5 * abs(decision.max()) / 10)
        
        return {
            'dialect': dialect_label,
            'confidence': confidence
        }
    except Exception as e:
        st.error(f"Error in dialect prediction: {str(e)}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_emotion_emoji(emotion):
    emoji_map = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 
                 'surprise': '😮', 'disgust': '🤢', 'neutral': '😐'}
    return emoji_map.get(emotion.lower(), '🤷')

def plot_emotion_distribution(proba_dict):
    emotions = list(proba_dict.keys())
    probabilities = list(proba_dict.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(emotions))
    bars = ax.barh(emotions, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title('Emotion Probability Distribution')
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

models, loaded = load_models()

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🎯 Live Prediction", "📊 Model Performance"])

if loaded:
    st.sidebar.success("✅ Models Loaded Successfully")
    st.sidebar.markdown("---")
    st.sidebar.metric("Emotion Accuracy", "81%")
    st.sidebar.metric("Dialect Accuracy", "86%")

# --- HOME PAGE ---
if page == "🏠 Home":
    st.title("🧠 Emotional Chatbot - Module 1")
    st.markdown("### Arabic Text Understanding Engine")
    st.info("This module detects **Emotions** and **Dialects** from Arabic text using Machine Learning.")

# --- LIVE PREDICTION PAGE ---
elif page == "🎯 Live Prediction":
    st.title("🎯 Live Prediction")
    
    user_text = st.text_area("Enter Arabic Text:", height=100, placeholder="اكتب مشاعرك هنا...")
    
    if st.button("Analyze", type="primary"):
        if user_text and loaded:
            with st.spinner("Analyzing..."):
                e_res = predict_emotion(user_text, models)
                d_res = predict_dialect(user_text, models)
                
            if e_res and d_res:
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    emoji = get_emotion_emoji(e_res['emotion'])
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1>{emoji}</h1>
                        <h3>{e_res['emotion'].title()}</h3>
                        <p>Confidence: {e_res['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h1>🗺️</h1>
                        <h3>{d_res['dialect']}</h3>
                        <p>Confidence: {d_res['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📊 Detailed Analysis")
                st.pyplot(plot_emotion_distribution(e_res['all_proba']))
                
                with st.expander("See Processed Text"):
                    st.code(e_res['cleaned_text'])

# --- PERFORMANCE PAGE ---
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Training Samples", "13,000+")
    col2.metric("Features (TF-IDF)", "5,000")
    st.dataframe(pd.DataFrame({
        'Model': ['Emotion (Random Forest)', 'Dialect (SVM)'],
        'Accuracy': ['81%', '86%']
    }), hide_index=True, use_container_width=True)
