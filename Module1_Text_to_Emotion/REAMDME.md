# <div align="center">🧠 Emotional Chatbot - Module 1



<sub>Arabic Text Understanding Engine</sub></div>

<div align="center">

</div>

---

## 📋 Project Overview

**Module 1** is the foundational component of the **Emotional Chatbot** project, designed to understand Arabic text with deep cultural and emotional intelligence. This module provides:

| Feature | Description | Accuracy |
| --- | --- | --- |
| **🎯 Emotion Detection** | Classifies **7 distinct emotional states** from Arabic text. | **81%** |
| **🗺️ Dialect Recognition** | Identifies major Arabic dialects (Egyptian, Levantine, Gulf, North African). | **86%** |
| **⚡ Real-time Processing** | Interactive Streamlit chatbot for live testing and model evaluation. | **~5ms** |

---

## 🚀 Quick Start

> [!IMPORTANT]
> Ensure you have **Python 3.8+** installed before proceeding.

### 1. Installation

**Clone the repository:**

```bash
git clone https://github.com/eslamalsaeed72-droid/Emotional-Arabic-Chatbot/new/main/Module1_Text_to_Emotion.git
cd emotional-chatbot-module-1

```

**Install dependencies:**

```bash
pip install -r requirements.txt

```

### 2. Run the Interactive Demo

```bash
streamlit run app.py

```

> **Access the Chatbot:** A new tab will open in your browser. Start testing emotions and dialects in real-time! ✨

---

## 📁 Project Structure

```text
module1/
├── Module1_Text_to_Emotion.ipynb    # 📓 Complete training & evaluation pipeline
├── requirements.txt                 # 📦 Python dependencies
├── app.py                           # 🚀 Streamlit interactive chatbot interface
├── models/
│   ├── emotions_models/             # 🧠 3 trained emotion detection models
│   │   ├── emotion_model_v1.pkl
│   │   ├── emotion_tfidf_v1.pkl
│   │   └── emotion_encoder_v1.pkl
│   └── dialect_models/              # 🌍 3 trained dialect recognition models
│       ├── dialect_model_v1.pkl
│       ├── dialect_tfidf_v1.pkl
│       └── dialect_encoder_v1.pkl
├── data/
│   ├── ArSAS.csv                    # Arabic Sarcasm Analysis Dataset
│   ├── AJGT.xlsx                    # Arabic Dialect Dataset
│   └── QADI.csv                     # Qatar Arabic Dialect Institute data
└── outputs/
    └── visualizations/              # 📊 18 publication-ready charts

```

---

## 🎯 Core Features

### 1. Emotion Detection Model

* **Algorithm:** Random Forest Classifier (200 trees, class-balanced)
* **Features:** 5,000 TF-IDF features from Arabic text
* **Classes (7 Emotions):**
* 😊 Joy
* 😢 Sadness
* 😠 Anger
* 😨 Fear
* 😮 Surprise
* 🤢 Disgust
* 😐 Neutral



**Performance Metrics:**

> **Accuracy:** 81% | **Precision:** 0.81 | **Recall:** 0.81 | **F1-Score:** 0.81

### 2. Dialect Recognition Model

* **Algorithm:** Linear SVM (SGDClassifier, Hinge Loss)
* **Features:** 5,000 TF-IDF features from Arabic text
* **Dialects (4 Regions):**
* 🏛️ **Egyptian** (MSA-influenced)
* 🌙 **Levantine** (Syrian, Lebanese, Palestinian)
* ⛽ **Gulf** (Saudi, UAE, Kuwaiti)
* 🌍 **North African** (Moroccan, Tunisian)



**Performance Metrics:**

> **Accuracy:** 86% | **Precision:** 0.86 | **Recall:** 0.86 | **F1-Score:** 0.86

### 3. Data Processing Pipeline

* **Text Cleaning:** Diacritics removal, normalization, tokenization.
* **Vectorization:** TF-IDF (5,000 features).
* **Preprocessing:** Arabic-specific stemming and lemmatization.
* **Dataset Size:** **13,000+ samples** from 3 authoritative sources.

---

## 🤖 Interactive Chatbot (Streamlit)

The `app.py` provides an interactive interface to test both models.

### ✨ Features

* ✅ **Real-time Emotion Detection:** Input any Arabic text & get emotion classification with confidence score.
* ✅ **Dialect Recognition:** Identify which Arabic dialect/region the text belongs to.
* ✅ **Model Comparison Dashboard:** Side-by-side performance metrics & confusion matrices.
* ✅ **Live Testing Interface:** Predefined test samples & custom text input.

### How to Use

1. Run `streamlit run app.py`
2. Select a test sample or enter custom Arabic text.
3. Click **"Predict"** to see real-time results.
4. Review confidence scores and feature importance.

---

## 📊 Comprehensive Evaluation

**Generated Visualizations (18 Total):** All visualizations saved in high resolution (300 DPI).

| Emotion Detection (10 Charts) | Dialect Recognition (8 Charts) |
| --- | --- |
| Class distribution (train vs test) | Class distribution (train vs test) |
| Confusion matrix (normalized) | Confusion matrix (normalized) |
| Per-class performance metrics | Per-class performance metrics |
| Accuracy by emotion class | Accuracy by dialect region |
| Prediction confidence distribution | Correct vs incorrect predictions |
| Top 20 important features | Test set class support |
| *...and more* | *...and more* |

---

## 🔬 Model Architecture Details

### Emotion Detection Model

```python
RandomForestClassifier(
    n_estimators=200,           # 200 decision trees
    max_depth=15,               # Depth limitation (prevent overfitting)
    min_samples_split=5,        # Minimum samples to split nodes
    min_samples_leaf=2,         # Minimum samples at leaf nodes
    max_features='sqrt',        # Feature subsampling
    class_weight='balanced',    # Handle imbalanced classes
    random_state=42,            # Reproducibility
    n_jobs=-1                   # Parallel processing
)

```

### Dialect Recognition Model

```python
SGDClassifier(
    loss='hinge',               # SVM loss function
    penalty='l2',               # L2 regularization
    alpha=1e-4,                 # Regularization strength
    class_weight='balanced',    # Handle imbalanced dialects
    max_iter=50,                # SGD iterations
    random_state=42,            # Reproducibility
    n_jobs=-1                   # Parallel processing
)

```

---

## 📈 Key Findings

### What Works Well ✅

* **Egyptian dialect** is consistently recognized (highest accuracy).
* **Joy emotion** detection is most reliable (clear linguistic patterns).
* **Model generalization** is solid across dialects.
* **TF-IDF features** capture meaningful emotion/dialect indicators.

### Areas for Improvement 🔄

* Mixed dialects (Levantine + Gulf code-switching) can cause confusion.
* Subtle emotions (Surprise vs Joy) need better discrimination.
* Sarcasm detection is challenging (requires context understanding).

### Recommendations 💡

1. **Module 2:** Add deep learning (BERT, AraBERT) for context understanding.
2. **Module 3:** Implement multi-emotion detection (mixed feelings).
3. **Module 4:** Deploy as web/mobile app with real-time feedback.

---

## 🛠️ Technical Stack

| Component | Technology |
| --- | --- |
| **ML Framework** | scikit-learn |
| **Text Processing** | NLTK, Arabic-specific stemming |
| **Vectorization** | TF-IDF (scikit-learn) |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Interactive UI** | Streamlit |
| **Model Serialization** | Pickle |
| **Language** | Python 3.8+ |

---

## 🚀 Usage Examples

### Example 1: Direct Model Usage

```python
from pickle import load

# Load models
emotion_model = load(open('models/emotions_models/emotion_model_v1.pkl', 'rb'))
emotion_vectorizer = load(open('models/emotions_models/emotion_tfidf_v1.pkl', 'rb'))

# Process text
text = "أنا سعيد جداً بهذا الخبر"
features = emotion_vectorizer.transform([text])

# Predict
emotion_pred = emotion_model.predict(features)
confidence = emotion_model.predict_proba(features).max()

print(f"Emotion: {emotion_pred}, Confidence: {confidence:.2%}")

```

### Example 2: Using Streamlit Interface

```bash
streamlit run app.py
# Then navigate to http://localhost:8501

```

---

## 📊 Performance Benchmarks

| Metric | Emotion Detection | Dialect Recognition |
| --- | --- | --- |
| **Accuracy** | 81% | 86% |
| **Precision** | 0.81 | 0.86 |
| **Recall** | 0.81 | 0.86 |
| **F1-Score** | 0.81 | 0.86 |
| **Training Time** | ~45 min | ~15 min |
| **Inference Time** | ~5ms | ~3ms |

---

## 🔐 Data Privacy & Ethics

<div align="center">

| **Privacy First** 🔒 | **Ethical AI** ⚖️ | **Open Source** 📖 |
| --- | --- | --- |
| No user data stored locally or remotely. | Balanced training across emotion classes. | Code available on GitHub. |
| All processing happens on-device. | Regional representation in dialect data. | Community-driven development. |
| No telemetry or tracking. | Transparent model outputs. | Reproducible results. |

</div>

---

## 🗓️ Roadmap

* [x] **Module 1 (COMPLETE):** Emotion detection (81%), Dialect recognition (86%), Interactive Chatbot.
* [ ] **Module 2 (IN PROGRESS):** Advanced emotional pattern recognition, Mixed emotion detection.
* [ ] **Module 3 (PLANNED):** Intelligent response generation, Context-aware conversations.
* [ ] **Module 4 (PLANNED):** Web platform deployment, Mobile app (iOS/Android).

---

## 🎯 Key Metrics & Statistics

* 📦 **Total Training Samples:** 13,000+
* 🧮 **Feature Dimensions:** 5,000 TF-IDF features
* 🎭 **Emotion Classes:** 7 distinct emotional states
* 🌍 **Dialect Groups:** 4 major Arabic-speaking regions
* 🤖 **Model Ensemble Size:** 6 trained models
* ⚡ **Inference Speed:** 5-8ms per prediction

---

## 📖 Citation

If you use this project in research, please cite:

```text
@project{emotional_chatbot_2026,
  title={Emotional Chatbot: AI That Truly Understands You},
  author={Eslam Alsaeed},
  year={2026},
  url={https://github.com/eslamalsaeed72-droid/Emotional-Arabic-Chatbot/blob/main/Module1_Text_to_Emotion}
}

```

---

## ⚖️ Legal Notice

> [!WARNING]
> This project is intended for **research and educational purposes**. Use responsibly and ethically. Do not use for harassment, discrimination, or harmful purposes.

<div align="center">

**Version:** 1.0.0 | **Last Updated:** January 3, 2026 | **Status:** ✅ Production Ready

*Building AI that listens. Building AI that cares. 💜*

</div>
