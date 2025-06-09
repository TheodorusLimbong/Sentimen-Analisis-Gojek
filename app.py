import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Section 1: Imports and Setup
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Gojek Sentiment Analyzer",
    page_icon=":bar_chart:",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Section 2: Resource Loading
import os

import pickle
import os

# Path to the vectorizer file (adjust as needed)
vectorizer_path = os.path.join('saved_models', 'tfidf_vectorizer.pkl')

# Load the vectorizer
with open(vectorizer_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Check if the vectorizer is fitted
print("Vocabulary size:", len(tfidf_vectorizer.vocabulary_))
print("IDF vector:", tfidf_vectorizer.idf_)
# Section 2: Resource Loading
@st.cache_resource
def load_models_and_tokenizers():
    """Load machine learning and deep learning models along with their vectorizers/tokenizers."""
    try:
        # Define the path to the saved_models folder relative to app.py
        base_path = os.path.join(os.path.dirname(__file__), 'saved_models')

        # Load ML model (Naive Bayes)
        with open(os.path.join(base_path, 'naive_bayes_model.pkl'), 'rb') as f:
            ml_model = pickle.load(f)
        
        # Load TF-IDF vectorizer
        with open(os.path.join(base_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Load deep learning model (GRU) using h5
        dl_model = load_model(os.path.join(base_path, 'gru_model.h5'))
        
        # Load tokenizer
        with open(os.path.join(base_path, 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(base_path, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        print("All resources loaded successfully")
        return ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None
# Section 3: Text Preprocessing
def clean_text(text):
    """Clean and preprocess input text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase, remove punctuation, numbers, and extra spaces
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# Section 4: Prediction Functions
def predict_with_ml_model(text, model, vectorizer):
    """Predict sentiment using the Naive Bayes model."""
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    sentiment_result = model.predict(vectorized_text)[0]
    
    # Get probabilities for Naive Bayes
    probabilities = model.predict_proba(vectorized_text)[0]
    
    return sentiment_result, probabilities

def predict_with_dl_model(text, model, tokenizer, label_encoder, max_length=100):
    """Predict sentiment using the GRU model."""
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    probabilities = model.predict(padded_sequence, verbose=0)[0]
    sentiment_idx = np.argmax(probabilities)
    sentiment_result = label_encoder.inverse_transform([sentiment_idx])[0]
    return sentiment_result, probabilities

# Section 5: UI Components
def display_prediction(sentiment, probabilities, model_type, classes):
    """Display sentiment prediction and confidence scores."""
    sentiment_colors = {'positive': '#28a745', 'neutral': '#007bff', 'negative': '#dc3545'}
    st.markdown(
        f"**Sentiment:** <span style='color:{sentiment_colors[sentiment]}'>{sentiment.upper()}</span>",
        unsafe_allow_html=True
    )
    
    st.write("**Confidence Scores:**")
    for label, prob in zip(classes, probabilities):
        st.progress(float(prob))
        st.write(f"{label.capitalize()}: {prob*100:.2f}%")
    
    # Add to prediction history
    st.session_state.prediction_history.append({
        'text': st.session_state.user_review,
        'sentiment': sentiment,
        'model': model_type,
        'probabilities': {label: float(prob) for label, prob in zip(classes, probabilities)}
    })
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history.pop(0)

def display_history():
    """Display recent prediction history."""
    if st.session_state.prediction_history:
        st.subheader("Recent Predictions")
        for idx, entry in enumerate(st.session_state.prediction_history):
            with st.expander(f"Prediction {idx+1}: {entry['sentiment'].capitalize()} ({entry['model']})"):
                st.write(f"**Review:** {entry['text']}")
                st.write(f"**Sentiment:** {entry['sentiment'].capitalize()}")
                st.write(f"**Model:** {entry['model']}")
                st.write("**Confidence Scores:**")
                for label, prob in entry['probabilities'].items():
                    st.write(f"{label.capitalize()}: {prob*100:.2f}%")

# Section 6: Main Application
def run_app():
    """Main function to run the Streamlit application."""
    # Load resources
    ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder = load_models_and_tokenizers()
    
    if not all([ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder]):
        st.error("Failed to load resources. Please check model files.")
        return
    
    # App header
    st.title("Gojek Sentiment Analyzer")
    st.markdown("Analyze the sentiment of Gojek app reviews using machine learning or deep learning models.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Machine Learning (Naive Bayes)", "Deep Learning (GRU)"]
    )
    
    # Tabs for input, results, and model info
    tab1, tab2, tab3 = st.tabs(["Input", "Results", "Model Info"])
    
    with tab1:
        st.subheader("Enter Your Review")
        st.session_state.user_review = st.text_area(
            "Type your review here:", 
            value="Aplikasi Gojek sangat membantu dan cepat.", 
            height=120
        )
        
        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
    
    with tab2:
        if st.session_state.get('analyze_clicked', False):
            with st.spinner("Processing your review..."):
                if model_choice == "Machine Learning (Naive Bayes)":
                    sentiment_result, probabilities = predict_with_ml_model(
                        st.session_state.user_review, ml_model, tfidf_vectorizer
                    )
                    classes = ml_model.classes_
                else:
                    sentiment_result, probabilities = predict_with_dl_model(
                        st.session_state.user_review, dl_model, tokenizer, label_encoder
                    )
                    classes = label_encoder.classes_
                
                display_prediction(sentiment_result, probabilities, model_choice, classes)
        
        display_history()
    
    with tab3:
        st.subheader("Model Information")
        if model_choice == "Machine Learning (Naive Bayes)":
            st.write("**Model:** Naive Bayes (MultinomialNB)")
            st.write("**Features:** TF-IDF (Term Frequency-Inverse Document Frequency)")
            st.write("**Description:** Uses a Multinomial Naive Bayes classifier with TF-IDF features to classify review sentiment.")
        else:
            st.write("**Model:** Gated Recurrent Unit (GRU)")
            st.write("**Features:** FastText Word Embeddings")
            st.write("**Description:** Uses a deep learning model with GRU layers for sentiment analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Gojek Sentiment Analysis Tool")

if __name__ == "__main__":
    run_app()
