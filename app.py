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
from pathlib import Path
from sklearn.utils.validation import check_is_fitted
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def load_models_and_tokenizers():
    try:
        base_dir = Path(__file__).parent
        saved_models_dir = base_dir / 'saved_models'
        
        # Check if directory exists
        if not saved_models_dir.exists():
            st.error(f"Model directory {saved_models_dir} does not exist.")
            logger.error(f"Model directory {saved_models_dir} not found.")
            return None, None, None, None, None
        
        # Define model paths
        paths = {
            'naive_bayes': saved_models_dir / 'naive_bayes_model.pkl',
            'tfidf_vectorizer': saved_models_dir / 'tfidf_vectorizer.pkl',
            'gru_model': saved_models_dir / 'gru_model.h5',
            'tokenizer': saved_models_dir / 'tokenizer.pkl',
            'label_encoder': saved_models_dir / 'label_encoder.pkl'
        }
        
        # Check if all files exist
        for name, path in paths.items():
            if not path.exists():
                st.error(f"File {name} not found at {path}.")
                logger.error(f"File {name} not found at {path}.")
                return None, None, None, None, None
        
        # Load Naive Bayes model
        with open(paths['naive_bayes'], 'rb') as f:
            ml_model = pickle.load(f)
            logger.info("Naive Bayes model loaded successfully.")
        
        # Load and verify TF-IDF vectorizer
        with open(paths['tfidf_vectorizer'], 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        # Verify TF-IDF vectorizer is fitted
        try:
            check_is_fitted(tfidf_vectorizer, attributes=['vocabulary_', 'idf_'])
            logger.info("TF-IDF vectorizer is fitted.")
            st.write("TF-IDF vectorizer loaded and fitted.")
        except Exception as e:
            st.error(
                f"TF-IDF vectorizer is not fitted: {str(e)}\n"
                "Please re-fit the vectorizer with your training data and re-save it. "
                "See instructions in the logs or run the provided fitting script."
            )
            logger.error(f"TF-IDF vectorizer is not fitted: {str(e)}")
            return None, None, None, None, None
        
        # Load GRU model
        try:
            dl_model = load_model(paths['gru_model'])
            logger.info("GRU model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load GRU model: {str(e)}")
            logger.error(f"Failed to load GRU model: {str(e)}")
            return None, None, None, None, None
        
        # Load tokenizer
        with open(paths['tokenizer'], 'rb') as f:
            tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully.")
        
        # Load label encoder
        with open(paths['label_encoder'], 'rb') as f:
            label_encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully.")
        
        return ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder
    
    except Exception as e:
        st.error(f"Unexpected error loading resources: {str(e)}")
        logger.error(f"Unexpected error loading resources: {str(e)}")
        return None, None, None, None, None

# Placeholder function to fit TF-IDF vectorizer (to be used separately)
def fit_and_save_tfidf_vectorizer(training_data, save_path='saved_models/tfidf_vectorizer.pkl'):
    """
    Fit a TF-IDF vectorizer with training data and save it.
    Args:
        training_data: List or array of text documents to fit the vectorizer.
        save_path: Path to save the fitted vectorizer.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(training_data)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f"TF-IDF vectorizer fitted and saved to {save_path}.")
        return vectorizer
    except Exception as e:
        logger.error(f"Failed to fit and save TF-IDF vectorizer: {str(e)}")
        raise

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
    try:
        cleaned_text = clean_text(text)
        logger.info(f"Cleaned text: {cleaned_text}")
        # Verify vectorizer is fitted before transforming
        check_is_fitted(vectorizer, attributes=['vocabulary_', 'idf_'])
        vectorized_text = vectorizer.transform([cleaned_text])
        sentiment_result = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        logger.info(f"ML prediction successful: {sentiment_result}")
        return sentiment_result, probabilities
    except Exception as e:
        st.error(f"Error in ML prediction: {str(e)}")
        logger.error(f"Error in ML prediction: {str(e)}")
        raise

def predict_with_dl_model(text, model, tokenizer, label_encoder, max_length=100):
    """Predict sentiment using the GRU model."""
    try:
        cleaned_text = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        probabilities = model.predict(padded_sequence, verbose=0)[0]
        sentiment_idx = np.argmax(probabilities)
        sentiment_result = label_encoder.inverse_transform([sentiment_idx])[0]
        logger.info(f"DL prediction successful: {sentiment_result}")
        return sentiment_result, probabilities
    except Exception as e:
        st.error(f"Error in DL prediction: {str(e)}")
        logger.error(f"Error in DL prediction: {str(e)}")
        raise

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
        st.error(
            "Failed to load resources. Please check model files.\n"
            "If the TF-IDF vectorizer is not fitted, re-fit it with your training data using the provided instructions."
        )
        logger.error("Failed to load all required resources.")
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
                try:
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
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    logger.error(f"Prediction failed: {str(e)}")
        
        display_history()
    
    with tab3:
        st.subheader("Model Information")
        if model_choice == "Machine Learning (Naive Bayes)":
            st.write("**Model:** Naive Bayes (MultinomialNB)")
            st.write("**Features:** TF-IDF (Term Frequency-Inverse Document Frequency)")
            st.write("**Description:** Uses a Multinomial Naive Bayes classifier with TF-IDIF features to classify review sentiment.")
        else:
            st.write("**Model:** Gated Recurrent Unit (GRU)")
            st.write("**Features:** FastText Word Embeddings")
            st.write("**Description:** Uses a deep learning model with GRU layers for sentiment analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Gojek Sentiment Analysis Tool")

if __name__ == "__main__":
    run_app()
