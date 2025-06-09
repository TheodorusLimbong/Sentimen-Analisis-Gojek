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
from sklearn.exceptions import NotFittedError

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
    """Load all models and tokenizers with comprehensive error handling."""
    try:
        base_dir = Path(__file__).parent
        saved_models_dir = base_dir / 'saved_models'
        
        # Check if directory exists
        if not saved_models_dir.exists():
            st.error(f"Model directory {saved_models_dir} does not exist.")
            st.info("Please ensure the 'saved_models' directory exists in the same folder as this script.")
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
        missing_files = []
        for name, path in paths.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            st.error("Missing model files:")
            for file in missing_files:
                st.error(f"• {file}")
            return None, None, None, None, None
        
        # Load models with individual error handling
        try:
            # Load Naive Bayes model
            with open(paths['naive_bayes'], 'rb') as f:
                ml_model = pickle.load(f)
            st.success("✓ Naive Bayes model loaded")
        except Exception as e:
            st.error(f"Failed to load Naive Bayes model: {str(e)}")
            return None, None, None, None, None
        
        try:
            # Load and verify TF-IDF vectorizer
            with open(paths['tfidf_vectorizer'], 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            
            # Check if vectorizer is fitted
            try:
                check_is_fitted(tfidf_vectorizer)
                st.success("✓ TF-IDF vectorizer loaded and fitted")
            except NotFittedError:
                st.error("❌ TF-IDF vectorizer is not fitted. Please ensure you saved a fitted vectorizer.")
                st.info("The vectorizer needs to be fitted on training data before saving.")
                return None, None, None, None, None
                
        except Exception as e:
            st.error(f"Failed to load TF-IDF vectorizer: {str(e)}")
            return None, None, None, None, None
        
        try:
            # Load GRU model
            dl_model = load_model(paths['gru_model'])
            st.success("✓ GRU model loaded")
        except Exception as e:
            st.error(f"Failed to load GRU model: {str(e)}")
            return None, None, None, None, None
        
        try:
            # Load tokenizer
            with open(paths['tokenizer'], 'rb') as f:
                tokenizer = pickle.load(f)
            st.success("✓ Tokenizer loaded")
        except Exception as e:
            st.error(f"Failed to load tokenizer: {str(e)}")
            return None, None, None, None, None
        
        try:
            # Load label encoder
            with open(paths['label_encoder'], 'rb') as f:
                label_encoder = pickle.load(f)
            st.success("✓ Label encoder loaded")
        except Exception as e:
            st.error(f"Failed to load label encoder: {str(e)}")
            return None, None, None, None, None
        
        # Verify vectorizer has required attributes
        required_attrs = ['vocabulary_', 'idf_']
        missing_attrs = [attr for attr in required_attrs if not hasattr(tfidf_vectorizer, attr)]
        if missing_attrs:
            st.error(f"TF-IDF vectorizer missing required attributes: {missing_attrs}")
            return None, None, None, None, None
        
        return ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder
    
    except Exception as e:
        st.error(f"Unexpected error loading resources: {str(e)}")
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
    try:
        # Verify vectorizer is fitted before using
        check_is_fitted(vectorizer)
        
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            st.warning("Input text is empty after cleaning. Please provide meaningful text.")
            return None, None
        
        # Transform text using fitted vectorizer
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Make prediction
        sentiment_result = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        return sentiment_result, probabilities
        
    except NotFittedError:
        st.error("TF-IDF vectorizer is not fitted. Cannot make predictions.")
        return None, None
    except Exception as e:
        st.error(f"Error in ML prediction: {str(e)}")
        return None, None

def predict_with_dl_model(text, model, tokenizer, label_encoder, max_length=100):
    """Predict sentiment using the GRU model."""
    try:
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            st.warning("Input text is empty after cleaning. Please provide meaningful text.")
            return None, None
        
        # Convert text to sequences
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        
        if not sequence or not sequence[0]:
            st.warning("Text could not be tokenized. The input might contain only unknown words.")
            return None, None
        
        # Pad sequences
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        # Make prediction
        probabilities = model.predict(padded_sequence, verbose=0)[0]
        sentiment_idx = np.argmax(probabilities)
        sentiment_result = label_encoder.inverse_transform([sentiment_idx])[0]
        
        return sentiment_result, probabilities
        
    except Exception as e:
        st.error(f"Error in DL prediction: {str(e)}")
        return None, None

# Section 5: UI Components
def display_prediction(sentiment, probabilities, model_type, classes):
    """Display sentiment prediction and confidence scores."""
    if sentiment is None or probabilities is None:
        st.error("Failed to make prediction. Please check your input and try again.")
        return
    
    sentiment_colors = {'positive': '#28a745', 'neutral': '#007bff', 'negative': '#dc3545'}
    color = sentiment_colors.get(sentiment, '#6c757d')
    
    st.markdown(
        f"**Sentiment:** <span style='color:{color}'>{sentiment.upper()}</span>",
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
        for idx, entry in enumerate(reversed(st.session_state.prediction_history)):
            with st.expander(f"Prediction {len(st.session_state.prediction_history)-idx}: {entry['sentiment'].capitalize()} ({entry['model']})"):
                st.write(f"**Review:** {entry['text']}")
                st.write(f"**Sentiment:** {entry['sentiment'].capitalize()}")
                st.write(f"**Model:** {entry['model']}")
                st.write("**Confidence Scores:**")
                for label, prob in entry['probabilities'].items():
                    st.write(f"{label.capitalize()}: {prob*100:.2f}%")

# Section 6: Main Application
def run_app():
    """Main function to run the Streamlit application."""
    # App header
    st.title("🚗 Gojek Sentiment Analyzer")
    st.markdown("Analyze the sentiment of Gojek app reviews using machine learning or deep learning models.")
    
    # Load resources
    with st.spinner("Loading models..."):
        ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder = load_models_and_tokenizers()
    
    if not all([ml_model, tfidf_vectorizer, dl_model, tokenizer, label_encoder]):
        st.error("❌ Failed to load required resources. Please check model files and try again.")
        st.info("**Troubleshooting steps:**")
        st.info("1. Ensure all model files exist in the 'saved_models' directory")
        st.info("2. Check that the TF-IDF vectorizer was fitted before saving")
        st.info("3. Verify all pickle files are not corrupted")
        return
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Machine Learning (Naive Bayes)", "Deep Learning (GRU)"],
        help="Select which model to use for sentiment analysis"
    )
    
    # Display model status
    st.sidebar.markdown("### Model Status")
    st.sidebar.success("✅ All models loaded successfully")
    
    # Tabs for input, results, and model info
    tab1, tab2, tab3 = st.tabs(["📝 Input", "📊 Results", "ℹ️ Model Info"])
    
    with tab1:
        st.subheader("Enter Your Review")
        st.session_state.user_review = st.text_area(
            "Type your review here:", 
            value="Aplikasi Gojek sangat membantu dan cepat.",
            height=120,
            help="Enter your Gojek app review in Indonesian or English"
        )
        
        # Input validation
        if len(st.session_state.user_review.strip()) < 5:
            st.warning("⚠️ Please enter a review with at least 5 characters for meaningful analysis.")
        
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
        
        if analyze_button:
            if len(st.session_state.user_review.strip()) < 5:
                st.error("Please enter a longer review for analysis.")
            else:
                st.session_state.analyze_clicked = True
    
    with tab2:
        if st.session_state.get('analyze_clicked', False):
            with st.spinner("🔄 Processing your review..."):
                try:
                    if model_choice == "Machine Learning (Naive Bayes)":
                        sentiment_result, probabilities = predict_with_ml_model(
                            st.session_state.user_review, ml_model, tfidf_vectorizer
                        )
                        classes = ml_model.classes_ if sentiment_result is not None else None
                    else:
                        sentiment_result, probabilities = predict_with_dl_model(
                            st.session_state.user_review, dl_model, tokenizer, label_encoder
                        )
                        classes = label_encoder.classes_ if sentiment_result is not None else None
                    
                    if sentiment_result is not None and probabilities is not None:
                        display_prediction(sentiment_result, probabilities, model_choice, classes)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
        
        # Always show history if available
        display_history()
    
    with tab3:
        st.subheader("📋 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Machine Learning Model")
            st.write("**Algorithm:** Naive Bayes (MultinomialNB)")
            st.write("**Features:** TF-IDF Vectorization")
            st.write("**Strengths:** Fast, interpretable, works well with text")
            st.write("**Best for:** Quick analysis, limited computational resources")
        
        with col2:
            st.markdown("### 🧠 Deep Learning Model")
            st.write("**Architecture:** Gated Recurrent Unit (GRU)")
            st.write("**Features:** FastText Word Embeddings")
            st.write("**Strengths:** Captures context, handles complex patterns")
            st.write("**Best for:** High accuracy, complex sentiment analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Gojek Sentiment Analysis Tool | Built with Streamlit")

if __name__ == "__main__":
    run_app()
