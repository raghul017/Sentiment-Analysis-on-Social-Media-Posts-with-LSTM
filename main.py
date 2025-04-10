import os
import pickle
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords if not already available
nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Load the trained model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Initialize session history if not exists
def init_history():
    if 'history' not in session:
        session['history'] = []
    return session['history']

# Text preprocessing function
def clean_text(text):
    """Cleans text by lowercasing and removing special characters while preserving important punctuation."""
    text = text.lower()
    # Keep important punctuation like !, ?, ., but remove other special characters
    text = re.sub(r'[^a-z\s!?.,]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction function
def predict_sentiment(text):
    """Predicts sentiment for a given text."""
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=150)  # Updated maxlen to match training
    prediction = model.predict(padded_sequence)[0][0]

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, float(prediction)

# Home and prediction route
@app.route("/", methods=["GET", "POST"])
def home():
    history = init_history()
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            result, confidence = predict_sentiment(text)
            confidence = round(confidence, 2) if confidence is not None else None
            
            # Add to history (keep only last 5 entries)
            history.insert(0, {
                'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate long texts
                'result': result,
                'confidence': confidence
            })
            history = history[:5]  # Keep only last 5 entries
            session['history'] = history

    return render_template("index.html", result=result, confidence=confidence, history=history)

# Clear history route
@app.route("/clear_history", methods=["POST"])
def clear_history():
    session['history'] = []
    return redirect(url_for('home'))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5002)
