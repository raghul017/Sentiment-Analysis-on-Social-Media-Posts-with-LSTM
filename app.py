from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and preprocess the dataset
def load_data():
    # Load the dataset
    df = pd.read_csv('data/twitter_training.csv')
    
    # Rename columns
    df.columns = ['id', 'entity', 'sentiment', 'text']
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert sentiment to binary (1 for positive, 0 for negative)
    df['sentiment'] = df['sentiment'].map({'Positive': 1, 'Negative': 0})
    
    # Remove neutral sentiments
    df = df[df['sentiment'].isin([0, 1])]
    
    return df

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

def create_model(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Load and preprocess data
    df = load_data()
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_text'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    
    # Pad sequences
    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating='post', padding='post')
    
    # Split data
    X = padded_sequences
    y = df['sentiment'].values
    
    # Create and train model
    model = create_model(10000, max_length)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train model
    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    return model, tokenizer, max_length

# Train the model and get tokenizer
model, tokenizer, max_length = train_model()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text")
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([processed_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(sequence, maxlen=max_length, truncating='post', padding='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]
        
        # Determine sentiment and confidence
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if sentiment == "Positive" else 1 - prediction
        
        return render_template("index.html", result=sentiment, confidence=confidence)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) 