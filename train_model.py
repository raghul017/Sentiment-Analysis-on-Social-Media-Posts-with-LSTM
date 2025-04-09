import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import pickle
import re

class DataGenerator(Sequence):
    def __init__(self, texts, labels, tokenizer, max_len, batch_size=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        
    def __len__(self):
        return (len(self.texts) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.texts))
        
        batch_texts = self.texts[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # Convert texts to sequences
        X = self.tokenizer.texts_to_sequences(batch_texts)
        X = pad_sequences(X, maxlen=self.max_len)
        
        return X, np.array(batch_labels)

def clean_text(text):
    """Cleans text by lowercasing and removing special characters while preserving important punctuation."""
    text = text.lower()
    # Keep important punctuation like !, ?, ., but remove other special characters
    text = re.sub(r'[^a-z\s!?.,]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_fasttext_data(file_path, max_samples=None):
    texts = []
    labels = []
    
    print("Loading FastText data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            # FastText format: __label__2 for positive, __label__1 for negative
            line = line.strip()
            if line:
                if line.startswith('__label__2'):
                    label = 1  # Positive
                elif line.startswith('__label__1'):
                    label = 0  # Negative
                else:
                    continue
                
                text = clean_text(line)
                if text:  # Only add if we have text after cleaning
                    texts.append(text)
                    labels.append(label)
            
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i} lines...")
    
    return texts, labels

# Load and preprocess the data
print("Loading data...")
texts, labels = load_fasttext_data('test.ft.txt', max_samples=50000)  # Start with 50k samples
print(f"Loaded {len(texts)} samples")

# Tokenization
print("Tokenizing texts...")
max_words = 30000  # Increased vocabulary size
max_len = 150  # Increased max length to capture more context

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Split the data
train_idx = int(len(texts) * 0.8)
train_texts = texts[:train_idx]
train_labels = labels[:train_idx]
val_texts = texts[train_idx:]
val_labels = labels[train_idx:]

# Create data generators
batch_size = 64  # Reduced batch size for better generalization
train_gen = DataGenerator(train_texts, train_labels, tokenizer, max_len, batch_size)
val_gen = DataGenerator(val_texts, val_labels, tokenizer, max_len, batch_size)

# Build the model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss',
                              patience=3,
                              restore_best_weights=True)

model_checkpoint = ModelCheckpoint('best_model.h5',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max')

# Train the model
print("Training model...")
history = model.fit(train_gen,
                   epochs=10,  # Reduced from 20 to 10
                   validation_data=val_gen,
                   callbacks=[early_stopping, model_checkpoint])

# Save the final model
model.save('lstm_model.h5')

print("\nModel training completed and saved as 'lstm_model.h5'")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Test the model
print("\nTesting the model with some example reviews:")
test_texts = [
    "This is absolutely fantastic! I love it!",
    "Terrible experience, would not recommend to anyone.",
    "The service was okay, but could be better.",
    "Amazing product, exceeded my expectations!",
    "Very disappointed with the quality."
]

# Preprocess test texts
test_texts = [clean_text(text) for text in test_texts]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_len)
predictions = model.predict(test_padded)

for text, pred in zip(test_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"\nText: {text}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {pred[0]:.2f})") 