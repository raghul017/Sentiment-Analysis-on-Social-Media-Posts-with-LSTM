import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle

# Expanded dataset with more diverse examples
texts = [
    # Positive examples (50 reviews)
    "This is amazing! I love it!",
    "Great product, highly recommended",
    "Excellent service and quality",
    "The best purchase I've ever made",
    "Outstanding performance",
    "Wonderful experience, will buy again",
    "Perfect solution for my needs",
    "Amazing customer support",
    "High quality product",
    "Very satisfied with the purchase",
    "Great value for money",
    "Excellent workmanship",
    "Beautiful design",
    "Fast delivery",
    "Exactly what I needed",
    "Your product is amazing",
    "This is exactly what I was looking for",
    "Superb quality and service",
    "Fantastic product, exceeded expectations",
    "Brilliant solution, highly recommended",
    "Outstanding value for money",
    "Excellent build quality",
    "Great performance overall",
    "Very happy with my purchase",
    "Perfect match for my needs",
    "Amazing features and functionality",
    "Top-notch customer service",
    "Exceptional product quality",
    "Wonderful user experience",
    "Great attention to detail",
    "Superior craftsmanship",
    "Excellent packaging",
    "Very reliable product",
    "Great durability",
    "Perfect size and fit",
    "Amazing color options",
    "Great battery life",
    "Excellent sound quality",
    "Very comfortable to use",
    "Great price point",
    "Perfect for beginners",
    "Amazing performance",
    "Great value proposition",
    "Excellent after-sales service",
    "Very intuitive design",
    "Great portability",
    "Perfect for professionals",
    "Amazing build quality",
    "Great versatility",
    "Excellent user interface",
    
    # Negative examples (50 reviews)
    "Terrible experience, would not recommend",
    "Waste of money, very disappointed",
    "Poor customer service",
    "Don't buy this product",
    "Completely useless",
    "Very poor quality",
    "Not worth the price",
    "Disappointing purchase",
    "Bad experience overall",
    "Would not recommend",
    "Poor build quality",
    "Terrible customer support",
    "Waste of time",
    "Not as described",
    "Very frustrating experience",
    "Horrible product quality",
    "Extremely disappointed",
    "Complete waste of money",
    "Terrible performance",
    "Very unreliable",
    "Poor design",
    "Not working as expected",
    "Bad quality materials",
    "Terrible user experience",
    "Very slow performance",
    "Poor battery life",
    "Not durable at all",
    "Terrible sound quality",
    "Very uncomfortable",
    "Overpriced product",
    "Difficult to use",
    "Poor functionality",
    "Terrible after-sales service",
    "Very confusing interface",
    "Not portable at all",
    "Poor value for money",
    "Terrible packaging",
    "Very noisy operation",
    "Poor ergonomics",
    "Not suitable for purpose",
    "Terrible durability",
    "Very heavy and bulky",
    "Poor color options",
    "Not energy efficient",
    "Terrible warranty service",
    "Very limited features",
    "Poor performance overall",
    "Not worth the investment",
    "Terrible product design",
    "Very poor reliability"
]

labels = [1] * 50 + [0] * 50  # 1 for positive, 0 for negative

# Enhanced tokenization parameters
max_words = 3000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# Split the data with more validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build an enhanced model with more layers
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.4),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Enhanced callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model with more epochs
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save('lstm_model.h5')

print("Model training completed and saved as 'lstm_model.h5'") 