
# **Program 2: Load Trained Model and Make Predictions**
import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to preprocess user input
def preprocess_input(text):
    text = text.lower()  
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

# Function to predict sentiment
def predict_sentiment(user_input):
    processed_text = preprocess_input(user_input)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=25)
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.4 else "Negative"
    return sentiment

# Example predictions
sample_texts = [
    "This movie was fantastic!",
    "It was an average movie, nothing special.",
]

for text in sample_texts:
    print(f"Review: {text}")
    print(f"Predicted Sentiment: {predict_sentiment(text)}\n")

# User input for prediction
while True:
    user_input = input("Enter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print(f"Predicted Sentiment: {predict_sentiment(user_input)}\n")
