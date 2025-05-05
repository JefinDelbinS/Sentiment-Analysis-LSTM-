import numpy as np
import pandas as pd
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2

# **Balanced Sentiment Dataset with Equal Positive & Negative Samples**
data = {
    "review": [
        "I loved the movie! The acting was great and the story was amazing.",
        "Horrible film. Waste of time. Poor script and terrible acting.",
        "Absolutely fantastic! Would watch again.",
        "It was okay, not the best but not the worst.",
        "I hated it. The worst movie I've ever seen.",
        "The visuals were stunning but the plot was boring.",
        "Worst movie ever. I regret watching it.",
        "Absolutely horrible experience.",
        "The film was too long and boring.",
        "Great storyline and amazing acting.",
        "Loved every moment! Highly recommend.",
        "Such a waste of time, do not watch.",
        "I absolutely hated this movie, waste of time.",  
        "Terrible direction, worst movie ever.",  
        "Not enjoyable, acting was horrible.",  
        "It was just okay, not great.",  
        "Loved the film, best one yet!",  
        "Amazing experience, highly recommended!",
        "It was an average movie, not bad but not great.",
        "Some parts were enjoyable, but overall it was boring.",
        "Decent film, could have been better.",
        "I have mixed feelings about this movie.",
        "I really enjoyed this film, would watch again!",  
        "One of the best movies I've seen in years.",  
        "This film was so touching, I loved it.",  
        "The cinematography was breathtaking!",  
        "What a fantastic experience, well done!",  
        "This movie was okay, some good and bad parts.",  
        "I didn’t enjoy this film, it was a letdown.",  
        "This movie was dull and dragged on forever.",  
        "I wouldn't recommend this to anyone.",  
        "The film was poorly directed and had bad acting.",
        "A masterpiece! Every scene was perfect.",
        "I had fun watching, highly entertaining!",
        "Completely lifeless, I almost fell asleep.",
        "The jokes didn’t land, boring humor.",
        "A must-watch, truly brilliant!",
        "A mediocre experience, nothing special.",
        "Too predictable, nothing exciting about it."
    ],
    "sentiment": [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0,
                  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# **2. Preprocess Text Data**
def preprocess_text(text):
    text = text.lower()  #Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df["cleaned_review"] = df["review"].apply(preprocess_text)

# **3. Tokenize and Convert to Sequences**
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["cleaned_review"])
X = tokenizer.texts_to_sequences(df["cleaned_review"])
X = pad_sequences(X, maxlen=25)  # Increased max length for better context

y = np.array(df["sentiment"])  # Labels (0 = Negative, 1 = Positive)

# **4. Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **5. Build Improved LSTM Model with Bidirectional Layers and L2 Regularization**
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=25),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# **6. Train the Model for More Epochs**
model.fit(X_train, y_train, epochs=150, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# **7. Evaluate Model**
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# **8. Make Predictions with Adjusted Threshold**
sample_text = ["This movie was fantastic!", "I did not like this film."]
sample_text = tokenizer.texts_to_sequences(sample_text)
sample_text = pad_sequences(sample_text, maxlen=25)
predictions = model.predict(sample_text)
predictions = ["Positive" if p > 0.4 else "Negative" for p in predictions]
print("Predictions:", predictions)
