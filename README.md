# 📊 LSTM Sentiment Analysis with Keras

This assignment project i had done a while back implements a sentiment analysis model using a **Bidirectional LSTM** in Keras. It is trained on a custom dataset of movie reviews with balanced positive and negative samples. The model predicts sentiment from user input in real time.

## 📁 Files

* **model.py** – Preprocesses text data, builds the LSTM model, trains and saves it.
* **predict.py** – Loads the trained model and tokenizer to predict sentiment of new text input.

## 🧠 Model Architecture

* **Embedding layer** (128 dimensions)
* **2 Bidirectional LSTM layers** with dropout and L2 regularization
* **Dense layers** with ReLU and Sigmoid activations
* **Binary classification**: Positive or Negative

## 🛠 Requirements

* Python 3.x
* TensorFlow
* NumPy
* Pandas
* scikit-learn
* pickle

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Model

```bash
python model.py
```

This will train the model and save `sentiment_model.h5` and `tokenizer.pkl`.

### 2. Predict Sentiment

```bash
python predict.py
```

Enter any review text to get a sentiment prediction.

## 🧪 Sample Output

```
Review: This movie was fantastic!
Predicted Sentiment: Positive

Review: It was an average movie, nothing special.
Predicted Sentiment: Negative
```

## 📌 Notes

* The model uses a simple custom dataset for demonstration.
* Accuracy may improve with larger or domain-specific datasets.

---

