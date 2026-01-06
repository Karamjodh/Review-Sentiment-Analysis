### 📊 Sentiment Analysis using Bidirectional RNN
---
### 📌 Project Overview

This project implements a Sentiment Analysis system using Deep Learning (Bidirectional RNN / LSTM) to classify movie reviews as positive or negative.
The model learns contextual meaning and negation directly from text data with minimal manual NLP preprocessing.

---

### 🧠 Key Features

Uses an Embedding Layer for dense word representations

Bidirectional RNN (LSTM) to capture both past and future context

Effectively handles negation (e.g., “not good”)

Trained on a large-scale benchmark dataset

Minimal preprocessing to preserve semantic structure

---

### 📂 Dataset Used
IMDB Movie Reviews (TensorFlow)

Source: TensorFlow/Keras built-in dataset

50,000 movie reviews

Pre-split into training and testing sets

Binary sentiment labels (positive / negative)

---

### ⚙️ Tech Stack

Python

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

---

### 🔄 Data Preprocessing

Lowercasing text

Padding sequences to fixed length

Integer encoding using Keras Tokenizer

Out-of-vocabulary (OOV) handling

⚠️ Stopword removal, stemming, and lemmatization are intentionally avoided to allow the model to learn contextual meaning directly.

---

### 🏗 Model Architecture

Embedding Layer
↓
Bidirectional RNN (LSTM)
↓
Dropout
↓
Dense (Sigmoid)

---

### 📈 Model Performance

Achieves ~80–83% accuracy on test data

Significantly improves over vanilla RNN models

Correctly captures sentiment polarity in the presence of negation

---

### 🚀 How to Run

Clone the repository

Install dependencies

pip install tensorflow numpy pandas scikit-learn


Run the notebook or script

Evaluate model performance on the test set

---

### 📌 Future Improvements

Add Attention mechanism

Use pretrained word embeddings

Hyperparameter tuning

Deploy the model using a REST API

---

### 👤 Author

Karamjodh Singh

---

### 📜 License

This project is intended for educational and research purposes.

---