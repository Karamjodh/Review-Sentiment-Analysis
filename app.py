import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Embedding

word_index = imdb.get_word_index()
reverse_word_index = {value : keys for keys,value in word_index.items()}

model = load_model("Models/BiDirectionalRNN_imdb.h5")

def decoded_review(encoded_review):
    decoded = " ".join([reverse_word_index.get(i-3,"?") for i in encoded_review])

def preprocess_text(text, maxlen=500):
    words = text.lower().split()
    encoded = []

    for w in words:
        idx = word_index.get(w, 2)  # OOV
        if idx < 10000:
            encoded.append(idx + 3)
        else:
            encoded.append(2)

    return sequence.pad_sequences([encoded], maxlen=maxlen)


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.6 else "Negative"
    return sentiment,prediction[0][0]

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a Review of a movie to classify it as Positive or Negative.")
st.write("(Please enter atleast 25 word for Efficient Results)")

user_input = st.text_area("Review")

if st.button("Classify"):
    sentiment, score = predict_sentiment(user_input)
    
    st.write(f"Sentiment : {sentiment}")
    st.write(f"Prediction Score : {score}")

else :
    st.write("Please Enter a Review.")