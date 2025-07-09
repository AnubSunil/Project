import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import streamlit as st

st.set_page_config(page_title="Next Word Predictor")
st.title("🧠 Next Word Predictor")
st.write("✅ App started... Loading model...")


# Load model and tokenizer
model = load_model('nextword_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1]

def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([sequence], maxlen=max_sequence_len, padding='pre')
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# Streamlit UI
st.title("🧠 Next Word Predictor")
sentence = st.text_input("Type your sentence:")

if sentence:
    next_word = predict_next_word(sentence)
    st.markdown(f"**Predicted next word:** `{next_word}`")
    st.markdown(f"**Full sentence:** {sentence} **{next_word}**")
