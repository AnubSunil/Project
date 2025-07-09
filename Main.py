#  STEP 1: Upload the dataset 
from google.colab import files
uploaded = files.upload()

import io
text = io.open(list(uploaded.keys())[0], 'r', encoding='utf-8').read().lower()
print("Sample Text:\n", text[:500])

#  STEP 2: Preprocessing
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import pickle

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split into inputs and labels
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

#  STEP 3: Build and Train the LSTM model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#  Train the model 
history = model.fit(xs, ys, epochs=300, verbose=1)
#  STEP 4: Save model and tokenizer
model.save('nextword_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print(" Model and tokenizer saved!")

#  STEP 5: Load model and tokenizer 
from keras.models import load_model
import numpy as np

model = load_model('nextword_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word(input_text):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    padded = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding='pre')
    prediction = model.predict(padded, verbose=0)
    predicted_index = np.argmax(prediction)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

