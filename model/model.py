import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Voorbeeld tekst dataset
text = "Ik hou van programmeren en leren over kunstmatige intelligentie."

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# Maak de trainingdata (volgorde van woorden)
sequences = tokenizer.texts_to_sequences([text])[0]
sequence_length = 2

X = []
y = []
for i in range(len(sequences) - sequence_length):
    X.append(sequences[i:i + sequence_length])
    y.append(sequences[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Maak het model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=sequence_length),
    LSTM(50),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train het model
model.fit(X, y, epochs=100, verbose=1)

# Voorspellen van het volgende woord
def predict_next_word(model, tokenizer, input_text):
    sequence = tokenizer.texts_to_sequences([input_text])[0]
    sequence = np.array(sequence[-sequence_length:]).reshape(1, sequence_length)
    pred = model.predict(sequence, verbose=0)
    predicted_word = tokenizer.index_word[np.argmax(pred)]
    return predicted_word
