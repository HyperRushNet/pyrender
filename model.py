import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping
import os

# Laad de dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Preprocessen van de tekst om sequenties van woorden te maken
def preprocess_data(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []

    # Maak sequenties van woorden
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(seq) for seq in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, total_words, max_sequence_length

# Bouw het LSTM model
def build_model(total_words, max_sequence_length):
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_length - 1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train het model en sla het op
def train_model():
    corpus = load_dataset('dataset.txt')
    X, y, tokenizer, total_words, max_sequence_length = preprocess_data(corpus)
    model = build_model(total_words, max_sequence_length)

    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=100, verbose=1, callbacks=[early_stopping])

    # Verkrijg het absolute pad naar de root directory van je project
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'model', 'my_model.h5')

    # Sla het model op
    model.save(model_path)

    # Sla de tokenizer op als pickle bestand
    with open(os.path.join(base_path, 'model', 'tokenizer.pkl'), 'wb') as tokenizer_file:
        import pickle
        pickle.dump(tokenizer, tokenizer_file)

    print("Model en tokenizer zijn succesvol getraind en opgeslagen!")

if __name__ == '__main__':
    train_model()
