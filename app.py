import tensorflow as tf
import pickle
import numpy as np
import os
from flask import Flask, request, jsonify

# Verkrijg het absolute pad naar de root directory van je project
base_path = os.path.dirname(os.path.abspath(__file__))

# Laad het model
model_path = os.path.join(base_path, 'model', 'my_model.h5')
model = tf.keras.models.load_model(model_path)

# Laad de tokenizer
with open(os.path.join(base_path, 'model', 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Flask applicatie
app = Flask(__name__)

# Functie om voorspellingen te doen
def predict_next_word(input_text):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=20, padding='pre')  # Padding
    pred = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(pred)
    predicted_word = tokenizer.index_word.get(predicted_word_index, None)
    return predicted_word

# Route om voorspellingen te doen
@app.route('/predict', methods=['GET'])
def predict():
    query = request.args.get('q', '')  # Haal de queryparameter op
    next_word = predict_next_word(query)  # Voorspel het volgende woord
    return jsonify({'next_word': next_word})

if __name__ == '__main__':
    app.run(debug=True)
