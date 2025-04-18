from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialiseer Flask app
app = Flask(__name__)

# Laad je getrainde model (zorg ervoor dat je model bestand hebt)
model = tf.keras.models.load_model('model/my_model.h5')

# Laad je tokenizer
tokenizer = Tokenizer()

# Functie om het volgende woord te voorspellen
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=10, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    return tokenizer.index_word[predicted[0]]

# API endpoint voor voorspelling via queryparameter
@app.route('/predict', methods=['GET'])
def predict():
    # Haal de tekst op uit de URL parameter 'q'
    input_text = request.args.get('q', '')  # Default naar lege string als er geen parameter is
    if input_text == '':
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    
    next_word = predict_next_word(input_text)
    return jsonify({"input_text": input_text, "next_word": next_word})

# Start de app
if __name__ == '__main__':
    app.run(debug=True)
