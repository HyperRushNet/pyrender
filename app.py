from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Laad je eigen getrainde model en tokenizer
model_name = "./model"  # Het pad naar je getrainde model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Stel CORS in om externe toegang mogelijk te maken
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Verkrijg de JSON-gegevens van de request
    input_text = data['text']  # Het tekstgedeelte dat wordt verstrekt in de POST request

    # Tokenizeer de inputtekst
    inputs = tokenizer(input_text, return_tensors='pt')

    # Gebruik het model om een voorspelling te genereren
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100)

    # Decodeer de voorspelde tekst
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"prediction": predicted_text})  # Geef de voorspelling terug als JSON

if __name__ == '__main__':
    app.run(debug=True)
