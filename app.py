import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM

# Maak Flask app
app = Flask(__name__)
CORS(app)  # Zorgt dat frontend van andere domeinen mag POSTen

# Model en tokenizer laden
model_name = "distilgpt2"  # Gebruik het kleinere model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocessing
def preprocess_text(text):
    # Tokenizeer de tekst
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

def generate_text(model, start_text, max_len=100):
    model.eval()
    inputs = preprocess_text(start_text)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Gebruik torch.no_grad() om geheugen te besparen tijdens inferentie
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_len, num_return_sequences=1)

    # Decodeer de output naar tekst
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')
    generated_text = generate_text(model, input_text, max_len=100)
    return jsonify({'generated_text': generated_text})

# Run voor local of Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
