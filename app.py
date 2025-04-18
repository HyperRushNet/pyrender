from flask import Flask, request, jsonify
from flask_cors import CORS  # Voeg deze regel toe voor CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Voeg CORS toe aan de app
CORS(app)  # Dit staat CORS toe voor alle domeinen. Je kunt het later beperken als je wilt.

# Laad het model en de tokenizer
model_name = "path/to/your/model"  # Vul hier je modelpad in
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']

    # Tokenizeer de invoer
    inputs = tokenizer(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100)

    # Decodeer de output en stuur de respons terug
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"prediction": predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
