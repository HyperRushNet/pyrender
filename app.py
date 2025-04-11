from flask import Flask, jsonify, request, send_from_directory
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Laad het voorgetrainde GPT-2 model en de tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Zorg ervoor dat het model in evaluatiemodus staat
model.eval()

# Serve de index.html file uit de static map
@app.route('/')
def serve_index():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'index.html')

# Route voor de /chat API
@app.route('/chat', methods=['POST'])
def chat():
    # Ontvang de vraag van de gebruiker
    data = request.json
    vraag = data.get('vraag', '')

    # Controleer of er een vraag is
    if not vraag:
        return jsonify({'antwoord': 'Ik heb geen vraag ontvangen.'})

    # Encode de vraag in tokens die het model kan verwerken
    inputs = tokenizer.encode(vraag, return_tensors="pt")

    # Genereer een antwoord van het model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode de output tokens terug naar tekst
    antwoord = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Geef het antwoord terug naar de frontend
    return jsonify({'antwoord': antwoord})

if __name__ == '__main__':
    # Draai de app op poort 10000
    app.run(debug=True, host='0.0.0.0', port=10000)
