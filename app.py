from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Laad het pre-trained GPT-2 model en de tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

app = Flask(__name__)

# Functie om tekst te genereren
def generate_response(prompt):
    # Zet de prompt om naar tokens
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Genereer tekst met GPT-2
    outputs = model.generate(
        inputs,
        max_length=50,  # Lengte van het antwoord
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,  # Controleert de samplingstrategie voor variatie
        top_k=50,
        temperature=0.7,  # Hoe creatief de output zal zijn
        do_sample=True,  # Samples de output, dus willekeurigheid
    )

    # Decodeer de gegenereerde tokens terug naar tekst
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/')
def home():
    return "Welkom bij de GPT-2 API!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    vraag = data.get('vraag', '')

    if vraag == "":
        return jsonify({"error": "Geen vraag gegeven!"}), 400

    antwoord = generate_response(vraag)
    return jsonify({"antwoord": antwoord})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
