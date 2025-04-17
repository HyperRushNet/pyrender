import torch
import requests
import re

# Functie om het dataset te verkrijgen
def get_ds():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    response = requests.get(url)
    response.raise_for_status()  # Zorg ervoor dat de request succesvol was

    lines = response.text.strip().split('\n')

    input_tensor = []
    target_tensor = []
    vocab = {}  # Gebruik een dictionary voor vocab in plaats van een set

    # Verwerk de lijnen en vul input_tensor, target_tensor en vocab
    for line in lines:
        input_line, target_line = line.split('\t')
        
        # Vul vocab bij met woorden en geef ze een index
        for word in input_line.split():
            if word not in vocab:
                vocab[word] = len(vocab)
        for word in target_line.split():
            if word not in vocab:
                vocab[word] = len(vocab)

        # Voeg de input_line en target_line toe aan de tensors
        input_tensor.append([vocab.get(word, 0) for word in input_line.split()])  # Gebruik 0 voor onbekende woorden
        target_tensor.append([int(re.sub(r'\D', '', word)) for word in target_line.split() if re.sub(r'\D', '', word) != ''])

    # Controleer of input_tensor leeg is
    if len(input_tensor) == 0:
        print("Waarschuwing: Geen invoer gevonden. Controleer de dataformaten.")

    # Zet input_tensor en target_tensor om naar PyTorch tensors
    input_tensor = torch.tensor(input_tensor, dtype=torch.long)
    target_tensor = torch.tensor(target_tensor, dtype=torch.long)

    return input_tensor, target_tensor, vocab


# Laad het model en vocabulaire (deze functie kan afhankelijk van je modelstructuur worden aangepast)
def load_model_and_vocab():
    # Laad het dataset
    input_tensor, target_tensor, vocab = get_ds()

    # Stel het model en andere nodige variabelen in (dit hangt af van je modelimplementatie)
    # Bijvoorbeeld, als je een model hebt:
    # model = YourModel()
    # model.load_state_dict(torch.load("your_model.pth"))
    # model.eval()

    return input_tensor, target_tensor, vocab


# Genereer een response (geeft bijvoorbeeld een vertaling of een voorspelling)
def generate_response(input_text):
    input_tensor, target_tensor, vocab = load_model_and_vocab()

    # Verwerk de input (afhankelijk van je modelstructuur)
    input_indices = [vocab.get(word, 0) for word in input_text.split()]
    input_tensor = torch.tensor([input_indices], dtype=torch.long)

    # Zorg ervoor dat je je model hier gebruikt om een output te genereren
    # Bijvoorbeeld, als je model een vertaling genereert:
    # output_tensor = model(input_tensor)
    
    # Veronderstel dat we de target_tensor gebruiken voor een vertaling
    output = target_tensor  # Dit is slechts een placeholder, vervang dit door je modeloutput.

    # Maak de output begrijpelijk
    output_text = " ".join([list(vocab.keys())[list(vocab.values()).index(idx)] for idx in output[0]])

    return output_text


# Flask applicatie om een webservice te draaien
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def api_generate():
    input_data = request.json.get("input", "")
    if not input_data:
        return jsonify({"error": "Geen invoer gegeven"}), 400

    response = generate_response(input_data)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
