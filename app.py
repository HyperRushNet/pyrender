import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Laad je model (verwijder de 'transformers' verwijzing)
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Veronderstel dat je vocab_size, embedding_dim, etc. hebt gedefinieerd zoals eerder
model = TextGenerationModel(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
model.load_state_dict(torch.load('./model/text_gen_model.pth'))
model.eval()

# Functie om tekst te genereren
def generate_text(start_text, length=100):
    model.eval()
    generated = start_text
    input_seq = torch.tensor([char_to_index[char] for char in start_text], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            _, top_index = torch.max(output[:, -1], dim=1)
            next_char = index_to_char[top_index.item()]
            generated += next_char
            input_seq = torch.cat([input_seq, top_index.unsqueeze(0)], dim=1)[:, 1:]

    return generated

# Stel CORS in
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    generated_text = generate_text(input_text, length=100)
    return jsonify({"prediction": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
