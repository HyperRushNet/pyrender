import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Definieer de vocabulaire en de vocab_size
vocab = ['<PAD>', '<UNK>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_index = {char: idx for idx, char in enumerate(vocab)}
index_to_char = {idx: char for idx, char in enumerate(vocab)}

# Vocabulaire grootte
vocab_size = len(vocab)

# Model definitie
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

# Laad je model
model = TextGenerationModel(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
model.load_state_dict(torch.load('./model/text_gen_model.pth'))
model.eval()

# Functie om tekst te genereren
def generate_text(start_text, length=100):
    model.eval()
    generated = start_text
    input_seq = torch.tensor([char_to_index.get(char, char_to_index['<UNK>']) for char in start_text], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            _, top_index = torch.max(output[:, -1], dim=1)
            next_char = index_to_char.get(top_index.item(), '<UNK>')
            generated += next_char
            input_seq = torch.cat([input_seq, top_index.unsqueeze(0)], dim=1)[:, 1:]

    return generated

# Stel CORS in
CORS(app)

# Flask route voor voorspelling
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    generated_text = generate_text(input_text, length=100)
    return jsonify({"prediction": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
