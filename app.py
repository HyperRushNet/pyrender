import torch
import torch.nn as nn
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import string

# Definieer het LSTM-model
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out)
        return out, hidden

# Laad het model
model_dir = './model'
model_load_path = os.path.join(model_dir, 'lstm_model.pth')

# Controleer of het model al bestaat
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"Modelbestand {model_load_path} niet gevonden!")

# Laad model en parameters
vocab = {c: i + 1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789 ')}
vocab_size = len(vocab) + 1

embedding_dim = 128
hidden_dim = 256
num_layers = 2
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Flask setup
app = Flask(__name__)
CORS(app)

# Preprocessing
def preprocess_text(text, vocab):
    return [vocab.get(c, 0) for c in text]

def decode_tokens(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ''.join([reverse_vocab.get(token, '?') for token in tokens])

# Genereer tekst
def generate_text(model, start_text, vocab, max_len=100):
    model.eval()
    hidden = None
    input_text = preprocess_text(start_text, vocab)
    input_tensor = torch.tensor(input_text).unsqueeze(1)

    output_tokens = input_text

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output = output[-1, :, :]
            next_token = torch.argmax(output).item()
            output_tokens.append(next_token)
            input_tensor = torch.tensor([[next_token]])

    return decode_tokens(output_tokens, vocab)

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')
    generated_text = generate_text(model, input_text, vocab, max_len=100)
    return jsonify({'generated_text': generated_text})

# Run voor local of Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
