import torch
import os
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn as nn

# Maak Flask app
app = Flask(__name__)
CORS(app)  # Zorgt dat frontend van andere domeinen mag POSTen

# Model definitie
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
vocab = {c: i + 1 for i, c in enumerate(string.ascii_lowercase + string.digits + ' ')}
vocab_size = len(vocab) + 1
embedding_dim = 128
hidden_dim = 256
num_layers = 2

model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)

# Pad naar modelbestand relatief
model_load_path = os.path.join(os.getcwd(), "model", "lstm_model.pth")
model.load_state_dict(torch.load(model_load_path))
model.eval()

# Functie om tekst te genereren
def generate_text(model, start_text, vocab, max_len=100):
    hidden = None
    input_text = [vocab.get(c, 0) for c in start_text]
    input_tensor = torch.tensor(input_text).unsqueeze(0)  # batch size 1

    output_tokens = input_text

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output = output[:, -1, :]  # Alleen de laatste output
            next_token = torch.argmax(output, dim=1).item()
            output_tokens.append(next_token)
            input_tensor = torch.tensor([[next_token]])

    reverse_vocab = {v: k for k, v in vocab.items()}
    generated_text = ''.join([reverse_vocab.get(token, '?') for token in output_tokens])
    return generated_text

# API endpoint voor voorspelling
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')
    generated_text = generate_text(model, input_text, vocab)
    return jsonify({'generated_text': generated_text})

# Run voor local of Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
