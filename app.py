import torch
import os
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import string

# Flask app init
app = Flask(__name__)
CORS(app)

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

# Preprocessing
def preprocess_text(text, vocab):
    return [vocab.get(c, 0) for c in text.lower()]

def decode_tokens(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ''.join([reverse_vocab.get(token, '?') for token in tokens])

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

# Vocab setup
vocab = {c: i + 1 for i, c in enumerate(string.ascii_lowercase + string.digits + ' ')}
vocab_size = len(vocab) + 1

# Model setup
embedding_dim = 128
hidden_dim = 256
num_layers = 2
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)

# Training (dummy data, klein & snel zodat Render het aan kan)
def train_model(model, vocab):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    texts = ["hello world", "deep learning", "flask api", "text generation", "openai gpt"]
    for epoch in range(1):  # slechts 1 epoch om resources te sparen
        for text in texts:
            input_text = preprocess_text(text, vocab)
            input_tensor = torch.tensor(input_text).unsqueeze(1)
            optimizer.zero_grad()
            output, _ = model(input_tensor, None)
            output = output.view(-1, vocab_size)
            target = input_tensor.view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

train_model(model, vocab)

# API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # force=True voor veiligheid
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        input_text = data['text']
        generated_text = generate_text(model, input_text, vocab, max_len=100)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
