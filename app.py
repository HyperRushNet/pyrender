import torch
import os
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import random
import string
from flask_cors import CORS

CORS(app)


app = Flask(__name__)

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

# Preprocessing en decoding
def preprocess_text(text, vocab):
    return [vocab.get(c, 0) for c in text]

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

# Vocab en model
vocab = {c: i + 1 for i, c in enumerate(string.ascii_lowercase + string.digits + ' ')}
vocab_size = len(vocab) + 1

embedding_dim = 128
hidden_dim = 256
num_layers = 2
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)

def train_model(model, vocab):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    random_texts = ["hello world", "deep learning", "flask api", "text generation", "openai gpt"]
    for epoch in range(5):  # kortere training
        for text in random_texts:
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


