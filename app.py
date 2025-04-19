import torch
import os
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import random
import string

# Flask app
app = Flask(__name__)

# Hardcode de poort naar 8080, omdat we geen omgevingvariabele gebruiken
port = int(os.environ.get('PORT', 8080))  # Gebruik de omgevingsvariabele PORT of 8080 als fallback

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

# Functie om tekst te preprocessen en om te zetten naar tokens
def preprocess_text(text, vocab):
    return [vocab.get(c, 0) for c in text]  # Zet elke karakter om naar een index

# Functie om de gegenereerde tokens om te zetten naar tekst
def decode_tokens(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ''.join([reverse_vocab.get(token, '?') for token in tokens])

# Functie voor tekstgeneratie
def generate_text(model, start_text, vocab, max_len=100):
    model.eval()
    hidden = None
    input_text = preprocess_text(start_text, vocab)
    input_tensor = torch.tensor(input_text).unsqueeze(1)  # Maak input geschikt voor LSTM (batch_size = 1)
    
    output_tokens = input_text

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output = output[-1, :, :]  # Neem de output van het laatste tijdstip

            # Verkrijg de waarschijnlijkheid van de volgende token
            next_token = torch.argmax(output).item()
            output_tokens.append(next_token)

            input_tensor = torch.tensor([[next_token]])

    return decode_tokens(output_tokens, vocab)

# Definieer vocabulaire
vocab = {c: i + 1 for i, c in enumerate(string.ascii_lowercase + string.digits + ' ')}
vocab_size = len(vocab) + 1  # +1 voor het onbekende teken (0-index)

# Maak model aan
embedding_dim = 128
hidden_dim = 256
num_layers = 2
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)

# Train een eenvoudig model (we gebruiken hier slechts enkele iteraties voor de demo)
def train_model(model, vocab):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # We trainen op willekeurige tekst voor dit voorbeeld
    random_texts = ["hello world", "deep learning", "flask api", "text generation", "openai gpt"]
    for epoch in range(100):  # Train voor een aantal epochs
        for text in random_texts:
            input_text = preprocess_text(text, vocab)
            input_tensor = torch.tensor(input_text).unsqueeze(1)  # Shape: (seq_len, batch_size)

            optimizer.zero_grad()
            output, _ = model(input_tensor, None)  # Geen initiale hidden state
            output = output.view(-1, vocab_size)  # (seq_len * batch_size, vocab_size)
            target = input_tensor.view(-1)  # Target is de "volgende" token in de tekst

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 and epoch > 0:
                print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
    print("Model training complete.")

# Train model (gewoon eens door de tekst heen)
train_model(model, vocab)

# Flask route voor de API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')
    
    # Genereer tekst met het model
    generated_text = generate_text(model, input_text, vocab, max_len=100)
    
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    # Start de Flask app op poort 8080
    print(f"Starting the app on port {PORT}")
    app.run(host='0.0.0.0', port=port)
