from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer

# Functie om vocab_size automatisch te bepalen
def get_vocab_size_from_input(user_input):
    # Laad een tokenizer (bijvoorbeeld BERT tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize de gebruikersinvoer (dit splitst de tekst in tokens)
    tokens = tokenizer.tokenize(user_input)

    # Verkrijg de unieke tokens
    unique_tokens = set(tokens)

    # De vocab_size is het aantal unieke tokens
    vocab_size = len(unique_tokens)
    return vocab_size

# Definieer het Text Generation Model
class TextGenerationModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(TextGenerationModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Initialiseer Flask applicatie
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Verkrijg de gebruikersinvoer uit het verzoek
    user_input = request.json.get("text", "")
    
    # Verkrijg de vocab_size van de gebruikersinvoer
    vocab_size = get_vocab_size_from_input(user_input)
    
    # Maak het model met de dynamisch berekende vocab_size
    model = TextGenerationModel(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
    
    # Hier kun je het model gebruiken om voorspellingen te doen (bijvoorbeeld genereren van tekst)
    # Hier is een placeholder voor een modelvoorspelling
    # output = model(input_tensor)  # Input tensor wordt gemaakt uit de gebruikersinvoer
    
    # Voor nu sturen we gewoon de vocab_size terug en een succesbericht
    return jsonify({"vocab_size": vocab_size, "message": "Model is ready for prediction"})

if __name__ == '__main__':
    app.run(debug=True)
