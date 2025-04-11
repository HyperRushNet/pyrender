import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Zorg ervoor dat je NLTK data hebt gedownload
nltk.download('punkt')

app = Flask(__name__)

# Eenvoudige tekstclassificatie met een feed-forward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Functie om de tekst om te zetten naar numerieke waarden
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Verlaag de tekst en tokenize
    return tokens

# Functie om de tekst om te zetten naar een vector (bijv. met one-hot encoding)
def text_to_vector(tokens, vocab):
    vector = np.zeros(len(vocab))
    for word in tokens:
        if word in vocab:
            index = vocab[word]
            vector[index] = 1
    return vector

# Stel de training data in (een eenvoudig voorbeeld)
train_texts = ["Hallo, hoe gaat het?", "Wat is jouw naam?", "Wat kan je doen?", "Wat is je favoriete kleur?"]
train_labels = ["groet", "vraag", "vraag", "vraag"]

# Maak een vocabulaire op basis van de trainingstekst
all_tokens = []
for text in train_texts:
    all_tokens.extend(preprocess_text(text))

vocab = {word: idx for idx, word in enumerate(set(all_tokens))}

# Encodeer labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Converteer de trainingstekst naar numerieke vectors
train_vectors = [text_to_vector(preprocess_text(text), vocab) for text in train_texts]

# Zet alles om naar PyTorch tensors
X_train = torch.tensor(train_vectors, dtype=torch.float32)
y_train = torch.tensor(train_labels_encoded, dtype=torch.long)

# Model instantiëren
input_size = len(vocab)
hidden_size = 10
num_classes = len(set(train_labels_encoded))

model = SimpleNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train het model (1 epoch voor snel voorbeeld)
def train_model():
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

# Train het model
for epoch in range(1):
    loss = train_model()
    print(f'Epoch {epoch+1}, Loss: {loss}')

# API endpoint voor classificatie
@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    vraag = data.get('vraag', '')

    if vraag == "":
        return jsonify({"error": "Geen vraag gegeven!"}), 400

    tokens = preprocess_text(vraag)
    vector = text_to_vector(tokens, vocab)
    vector = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)  # Voeg batch-dimensie toe

    model.eval()  # Zet het model in evaluatiemodus
    with torch.no_grad():
        output = model(vector)
        _, predicted = torch.max(output, 1)

    label = label_encoder.inverse_transform([predicted.item()])[0]
    return jsonify({"categorie": label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
