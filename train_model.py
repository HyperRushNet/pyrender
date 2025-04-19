import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string
import os

# LSTM Model voor tekstgeneratie
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out)
        return out, hidden

# Dataset voorbereiden
class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_length=100):
        self.text = text
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        x = [self.vocab.get(c, 0) for c in self.text[idx: idx + self.seq_length]]
        y = [self.vocab.get(c, 0) for c in self.text[idx + 1: idx + self.seq_length + 1]]
        return torch.tensor(x), torch.tensor(y)

# Tekst voor training
data_path = os.path.join(os.getcwd(), "data", "dataset.txt")  # Zet het bestand relatief
with open(data_path, "r") as file:
    text = file.read()

# Vocab aanmaken (alle unieke karakters in de tekst)
vocab = {c: i + 1 for i, c in enumerate(string.ascii_lowercase + string.digits + ' ')}
vocab_size = len(vocab) + 1

# Modelparameters
embedding_dim = 128
hidden_dim = 256
num_layers = 2
batch_size = 64
epochs = 10

# Dataset en dataloader
dataset = TextDataset(text, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model en training setup
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Trainen van het model
def train_model():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        hidden = None
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            output, hidden = model(x_batch, hidden)
            loss = criterion(output.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Train het model
train_model()

# Sla het model op
model_save_path = os.path.join(os.getcwd(), "model", "lstm_model.pth")  # Relatief opslaan
torch.save(model.state_dict(), model_save_path)
