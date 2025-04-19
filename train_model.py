import torch
import torch.nn as nn
import torch.optim as optim
import os
from datasets import load_dataset
from torch.utils.data import Dataset

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

# Preprocessing
def preprocess_text(text, vocab):
    return [vocab.get(c, 0) for c in text]

def decode_tokens(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ''.join([reverse_vocab.get(token, '?') for token in tokens])

# Dataset inladen en voorbereiden
dataset = load_dataset("text", data_files={"train": "data/dataset.txt"})

# Vocab instellen
vocab = {c: i + 1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789 ')}
vocab_size = len(vocab) + 1  # voor padding

# Modelinstellingen
embedding_dim = 128
hidden_dim = 256
num_layers = 2

# Maak een instance van het model
model = TextGenerationModel(vocab_size, embedding_dim, hidden_dim, num_layers)

# Dataset voorbereiden
class TextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = texts
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        return torch.tensor(preprocess_text(text, self.vocab))

train_texts = dataset['train']['text']

train_dataset = TextDataset(train_texts, vocab)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# Training
def train_model(model, train_loader, vocab_size, num_epochs=5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            # Veronderstel dat we de tekst als input gebruiken
            input_text = batch
            output, _ = model(input_text, None)

            # Target is de tekst verschoven met 1 (volgende token)
            target = input_text[:, 1:].contiguous().view(-1)
            output = output[:, :-1, :].contiguous().view(-1, vocab_size)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Train het model
train_model(model, train_loader, vocab_size)

# Zorg ervoor dat de map voor het model bestaat
model_dir = './model'
os.makedirs(model_dir, exist_ok=True)

# Sla het model op
model_save_path = os.path.join(model_dir, 'lstm_model.pth')
torch.save(model.state_dict(), model_save_path)

print("Model opgeslagen!")
