import os
import pickle
import torch
from torch import nn
from torch.optim import Adam
from model.Seq2Seq import Seq2Seq, Encoder, Decoder
from get_data import get_ds

# Controleer of de modelmap bestaat, maak hem anders aan
if not os.path.exists('model'):
    os.makedirs('model')

# Verkrijg de gegevens
input_tensor, target_tensor, vocab = get_ds()

# Controleer of de data geldig is
if len(input_tensor) == 0 or len(target_tensor) == 0:
    print("Error: Geen trainingsdata beschikbaar.")
    exit(1)

# Initialiseer model, encoder, en decoder
embedding_dim = 256
hidden_dim = 512
encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
model = Seq2Seq(encoder, decoder)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train het model
for epoch in range(10):
    model.train()
    total_loss = 0
    for i in range(0, len(input_tensor), 64):  # Batches van 64
        inputs = input_tensor[i:i + 64]
        targets = target_tensor[i:i + 64]
        
        output = model(inputs, targets)
        loss = loss_fn(output.view(-1, len(vocab)), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(input_tensor) // 64)
    print(f"Epoch {epoch + 1}/10, Gemiddelde verlies: {avg_loss:.4f}")

# Sla het model en vocab op
torch.save(model.encoder.state_dict(), 'model/encoder.pt')
torch.save(model.decoder.state_dict(), 'model/decoder.pt')

# Sla vocab op in een pickle bestand
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Model getraind en opgeslagen.")
