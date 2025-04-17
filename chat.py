import torch
import pickle
from torch import nn
from torch.optim import Adam
from model.Seq2Seq import Seq2Seq, Encoder, Decoder  # Importeren van Encoder, Decoder en Seq2Seq
from get_data import get_ds

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Verkrijg de training data
input_tensor, target_tensor, vocab = get_ds()

# Initialiseer de Encoder en Decoder (indien nodig)
encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)

# Initialiseer het Seq2Seq model
model = Seq2Seq(encoder, decoder)

# Definieer de optimizer en loss function
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Train het model
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(input_tensor), batch_size):
        # Haal een batch van de data
        inputs = input_tensor[i:i + batch_size]
        targets = target_tensor[i:i + batch_size]

        # Voer een forward pass uit
        output = model(inputs, targets)
        loss = loss_fn(output.view(-1, len(vocab)), targets.view(-1))

        # Voer backpropagation uit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Sla het model op
torch.save(model.encoder, 'model/encoder.pt')
torch.save(model.decoder, 'model/decoder.pt')

# Sla de vocab op
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
