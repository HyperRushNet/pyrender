import torch
import pickle
from torch import nn
from torch.optim import Adam
from Seq2Seq import seq2seq
from training_data import get_data

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Verkrijg de training data
input_tensor, target_tensor, vocab = get_data()

# Initialiseer het model
model = Seq2Seq(input_dim=len(vocab), output_dim=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
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
        loss = loss_fn(output, targets)

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
