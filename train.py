import torch
import pickle
from torch import nn
from torch.optim import Adam
from model.Seq2Seq import Seq2Seq, Encoder, Decoder
from get_data import get_ds
import os

# Controleer of de map bestaat, zo niet maak hem aan
if not os.path.exists('model'):
    os.makedirs('model')

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Verkrijg de training data
input_tensor, target_tensor, vocab = get_ds()

# Debug print statements om de data te controleren
print(f"Input Tensor Length: {len(input_tensor)}")
print(f"Target Tensor Length: {len(target_tensor)}")
print(f"Vocabulary Size: {len(vocab)}")

# Controleer of de data leeg is
if len(input_tensor) == 0 or len(target_tensor) == 0:
    print("Error: No training data available.")
    exit(1)

# Zorg ervoor dat het model naar de juiste device wordt gestuurd (GPU of CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialiseer het model
encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
model = Seq2Seq(encoder, decoder)
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Train het model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Variabele om de loss per epoch te accumuleren
    for i in range(0, len(input_tensor), batch_size):
        # Haal een batch van de data
        inputs = input_tensor[i:i + batch_size].to(device)
        targets = target_tensor[i:i + batch_size].to(device)

        # Voer een forward pass uit
        output = model(inputs, targets)
        
        # Debug: Controleer de vormen van de inputs, outputs, en targets
        print(f"Epoch {epoch + 1}, Step {i + 1}/{len(input_tensor)}, Output shape: {output.shape}, Targets shape: {targets.shape}")

        # Bereken de loss
        loss = loss_fn(output.view(-1, len(vocab)), targets.view(-1))

        # Voer backpropagation uit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumuleer de loss voor de epoch

    # Print de loss na elke epoch
    avg_loss = total_loss / (len(input_tensor) // batch_size)  # Gemiddelde loss per epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Sla het model op (encoder, decoder)
torch.save(model.encoder.state_dict(), 'model/encoder.pt')
torch.save(model.decoder.state_dict(), 'model/decoder.pt')

# Sla de vocab op
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
