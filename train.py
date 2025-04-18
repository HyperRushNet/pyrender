import torch
import pickle
from torch import nn
from torch.optim import Adam
from model.Seq2Seq import Seq2Seq, Encoder, Decoder
from get_data import get_ds

# Hyperparameters
embedding_dim = 256
hidden_dim = 512
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Verkrijg de training data
input_tensor, target_tensor, vocab = get_ds()

# Initialiseer het model
encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
model = Seq2Seq(encoder, decoder)
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Zorg ervoor dat het model naar de juiste device wordt gestuurd (GPU of CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train het model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Variable to accumulate loss for each epoch
    for i in range(0, len(input_tensor), batch_size):
        # Haal een batch van de data
        inputs = input_tensor[i:i + batch_size].to(device)
        targets = target_tensor[i:i + batch_size].to(device)

        # Voer een forward pass uit
        output, encoder_hidden = model(inputs, targets)

        # Debug print voor de vorm van de encoder output
        print(f"Encoder hidden state shape: {encoder_hidden[0].shape}")  # Controleer de vorm

        # Bereken de loss
        loss = loss_fn(output.view(-1, len(vocab)), targets.view(-1))

        # Voer backpropagation uit
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumulate loss for the epoch

    # Print de loss na elke epoch
    avg_loss = total_loss / (len(input_tensor) // batch_size)  # Gemiddelde loss per epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Sla het model op
torch.save(model.encoder.state_dict(), 'model/encoder.pt')
torch.save(model.decoder.state_dict(), 'model/decoder.pt')

# Sla de vocab op
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
