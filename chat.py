import os
import torch
import pickle
from torch import nn
from torch.optim import Adam
from model.Seq2Seq import Seq2Seq, Encoder, Decoder  # Correcte import van Seq2Seq
from get_data import get_ds

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

# Initialiseer het model
encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
model = Seq2Seq(encoder, decoder)
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

        # Print de loss voor elke batch
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i // batch_size + 1}, Loss: {loss.item()}")

# Sla het model op (gebruik state_dict voor het opslaan van de gewichten)
torch.save(model.encoder.state_dict(), 'model/encoder.pt')
torch.save(model.decoder.state_dict(), 'model/decoder.pt')

# Sla de vocab op
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Functie om model en vocabulaire te laden voor de Flask-app
def load_model_and_vocab():
    # Laad de encoder en decoder
    encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
    decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
    model = Seq2Seq(encoder, decoder)
    
    # Laad de gewichten van het model
    model.encoder.load_state_dict(torch.load('model/encoder.pt'))
    model.decoder.load_state_dict(torch.load('model/decoder.pt'))
    
    # Laad de vocabulaire
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    model.eval()  # Zet het model in evaluatiemodus
    return model, vocab

# Functie om het antwoord te genereren
def generate_response(user_input, model, vocab):
    # Zet de input om in een tensor (deze stap hangt af van je vocabulaire en de inputvorm)
    input_tensor = torch.tensor([vocab[word] for word in user_input.split()]).unsqueeze(0)  # Dit is een eenvoudige conversie

    # Verkrijg het modelantwoord
    with torch.no_grad():
        output = model(input_tensor, input_tensor)  # Dit zou de 'input_tensor' en 'target_tensor' moeten zijn

    # Verwerk het modeloutput (bijvoorbeeld converteer het naar tekst)
    output_words = [vocab.itos[idx] for idx in output.argmax(dim=-1).squeeze().cpu().numpy()]  # Zet het modeloutput om naar tekst

    return ' '.join(output_words)
