import os
import torch
import pickle
from torch import nn
from torch.optim import Adam
from model.Seq2Seq import Seq2Seq, Encoder, Decoder
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

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Sla het model op
torch.save(model.encoder.state_dict(), 'model/encoder.pt')
torch.save(model.decoder.state_dict(), 'model/decoder.pt')

# Sla de vocab op
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)


def tensor_from_sentence(vocab, sentence):
    # Zet de zin om naar een lijst van indices, update het vocabulaire als een woord niet bestaat
    indices = []
    for word in sentence.split(' '):
        # Voeg het woord toe aan vocab als het nog niet bestaat
        if word not in vocab:
            vocab[word] = len(vocab)
            print(f"Adding new word to vocab: {word}")
        indices.append(vocab[word])
    
    # Zet de lijst van indices om naar een tensor
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)


def generate_response(user_input, encoder, decoder, vocab):
    print(f"User input: {user_input}")

    # Converteer de user input naar indices
    input_tensor = tensor_from_sentence(vocab, user_input)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    # Verkrijg de verborgen toestand van de encoder
    encoder_hidden = encoder(input_tensor)

    # Begin de decodering met <SOS>
    decoder_input = torch.tensor([[vocab.get('<SOS>', len(vocab))]]).to(device)  # Start met SOS
    decoded_words = []

    for _ in range(10):  # Aantal stappen in de decodering
        decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden)
        topv, topi = decoder_output.topk(1)
        ni = topi.squeeze().item()

        # Voeg het gegenereerde woord toe aan de lijst van gedecodeerde woorden
        decoded_words.append(ni)

        # Stop als het EOS-token is bereikt
        if ni == vocab.get('<EOS>', len(vocab) - 1):
            break

        # De nieuwe decoder input is het laatst gegenereerde token
        decoder_input = torch.tensor([[ni]]).to(device)

    # Zet de output indices om naar woorden
    reverse_vocab = {v: k for k, v in vocab.items()}
    output_words = [reverse_vocab.get(idx, '<UNK>') for idx in decoded_words]

    # Return het antwoord
    return ' '.join(output_words)


def load_model_and_vocab():
    # Laad het vocabulaire
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Laad het model
    encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
    decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
    model = Seq2Seq(encoder, decoder)

    # Laad de opgeslagen gewichten
    encoder.load_state_dict(torch.load('model/encoder.pt'))
    decoder.load_state_dict(torch.load('model/decoder.pt'))

    return encoder, decoder, vocab
