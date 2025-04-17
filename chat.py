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

        # Print de loss voor elke batch
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i // batch_size + 1}, Loss: {loss.item()}")

# Sla het model op (gebruik state_dict voor het opslaan van de gewichten)
torch.save(model.encoder.state_dict(), 'model/encoder.pt')
torch.save(model.decoder.state_dict(), 'model/decoder.pt')

# Sla de vocab op
with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)


def tensor_from_sentence(vocab, sentence):
    # Zet de zin om naar een lijst van indices, update het vocabulaire als een woord niet bestaat
    indices = []
    for word in sentence.split(' '):
        if word not in vocab:
            print(f"Adding new word to vocab: {word}")
            vocab[word] = len(vocab)
        indices.append(vocab[word])

    print(f"Generated indices: {indices}")
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)


def generate_response(user_input, encoder, decoder, vocab):
    print(f"Current vocab size: {len(vocab)}")
    print(f"User input: {user_input}")

    try:
        input_tensor = tensor_from_sentence(vocab, user_input)
    except KeyError as e:
        return f"Fout bij het omzetten van de input naar tensor: {e}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    # Initialiseer de verborgen toestand van de encoder
    encoder_hidden = encoder(input_tensor)

    # Start met de decoderinvoer als <SOS>
    decoder_input = torch.tensor([[vocab.get('<SOS>', 0)]]).to(device)

    # Initialiseer de verborgen toestand van de decoder
    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(10):  # Aantal stappen in de decodering (max 10)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        ni = topi.squeeze().item()

        # Voeg het gegenereerde woord toe aan de lijst
        decoded_words.append(vocab.get(ni, '<UNK>'))

        # Stop als we het <EOS> token bereiken
        if ni == vocab.get('<EOS>', -1):
            break

        # De nieuwe decoderinput is het laatst voorspelde token
        decoder_input = torch.tensor([[ni]]).to(device)

    response = ' '.join(decoded_words)
    
    # Update vocab en sla op
    with open('model/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    return response


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
