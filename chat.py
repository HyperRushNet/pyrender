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

# Debug print statements om de data te controleren
print(f"Input Tensor Length: {len(input_tensor)}")
print(f"Target Tensor Length: {len(target_tensor)}")
print(f"Vocabulary Size: {len(vocab)}")

# Controleer of de data leeg is
if len(input_tensor) == 0 or len(target_tensor) == 0:
    print("Error: No training data available.")
    exit(1)

# Initialiseer het model
encoder = Encoder(vocab_size=len(vocab), hidden_size=hidden_dim)
decoder = Decoder(vocab_size=len(vocab), hidden_size=hidden_dim)
model = Seq2Seq(encoder, decoder)
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Train het model
train_model = True  # Zet dit op True om te trainen

if train_model:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0  # Variable to accumulate loss for each epoch
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

            total_loss += loss.item()  # Accumulate loss for the epoch

        # Print de loss na elke epoch
        avg_loss = total_loss / (len(input_tensor) // batch_size)  # Gemiddelde loss per epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Sla het model op (encoder, decoder)
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
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)

def generate_response(user_input, encoder, decoder, vocab):
    print(f"User input: {user_input}")
    input_tensor = tensor_from_sentence(vocab, user_input)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    # Verkrijg de verborgen toestand van de encoder
    encoder_output, encoder_hidden = encoder(input_tensor)

    # Debug: Controleer de vorm van de encoderoutput
    print(f"Encoder hidden state shape: {encoder_hidden[0].shape}")  # Controleer de vorm van de eerste laag van de encoder

    decoder_input = torch.tensor([[vocab.get('<SOS>', 0)]]).to(device)  # Gebruik <SOS> token voor de decoder input

    # De eerste verborgen toestand van de encoder wordt gebruikt voor de decoder
    decoder_hidden = encoder_hidden[0]  # Neem de eerste waarde van de encoderoutput als verborgen toestand

    decoded_words = []

    for di in range(10):  # Aantal stappen in de decodering (max 10)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # Voer de decoder door
        topv, topi = decoder_output.topk(1)  # Haal het meest waarschijnlijke token uit
        ni = topi.squeeze().item()

        decoded_words.append(vocab.get(ni, '<UNK>'))

        if ni == vocab.get('<EOS>', -1):  # Stop als je het <EOS> token bereikt
            break

        decoder_input = torch.tensor([[ni]]).to(device)  # Geef het nieuwe token door aan de decoder

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
