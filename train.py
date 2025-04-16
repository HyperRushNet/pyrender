# train.py

import torch
import torch.optim as optim
import random
from chat import load_data, Vocabulary, EncoderRNN, DecoderRNN, train

# Functie om te trainen en model op te slaan
def train_model():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    pairs = load_data(url)
    vocab = Vocabulary()

    # Voeg woorden toe aan de vocabulaire
    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])

    # Maak encoder en decoder modellen
    hidden_size = 256
    encoder = EncoderRNN(vocab.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, vocab.n_words).to(device)

    # Stel optimizer en verliesfunctie in
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))
    criterion = nn.NLLLoss()

    # Start het trainen
    for epoch in range(1000):
        pair = random.choice(pairs)
        input_tensor = torch.tensor(indexes_from_sentence(vocab, pair[0]), dtype=torch.long, device=device).view(-1, 1)
        target_tensor = torch.tensor(indexes_from_sentence(vocab, pair[1]), dtype=torch.long, device=device).view(-1, 1)

        loss = train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Sla de getrainde modellen op
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')
    print("Training finished and models saved.")

if __name__ == '__main__':
    train_model()
