import torch
import torch.nn as nn
import torch.optim as optim
import requests
import random
import numpy as np
import torch.nn.functional as F

# Definieer de Vocabulary klasse om de vocabulaire op te bouwen
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# Functie om de dataset van een URL te lezen
def read_data_from_url(url):
    pairs = []
    response = requests.get(url)
    response.raise_for_status()  # Controleer op fouten

    # Lees de inhoud van de URL (de dataset)
    lines = response.text.splitlines()
    for line in lines:
        # Split de regels in vraag en antwoord
        input_sentence, target_sentence = line.strip().split('\t')
        pairs.append((input_sentence, target_sentence))
    
    return pairs

# Functie om de data en vocabulaire voor te bereiden
def prepare_data_from_url(url):
    pairs = read_data_from_url(url)
    vocab = Vocabulary()

    # Voeg woorden toe aan vocabulaire
    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])

    return pairs, vocab

# Definieer de Encoder klasse voor het Seq2Seq-model
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Definieer de Decoder klasse voor het Seq2Seq-model
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Definieer het Seq2Seq-model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, embed_size, hidden_size)
        self.decoder = Decoder(output_size, embed_size, hidden_size)

    def forward(self, input_seq, target_seq):
        encoder_hidden, encoder_cell = self.encoder(input_seq)
        decoder_input = target_seq[0]  # Start token
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        outputs = []

        for t in range(1, target_seq.size(0)):  # Start van 1 omdat 0 de starttoken is
            output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs.append(output)
            decoder_input = target_seq[t]  # Volgende token

        return torch.stack(outputs)

# Trainfunctie voor het model
def train_model(model, train_data, vocab, epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_data)
        for input_seq, target_seq in train_data:
            input_indices = torch.tensor([vocab.word2index[word] for word in input_seq.split(' ')], dtype=torch.long)
            target_indices = torch.tensor([vocab.word2index[word] for word in target_seq.split(' ')], dtype=torch.long)

            # Voeg batch-dimensie toe
            input_indices = input_indices.unsqueeze(0)
            target_indices = target_indices.unsqueeze(0)

            optimizer.zero_grad()
            output = model(input_indices, target_indices)  # Forward pass

            # Compute de verliesfunctie (cross entropy loss)
            loss = criterion(output.view(-1, vocab.n_words), target_indices.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}')

# De functie om de data voor te bereiden en het model te trainen
def train_from_url(url):
    # Laad de data en vocabulaire
    pairs, vocab = prepare_data_from_url(url)
    
    # Initialiseer het model
    embed_size = 256
    hidden_size = 512
    model = Seq2Seq(vocab.n_words, vocab.n_words, embed_size, hidden_size)

    # Train het model
    train_model(model, pairs, vocab, epochs=10, learning_rate=0.001)

# Voorbeeld van het trainen met de data van de URL
url = "https://hyperrushnet.github.io/ai-training/data/ds1.txt"  # De URL van je dataset
train_from_url(url)
