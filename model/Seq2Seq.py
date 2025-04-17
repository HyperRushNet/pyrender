import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq):
        # Input naar embedding laag
        embedded = self.embedding(input_seq)
        # Forward pass door de RNN (GRU)
        _, hidden = self.rnn(embedded)
        return hidden  # Geef de verborgen toestand terug

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)  # Embedding van de input
        output, hidden = self.rnn(embedded, hidden)  # RNN output met verborgen toestand
        output = self.fc(output)  # Devolueer de output naar het vocabulaire
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        hidden = self.encoder(input_seq)  # Encoder verwerkt input_seq en geeft hidden toestand
        output, _ = self.decoder(target_seq, hidden)  # Decoder verwerkt target_seq met hidden toestand
        return output
