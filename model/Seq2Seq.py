import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, embedding_dim=256):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)

    def init_hidden(self, batch_size):
        """Initialiseer de verborgen toestanden van de GRU naar nullen"""
        return torch.zeros(1, batch_size, self.hidden_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        # De GRU gebruikt de initialisatie van de hidden state
        hidden = self.init_hidden(input_seq.size(0))  # batch_size
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, embedding_dim=256):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        # Encoder ontvangt de input_seq en genereert hidden states
        encoder_output, encoder_hidden = self.encoder(input_seq)
        
        # Decoder gebruikt de verborgen toestand van de encoder als zijn initiële verborgen toestand
        output, _ = self.decoder(target_seq, encoder_hidden)
        return output
