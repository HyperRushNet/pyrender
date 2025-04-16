# chat.py

import torch
import torch.nn as nn
import requests
import random

# Constants
MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper classes and functions
class Vocabulary:
    def __init__(self):
        self.word2index = {"SOS": SOS_token, "EOS": EOS_token}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index.get(word, 0) for word in sentence.split()] + [EOS_token]

def sentence_from_indexes(vocab, indexes):
    return ' '.join([vocab.index2word.get(index, '?') for index in indexes])

def load_data(url):
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    pairs = [line.split('\t') for line in lines if '\t' in line]
    return pairs

# Simple encoder and decoder models
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Model initialization
def initialize_model(pairs):
    vocab = Vocabulary()
    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])

    hidden_size = 256
    encoder = EncoderRNN(vocab.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, vocab.n_words).to(device)
    return encoder, decoder, vocab

# Response generator
def generate_response(user_input, encoder, decoder, vocab):
    with torch.no_grad():
        input_tensor = torch.tensor(indexes_from_sentence(vocab, user_input), dtype=torch.long, device=device).view(-1, 1)
        encoder_hidden = encoder.initHidden()

        for ei in range(input_tensor.size(0)):
            _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([SOS_token], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for _ in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            next_word_index = topi.item()
            if next_word_index == EOS_token:
                break
            decoded_words.append(vocab.index2word.get(next_word_index, '?'))
            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

# Laad het model & vocab na training
def load_model_and_vocab():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    pairs = load_data(url)
    vocab = Vocabulary()

    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])

    hidden_size = 256
    encoder = EncoderRNN(vocab.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, vocab.n_words).to(device)

    # Laad getrainde gewichten
    encoder.load_state_dict(torch.load('encoder.pt', map_location=device))
    decoder.load_state_dict(torch.load('decoder.pt', map_location=device))

    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab
