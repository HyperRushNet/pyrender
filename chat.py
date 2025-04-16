import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import string
from collections import defaultdict

# Vooraf gedefinieerde parameters
MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Laad en verwerk de dataset
def load_data(filename):
    pairs = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            input_sentence, output_sentence = line.strip().split('\t')
            pairs.append((input_sentence, output_sentence))
    return pairs

# Tokenizeer een zin
def tokenize(sentence):
    return list(sentence)

# Maak een vocabulaire
def build_vocab(pairs):
    vocab = defaultdict(lambda: len(vocab))
    vocab['<SOS>'] = SOS_token
    vocab['<EOS>'] = EOS_token
    for pair in pairs:
        for sentence in pair:
            for word in tokenize(sentence):
                vocab[word]
    return vocab

# Zet een zin om in indices
def indexes_from_sentence(vocab, sentence):
    return [vocab[word] for word in tokenize(sentence)] + [EOS_token]

# Zet een lijst van indices om in een zin
def sentence_from_indexes(vocab, indexes):
    reverse_vocab = {index: word for word, index in vocab.items()}
    return ''.join([reverse_vocab[index] for index in indexes if index not in [SOS_token, EOS_token]])

# Definieer het Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Definieer het Decoder model
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_step, hidden):
        embedded = self.embedding(input_step).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Initialiseer het model
def initialize_model(pairs):
    vocab = build_vocab(pairs)
    input_lang_size = len(vocab)
    output_lang_size = len(vocab)
    hidden_size = 256

    encoder = Encoder(input_lang_size, hidden_size).to(device)
    decoder = Decoder(hidden_size, output_lang_size).to(device)

    return encoder, decoder, vocab

# Train het model
def train(encoder, decoder, pairs, vocab, n_iters=1000, print_every=100):
    encoder.train()
    decoder.train()

    for iter in range(1, n_iters + 1):
        training_pair = random.choice(pairs)
        input_tensor = torch.tensor(indexes_from_sentence(vocab, training_pair[0]), dtype=torch.long, device=device).view(-1, 1)
        target_tensor = torch.tensor(indexes_from_sentence(vocab, training_pair[1]), dtype=torch.long, device=device).view(-1, 1)

        encoder_hidden = encoder.initHidden()

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0
        decoder_input = torch.tensor([SOS_token], device=device)

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_hidden = encoder_hidden

        for di in range(target_tensor.size(0)):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += nn.NLLLoss()(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        if iter % print_every == 0:
            print(f"Iteration {iter} Loss: {loss.item() / target_tensor.size(0)}")

# Genereer een antwoord
def generate_response(encoder, decoder, vocab, input_sentence):
    with torch.no_grad():
        input_tensor = torch.tensor(indexes_from_sentence(vocab, input_sentence), dtype=torch.long, device=device).view(-1, 1)
        encoder_hidden = encoder.initHidden()

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([SOS_token], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            decoded_words.append(vocab[decoder_input.item()])

            if decoder_input.item() == EOS_token:
                break

        return ''.join(decoded_words)

# Hoofdfunctie
def main():
    pairs = load_data('https://hyperrushnet.github.io/ai-training/data/ds1.txt')
    encoder, decoder, vocab = initialize_model(pairs)
    train(encoder, decoder, pairs, vocab)
    while True:
        input_sentence = input("You: ")
        if input_sentence.lower() in ['quit', 'exit']:
            break
        response = generate_response(encoder, decoder, vocab, input_sentence)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
