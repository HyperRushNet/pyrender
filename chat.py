import torch
import torch.nn as nn
import torch.optim as optim

# Vocabulary class to manage the vocabulary
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.n_words = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

# Example Encoder-Decoder Model class
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_rnn = nn.LSTM(embed_size, hidden_size)
        self.decoder_rnn = nn.LSTM(embed_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, target_seq):
        embedded_input = self.embedding(input_seq)
        encoder_output, (hidden, cell) = self.encoder_rnn(embedded_input)
        decoder_output, _ = self.decoder_rnn(embedded_input, (hidden, cell))
        output = self.output_layer(decoder_output)
        return output

# Example function to load model and vocab
def load_model_and_vocab():
    vocab = Vocabulary()
    # Load your model here
    encoder = Seq2Seq(vocab.n_words, embed_size=256, hidden_size=512)
    decoder = Seq2Seq(vocab.n_words, embed_size=256, hidden_size=512)
    return encoder, decoder, vocab
