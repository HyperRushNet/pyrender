import torch
from Seq2Seq import Encoder, Decoder, Seq2Seq
from vocab import Vocabulary

def load_model_and_vocab():
    encoder = Encoder(vocab_size=10000)  # stel vocab_size in op basis van je model
    decoder = Decoder(vocab_size=10000)
    encoder.load_state_dict(torch.load('model/encoder.pt'))
    decoder.load_state_dict(torch.load('model/decoder.pt'))
    vocab = Vocabulary.load('model/vocab.pkl')
    return encoder, decoder, vocab

def generate_response(input_text, encoder, decoder, vocab):
    input_seq = torch.tensor([vocab.word2index[word] for word in input_text.split()]).unsqueeze(0)
    hidden = encoder(input_seq)
    output, _ = decoder(input_seq, hidden)
    response = ' '.join([vocab.index2word[idx.item()] for idx in output.argmax(dim=2).squeeze()])
    return response
