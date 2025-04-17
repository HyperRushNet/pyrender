import torch
import pickle
from torch import nn

### Zelfgemaakte Encoder-Decoder ter vervanging van `seq2seq.Seq2Seq` ###
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input_tensor):
        embedded = self.embedding(input_tensor)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_tensor, hidden, encoder_outputs):
        embedded = self.embedding(input_tensor).unsqueeze(0)
        output, (hidden, cell) = self.lstm(embedded, hidden)
        output = self.out(output)
        output = self.softmax(output)
        return output, (hidden, cell)

### Originele functionaliteit (aangepast voor onze nieuwe klassen) ###
def load_model_and_vocab():
    # Laad vocab
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Laad model (nu met onze eigen Encoder/Decoder)
    encoder = torch.load('model/encoder.pt')
    decoder = torch.load('model/decoder.pt')
    
    return encoder, decoder, vocab

def generate_response(user_input, encoder, decoder, vocab):
    # Preprocess de input
    input_tensor = torch.tensor([vocab.get(word, vocab['<unk>']) for word in user_input.split()]).unsqueeze(0)

    # Verwerk de input door het model
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = encoder(input_tensor)
        decoder_input = torch.tensor([vocab['<sos>']])
        decoder_hidden = (hidden, cell)

        output_words = []
        for _ in range(100):  # Max output length
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == vocab['<eos>']:
                break

            output_words.append(vocab.get(decoder_input.item(), '<unk>'))

        return ' '.join(output_words)
