import torch
import pickle
from Seq2Seq import seq2seq
from torch import nn
from torch.optim import Adam

# Functie om het model en vocab te laden
def load_model_and_vocab():
    # Laad vocab
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Laad model
    encoder = torch.load('model/encoder.pt')
    decoder = torch.load('model/decoder.pt')
    
    return encoder, decoder, vocab

# Functie om een response van het model te genereren
def generate_response(user_input, encoder, decoder, vocab):
    # Preprocess de input
    input_tensor = torch.tensor([vocab.get(word, vocab['<unk>']) for word in user_input.split()]).unsqueeze(0)

    # Verwerk de input door het model
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([vocab['<sos>']])
        decoder_hidden = encoder_hidden

        output_words = []
        for _ in range(100):  # Max output length
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == vocab['<eos>']:
                break

            output_words.append(vocab.get(decoder_input.item(), '<unk>'))

        return ' '.join(output_words)
