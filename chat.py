import torch
import torch.nn as nn
import pickle
import numpy as np
from model import Seq2Seq, Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_vocab():
    # Laad het vocabulaire
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Laad de encoder en decoder modellen
    encoder = torch.load('model/encoder.pt', map_location=device)
    decoder = torch.load('model/decoder.pt', map_location=device)
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, vocab

def generate_response(user_input, encoder, decoder, vocab):
    # Zet de input om naar tokens en voer padding uit
    input_tokens = [vocab.word2index.get(word, vocab.word2index['<UNK>']) for word in user_input.split()]
    input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Zet de input door de encoder
    encoder_hidden = encoder.init_hidden(input_tensor.size(0))
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    
    # Initialiseer de decoder
    decoder_input = torch.tensor([vocab.word2index['<SOS>']], device=device).unsqueeze(0)
    decoder_hidden = encoder_hidden
    
    decoded_words = []
    for di in range(50):  # Beperk de lengte van de output
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Verkrijg het volgende woord
        
        if decoder_input.item() == vocab.word2index['<EOS>']:
            break
        
        decoded_words.append(vocab.index2word[decoder_input.item()])
    
    return ' '.join(decoded_words)
