# Zorg ervoor dat de benodigde imports bovenaan staan
import torch
import torch.nn as nn
import torch.optim as optim
import random
import requests

MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Het laad- en vocabulaireproces moet in de functie zitten om niet elke keer opnieuw geladen te worden.
def load_model_and_vocab():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    pairs = load_data(url)
    encoder, decoder, vocab = initialize_model(pairs)
    return encoder, decoder, vocab

# De rest van je modelcode blijft hetzelfde

def generate_response(user_input, encoder, decoder, vocab):
    # De implementatie van je generate_response functie blijft hetzelfde.
    # Gebruik encoder, decoder en vocab zoals voorheen.
    with torch.no_grad():
        input_tensor = torch.tensor(indexes_from_sentence(vocab, user_input), dtype=torch.long, device=device).view(-1, 1)
        encoder_hidden = encoder.initHidden()

        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([SOS_token], device=device)
        decoder_hidden = encoder_hidden

        decoded_indices = []
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_token:
                break
            decoded_indices.append(decoder_input.item())

        return sentence_from_indexes(vocab, decoded_indices)
