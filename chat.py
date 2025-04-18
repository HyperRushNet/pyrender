import torch
import pickle
from model.Seq2Seq import Seq2Seq, Encoder, Decoder

# Zorg ervoor dat het model naar de juiste device wordt gestuurd (GPU of CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Laad het vocabulaire en model
def load_model_and_vocab():
    # Laad het vocabulaire
    with open('model/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Laad het model
    encoder = Encoder(vocab_size=len(vocab), hidden_size=512)
    decoder = Decoder(vocab_size=len(vocab), hidden_size=512)
    model = Seq2Seq(encoder, decoder)
    model.to(device)  # Verplaats het model naar de juiste device

    # Laad de opgeslagen gewichten
    encoder.load_state_dict(torch.load('model/encoder.pt'))
    decoder.load_state_dict(torch.load('model/decoder.pt'))

    return encoder, decoder, vocab

# Functie om een reactie van de gebruiker te genereren
def generate_response(user_input, encoder, decoder, vocab):
    print(f"User input: {user_input}")
    input_tensor = tensor_from_sentence(vocab, user_input)
    
    input_tensor = input_tensor.to(device)  # Verplaats de input tensor naar de juiste device

    # Verkrijg de verborgen toestand van de encoder
    encoder_output, encoder_hidden = encoder(input_tensor)

    # Debug: Controleer de vorm van de encoderoutput
    print(f"Encoder hidden state shape: {encoder_hidden[0].shape}")  # Controleer de vorm van de eerste laag van de encoder

    decoder_input = torch.tensor([[vocab.get('<SOS>', 0)]]).to(device)  # Gebruik <SOS> token voor de decoder input

    # De eerste verborgen toestand van de encoder wordt gebruikt voor de decoder
    decoder_hidden = encoder_hidden[0]  # Neem de eerste waarde van de encoderoutput als verborgen toestand

    decoded_words = []

    for di in range(10):  # Aantal stappen in de decodering (max 10)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # Voer de decoder door
        topv, topi = decoder_output.topk(1)  # Haal het meest waarschijnlijke token uit
        ni = topi.squeeze().item()

        decoded_words.append(vocab.get(ni, '<UNK>'))

        if ni == vocab.get('<EOS>', -1):  # Stop als je het <EOS> token bereikt
            break

        decoder_input = torch.tensor([[ni]]).to(device)  # Geef het nieuwe token door aan de decoder

    response = ' '.join(decoded_words)

    return response
