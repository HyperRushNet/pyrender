import os
import pickle
import torch
from model.Seq2Seq import Seq2Seq, Encoder, Decoder

# Zorg ervoor dat het model naar de juiste device wordt gestuurd (GPU of CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Zorg ervoor dat de tijdelijke map wordt aangemaakt
tmp_dir = './tmp'
os.makedirs(tmp_dir, exist_ok=True)  # Maakt de map aan als deze nog niet bestaat

# Laad het vocabulaire en model
def load_model_and_vocab():
    vocab_path = os.path.join(tmp_dir, 'vocab.pkl')
    encoder_path = os.path.join(tmp_dir, 'encoder.pt')
    decoder_path = os.path.join(tmp_dir, 'decoder.pt')

    # Laad het vocabulaire uit de tijdelijke map
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Laad het model uit de tijdelijke map
    encoder = Encoder(vocab_size=len(vocab), hidden_size=512)
    decoder = Decoder(vocab_size=len(vocab), hidden_size=512)
    model = Seq2Seq(encoder, decoder)
    model.to(device)  # Verplaats het model naar de juiste device

    # Laad de opgeslagen gewichten
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    return encoder, decoder, vocab
