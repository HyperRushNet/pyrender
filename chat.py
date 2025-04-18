import os
import pickle
import torch

# Zorg ervoor dat de tmp map bestaat
tmp_dir = './tmp'
os.makedirs(tmp_dir, exist_ok=True)  # Maakt de map als deze nog niet bestaat

# Pad naar het vocab bestand
vocab_file = os.path.join(tmp_dir, 'vocab.pkl')

# Laad het vocab bestand
def load_model_and_vocab():
    try:
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Het bestand '{vocab_file}' is niet gevonden.")
    
    # Laad modelgewichten uit de model map
    model_dir = './model'
    model_weights_path = os.path.join(model_dir, 'model_weights.pth')
    
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Het bestand '{model_weights_path}' is niet gevonden.")

    # Laad het model
    model = torch.load(model_weights_path)
    model.eval()  # Zet het model in evaluatiemodus
    return model, vocab
