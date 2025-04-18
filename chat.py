import os
import pickle
import torch

def load_model_and_vocab():
    # Pad naar de model map
    model_dir = './model'
    
    # Bestanden in de model map
    vocab_file = os.path.join(model_dir, 'vocab.pkl')
    model_weights_path = os.path.join(model_dir, 'model_weights.pth')

    # Debugging: Controleer of de bestanden bestaan
    print(f"Bestand vocab: {os.path.exists(vocab_file)}")
    print(f"Bestand model: {os.path.exists(model_weights_path)}")

    # Laad het vocab bestand
    try:
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Het bestand '{vocab_file}' is niet gevonden.")

    # Controleer of het model-bestand bestaat
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Het bestand '{model_weights_path}' is niet gevonden.")

    # Laad het model
    model = torch.load(model_weights_path)
    model.eval()  # Zet het model in evaluatiemodus
    return model, vocab
