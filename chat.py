import os
import pickle

def load_model_and_vocab():
    # Pad naar de model map
    model_dir = './model'
    
    # Bestanden in de model map
    vocab_file = os.path.join(model_dir, 'vocab.pkl')
    model_file = os.path.join(model_dir, 'model.pkl')

    # Debugging: Controleer of de bestanden bestaan
    print(f"Bestand vocab: {os.path.exists(vocab_file)}")
    print(f"Bestand model: {os.path.exists(model_file)}")

    # Laad het vocab bestand
    try:
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Het bestand '{vocab_file}' is niet gevonden.")

    # Laad het model bestand
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Het bestand '{model_file}' is niet gevonden.")

    # Retourneer model en vocab
    return model, vocab
