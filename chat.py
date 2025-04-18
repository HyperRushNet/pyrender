import os
import pickle

def load_model_and_vocab():
    # Definieer het pad naar de modelbestanden
    tmp_dir = './model'

    # Zorg ervoor dat de map bestaat
    if not os.path.exists(tmp_dir):
        raise FileNotFoundError(f"De map '{tmp_dir}' bestaat niet.")

    # Probeer de vocab en het model bestand te laden
    try:
        with open(f'{tmp_dir}/vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Het bestand '{tmp_dir}/vocab.pkl' is niet gevonden.")

    try:
        with open(f'{tmp_dir}/model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Het bestand '{tmp_dir}/model.pkl' is niet gevonden.")

    # Retourneer geladen model en vocab
    return model, vocab
