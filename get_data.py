import torch
import requests
import re

def get_ds():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    response = requests.get(url)
    response.raise_for_status()  # Zorg ervoor dat de request succesvol was

    lines = response.text.strip().split('\n')

    input_tensor = []
    target_tensor = []
    vocab = set()

    for line in lines:
        input_line, target_line = line.split('\t')
        # Alleen numerieke waarden toevoegen aan target_tensor
        target_tensor.append([int(re.sub(r'\D', '', word)) for word in target_line.split() if re.sub(r'\D', '', word) != ''])
        vocab.update(input_line.split())
        vocab.update(target_line.split())

    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
    input_tensor = torch.tensor(input_tensor)
    target_tensor = torch.tensor(target_tensor)

    return input_tensor, target_tensor, vocab
