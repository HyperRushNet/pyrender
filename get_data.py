import torch
import re
import requests

def get_ds():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.strip().split('\n')

    input_tensor = []
    target_tensor = []
    vocab = set()

    for line in lines:
        input_line, target_line = line.split('\t')
        # Voeg numerieke waarden toe aan target_tensor en woorden aan vocab
        target_tensor.append([int(re.sub(r'\D', '', word)) for word in target_line.split() if re.sub(r'\D', '', word) != ''])
        vocab.update(input_line.split())
        vocab.update(target_line.split())

    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
    
    # Nu moeten we input_tensor als een lijst van indices maken
    for line in lines:
        input_line, _ = line.split('\t')
        input_tensor.append([vocab[word] for word in input_line.split()])

    # Zet input_tensor en target_tensor om naar tensors van het juiste type
    input_tensor = torch.tensor(input_tensor, dtype=torch.long)
    target_tensor = torch.tensor(target_tensor, dtype=torch.long)

    return input_tensor, target_tensor, vocab
