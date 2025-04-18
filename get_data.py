import torch
from torch.nn.utils.rnn import pad_sequence
import requests
import re

def get_ds():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.strip().split('\n')

    input_tensor = []
    target_tensor = []
    vocab = set()  # Een set voor unieke woorden

    # Verzamel alle woorden in vocab
    for line in lines:
        input_line, target_line = line.split('\t')
        
        input_tokens = input_line.split()
        target_tokens = target_line.split()

        vocab.update(input_tokens)
        vocab.update(target_tokens)

    # Maak vocab een gesorteerde lijst
    vocab = sorted(vocab)

    # Maak een woordenboek om snel indices op te halen
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    for line in lines:
        input_line, target_line = line.split('\t')

        input_tokens = input_line.split()
        target_tokens = target_line.split()

        # Convert de input tokens naar indices
        input_tensor.append(torch.tensor([word_to_idx.get(word, 0) for word in input_tokens], dtype=torch.long))

        # Voeg numerieke waarden toe aan target_tensor
        target_tensor.append([int(re.sub(r'\D', '', word)) for word in target_tokens if re.sub(r'\D', '', word) != ''])

    # Pad de input_tensor zodat alle sequenties dezelfde lengte hebben
    input_tensor = pad_sequence(input_tensor, batch_first=True, padding_value=0)

    target_tensor = torch.tensor(target_tensor, dtype=torch.long)

    return input_tensor, target_tensor, word_to_idx
