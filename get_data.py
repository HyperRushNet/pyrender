import torch
import requests
import re

def get_ds():
    url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
    response = requests.get(url)
    response.raise_for_status()

    lines = response.text.strip().split('\n')

    input_tensor = []
    target_tensor = []
    vocab = set()

    for line in lines:
        try:
            # Verwacht dat elke regel 2 delen heeft gescheiden door een tab
            input_line, target_line = line.split('\t')

            # Voeg de input line toe aan input_tensor
            # Verdeel de inputregel in woorden en voeg deze toe aan de vocab
            input_tensor.append([word for word in input_line.split() if word.strip() != ''])
            vocab.update(input_line.split())

            # Verwerk de target_line naar numerieke waarden
            target_tensor.append([int(re.sub(r'\D', '', word)) for word in target_line.split() if re.sub(r'\D', '', word) != ''])
            vocab.update(target_line.split())

        except ValueError:
            # Fout afhandelen als een regel niet in 2 delen kan worden gesplitst
            print(f"Skipping invalid line: {line}")
            continue

    # Maak een vocabulaire van woorden naar indices
    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
    
    # Zet input_tensor om naar een tensor van indices
    input_tensor = [torch.tensor([vocab[word] for word in sentence], dtype=torch.long) for sentence in input_tensor]
    target_tensor = torch.tensor(target_tensor, dtype=torch.long)

    return input_tensor, target_tensor, vocab
