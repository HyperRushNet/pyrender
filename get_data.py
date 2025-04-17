import torch
import requests
import re  # Vergeet deze import niet!

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

        # Voeg numerieke waarden toe aan target_tensor
        target_tensor.append([int(re.sub(r'\D', '', word)) for word in target_line.split() if re.sub(r'\D', '', word) != ''])

        # Vul input_tensor en vocab met de juiste waarden
        input_tensor.append([vocab.get(word, 0) for word in input_line.split()])  # Gebruik 0 voor onbekende woorden
        vocab.update(input_line.split())
        vocab.update(target_line.split())

    # Zorg ervoor dat vocab een gesorteerde index heeft
    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}

    # Als input_tensor leeg is, geef dan een waarschuwing en voeg een standaardinvoer toe
    if len(input_tensor) == 0:
        print("Waarschuwing: Geen invoer gevonden. Controleer de dataformaten.")
        default_input = "Hallo, hoe gaat het?"  # Standaardinvoer
        input_tensor = [[vocab.get(word, 0) for word in default_input.split()]]  # Standaard input als tensor

    # Zorg ervoor dat de tensoren correct zijn voor input en target
    input_tensor = torch.tensor(input_tensor, dtype=torch.long)
    target_tensor = torch.tensor(target_tensor, dtype=torch.long)

    return input_tensor, target_tensor, vocab

# Verifieren of de functie werkt:
input_tensor, target_tensor, vocab = get_ds()
print(f"Input tensor: {input_tensor}")
print(f"Target tensor: {target_tensor}")
print(f"Vocabulary: {vocab}")
