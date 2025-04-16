import torch
import torch.nn as nn
import torch.nn.functional as F
import string
import random

# Karakterset en mappings
all_chars = string.ascii_letters + " .,;'-"
n_chars = len(all_chars)
char_to_ix = {ch: i for i, ch in enumerate(all_chars)}
ix_to_char = {i: ch for i, ch in enumerate(all_chars)}

# Eenvoudig RNN-model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Initialiseer model
hidden_size = 128
model = CharRNN(n_chars, hidden_size, n_chars)

# Laad getraind model als beschikbaar
try:
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
except FileNotFoundError:
    print("Waarschuwing: 'model.pth' niet gevonden. Zorg ervoor dat het model is getraind en opgeslagen.")

# Functie om tekst te genereren
def generate_response(start_str='Hello', length=100):
    with torch.no_grad():
        input = char_tensor(start_str[0])
        hidden = model.init_hidden()
        output_str = start_str

        for i in range(length):
            output, hidden = model(input, hidden)
            output_dist = output.data.view(-1).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            predicted_char = ix_to_char[top_i.item()]
            output_str += predicted_char
            input = char_tensor(predicted_char)

        return output_str

# Helperfunctie om karakter naar tensor te converteren
def char_tensor(char):
    tensor = torch.zeros(1, n_chars)
    if char in char_to_ix:
        tensor[0][char_to_ix[char]] = 1
    return tensor
