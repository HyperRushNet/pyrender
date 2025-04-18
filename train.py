import os
import pickle
import torch

# Pad voor opslaan van vocab en model
tmp_dir = './tmp'
os.makedirs(tmp_dir, exist_ok=True)

model_dir = './model'
os.makedirs(model_dir, exist_ok=True)

# Voorbeeld vocab en model simulatie
vocab = {"word1": 1, "word2": 2, "word3": 3}  # Vervang door werkelijke vocab
model_weights = torch.nn.Linear(10, 10)  # Vervang door je werkelijke model

# Sla vocab op
with open(f'{tmp_dir}/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Sla model op
torch.save(model_weights.state_dict(), f'{model_dir}/model_weights.pth')

print(f"Model en vocab opgeslagen in {tmp_dir} en {model_dir}")
