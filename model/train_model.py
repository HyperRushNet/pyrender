from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Laad dataset
dataset = load_dataset("text", data_files={"train": "data/dataset.txt"})

# Laad een pre-trained model en tokenizer (bijvoorbeeld GPT-2)
model_name = "gpt2"  # Je kunt hier een ander model kiezen
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Voorbereiden van de dataset (tokenizeer de tekst)
def encode(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

dataset = dataset.map(encode, batched=True)

# Configuratie voor training
training_args = TrainingArguments(
    output_dir="./model",          # Waar de modelbestanden worden opgeslagen
    num_train_epochs=3,            # Aantal trainingsepochs
    per_device_train_batch_size=4, # Batchgrootte
    save_steps=10_000,             # Opslaan elke 10.000 stappen
    save_total_limit=2,            # Aantal opgeslagen modellen
)

# Train de model
trainer = Trainer(
    model=model,                         # Het model dat je wilt trainen
    args=training_args,                  # De training argumenten
    train_dataset=dataset["train"],      # Dataset
)

trainer.train()

# Sla het model op na het trainen
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
