from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Maak de Flask-app
app = Flask(__name__)

# Gebruik een kleiner model (sneller en minder geheugenintensief)
MODEL_NAME = "distilgpt2"

# Laad tokenizer en model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Zorg dat het model in eval mode staat
model.eval()

# Functie om een antwoord te genereren
def generate_answer(question):
    input_text = question + " Antwoord:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Genereer output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded[len(input_text):].strip()
    return answer

@app.route('/')
def home():
    return 'Welkom bij de dynamische vraag-antwoord bot! Gebruik /vraag?q=je_vraag om een vraag te stellen.'

@app.route('/vraag', methods=['GET'])
def vraag():
    user_question = request.args.get('q', '')
    if user_question:
        print(f"[LOG] Vraag ontvangen: {user_question}")
        answer = generate_answer(user_question)
        return jsonify({'vraag': user_question, 'antwoord': answer})
    else:
        return jsonify({'error': 'Geen vraag opgegeven. Geef een vraag mee in de URL, bijvoorbeeld: /vraag?q=wat is je naam?'})

# Start de Flask-app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
