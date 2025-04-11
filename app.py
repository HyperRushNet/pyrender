from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Maak de Flask-app
app = Flask(__name__)

# Laad het GPT-2 model en tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Functie om een vraag om te zetten in een antwoord met behulp van GPT-2
def generate_answer(question):
    input_text = question + " Antwoord:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer[len(input_text):]
    return answer.strip()

@app.route('/')
def home():
    return 'Welkom bij de dynamische vraag-antwoord bot! Gebruik /vraag?q=je_vraag om een vraag te stellen.'

@app.route('/vraag', methods=['GET'])
def vraag():
    user_question = request.args.get('q', '')
    if user_question:
        answer = generate_answer(user_question)
        return jsonify({'vraag': user_question, 'antwoord': answer})
    else:
        return jsonify({'error': 'Geen vraag opgegeven. Geef een vraag mee in de URL, bijvoorbeeld: /vraag?q=wat is je naam?'})

# Start de Flask-app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
