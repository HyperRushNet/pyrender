from flask import Flask, request, jsonify
from transformers import pipeline, set_seed

app = Flask(__name__)

# Laad het model één keer bij opstart
generator = pipeline('text-generation', model='distilgpt2')
set_seed(42)

@app.route('/')
def home():
    return 'Local NLG API is running!'

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        result = generator(prompt, max_length=100, num_return_sequences=1)
        return jsonify({'response': result[0]['generated_text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
