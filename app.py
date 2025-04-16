from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import generate_response, load_data, initialize_model

app = Flask(__name__)
CORS(app)  # CORS inschakelen voor alle routes

# Laad het model en vocabulaire bij het opstarten van de server
url = 'https://hyperrushnet.github.io/ai-training/data/ds1.txt'
pairs = load_data(url)
encoder, decoder, vocab = initialize_model(pairs)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'Geen invoer ontvangen'}), 400
    response = generate_response(user_input, encoder, decoder, vocab)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
