from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import generate_response, load_model_and_vocab

app = Flask(__name__)
CORS(app)

# Laad het model en vocab bij het opstarten
encoder, decoder, vocab = load_model_and_vocab()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET'])
def chat():
    user_input = request.args.get('q', '')
    if not user_input:
        return jsonify({'error': 'Geen vraag ontvangen, gebruik ?q=<je vraag>'}), 400
    try:
        response = generate_response(user_input, encoder, decoder, vocab)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'Er is een fout opgetreden: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
