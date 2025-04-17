from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import generate_response, load_model_and_vocab

app = Flask(__name__)
CORS(app)  # CORS inschakelen voor alle routes

# Laad het model en vocab bij het opstarten van de app
encoder, decoder, vocab = load_model_and_vocab()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET'])
def chat():
    # Haal de vraag op uit de queryparameter ?q=
    user_input = request.args.get('q', '')  # Verkrijg de parameter 'q' uit de URL

    if not user_input:
        return jsonify({'error': 'Geen vraag ontvangen, gebruik ?q=<je vraag>'}), 400

    try:
        # Genereer het antwoord via je chatbotfunctie
        response = generate_response(user_input, encoder, decoder, vocab)
        return jsonify({'response': response})  # Geef het antwoord terug in de response body
    except Exception as e:
        # Als er iets misgaat, geef een 500 error met een gedetailleerde foutmelding
        return jsonify({'error': f'Er is een fout opgetreden: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
