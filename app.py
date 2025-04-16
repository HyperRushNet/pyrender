from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import generate_response

app = Flask(__name__)
CORS(app)  # CORS inschakelen voor alle routes

@app.route('/')
def index():
    return render_template('index.html')

# Route voor het ophalen van de vraag via de query parameter
@app.route('/chat', methods=['GET'])
def chat():
    # Haal de vraag op uit de queryparameter ?q=
    user_input = request.args.get('q', '')  # Verkrijg de parameter 'q' uit de URL

    if not user_input:
        return jsonify({'error': 'Geen vraag ontvangen, gebruik ?q=<je vraag>'}), 400
    
    # Genereer het antwoord via je chatbotfunctie
    response = generate_response(user_input)
    
    # Geef het antwoord terug in de response body
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
