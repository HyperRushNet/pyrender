from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import generate_response

app = Flask(__name__)
CORS(app)  # CORS inschakelen voor alle routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'Geen invoer ontvangen'}), 400
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
