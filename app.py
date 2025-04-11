from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hallo vanaf je AI backend op Render!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    vraag = data.get('vraag', '')
    
    antwoord = f"Je vroeg: {vraag}. Hier komt je AI antwoord 😉"
    
    return jsonify({"antwoord": antwoord})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
