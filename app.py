from flask import Flask, jsonify
from chat import load_model_and_vocab

app = Flask(__name__)

@app.route('/')
def index():
    try:
        # Laad het model en vocabulaire
        model, vocab = load_model_and_vocab()
        return jsonify({
            "status": "success",
            "message": "Model en vocab zijn succesvol geladen."
        })
    except FileNotFoundError as e:
        # Als de bestanden niet gevonden worden, geef een foutmelding
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404

if __name__ == "__main__":
    app.run(debug=True)
