from flask import Flask, jsonify
from chat import load_model_and_vocab

app = Flask(__name__)

# Laad model en vocab
try:
    model, vocab = load_model_and_vocab()
except FileNotFoundError as e:
    print(f"Fout: {e}")
    model = None
    vocab = None

@app.route('/')
def home():
    if model is None or vocab is None:
        return jsonify({"error": "Model of vocab bestanden niet gevonden"}), 500
    return jsonify({"message": "Model en vocab geladen!"})

if __name__ == '__main__':
    app.run(debug=True)
