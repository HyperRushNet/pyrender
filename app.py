from flask import Flask
from chat import load_model_and_vocab

app = Flask(__name__)

@app.route('/')
def home():
    model, vocab = load_model_and_vocab()
    return "Model en vocab succesvol geladen!"

if __name__ == '__main__':
    app.run(debug=True)
