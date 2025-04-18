from flask import Flask, request, jsonify
from model import model, tokenizer, predict_next_word

app = Flask(__name__)

@app.route('/')
def home():
    return "Welkom bij de woordvoorspeller API!"

@app.route('/predict', methods=['GET'])
def predict():
    input_text = request.args.get('q')
    
    if input_text:
        predicted_word = predict_next_word(model, tokenizer, input_text)
        return jsonify({"input_text": input_text, "predicted_word": predicted_word})
    else:
        return jsonify({"error": "Geen tekst opgegeven. Gebruik ?q=parameter."}), 400

if __name__ == "__main__":
    app.run(debug=True)
