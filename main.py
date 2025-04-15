import os
from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Voorbeeld: simpele NLG-functie
def genereer_zin(onderwerp):
    werkwoorden = ["houdt van", "haat", "eet", "bekijkt", "bespreekt"]
    dingen = ["pannenkoeken", "AI", "de zon", "code", "muziek"]
    werkwoord = random.choice(werkwoorden)
    ding = random.choice(dingen)
    return f"{onderwerp} {werkwoord} {ding}."

@app.route("/")
def home():
    return "Welkom bij je NLG API!"

@app.route("/nlg", methods=["POST"])
def nlg():
    data = request.get_json()
    onderwerp = data.get("onderwerp", "De gebruiker")
    zin = genereer_zin(onderwerp)
    return jsonify({"zin": zin})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
