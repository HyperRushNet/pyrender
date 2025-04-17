from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import generate_response, load_model_and_vocab

app = Flask(__name__)
CORS(app)

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

# === chat.py ===
import torch
from Seq2Seq import Encoder, Decoder
from vocab import Vocabulary

def load_model_and_vocab():
    vocab = Vocabulary.load('vocab.py')
    encoder = Encoder(vocab)
    decoder = Decoder(vocab)
    encoder.load_state_dict(torch.load('model/encoder.pt'))
    decoder.load_state_dict(torch.load('model/decoder.pt'))
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab

def generate_response(input_text, encoder, decoder, vocab):
    tokens = input_text.split()
    indices = [vocab.word2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor(indices).unsqueeze(1)
    hidden = encoder(input_tensor)
    output, _ = decoder(input_tensor, hidden)
    top_indices = output.argmax(dim=2).squeeze().tolist()
    response = ' '.join([vocab.index2word[i] for i in top_indices])
    return response

# === train.py ===
import torch
import torch.optim as optim
from Seq2Seq import Encoder, Decoder, Seq2Seq
from vocab import Vocabulary

with open('training_data/ds1.txt') as f:
    lines = f.read().splitlines()

pairs = [line.split('\t') for line in lines if '\t' in line]
vocab = Vocabulary([p[0] + ' ' + p[1] for p in pairs])

encoder = Encoder(vocab)
decoder = Decoder(vocab)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    total_loss = 0
    for input_text, target_text in pairs:
        input_tensor = torch.tensor([vocab.word2index[w] for w in input_text.split()]).unsqueeze(1)
        target_tensor = torch.tensor([vocab.word2index[w] for w in target_text.split()]).unsqueeze(1)
        optimizer.zero_grad()
        output = model(input_tensor, target_tensor)
        loss = loss_fn(output.squeeze(), target_tensor.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(encoder.state_dict(), 'model/encoder.pt')
torch.save(decoder.state_dict(), 'model/decoder.pt')

# === Seq2Seq.py ===
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab.size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab.size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab.size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        hidden = self.encoder(input_seq)
        output, _ = self.decoder(target_seq, hidden)
        return output
