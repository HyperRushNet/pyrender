import torch
import torch.nn as nn
import torch.optim as optim
from chat import Seq2Seq, Vocabulary

# Function to train the model
def train_model(model, train_data, vocab, epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for input_seq, target_seq in train_data:
            optimizer.zero_grad()
            output = model(input_seq, target_seq)
            loss = criterion(output.view(-1, vocab.n_words), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}')

# Example of training data (to be replaced with actual data)
train_data = [("hello world", "world hello")]
vocab = Vocabulary()
model = Seq2Seq(vocab.n_words, embed_size=256, hidden_size=512)
train_model(model, train_data, vocab)
