import torch
import torch.optim as optim
from Seq2Seq import Encoder, Decoder, Seq2Seq
from torch.utils.data import DataLoader
from vocab import Vocabulary

# Laad de trainingsdata en vocabulaire
# Veronderstel dat je get_data van de data zelf hebt
data = get_data('training_data/ds1.txt')
vocab = Vocabulary(data)
train_loader = DataLoader(data, batch_size=32, shuffle=True)

# Initialiseer het model
encoder = Encoder(vocab.size)
decoder = Decoder(vocab.size)
model = Seq2Seq(encoder, decoder)

# Definieer de optimizer en loss functie
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

# Train het model
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_seq, target_seq = batch
        optimizer.zero_grad()
        output_seq = model(input_seq, target_seq)
        loss = loss_fn(output_seq.view(-1, vocab.size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

# Sla het getrainde model op
torch.save(encoder.state_dict(), 'model/encoder.pt')
torch.save(decoder.state_dict(), 'model/decoder.pt')
torch.save(vocab, 'model/vocab.pkl')
