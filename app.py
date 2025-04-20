import numpy as np

# Zelf-attentie mechanisme
def self_attention(query, key, value):
    # Bereken de score (dot-product)
    scores = np.matmul(query, key.T)  # Afmeting: (seq_len, seq_len)
    
    # Voeg de softmax toe om de waarschijnlijkheden te krijgen
    attention_weights = softmax(scores)
    
    # Vermenigvuldig de gewichten met de value
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

# Softmax functie voor normalisatie
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Positional encoding toevoegen (gebaseerd op de originele Transformer paper)
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Sin voor even
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Cos voor oneven
    return pos_encoding

# Lineaire laag (fully connected)
def linear(x, weights, bias):
    return np.dot(x, weights) + bias

# Feedforward netwerk (2 lagen)
def feedforward(x, d_model, ff_size):
    # Eerste laag
    w1 = np.random.randn(d_model, ff_size)
    b1 = np.random.randn(ff_size)
    x = np.maximum(0, linear(x, w1, b1))  # ReLU activatie
    
    # Tweede laag
    w2 = np.random.randn(ff_size, d_model)
    b2 = np.random.randn(d_model)
    x = linear(x, w2, b2)
    
    return x

# Encoder blok (self-attention + feedforward)
def transformer_encoder(x, d_model, n_heads, ff_size):
    seq_len = x.shape[0]
    
    # Positional encoding toevoegen aan de input
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc  # Input + Positional encoding
    
    # Zelf-attentie
    query = np.random.randn(seq_len, d_model)
    key = np.random.randn(seq_len, d_model)
    value = x
    attention_output, _ = self_attention(query, key, value)
    
    # Residual connectie
    x = x + attention_output
    
    # Feedforward netwerk
    x = feedforward(x, d_model, ff_size)
    
    return x

# Model initialisatie
seq_len = 10  # Sequentie lengte
d_model = 64  # Dimensie van embedding
n_heads = 8   # Aantal heads in multi-head attention
ff_size = 256 # Grootte van het feedforward netwerk

# Input data (bijvoorbeeld tokenized woorden)
x = np.random.randn(seq_len, d_model)

# Encoder doorgeven
output = transformer_encoder(x, d_model, n_heads, ff_size)

print("Transformer output shape:", output.shape)
