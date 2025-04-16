import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_tensor, hidden_state):
        encoder_out, encoder_hidden = self.encoder(input_tensor, hidden_state)
        decoder_out, decoder_hidden = self.decoder(encoder_out, encoder_hidden)
        output = self.output_layer(decoder_out)
        return output, decoder_hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
