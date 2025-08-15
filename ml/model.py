import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMAutoencoder, self).__init__()
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Map hidden state to latent space
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        # Map latent space back to hidden state for decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        # Output layer to reconstruct input
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        # Encoder
        enc_out, (h_n, c_n) = self.encoder_lstm(x)  # h_n: (1, batch, hidden_dim)
        # Take last hidden state and map to latent
        latent = self.hidden_to_latent(h_n[-1])  # (batch, latent_dim)
        # Repeat latent for each time step
        latent_seq = latent.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)
        # Map latent back to hidden state for decoder
        dec_h_0 = self.latent_to_hidden(latent).unsqueeze(0)  # (1, batch, hidden_dim)
        dec_c_0 = torch.zeros_like(dec_h_0)  # (1, batch, hidden_dim)
        # Decoder
        dec_out, _ = self.decoder_lstm(latent_seq, (dec_h_0, dec_c_0))  # (batch, seq_len, hidden_dim)
        # Reconstruct
        out = self.output_layer(dec_out)  # (batch, seq_len, input_dim)
        return out
