import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, n_features)
        """
        # First LSTM => (batch_size, seq_len, hidden_dim)
        x, _ = self.rnn1(x)

        # Second LSTM => final hidden state shape: (1, batch_size, embedding_dim)
        x, (hidden_n, _) = self.rnn2(x)

        # Remove the extra "num_layers" dimension => (batch_size, embedding_dim)
        hidden_n = hidden_n.squeeze(0)

        return hidden_n


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        """
        x shape: (batch_size, input_dim)

        We'll replicate each embedding along the time dimension (seq_len)
        so the decoder can reconstruct a full sequence.
        """
        # Insert a time dimension: (batch_size, 1, input_dim)
        # Repeat => (batch_size, seq_len, input_dim)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        # First LSTM => (batch_size, seq_len, input_dim)
        x, _ = self.rnn1(x)

        # Second LSTM => (batch_size, seq_len, hidden_dim)
        x, _ = self.rnn2(x)

        # Flatten => (batch_size * seq_len, hidden_dim)
        x = x.reshape(-1, self.hidden_dim)

        # Map each time step to n_features => (batch_size * seq_len, n_features)
        x = self.output_layer(x)

        # Reshape back to (batch_size, seq_len, n_features)
        x = x.reshape(-1, self.seq_len, self.n_features)
        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        """
        Expects x shape: (batch_size, seq_len, n_features).
        """
        # Encode => (batch_size, embedding_dim)
        x = self.encoder(x)
        # Decode => (batch_size, seq_len, n_features)
        x = self.decoder(x)
        return x