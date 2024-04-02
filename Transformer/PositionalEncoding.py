import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000, device: torch.device = torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        dim = torch.arange(d_model, dtype=torch.float, device=device).reshape(1, 1, -1)

        phase = pos / (10000 ** (dim / dimension))

        #div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(phase)
        pe[:, 1::2] = torch.cos(phase)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)