
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, num_head: int, dropout: float):
        super().__init__()
        # TODO
        head_size = dim_model // num_head
        self.attention = MultiHeadAttention(num_head, dim_model, head_size, dropout)
        self.ffn = FeedForward(dim_model, dropout)
        self.norm1_q = nn.LayerNorm(dim_model)
        self.norm1_k = nn.LayerNorm(dim_model)
        self.norm1_v = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, q, k, v, mask=None):
        q = q + self.attention(self.norm1_q(q), self.norm1_k(k), self.norm1_v(v), mask)
        q = q + self.ffn(self.norm2(q))
        return q


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dim_model: int, num_heads: int, out_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim_model, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim_model)
        self.fc = nn.Linear(dim_model, out_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        batch_size, num_tokens, dim = x.shape
        x += position_encoding(num_tokens, dim, x.device)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        x = self.norm(x)
        x = self.fc(x)
        return x