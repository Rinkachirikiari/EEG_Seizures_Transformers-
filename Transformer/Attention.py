import copy
import math

import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F

seaborn.set_context(context="talk")


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention


       d_k: dimension de la clé
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -torch.inf)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionHead(nn.Module):
    def __init__(self, dim_embed: int, head_size: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.k_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.v_proj = nn.Linear(dim_embed, head_size, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Project the query, key, and value
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)

        if mask is not None:
            # Assuming mask is applied to query, it should be compatible with the scores shape
            mask = mask.unsqueeze(1)

        # Compute Scaled Dot Product Attention
        attn_output, p_attn = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask, self.dropout)

        return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_embed: int, head_size: int, dropout: float):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(dim_embed, head_size, dropout) for _ in range(num_heads)])
        self.output_linear = nn.Linear(num_heads * head_size, dim_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        return self.dropout(self.output_linear(out))


