import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class AttentionHead(nn.Module):
    def __init__(self, dim_embed: int, head_size: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.k_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.v_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        attention = q.bmm(k.transpose(-2, -1))
        if mask is not None:
            attention[:, mask[:T, :T].logical_not()] = -torch.inf
        scale = q.size(-1) ** 0.5
        softmax = F.softmax(attention / scale, dim=-1)
        softmax = self.dropout(softmax)
        return softmax.bmm(v)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_embed: int, head_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_embed, head_size, dropout) for _ in range(num_heads)])
        self.fc = nn.Linear(num_heads * head_size, dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        return self.dropout(self.fc(out))
    

class FeedForward(nn.Module):
    def __init__(self, dim_embed: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_embed, 4 * dim_embed),
            nn.ReLU(),
            nn.Linear(4 * dim_embed, dim_embed),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.ffn(x)
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_embed: int, num_head: int, dropout: float):
        super().__init__()
        
        head_size = dim_embed // num_head
        self.attention = MultiHeadAttention(num_head, dim_embed, head_size, dropout)
        self.ffn = FeedForward(dim_embed, dropout)
        self.norm1_q = nn.LayerNorm(dim_embed)
        self.norm1_k = nn.LayerNorm(dim_embed)
        self.norm1_v = nn.LayerNorm(dim_embed)
        self.norm2 = nn.LayerNorm(dim_embed)
        
    def forward(self, q, k, v, mask=None):
        q = q + self.attention(self.norm1_q(q), self.norm1_k(k), self.norm1_v(v), mask)
        q = q + self.ffn(self.norm2(q))
        return q
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dim_embed: int, num_heads: int, out_size: int, dropout: float):
        super().__init__()
        # TODO
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim_embed, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim_embed)
        self.fc = nn.Linear(dim_embed, out_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x, mask=None):
        B, C, T = x.shape 
        print(f"batch_size : ",B)
        print(f"Sequence length : ",C)
        print(f"dimension embeddings : ",T)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        x = self.norm(x)
        x = self.fc(x)
        print(f'Shape output \t\t= {tuple(x.shape)}')
        return x




class TransformerClassifier(nn.Module):
    def __init__(self, input, num_classes, hidden_sizes=[512, 256], dropout=0.1):
        super(TransformerClassifier, self).__init__()
        layers = []
        #input_dim = input.size()[-1]
        input_dim = input

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        #x.shape = (batch_size, seq_length, hidden_size)

        pooled_output = x.mean(dim=1)  
        logits = self.classifier(pooled_output)
        return F.sigmoid(logits)


