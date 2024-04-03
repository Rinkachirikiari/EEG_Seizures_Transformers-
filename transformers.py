import math

import numpy as np

import torch
import torch.random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt


class AttentionHead(nn.Module):
    def __init__(self, dim_embed: int, head_size: int, dropout: float):
        super().__init__()

        # Trois couches linéaires sans biais
        self.q_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.k_proj = nn.Linear(dim_embed, head_size, bias=False)
        self.v_proj = nn.Linear(dim_embed, head_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape  # B for batch, T for Time, C for Channels
        
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        
        attention = q.bmm(k.transpose(-2, -1))  # ou query @ key.transpose(1, 2) # bmm = batch matmul
        
        
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
        
    
class FeedForwardBlock(nn.Sequential):
        def __init__(self, dim_embed:int, expansion:int, drop:float):
            super().__init__(
                nn.Linear(dim_embed, dim_embed*expansion),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(dim_embed*expansion, dim_embed),
            )


class GELU(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))
        
class ResidualAdd(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x, **kwargs):
            res = x
            x = self.fn(x, **kwargs)
            x += res
            return x
        

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 dim_embed,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_embed),
                MultiHeadAttention(dim_embed, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_embed),
                FeedForwardBlock(
                    dim_embed, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
