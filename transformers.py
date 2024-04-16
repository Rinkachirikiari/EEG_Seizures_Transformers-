import math

import numpy as np

import torch
import torch.random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

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
        
        def forward(self, x, mask=None):
            
            out = torch.cat([h(x, x, x, mask) for h in self.heads], dim=-1)
            
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
                 head_size = 512,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_embed),
                MultiHeadAttention(num_heads, dim_embed, head_size, drop_p),
                nn.Dropout(drop_p)
            )),

            ResidualAdd(nn.Sequential(
                nn.LayerNorm(dim_embed),
                FeedForwardBlock(
                    dim_embed, expansion=forward_expansion, drop=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, dim_embed):
        super().__init__(*[TransformerEncoderBlock(dim_embed) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, dim_embed, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(dim_embed),
            nn.Linear(dim_embed, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(1018*40, 256), # Il faudra vérifier l'input size de nos données (dans ce cas là c'est output du Transformers)
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes) # On a 2 classes !
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out
