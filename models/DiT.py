import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.AR import *


"""
╭ CONVENTIONS ────────────────────────────────────────────────────────────────╮
│ ├─• B        ▶ batch size                                                   │
│ ├─• T        ▶ number of tokens in a batch i.e. length of a sequence        │
│ ├─• C        ▶ embedding dimension of each token                            │
│ │                                                                           │
│ ├─• H        ▶ number of heads                                              │
│ ├─• V        ▶ vocabulary size                                              │
│ │                                                                           │
│ ├─• cond_dim ▶ output size of the TimestepEmbedding layer                   │
│ ╰─• f_dim    ▶ initial embedding size for of the frequency                  │
╰─────────────────────────────────────────────────────────────────────────────╯
"""



# ╭───────────────────────────────────────────────────────────────────────────╮
# │                              Embedding Layer                              │
# ╰───────────────────────────────────────────────────────────────────────────╯

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, cond_dim, f_dim=256, max_period = 10_000):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(f_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.f_dim = f_dim

        half = self.f_dim // 2
        arange = torch.arange(0, half, dtype=torch.get_default_dtype())         # f_dim//2
        freqs = torch.exp(- math.log(max_period)* arange / half )               # f_dim//2
        self.register_buffer('freqs', freqs)  


    def forward(self, t):
        args = t[:, None].float() * self.freqs[None]                            # B f_dim//2
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)       # B (f_dim//2)*2 != B f_dim if f_dim is odd
        embedding = F.pad(embedding, (0, self.f_dim % 2))                       # B f_dim

        return self.FFN(embedding)                                              # B cond_dim





# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                   Blocks                                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def add_scale(x:     torch.Tensor,
              shift: torch.Tensor, 
              scale: torch.Tensor
              ) -> torch.Tensor:
    return x * (1 + scale) + shift



class DiTBlock(nn.Module):                                                      # Paper: ┬ Scalable Diffusion Models 
    def __init__(self, C, H, cond_dim, p_dropout=0.1, FFN_ratio=4):             #        ├ with Transformers
        super().__init__()                                                      #        ╰ https://arxiv.org/pdf/2212.09748
        self.H         = H

        self.norm1     = LayerNorm(C)
        self.attention = MultiHeadAttention(C, H, p_dropout=p_dropout, 
                                            is_causal=False)

        self.norm2     = LayerNorm(C)
        self.FFN       = FeedForward(C, FFN_ratio)
        self.dropout   = nn.Dropout(p_dropout)

        self.ALN       = nn.Linear(cond_dim, 6 * C)                             # ◀╮ Adaptive Layer Normalization, used
        self.ALN.weight.data.zero_()                                            #  │ for conditioning. Initialized at
        self.ALN.bias.data.zero_()                                              #  ╰ zero


    def forward(self, x, rotary_cos_sin, conditioning, seqlens=None):
        ALN = self.ALN(conditioning)[:, None]                                   # ◀─ Forward of the Adaptive Normalization Layer: 
        (shift_att, scale_att, gate_att,                                        # ◀╮ we get the shift, scale and gate tensors for both
         shift_FFN, scale_FFN, gate_FFN) = ALN.chunk(6, dim=2)                  #  ╰ the attention and the feed forward layer

        x_as = add_scale(self.norm1(x), shift_att, scale_att)                   # ◀─ scale and shift conditioning after normalizaton 
        x = x + gate_att*self.attention(x_as, rotary_cos_sin, seqlens)          # ◀─ add the computed attention of x_as to the original x
        
        x_as = add_scale(self.norm2(x), shift_FFN, scale_FFN)                   # ◀─ scale and shift conditioning after the attention 
        x = x + gate_FFN*self.dropout(self.FFN(x_as))                           # ◀─ Add the computed FFN of x_as to the previous x

        return x
    


class DiTLastBlock(nn.Module):
    def __init__(self, C, V, cond_dim):
        super().__init__()
        self.norm   = LayerNorm(C)

        self.linear = nn.Linear(C, V)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.ALN    = nn.Linear(cond_dim, 2 * C)
        self.ALN.weight.data.zero_()
        self.ALN.bias.data.zero_()


    def forward(self, x, conditioning):
        shift, scale = self.ALN(conditioning)[:, None].chunk(2, dim=2)

        x = add_scale(self.norm(x), shift, scale) 
        x = self.linear(x)
        return x





# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                    DIT                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class DIT(AR):                     
    def __init__(self, 
                V: int,                                                         # ◀ vocabulary size
                C: int = 128,                                                   # ◀ embedding dimension
                H: int = 4,                                                     # ◀ number of heads
                cond_dim: int = 32,                                             # ◀ internal dimension for conditioning
                N: int = 3,                                                     # ◀ number of blocks
                p: float = 0.1,                                                 # ◀ probability of dropout
                name: str = 'DiT'                                               # ◀ name of the model
                ):
        super().__init__()

        self.name = name

        self.embedding = EmbeddingLayer(C, V)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary    = Rotary(C // H)
       
        blocks         = [DiTBlock(C, H, cond_dim, p) for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        
        self.output    = DiTLastBlock(C, V, cond_dim)


    def forward(self, indices, sigma):
        x = self.embedding(indices)

        conditioning = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary(x)

        with torch.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, conditioning, seqlens=None)

            x = self.output(x, conditioning)

        return x