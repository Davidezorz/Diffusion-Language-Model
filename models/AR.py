import math

import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import warnings

from models.base_model import *



"""
╭ CONVENTIONS ────────────────────────────────────────────────────────────────╮
│ ├─• B     ▶ batch size                                                      │
│ ├─• T     ▶ number of tokens in a batch -> length of a sequence/sentence    │
│ ├─• C     ▶ embedding dimension of each token                               │
│ │                                                                           │
│ ├─• H     ▶ number of heads                                                 │
│ ╰─• V     ▶ vocabulary size                                                 │
╰─────────────────────────────────────────────────────────────────────────────╯
"""





# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                    AR                                     │
# ╰───────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AR(BaseModel):
    def __init__(self, 
                V: int,                   # ◀ vocabulary size
                C: int = 128,             # ◀ embedding dimension
                H: int = 4,               # ◀ number of heads
                N: int = 3,               # ◀ number of blocks
                p: float = 0.1,           # ◀ probability of dropout
                name: str = 'AR'
                ):
        super().__init__(name)
        
        self.embedding = EmbeddingLayer(C, V)
        self.rotary    = Rotary(C // H)
       
        blocks         = [Block(C, H, p) for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        
        self.output    = LastBlock(C, V)


    def forward(self, indices, seqlens=None):
        x = self.embedding(indices)
        rotary_cos_sin = self.rotary(x)


        for block in self.blocks:
            x = block(x, rotary_cos_sin, seqlens=seqlens)

        logits = self.output(x)

        return logits