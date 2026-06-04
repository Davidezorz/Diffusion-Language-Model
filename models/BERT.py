import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

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
# │                                   BERT                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬


class BERT(BaseModel):
    def __init__(self, 
                V: int,                                                         # ◀ vocabulary size
                C: int = 128,                                                   # ◀ embedding dimension
                H: int = 4,                                                     # ◀ number of heads
                N: int = 3,                                                     # ◀ number of blocks
                p: float = 0.1,                                                 # ◀ probability of dropout
                name: str = 'BERT'
                ):
        super().__init__(name)
        
        self.embedding = EmbeddingLayer(C, V)
        self.rotary     = Rotary(C // H, base=160000.0)
        self.embed_norm = LayerNorm(C)
       
        blocks         = [Block(C, H, p, FFN_ratio=1.5, is_causal=False) 
                          for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        blocks[0].norm1 = nn.Identity()
        self.final_norm = LayerNorm(C) 

        self.output    = LastBlock(C, V)
        self.output.linear.weight = self.embedding.embedding                    # Weight Tying


    def last_hidden(self, indices, seqlens=None):
        x = self.embedding(indices)
        x = self.embed_norm(x)

        rotary_cos_sin = self.rotary(x)

        for i, block in enumerate(self.blocks):
            x = block(x, rotary_cos_sin, seqlens=seqlens)
        
        x = self.final_norm(x)           
        return x


    def forward(self, indices, seqlens=None):
        x = self.last_hidden(indices, seqlens)
        logits = self.output(x)          
        return logits
    

    @torch.no_grad()
    def generate(self, indices, seqlens=None, mask_id=None):
        self.eval()
        logits = self(indices, seqlens=seqlens)

        masked_index = torch.where(indices == mask_id)[1]
        indices[:, masked_index] = logits[:, masked_index].argmax(dim=-1)

        return indices
    




"""
# If you use:
# ```
# model = AutoModelForMaskedLM.from_pretrained(
#     "answerdotai/ModernBERT-base",
#     device_map="auto",
#     attn_implementation="sdpa"
# )
# ```
# you should use the version below:


class BERT(BaseModel):
    def __init__(self, 
                V: int,                                                         # ◀ vocabulary size
                C: int = 128,                                                   # ◀ embedding dimension
                H: int = 4,                                                     # ◀ number of heads
                N: int = 3,                                                     # ◀ number of blocks
                p: float = 0.1,                                                 # ◀ probability of dropout
                name: str = 'BERT'
                ):
        super().__init__(name)
        
        self.embedding = EmbeddingLayer(C, V)
        self.rotary1    = Rotary(C // H, base=160000.0)
        self.rotary2    = Rotary(C // H, base=160000.0)
        self.embed_norm = LayerNorm(C)
       
        blocks         = [Block(C, H, p, FFN_ratio=1.5, is_causal=False) 
                          for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        blocks[0].norm1 = nn.Identity()
        self.final_norm = LayerNorm(C) 

        self.output    = LastBlock(C, V)


    def last_hidden(self, indices, seqlens=None):
        x = self.embedding(indices)
        x = self.embed_norm(x)

        rotary_cos_sin_global = self.rotary1(x)
        rotary_cos_sin_local  = self.rotary2(x)

        for i, block in enumerate(self.blocks):
            rotary_cos_sin = rotary_cos_sin_global if i % 3 == 0 else rotary_cos_sin_local
            x = block(x, rotary_cos_sin, seqlens=seqlens)
        
        x = self.final_norm(x)           
        return x
"""