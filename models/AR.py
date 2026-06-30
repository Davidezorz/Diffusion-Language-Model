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
# │                                    AR                                     │
# ╰───────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AR(BaseModel):
    def __init__(self, 
                V: int,                                                         # ◀ vocabulary size
                C: int = 128,                                                   # ◀ embedding dimension
                H: int = 4,                                                     # ◀ number of heads
                N: int = 3,                                                     # ◀ number of blocks
                p: float = 0.1,                                                 # ◀ probability of dropout
                name: str = 'AR'
                ):
        super().__init__(name)
        
        self.embedding  = EmbeddingLayer(C, V)
        self.embed_norm = LayerNorm(C)
        self.rotary     = Rotary(C // H, base=160000.0)
        self.rotary2    = Rotary(C // H, base=10000.0)
       
        blocks         = [Block(C, H, p, FFN_ratio=1.5, is_causal=True) 
                          for _ in range(N)]
        blocks[0].norm1 = nn.Identity()
        self.blocks    = nn.ModuleList(blocks)
        
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
    def generate(self, ids, n_tokens, temperature=1, tokenizer=None):
        self.eval()
        for _ in range(n_tokens):
            
            logits = self(ids)                                                  # get the logits
            logits = logits[:, -1, :]                                           # B C

            sorted_idx= torch.argsort(-logits)
            print(tokenizer.decode(sorted_idx[0, :5].tolist()))
            probs = F.softmax(logits/temperature, dim=-1)                       # apply softmax to get probabilities
            
            id_next = torch.multinomial(probs, num_samples=1)                   # B 1  -> sample from the distribution
            ids = torch.cat((ids, id_next), dim=1)                              # B T+1
        return ids
