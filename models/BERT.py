from models.AR import *


class BERT(AR):
    def __init__(self, 
                V: int,                   # ◀ vocabulary size
                C: int = 128,             # ◀ embedding dimension
                H: int = 4,               # ◀ number of heads
                N: int = 3,               # ◀ number of blocks
                p: float = 0.1,           # ◀ probability of dropout
                name: str = 'BERT'
                ):
        super().__init__()
        self.name = name
        
        self.embedding = EmbeddingLayer(C, V)
        self.rotary    = Rotary(C // H)
       
        blocks         = [Block(C, H, p, is_causal=False) for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        
        self.output    = LastBlock(C, V)

