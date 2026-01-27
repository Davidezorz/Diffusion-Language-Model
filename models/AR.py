import math

import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

if torch.cuda.is_available():
    import flash_attn
    import flash_attn.layers.rotary


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




# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                Rotary  PE                                 │
# ╰───────────────────────────────────────────────────────────────────────────╯

class Rotary(torch.nn.Module):
    def __init__(self, c, base=10_000):
        super().__init__()
        dtype = torch.get_default_dtype()
        inv_freq = 1. / (base ** (torch.arange(0, c, 2, dtype=dtype) / c))
        self.register_buffer('inv_freq', inv_freq)
        
        self.T_cached = -1                                                      # we will store the cos and sin values for the max T yet

    
    def forward(self, x):
        T = x.shape[1]
        dtype = self.inv_freq.dtype

        if self.T_cached < T:
            t = torch.arange(T, dtype=dtype, device=x.device)                   # T
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())           # T c//2, first row is t[0]*inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)                             # T c

            self.cos = repeat(emb.cos(), 'T c -> 1 T 3 1 c')                    # 1 T 3 1 c
            self.sin = repeat(emb.sin(), 'T c -> 1 T 3 1 c')                    # 1 T 3 1 c
            
            self.cos[:,:,2,:,:].fill_(1.)                                       # ◀─┬ This makes the transformation 
            self.sin[:,:,2,:,:].fill_(0.)                                       # ◀─╯ on values an identity
            
            self.T_cached = T                                                   # update T_cached

        cos = self.cos[:, :T, :, :, :]                                          # ◀─┬ cut based on the 
        sin = self.sin[:, :T, :, :, :]                                          # ◀─╯ token length

        return cos, sin



def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                             Embedding Layers                              │
# ╰───────────────────────────────────────────────────────────────────────────╯

class EmbeddingLayer(nn.Module):
    def __init__(self, C, V):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((V, C)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):                                                       # B T                                              
        return self.embedding[x]                                                # B T C





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                Layer Norm                                 │
# ╰───────────────────────────────────────────────────────────────────────────╯

class LayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.scale = nn.Parameter(torch.ones([C]))
        self.bias  = nn.Parameter(torch.zeros([C]))
        self.C     = C
    
    
    def forward(self, x):
        x = F.layer_norm(x.float(), [self.C])
        return x * self.scale[None, None, :] + self.bias[None, None, :]





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                           Multi Head Attention                            │
# ╰───────────────────────────────────────────────────────────────────────────╯

class MultiHeadAttention(nn.Module):
    def __init__(self, C: int = 256, H: int = 8, p_dropout: float = 0.1):
        super().__init__()
        assert C % H == 0, "embedding dimension C must be divisible by the number of heads H"
        
        self.C, self.H = C, H
        self.c         = int(C // H)

        self.W_qkv     = nn.Linear(C, 3 * C, bias=False)
        self.W_o       = nn.Linear(C, C) 
        self.dropout   = nn.Dropout(p_dropout)

        cuda = torch.cuda.is_available()
        self.attention = self._attention_cuda if cuda else self._attention

        self.register_buffer("mask", None)
        

    def forward(self, x, rotary_cos_sin, seqlens):
        """ 
        x:              B, T C, tensor input
        rotary_cos_sin: cosine and sine tensor from Rotary class
        seqlens:        B T, how long is each sequence 
        """
        qkv = self.W_qkv(x)                                                     # B T (three C)
        qkv = rearrange(qkv, 'B T (three H c) -> B T three H c',                # B T three H c
                        three=3, H=self.H)
        
        x = self.attention(qkv, rotary_cos_sin, seqlens)                        # B T C 
        x = self.dropout(self.W_o(x))                                           # B T C 

        return x    


    def _attention(self, qkv, rotary_cos_sin, seqlens):
        B, T, _, H, c = qkv.shape 
        cos, sin = rotary_cos_sin                                               #  ╭ rotary positional embedding 
        qkv = qkv * cos + rotate_half(qkv) * sin                                # ◀╯ B T three H c  

        qkv = rearrange(qkv, 'B T three H c -> B three H T c')
        q, k, v = qkv.unbind(dim=1)                                             # B three H S c -> 3 * B H T c

        c = q.shape[-1]                                                         #   ╭ compute attention
        attn_scores = (q @ k.transpose(-2, -1)) * (c ** -0.5)                   # ◀─┤ B H T T

        mask = self._create_causal_mask(T, q.device)                            # ◀─┤ T T
        
        if seqlens is not None:                                                 # ◀── handle sequence length
            mask_seqlen = self._create_seqlens_mask(seqlens, T, q.device)       
            mask =  mask | mask_seqlen           
        
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))              # ◀─┤ B H T T 
        attn_probs = F.softmax(attn_scores, dim=-1)                             # ◀─┤ B H T T 
        x = attn_probs @ v                                                      # ◀─╯ B H T c

        return rearrange(x, 'B H T c -> B T (H c)')                             # B T C


    def _attention_cuda(self, qkv, rotary_cos_sin, seqlens):
        B, T, _, H, c = qkv.shape
        device        = qkv.device
   
        cos, sin = rotary_cos_sin                                               #   ╭ rotary positional embedding  
        cos = cos[0, :, 0, 0, :cos.shape[-1]//2].to(qkv.dtype)                  # ◀─┤  T c//2
        sin = sin[0, :, 0, 0, :sin.shape[-1]//2].to(qkv.dtype)                  # ◀─┤  T c//2
        qkv = flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)     # ◀─╯  B T 3 H c
        
        qkv = rearrange(qkv, 'B T ... -> (B T) ...')                            # (B T) 3 H c
        cu_seqlens, max_seqlen = self.compute_cu_seqlens(seqlens, B, T, device)

        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(   #   ╭ compute attention 
            qkv, cu_seqlens, max_seqlen, 0., causal=True )                      # ◀─╯ (B T) 3 H c
        
        return rearrange(x, '(B T) H c -> B T (H c)', B=B)                      # B T C
        

    def compute_cu_seqlens(self, seqlens, B, T, device):                        # ◀── B + 1, compute the cumulative sequence length
        if seqlens == None:
            cu_seqlens = torch.arange(0, (B+1)*T, T, 
                                      dtype=torch.int32, device=device)
            max_seqlen = T
        else:
            zero = torch.tensor([0], dtype=torch.int32, device=device)
            cu_seqlens = torch.cat((zero, seqlens.cumsum(0)))
            max_seqlen = seqlens.max().item()

        return cu_seqlens, max_seqlen

    
    def _create_causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)          # Create causal mask for autoregressive attention
        return mask.bool() 
    

    def _create_seqlens_mask(self, seqlens, T, device):
        indices = torch.arange(T, dtype=torch.int32, device=device)
        mask_row = indices.unsqueeze(0) >= seqlens.unsqueeze(1)
        mask = mask_row.unsqueeze(1).expand(-1, T, -1)
        return mask.bool()


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                           Feed Forward Network                            │
# ╰───────────────────────────────────────────────────────────────────────────╯

class FeedForward(nn.Module):
    def __init__(self, C: int = 64, factor: int = 4):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(C, factor*C),
            nn.GELU(approximate='tanh'),
            nn.Linear(factor*C, C)
        )
    
    def forward(self, x):
        return self.FFN(x)





# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                  Blocks                                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛


class Block(nn.Module):                                               
    def __init__(self, C, H, p_dropout=0.1, FFN_ratio=4):      
        super().__init__()                                               
        self.H         = H

        self.norm1     = LayerNorm(C)
        self.attention = MultiHeadAttention(C, H)
        self.dropout1  = nn.Dropout(p_dropout)

        self.norm2     = LayerNorm(C)
        self.FFN       = FeedForward(C, FFN_ratio)
        self.dropout2  = nn.Dropout(p_dropout)


    def forward(self, x, rotary_cos_sin, seqlens=None):
        x = x + self.dropout1(self.attention(self.norm1(x), 
                              rotary_cos_sin, seqlens))                         # ◀─ add the computed attention of x_as to the original x
        x = x + self.dropout2(self.FFN(self.norm2(x)))                          # ◀─ Add the computed FFN of x_as to the previous x

        return x
    


class LastBlock(nn.Module):
    def __init__(self, C, V):
        super().__init__()
        self.norm   = LayerNorm(C)

        self.linear = nn.Linear(C, V)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.norm(x))
        return x
  




# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                    AR                                     │
# ╰───────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class AR(nn.Module):
    def __init__(self, 
                V: int,                   # ◀ vocabulary size
                C: int = 128,             # ◀ embedding dimension
                H: int = 4,               # ◀ number of heads
                N: int = 3,               # ◀ number of blocks
                p: float = 0.1,           # ◀ probability of dropout
                name: str = 'AR'
                ):
        super().__init__()
        self.name = name
        
        self.embedding = EmbeddingLayer(C, V)
        self.rotary    = Rotary(C // H)
       
        blocks         = [Block(C, H, p) for _ in range(N)]
        self.blocks    = nn.ModuleList(blocks)
        
        self.output    = LastBlock(C, V)


    def save(self, folder='.weights/', name=None):                              # ◀┬─ save model 
        name = self.name + '.pth' if name else 'model.pth'                      #  │  weights
        file = folder + name                                                    #  │
        torch.save(self.state_dict(), file)                                     #  ╯


    def load(self, folder='.weights/', name=None):                              # ◀┬─ load model
        name = self.name + '.pth' if name else 'model.pth'                      #  │  weights
        file = folder + name                                                    #  │ 
        try:                                                                    #  │ 
            self.load_state_dict(torch.load(file, weights_only=True))           #  │
            print('model loaded')                                               #  │
        except Exception as e:                                                  #  │
            print("Model weights not avaiable \n\n", e)                         #  ╯


    def forward(self, indices, seqlens=None):
        x = self.embedding(indices)
        rotary_cos_sin = self.rotary(x)


        for block in self.blocks:
            x = block(x, rotary_cos_sin, seqlens=seqlens)

        logits = self.output(x)

        return logits