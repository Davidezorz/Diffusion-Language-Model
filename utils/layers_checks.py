"""
TOTALLY HARD CODED, ONLY USED DURING DEVELOPMENT
>>> NOT USED IN THE FINAL CODE <<<

Utilities for validating custom BERT and autoregressive transformer
implementations against their Hugging Face counterparts.

This module provides:

- State-dict translation utilities for loading Hugging Face checkpoints
  into custom architectures.
- Layer-by-layer diagnostic comparisons between reference and custom models.
- End-to-end logits validation after weight transfer.
- Rotary positional embedding (RoPE) consistency checks.

These functions are intended for debugging and verification during model
reimplementation and weight-porting, ensuring numerical equivalence between
the original Hugging Face models and the custom PyTorch implementations.
"""

from transformers import AutoModelForCausalLM
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat 
from models.base_model import *





def compare_encoder_layers(hf_model, my_model, num_layers=22):
    hf_model.eval()
    my_model.eval()

    hf_model = hf_model.model if hasattr(hf_model, 'model') else hf_model

    batch_size = 1
    seq_len = 8
    dummy_input_ids = torch.randint(0, 50368, (batch_size, seq_len))

    print(f"\n--- MULTI-LAYER DIAGNOSTIC ({num_layers} LAYERS) ---")
    
    with torch.no_grad():
        hf_captured_attn = {}
        hf_captured_block = {}
        hooks = []

        # ---------------------------------------------------------
        # 1. SETUP DYNAMIC WIRETAPS
        # ---------------------------------------------------------
        def get_attn_hook(idx):
            return lambda m, i, o: hf_captured_attn.update({idx: o[0] if isinstance(o, tuple) else o})

        def get_block_hook(idx):
            return lambda m, i, o: hf_captured_block.update({idx: o[0] if isinstance(o, tuple) else o})

        # Attach hooks to the first N layers
        for i in range(num_layers):
            hooks.append(hf_model.layers[i].attn.register_forward_hook(get_attn_hook(i)))
            hooks.append(hf_model.layers[i].register_forward_hook(get_block_hook(i)))

        # Run the full HF model safely
        hf_final = hf_model(dummy_input_ids).last_hidden_state
        
        # Cleanup hooks
        for h in hooks: h.remove()

        base_model = hf_model.model if hasattr(hf_model, 'model') else hf_model
        print("\n--- ETTIN ENCODER RoPE ---")
        print(f"Layer Types (First 6): {base_model.config.layer_types[:6]}")
        print(f"RoPE Parameters:       {base_model.config.rope_parameters}")
        print("-----------------------------\n")

        # ---------------------------------------------------------
        # 2. RUN CUSTOM MODEL & COMPARE
        # ---------------------------------------------------------
        hf_emb = hf_model.embeddings.tok_embeddings(dummy_input_ids)
        my_emb = my_model.embedding(dummy_input_ids)
        hf_norm = hf_model.embeddings.norm(hf_emb)
        my_norm = my_model.embed_norm(my_emb)
        
        print(f"Raw Embeddings:    Max Diff = {(hf_emb - my_emb).abs().max().item():.2e}")
        print(f"Embedding Norm:    Max Diff = {(hf_norm - my_norm).abs().max().item():.2e}\n")

        # Generate BOTH sets of waves
        try:
            rotary   = my_model.rotary (my_norm)
        except:
            rotary = None
            rotary1  = my_model.rotary1(my_norm)
            rotary2  = my_model.rotary2(my_norm)
        
        x_custom = my_norm 

        for i in range(num_layers):
            # Apply your alternating logic in the test loop!
            if rotary is None:
                current_rope = rotary1 if i % 3 == 0 \
                                else rotary2
            else:
                current_rope = rotary
            
            my_attn_output = my_model.blocks[i].attention(my_model.blocks[i].norm1(x_custom), current_rope, None)
            x_custom = my_model.blocks[i](x_custom, current_rope, None)

            diff_attn = (hf_captured_attn[i] - my_attn_output).abs().max().item()
            diff_block = (hf_captured_block[i] - x_custom).abs().max().item()

            print(f"Layer {i} Attention: Max Diff = {diff_attn:.2e}")
            print(f"Layer {i} Complete:  Max Diff = {diff_block:.2e}")
            
            if diff_block > 1e-4:
                print(f"   --> ⚠️ DIVERGENCE DETECTED AT LAYER {i}!")
                break 
        try:
            diff = (hf_final - my_model.last_hidden(dummy_input_ids))
        except:
            diff = (hf_final - my_model.last_hidden(dummy_input_ids)[0])
        print(f"\nFinal Hidden States: Max Diff = {diff.abs().max().item():.2e}")
    




def compare_decoder_layers(hf_model, my_model, num_layers=22):
    hf_model.eval()
    my_model.eval()

    batch_size = 1
    seq_len = 8
    dummy_input_ids = torch.randint(0, 50368, (batch_size, seq_len))

    print(f"\n--- MULTI-LAYER DIAGNOSTIC DECODER ({num_layers} LAYERS) ---")
    
    with torch.no_grad():
        hf_captured_attn = {}
        hf_captured_block = {}
        hooks = []

        # ---------------------------------------------------------
        # 1. SETUP DYNAMIC WIRETAPS
        # ---------------------------------------------------------
        # Hugging Face wraps the transformer inside `.model` for CausalLMs
        base_model = hf_model.model if hasattr(hf_model, 'model') else hf_model

        def get_attn_hook(idx):
            return lambda m, i, o: hf_captured_attn.update({idx: o[0] if isinstance(o, tuple) else o})

        def get_block_hook(idx):
            return lambda m, i, o: hf_captured_block.update({idx: o[0] if isinstance(o, tuple) else o})

        # Attach hooks to the first N layers
        for i in range(num_layers):
            hooks.append(base_model.layers[i].attn.register_forward_hook(get_attn_hook(i)))
            hooks.append(base_model.layers[i].register_forward_hook(get_block_hook(i)))

        # Run the full HF model safely (ask it to return hidden states so we can check the final norm)
        hf_outputs = hf_model(dummy_input_ids, output_hidden_states=True)
        hf_final_hidden = hf_outputs.hidden_states[-1]
        hf_logits = hf_outputs.logits
        
        # Cleanup hooks
        for h in hooks: h.remove()

        # ---------------------------------------------------------
        # 2. RUN CUSTOM MODEL & COMPARE
        # ---------------------------------------------------------
        hf_emb = base_model.embeddings.tok_embeddings(dummy_input_ids)
        my_emb = my_model.embedding(dummy_input_ids)
        hf_norm = base_model.embeddings.norm(hf_emb)
        my_norm = my_model.embed_norm(my_emb)
        
        print(f"Raw Embeddings:    Max Diff = {(hf_emb - my_emb).abs().max().item():.2e}")
        print(f"Embedding Norm:    Max Diff = {(hf_norm - my_norm).abs().max().item():.2e}\n")

        # Generate BOTH sets of waves
        try:
            rotary   = my_model.rotary (my_norm)
        except:
            rotary = None
            rotary1  = my_model.rotary1(my_norm)
            rotary2  = my_model.rotary2(my_norm)
        
        x_custom = my_norm 

        base_model = hf_model.model if hasattr(hf_model, 'model') else hf_model
        print("\n--- ETTIN DECODER RoPE ---")
        print(f"Layer Types (First 6): {base_model.config.layer_types[:6]}")
        print(f"RoPE Parameters:       {base_model.config.rope_parameters}")
        print("-----------------------------\n")

        for i in range(num_layers):        
            # Apply your alternating logic in the test loop!
            if rotary is None:
                current_rope = rotary1 if i % 3 == 0 \
                                else rotary2
            else:
                current_rope = rotary  
                  
            my_attn_output = my_model.blocks[i].attention(my_model.blocks[i].norm1(x_custom), current_rope, None)
            x_custom = my_model.blocks[i](x_custom, current_rope, None)

            diff_attn = (hf_captured_attn[i] - my_attn_output).abs().max().item()
            diff_block = (hf_captured_block[i] - x_custom).abs().max().item()

            print(f"Layer {i} Attention: Max Diff = {diff_attn:.2e}")
            print(f"Layer {i} Complete:  Max Diff = {diff_block:.2e}")
            
            if diff_block > 1e-4:
                print(f"   --> ⚠️ DIVERGENCE DETECTED AT LAYER {i}!")
                break

        # We need to manually apply your final norm and head to compare the end results
        my_final_hidden = my_model.last_hidden(dummy_input_ids)
        my_logits = my_model(dummy_input_ids)

        print(f"\nFinal Hidden States: Max Diff = {(hf_final_hidden - my_final_hidden).abs().max().item():.2e}")
        print(f"Vocabulary Logits:   Max Diff = {(hf_logits - my_logits).abs().max().item():.2e}")





def compare_rope_embeddings(hf_model, my_model):
    print("\n--- ROPE ANALYSIS ---")
    
    # 1. Extract the raw frequencies from memory
    hf_freq = hf_model.rotary_emb.full_attention_inv_freq
    my_freq = my_model.rotary.inv_freq
    
    print(f"HF Frequencies (First 4): {hf_freq[:4].tolist()}")
    print(f"My Frequencies (First 4): {my_freq[:4].tolist()}")
    
    # 2. Generate the waves
    dummy_input = torch.zeros(1, 8, 768)
    position_ids = torch.arange(8).unsqueeze(0)
    
    with torch.no_grad():
        hf_cos, _ = hf_model.rotary_emb(dummy_input, position_ids, 'full_attention')
        my_cos, _ = my_model.rotary(dummy_input)
    
    # 3. Print Token 1 (where t=1, so we can see the wave in action)
    print(f"\nHF Cosine at Token 1: {hf_cos[0, 1, :4].tolist()}")
    print(f"\nHF Cosine at Token 2: {hf_cos[0, 2, :4].tolist()}")
    
    # Your cos is packed as [1, T, 3, 1, c], so we pull T=1, Q-block=0, Head=0
    print(f"My Cosine at Token 1: {my_cos[0, 1, 0, 0, :4].tolist()}")
    print(f"My Cosine at Token 2: {my_cos[0, 2, 0, 0, :4].tolist()}")

