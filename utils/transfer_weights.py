from utils.layers_checks import *

def trasfer_weights(
    target_model,
    source_model,
    weight_translator,
    run_validation=True,
    show_layers=False,
):
    """
    Transfer weights from a source model into a target model.

    Args:
        - target_model:         Custom PyTorch model receiving the weights.
        - source_model:         Reference model providing the original weights.
        - weight_translator:    Function that converts the source state dict
                                into the target model's parameter format.
        - run_validation:       If True, compare models logits after loading.
        - show_layers:          If True, print the layer names of both models.

    Returns:
        The target model with translated weights loaded.
    """
    source_state_dict = source_model.state_dict()

    if show_layers:
        print("\n\nSource model")
        print_model_layers(source_state_dict)

        print("\n\nTarget model")
        print_model_layers(target_model.state_dict())

    translated_state_dict = weight_translator(source_state_dict)                # get the tranlated state dict
    target_model.load_state_dict(translated_state_dict, strict=False)           # strict=False allows architecture differences.

    print("SUCCESS: Parameters loaded into target model.")

    if run_validation:
        compare_logits(source_model, target_model)
        print()

    return target_model





def print_model_layers(state_dict, limit=None):
    outputs = []
    total_params = 0
    keys = list(state_dict.keys())

    for key in keys:
        shape = list(state_dict[key].shape)
        outputs.append(f"{key:<50} -> Shape: {shape}")
        total_params += state_dict[key].numel()

    print("\n".join(outputs[:limit]))
    if limit is not None and limit < len(keys):
        print("...")

    print(f"\nTotal parameters: {total_params:,}")





def translate_weights_encoder(hf_weights, N_blocks=22):
    """Map out ModernBERT state dict in our BERT state dict"""
    custom_dict = {}
    
    # 1. Base Model - Embeddings
    custom_dict["embedding.embedding"] = hf_weights["model.embeddings.tok_embeddings.weight"]
    custom_dict["embed_norm.scale"] = hf_weights["model.embeddings.norm.weight"]
    
    # 2. Base Model - Blocks
    for i in range(N_blocks):
        custom_dict[f"blocks.{i}.attention.W_qkv.weight"] = hf_weights[f"model.layers.{i}.attn.Wqkv.weight"]
        custom_dict[f"blocks.{i}.attention.W_o.weight"] = hf_weights[f"model.layers.{i}.attn.Wo.weight"]
        
        if i > 0:
            custom_dict[f"blocks.{i}.norm1.scale"] = hf_weights[f"model.layers.{i}.attn_norm.weight"]
        custom_dict[f"blocks.{i}.norm2.scale"] = hf_weights[f"model.layers.{i}.mlp_norm.weight"]
        
        custom_dict[f"blocks.{i}.FFN.Wi.weight"] = hf_weights[f"model.layers.{i}.mlp.Wi.weight"]
        custom_dict[f"blocks.{i}.FFN.Wo.weight"] = hf_weights[f"model.layers.{i}.mlp.Wo.weight"]
        
    # 3. Base Model - Final Norm
    custom_dict["final_norm.scale"] = hf_weights["model.final_norm.weight"]
    
    # 4. MLM Prediction Head
    custom_dict["output.dense.weight"] = hf_weights["head.dense.weight"]
    custom_dict["output.norm.scale"] = hf_weights["head.norm.weight"]
    
    # 5. Final Decoder (Weights are tied to the embeddings in ModernBERT!)
    custom_dict["output.linear.weight"] = hf_weights["model.embeddings.tok_embeddings.weight"]
    # custom_dict["output.linear.weight"] = hf_weights["decoder.weight"]
    custom_dict["output.linear.bias"] = hf_weights["decoder.bias"]
    
    return custom_dict





def translate_weights_decoder(hf_weights, N_blocks=22):
    """Maps the Ettin 150M state dict into your custom PyTorch architecture"""
    custom_dict = {}
    
    # 1. Base Model - Embeddings
    custom_dict["embedding.embedding"] = hf_weights["model.embeddings.tok_embeddings.weight"]
    custom_dict["embed_norm.scale"]    = hf_weights["model.embeddings.norm.weight"]
    
    # 2. Base Model - Blocks
    for i in range(N_blocks):
        # ◀─ THE QKV PACKING TRICK ─▶
        q = hf_weights[f"model.layers.{i}.attn.q_proj.weight"]
        k = hf_weights[f"model.layers.{i}.attn.k_proj.weight"]
        v = hf_weights[f"model.layers.{i}.attn.v_proj.weight"]
        
        # Concatenate them vertically (dim=0) to form a [2304, 768] matrix!
        custom_dict[f"blocks.{i}.attention.W_qkv.weight"] = torch.cat([q, k, v], dim=0)
        
        # Standard Output Projection
        custom_dict[f"blocks.{i}.attention.W_o.weight"]   = hf_weights[f"model.layers.{i}.attn.Wo.weight"]
        
        # Layer Normalizations
        if i > 0:
            custom_dict[f"blocks.{i}.norm1.scale"]        = hf_weights[f"model.layers.{i}.attn_norm.weight"]
        custom_dict[f"blocks.{i}.norm2.scale"]            = hf_weights[f"model.layers.{i}.mlp_norm.weight"]
        
        # Feed Forward Network (SwiGLU/GeGLU)
        custom_dict[f"blocks.{i}.FFN.Wi.weight"]          = hf_weights[f"model.layers.{i}.mlp.Wi.weight"]
        custom_dict[f"blocks.{i}.FFN.Wo.weight"]          = hf_weights[f"model.layers.{i}.mlp.Wo.weight"]
        
    # 3. Base Model - Final Residual Norm
    custom_dict["final_norm.scale"] = hf_weights["model.final_norm.weight"]
    
    # 4. The Causal/MLM Prediction Head (Notice the key is 'lm_head' instead of 'head')
    custom_dict["output.dense.weight"] = hf_weights["lm_head.dense.weight"]
    custom_dict["output.norm.scale"]   = hf_weights["lm_head.norm.weight"]
    
    # 5. Final Vocabulary Decoder
    custom_dict["output.linear.weight"] = hf_weights["decoder.weight"]
    custom_dict["output.linear.bias"]   = hf_weights["decoder.bias"]
    
    return custom_dict





def compare_logits(hf_model, my_model, max_vocabualary_idx=50368):
    hf_model.eval()
    my_model.eval()

    batch_size = 1
    seq_len = 8
    dummy_input_ids = torch.randint(0, max_vocabualary_idx, 
                                    (batch_size, seq_len))

    print(f"\n--- LOGITS DIAGNOSTIC ---")
    with torch.no_grad():
        # Get HF Logits
        hf_outputs = hf_model(dummy_input_ids)
        hf_logits = hf_outputs.logits
        
        # Get Custom Logits
        my_logits = my_model(dummy_input_ids)

        diff = (hf_logits - my_logits).abs().max().item()
        print(f"Vocabulary Logits: Max Diff = {diff:.2e}")
        
        if diff < 1e-4:
            print("✅ The engine is complete and ready for training!")
        else:
            print("❌ Mismatch in the LM Head.")