import numpy as np
import torch
import lightning as L

from omegaconf import OmegaConf


from models.AR import AR
from models.BERT import BERT
from GPT_Lightning import GPT

from models.DiT import DiT
from diffusion_lightning import Diffusion
import data_processing.samplers as samplers
import utils.utils

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
import readline  # Add this at the top of your file to enable robust terminal history!


banner = """   
 ██████╗██╗  ██╗ █████╗ ████████╗██████╗  ██████╗  ██████╗ ████████╗
██╔════╝██║  ██║██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗██╔═══██╗╚══██╔══╝
██║     ███████║███████║   ██║   ██████╔╝██║   ██║██║   ██║   ██║   
██║     ██╔══██║██╔══██║   ██║   ██╔══██╗██║   ██║██║   ██║   ██║   
╚██████╗██║  ██║██║  ██║   ██║   ██████╔╝╚██████╔╝╚██████╔╝   ██║   
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝  ╚═════╝    ╚═╝                                                                                      
"""


def chat_interface(model, device="cpu"):
    print("🤖 Chat session started! (Type 'quit' or 'exit' to stop)\n")
    print('Hi! How can I assist you today?')
    bos_id = model.tokenizer.bos_token_id
    eos_id = model.tokenizer.eos_token_id
    tok_kwargs = {'clean_up_tokenization_spaces': False,
                  'add_special_tokens':           False}
    
    # Initialize the context tensor with the BOS token already inside it
    context_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    
    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break 
            
        if user_input.lower() in ['quit', 'exit']:
            print("Ending chat...")
            break
        elif user_input.lower() in ['clear']:
            context_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            continue
            
        user_text = f"User: {user_input}"
        user_tok = model.tokenizer.encode(user_text, **tok_kwargs)
        assistant_prefix = model.tokenizer.encode("Assistant: ",  **tok_kwargs)
        
        # Combine: User: ... [EOS] Assistant:
        # Notice how clean this is now! We just build the turn and append it.
        new_turn_list = user_tok + [eos_id] + assistant_prefix
        new_input_ids = torch.tensor([new_turn_list], dtype=torch.long, device=device)
        
        # Append the new turn directly to the context (which already has [BOS])
        context_ids = torch.cat((context_ids, new_input_ids), dim=1)
            
        print("Bot: ", end="", flush=True)
        
        generated_tokens = []
        for token_id in model.generate_stream(context_ids, n_tokens=150, temperature=0.7):
            generated_tokens.append(token_id)
            word = model.tokenizer.decode([token_id],  **tok_kwargs)
            print(word, end="", flush=True)
            
        print() 
        
        if generated_tokens:
            if generated_tokens[-1] != eos_id:
                generated_tokens.append(eos_id)
                
            response_ids = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            context_ids = torch.cat((context_ids, response_ids), dim=1)


def main():
    print(f'{banner}\n')
    
    config = OmegaConf.load("config.yaml")                                      # get the config
    mode   = config.mode

    backbones = {"AR": AR,  "BERT": BERT,      "DiT": DiT}
    models    = {"AR": GPT, "BERT": Diffusion, "DiT": Diffusion}

    device      = utils.utils.getDevice()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    ) 

    # load model
    backbone = backbones[mode](V = len(tokenizer),                              # ◀ vocabulary size
                               C = config.backbone.C,                           # ◀ embedding dimension
                               H = config.backbone.H,                           # ◀ number of heads
                               N = config.backbone.N,                           # ◀ number of blocks
                              )

    model = models[mode].load_from_checkpoint(
            config.checkpoint,
            backbone=backbone,
            tokenizer=tokenizer,
            T=config.backbone.T,
        ).to(device)
    

    # chat
    chat_interface(model, device)

if __name__ == '__main__':
    main()