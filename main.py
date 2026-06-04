import torch
import lightning as L

import datasets 

from data_processing.data_manager import DataManager
import utils.utils

from models.AR import AR
from models.BERT import BERT
from gpt_lightning import GPT

from models.DiT import DiT
from diffusion_lightning import Diffusion
import data_processing.samplers as samplers

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat 
from models.base_model import *

from utils.transfer_weights import *



def load_ModernBERT():
    print("Downloading ModernBERT weights...")
    hf_model = AutoModelForMaskedLM.from_pretrained(
        "jhu-clsp/ettin-encoder-150m"
        )
    return hf_model



def load_ModernBERTDecoder():
    print("Downloading load_ModernBERTDecoder weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "jhu-clsp/ettin-decoder-150m"
        )
    return hf_model


def test_model(model, tokenizer, mode):
    inputs_dict =  {"AR":   "The capital of Italy is",
                    "BERT": "The capital of France is [MASK], " +\
                            "while the capital of Italy is [MASK]", 
                    "DiT":  "The capital of France is [MASK], " +\
                            "while the capital of Italy is [MASK]"}

    inputs = tokenizer(inputs_dict[mode], return_tensors="pt")
    
    print(inputs['input_ids'])
    print(inputs['input_ids'].shape)

    if mode == "AR":
        output = model.generate(inputs['input_ids'][:, :-1], n_tokens=25, 
                                temperature=0.5, tokenizer=tokenizer)
    else:
        output = model.generate(inputs['input_ids'], mask_id=tokenizer.mask_token_id)
    print(tokenizer.decode(output))




def main():
    print('main online\n')
    backbones = {"AR": AR, "BERT": BERT, "DiT": DiT}
    models    = {"AR": GPT, "BERT": Diffusion, "DiT": Diffusion}

    load_fn   = {"AR":   load_ModernBERTDecoder, 
                 "BERT": load_ModernBERT, 
                 "DiT":  load_ModernBERT}

    n_processes = 4
    caching_directory = '.data/'
    B, T, C = 8, 128, 768
    N, H = 22, 12
    device = utils.utils.getDevice()


    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    )
    V = len(tokenizer)
    
    # -------------------------------------------------------------------------
    mode = "AR"
    # -------------------------------------------------------------------------

    

    dataset = datasets.load_dataset("Trelis/tiny-shakespeare", 
                                    cache_dir=caching_directory)
    dataset = dataset.rename_column("Text", "text")

    print('\nFirst 150 element')
    print(repr(dataset['train'][0]['text'][:150]))


    data_manager = DataManager(caching_directory, tokenizer, n_processes)
    tokens = data_manager.tokenize(dataset['train'])
    process_tokens = data_manager.group_texts(tokens, T)
    process_tokens = process_tokens.with_format('torch')

    # -------------------------------------------------------------------------

    print('\nexample')
    i = -1
    print(process_tokens['input_ids'][i])
    print(process_tokens['output_ids'][i])

    print(repr(data_manager.tokenizer.decode(process_tokens['input_ids'][i])))
    print(repr(data_manager.tokenizer.decode(process_tokens['output_ids'][i])))

    print()
    print(len(process_tokens['input_ids']))

    
    n_pad, tot = 0, 0
    pad_str = data_manager.tokenizer.pad_token
    pad = torch.tensor(data_manager.tokenizer.encode(pad_str, add_special_tokens=False))

    for ids in process_tokens['input_ids']:
        are_pad = ids == pad
        n_pad += are_pad.sum()
        tot += len(ids)
    
    print(f"n_pad: {n_pad}")
    print(f"tot:   {tot}")
    print(f"p:     {n_pad/tot*100: .2f}%")


    # -------------------------------------------------------------------------
    
    sampler_cls = samplers.RandomFaultTolerantSampler
    train_loader = data_manager.getTrainloader(process_tokens, B, sampler_cls)


    print(f"\nTesting grainloader:")
    for batch in train_loader:
        seqlens = batch.get('attention_mask')
        print(f"input_ids:  {batch['input_ids'].shape}")
        print(f"output_ids: {batch['output_ids'].shape}")
        print(f"seqlens:    {seqlens.shape}\n")
        break
    

    
    # -------------------------------------------------------------------------
    # defining backbone and loading weights
    # -------------------------------------------------------------------------
    backbone = backbones[mode](V = len(data_manager.tokenizer),       # ◀ vocabulary size
                               C = C,                                 # ◀ embedding dimension
                               H = H,                                 # ◀ number of heads
                               N = N,                                 # ◀ number of blocks
                              )
    translate_weights_dict = {"AR":   translate_weights_decoder,
                              "BERT": translate_weights_encoder, 
                              "DiT":  translate_weights_encoder}

    hf_model = load_fn[mode]()

    trasfer_weights(backbone, hf_model, translate_weights_dict[mode], 
                    run_validation=True, show_layers=False)

    test_model(backbone, tokenizer, mode)
    # -------------------------------------------------------------------------

        
    print(f"\n{mode} parameters: {utils.utils.numberOfparameters(backbone)}")

    model = models[mode](backbone, data_manager.tokenizer, T=512).to(device)
    print("\n\n")
    try:
        print(f"Model noise device {model.noise.sigma_max.device}")
    except:
        pass

    print('\nmodel testing:')
    start_token = tokenizer(tokenizer.bos_token, return_tensors="pt", 
                            add_special_tokens=False)['input_ids']
    gen = model.generate(start_token.to(device), 100)
    print(model.tokenizer.decode(gen[0]))



    print("\ntraining:")
    trainer = L.Trainer(
        max_epochs=3,  
        accelerator=device,
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10  # Log to console every 10 steps
    )


    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=train_loader
    )
    
    model.to(device)
    print('\nmodel testing:')
    gen = model.generate(start_token.to(device), 200)
    print(model.tokenizer.decode(gen[0]))
    





if __name__ == '__main__':
    main()