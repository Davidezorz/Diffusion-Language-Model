import numpy as np
import torch
import lightning as L

import matplotlib.pyplot as plt
import datasets 
from omegaconf import OmegaConf

from data_processing.data_manager import DataManagerPreTrain, DataManagerQA
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
    if mode == 'AR':
        text1 = "I love the italian cities!"
        text2 = "The capital of Italy is"
    elif mode in ['BERT', 'DiT']:
        text1 = "I love the italian cities!"
        text2 = "The capital of Italy is [MASK][MASK]"
    else:
        raise KeyError
        
    tokenizer_kwargs = {'return_tensors': "pt", 'add_special_tokens': False}
    input1 = tokenizer(text1, **tokenizer_kwargs)['input_ids']
    input2 = tokenizer(text2, **tokenizer_kwargs)['input_ids']

    BOS = torch.tensor([[tokenizer.bos_token_id]]).expand(1, -1)
    EOS = torch.tensor([[tokenizer.eos_token_id]]).expand(1, -1)

    inputs = torch.cat([BOS, input1, EOS, BOS, input2], dim=-1)
    print(inputs.shape)
    print(inputs)
    print(tokenizer.decode(inputs))
    print("\n\n")

    if mode == "AR":
        output = model.generate(inputs, n_tokens=25, 
                                temperature=0.5, tokenizer=tokenizer)
    else:
        output = model.generate(inputs, mask_id=tokenizer.mask_token_id)
    print(tokenizer.decode(output))




def load_shakespeare(caching_directory, print_first_lines=False):
    dataset = datasets.load_dataset("Trelis/tiny-shakespeare", 
                                    cache_dir=caching_directory)
    dataset = dataset.rename_column("Text", "text")

    if print_first_lines == True:
        print('\nFirst 150 element')
        print(repr(dataset['train'][0]['text'][:150]))

    return dataset



def print_token_examples(process_tokens, tokenizer,
                         keys = ['input_ids']):
    print('\nexample of the tokenized dataset (encoded and decoded): \n')
    i = -7
    for key in keys:
        print(f"first ids: {process_tokens[i][key][:20]}")
        print(f"last  ids: {process_tokens[i][key][-20:]}")
        print(f"len:       {process_tokens[i][key].shape}")
        print("Detokenized token: ")
        print(repr(tokenizer.decode(process_tokens[key][i])))

        print()
        print(f"len process_tokens: {len(process_tokens['input_ids'])}\n\n")



def count_pad(process_tokens, tokenizer):
    n_pad, tot = 0, 0
    pad_str = tokenizer.pad_token
    pad = torch.tensor(tokenizer.encode(pad_str, add_special_tokens=False))

    for ids in process_tokens['input_ids']:
        are_pad = ids == pad
        n_pad += are_pad.sum()
        tot += len(ids)
    
    print("PAD analysis:")
    print(f"n_pad: {n_pad}")
    print(f"tot:   {tot}")
    print(f"p:     {n_pad/tot*100: .2f}% \n")




def test_train_loader(train_loader):
    print(f"\nTesting grainloader:")
    for batch in train_loader:
        seqlens = batch.get('attention_mask')
        print(f"input_ids:  {batch['input_ids'].shape}")
        print(f"output_ids: {batch['output_ids'].shape}")
        print(f"seqlens:    {seqlens.shape}\n")
        break
    



def main():
    print('main online\n')
    config = OmegaConf.load("config.yaml")                                      # get the config
    mode   = config.mode

    backbones = {"AR": AR,  "BERT": BERT,      "DiT": DiT}
    models    = {"AR": GPT, "BERT": Diffusion, "DiT": Diffusion}

    load_fn   = {"AR":   load_ModernBERTDecoder, 
                 "BERT": load_ModernBERT, 
                 "DiT":  load_ModernBERT}

    n_processes = 4
    device      = utils.utils.getDevice()
    caching_directory = config.caching_directory

    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    ) 

    # -------------------------------------------------------------------------
    """
    dataset = load_shakespeare(caching_directory)

    data_manager = DataManagerPreTrain(caching_directory, tokenizer, n_processes)
    tokens = data_manager.tokenize(dataset['train'])
    process_tokens = data_manager.group_texts(tokens, config.backbone.T)
    process_tokens = process_tokens.with_format('torch')

    print_token_examples(process_tokens, tokenizer)
    """

    dataset = load_smoltalk()
    """
    i = 0
    for messages in reversed(dataset["messages"]):
        print("\n")
        print("--"*40)
        for msg in messages:
            role, content = msg["role"], msg["content"]
            print(f"---------- role: {role}\n {content}")
        if i>3:
            break
        i+=1
    """
    data_manager = DataManagerQA(caching_directory, tokenizer, n_processes)
    tokens = data_manager.tokenize(dataset)
    process_tokens = data_manager.group_texts_ar(tokens, config.backbone.T)

    process_tokens = process_tokens.select(range(1050))
    process_tokens = process_tokens.with_format('torch')

    print_token_examples(process_tokens, tokenizer)

    # -------------------------------------------------------------------------
    
    sampler_cls = samplers.RandomFaultTolerantSampler
    train_loader = data_manager.getTrainloader(process_tokens, config.backbone.B, 
                                               sampler_cls)


    test_train_loader(train_loader)
    count_pad(process_tokens, tokenizer)

    # -------------------------------------------------------------------------
    # defining backbone and loading weights
    # -------------------------------------------------------------------------
    backbone = backbones[mode](V = len(data_manager.tokenizer),                 # ◀ vocabulary size
                               C = config.backbone.C,                           # ◀ embedding dimension
                               H = config.backbone.H,                           # ◀ number of heads
                               N = config.backbone.N,                           # ◀ number of blocks
                              )
    translate_weights_dict = {"AR":   translate_weights_decoder,
                              "BERT": translate_weights_encoder, 
                              "DiT":  translate_weights_encoder}

    hf_model = load_fn[mode]()                                                  # hugghingface model

    trasfer_weights(backbone, hf_model, translate_weights_dict[mode], 
                    run_validation=True, show_layers=False)

    test_model(backbone, tokenizer, mode)
    # -------------------------------------------------------------------------
        
    print(f"\n{mode} parameters: {utils.utils.numberOfparameters(backbone)}")

    model = models[mode](backbone, data_manager.tokenizer, 
                         T=config.backbone.T).to(device)
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
        max_epochs=1,  
        accelerator=device,
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10  # Log to console every 10 steps
    )


    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        # val_dataloaders=train_loader
    )
    
    model.to(device)
    print('\nmodel testing:')
    #gen = model.generate(start_token.to(device), 200)
    # print(model.tokenizer.decode(gen[0]))
    



    text1 = "User: What is the capital of France?"
    text2 = "Assistant: "
        
    tokenizer_kwargs = {'return_tensors': "pt", 'add_special_tokens': False}
    input1 = tokenizer(text1, **tokenizer_kwargs)['input_ids']
    input2 = tokenizer(text2, **tokenizer_kwargs)['input_ids']

    BOS = torch.tensor([[tokenizer.bos_token_id]]).expand(1, -1)
    EOS = torch.tensor([[tokenizer.eos_token_id]]).expand(1, -1)

    inputs = torch.cat([BOS, input1, EOS, input2], dim=-1)
    gen = model.generate(inputs.to(device), 200)
    print(model.tokenizer.decode(gen[0]))



def load_smoltalk():
    ds = datasets.load_dataset("HuggingFaceTB/smoltalk",
                               "all",
                               split="train[:10%]",                             # TODO: change it back to 'train'
                               cache_dir=".data"
                              )
    # print(ds.cache_files)

    return ds



def test_smoltalk(ds, tokenizer):
    """
    for conversation in ds["messages"]:
        for msg in conversation:
            print(f'----{msg["role"]}:----\n{msg["content"]}')
        
    return
    """
    """
    if "capital of Italy" in msg["content"]:
        print(f'----{msg["role"]}:----\n{msg["content"]}')
        return
    """ 
    

    BOS = tokenizer.bos_token
    EOS = tokenizer.eos_token

    lengths = []
    for conversation in ds["messages"]:
        # Convert the conversation into plain text
        text = BOS + "\n".join(
            f"User: {msg["role"]}{EOS}Assistant: {msg["content"]}{EOS}"
            for msg in conversation
        )

        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        lengths.append(n_tokens)

    lengths = np.asarray(lengths)

    print()
    print(f"Min:             {lengths.min()}")
    print(f"Max:             {lengths.max()}")
    print(f"Mean:            {lengths.mean():.1f}\n")
    print(f"Median:          {np.median(lengths):.1f}")
    for q in [50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
        print(f"{q}th percentile: {np.percentile(lengths, q):.1f}")
    print("\n\n")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(lengths, bins=100, edgecolor="black", alpha=0.7)
    ax1.axvline(np.median(lengths), color="red", ls="--", label="Median")
    ax1.axvline(np.percentile(lengths, 95), color="orange", ls="--", label="95th percentile")
    ax1.set_xlabel("Number of tokens")
    ax1.set_ylabel("Number of conversations")
    ax1.set_title("Length Distribution")
    ax1.legend()

    # CDF
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    ax2.plot(sorted_lengths, cdf, lw=2)
    ax2.set_xlabel("Conversation length (tokens)")
    ax2.set_ylabel("Fraction of conversations")
    ax2.set_title("CDF")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2048*2)

    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
    # smoltalk = load_smoltalk()

    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    )
    #test_smoltalk(smoltalk, tokenizer)

    print("Special Tokens:")
    for token in tokenizer.all_special_tokens:
        token_encoded=tokenizer(token, add_special_tokens=False)
        print(f"{token: <8} --> {token_encoded['input_ids']}")

