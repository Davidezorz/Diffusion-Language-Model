import torch
import lightning as L

import datasets 

from data_processing.data_manager import DataManager
import utils.utils
from models.GPT import GPT
from models.AR import AR



if __name__ == '__main__':
    print('online\n')
    n_processes = 4
    caching_directory = '.data/'
    B, T, C = 8, 128, 64
    N, H = 2, 4
    device = utils.utils.getDevice()
    

    dataset = datasets.load_dataset("Trelis/tiny-shakespeare", 
                                    cache_dir=caching_directory)
    dataset = dataset.rename_column("Text", "text")

    print('\nFirst 150 element')
    print(repr(dataset['train'][0]['text'][:150]))


    data_manager = DataManager(caching_directory, n_processes)
    tokens = data_manager.tokenize(dataset['train'])
    process_tokens = data_manager.group_texts(tokens, T)
    process_tokens = process_tokens.with_format('torch')


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
    pad = torch.tensor(data_manager.tokenizer.encode(pad_str))

    for ids in process_tokens['input_ids']:
        are_pad = ids == pad
        n_pad += are_pad.sum()
        tot += len(ids)
    
    print(f"n_pad: {n_pad}")
    print(f"tot:   {tot}")
    print(f"p:     {n_pad/tot*100: .2f}%")

    train_loader = data_manager.getTrainloader(process_tokens, B)
    

    model = AR( V = len(data_manager.tokenizer),        # ◀ vocabulary size
                C = C,                                  # ◀ embedding dimension
                H = H,                                  # ◀ number of heads
                N = N,                                  # ◀ number of blocks
                )
    gpt = GPT(data_manager.tokenizer, model, B).to(device)

    print('\nmodel testing:')
    gen = gpt.generate(torch.tensor([[1]], device=device), 20)
    print(gpt.tokenizer.decode(gen[0]))


    print("\ntraining:")
    trainer = L.Trainer(
        max_epochs=3,  
        accelerator=device,
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10  # Log to console every 10 steps
    )


    trainer.fit(
        model=gpt, 
        train_dataloaders=train_loader
    )
    
    gpt.to(device)
    print('\nmodel testing:')
    gen = gpt.generate(torch.tensor([[1]], device=device), 200)
    print(gpt.tokenizer.decode(gen[0]))