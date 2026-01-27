import torch
import functools
from transformers import AutoTokenizer
import os



class DataManager():
    def __init__(self, caching_directory, n_processes: int = 1):
        if n_processes<=1: n_processes=1; print(f'using {n_processes} cpu')

        self.caching_directory = caching_directory
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.EOS = self.tokenizer.encode(self.tokenizer.eos_token)[0]
        self.BOS = self.tokenizer.encode(self.tokenizer.bos_token)[0]
        self.n_processes = n_processes

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '<MASK>'})
            
        self.vocab_size = self.tokenizer.vocab_size


    def _tokenize(self, dataset):
        tokens = self.tokenizer(
            dataset["text"],
            add_special_tokens=False,
            padding=False,                                                      # leave unpadded; pad in DataLoader later
            truncation=False,                                                   # truncation will happen in grouping
            return_attention_mask=False,
            return_token_type_ids=False
        )        
        return tokens
    

    def tokenize(self, dataset):
        cache_file = os.path.join(self.caching_directory + 'tokenized/', 
                                  "tokenized_data.arrow")

        tokenized_dataset = dataset.map(
            self._tokenize,
            batched=True,
            num_proc=self.n_processes,
            remove_columns=["text"],
            desc="Tokenizing",
            load_from_cache_file=True,
            cache_file_name=cache_file 
        )
        return tokenized_dataset



    def _group_texts(self, dataset, T):
        input_blocks = []
        attn_masks = []
        output_blocks = []

        for ids in dataset['input_ids']:
            ids = [self.BOS] + ids + [self.EOS]                                 # concatenate 'start' and 'end' tokens
            total_length = ((len(ids)-1)// T) * T                               # Compute the number of tokens in batch
            
            for i in range(0, total_length, T):                                 # Split into blocks of size T
                input_blocks.append( ids[i   :i+T])
                output_blocks.append(ids[i+1 :i+T+1])
                attn_masks.append(torch.ones(T))

            length = len(ids[total_length:])                                    # manage the last incompelte block
            if length > 0:
                PADs = [self.tokenizer.pad_token_id] * (T-length) 
                in_block  = ids[total_length:  ] + PADs
                out_block = ids[total_length+1:] + PADs + \
                            [self.tokenizer.pad_token_id]
                mask = torch.zeros(T)
                mask[:length] = 1

                input_blocks.append(in_block)
                output_blocks.append(out_block)
                attn_masks.append(mask)

        return {'input_ids':        input_blocks, 
                'attention_mask':   attn_masks, 
                'output_ids':       output_blocks}


    def group_texts(self, dataset, T):
        group_texts = functools.partial(self._group_texts, T=T) 

        cache_file = os.path.join(self.caching_directory + 'grouped/',
                                  f"grouped_data_T{T}.arrow")

        chunked_dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=self.n_processes,
            load_from_cache_file=True,
            cache_file_name=cache_file, 
            desc='Grouping'
        )
        return chunked_dataset
    

    def getTrainloader(self, data, B):
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=B,
            num_workers=self.n_processes,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True)
        train_loader.tokenizer = self.tokenizer
        
        return train_loader



