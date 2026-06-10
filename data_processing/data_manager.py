import torch
import functools
import os



# ╭───────────────────────────────────────────────────────────────────────────╮
# │                      Preprocessing for pretraining                        │
# ╰───────────────────────────────────────────────────────────────────────────╯


class DataManagerPreTrain():
    def __init__(self, caching_directory, tokenizer, n_processes: int = 1):
        if n_processes<=1: n_processes=1; print(f'using {n_processes} cpu')

        self.caching_directory = caching_directory
        self.tokenizer = tokenizer
        self.EOS = self.tokenizer.encode(self.tokenizer.eos_token, 
                                         add_special_tokens=False)[0]
        self.BOS = self.tokenizer.encode(self.tokenizer.bos_token, 
                                         add_special_tokens=False)[0]
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
            load_from_cache_file=False,
            cache_file_name=cache_file 
        )
        return tokenized_dataset


    def _group_texts(self, dataset, T):
        input_blocks  = []
        attn_masks    = []
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
                attn_masks.append(mask)
                output_blocks.append(out_block)

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
            load_from_cache_file=False,
            cache_file_name=cache_file, 
            desc='Grouping'
        )
        return chunked_dataset
    

    def getTrainloader(self, data, B, sampler_cls=None):
        shuffle = True if sampler_cls is None else False
        sampler = None if sampler_cls is None else sampler_cls(data) 

        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=B,
            num_workers=self.n_processes,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
            persistent_workers=True)
        
        train_loader.tokenizer = self.tokenizer
        return train_loader





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                  Preprocessing for question answer task                   │
# ╰───────────────────────────────────────────────────────────────────────────╯

import torch
import functools
import os

class DataManagerQA(DataManagerPreTrain):

    def __init__(self, caching_directory, tokenizer, 
                 mode: str ='AR', n_processes: int = 1):
        super().__init__(caching_directory, tokenizer, n_processes)
        group_fn_dict = {'AR':   self.group_texts_ar,
                         'BERT': self.group_texts_dit,
                         'DiT':  self.group_texts_dit
}
        self.group_texts = group_fn_dict[mode]

    def _tokenize(self, dataset):
        """ Tokeinze the conversion, in question and answer split """
        tokenizer = self.tokenizer
        conversations = []
        IGNORE = -100

        # Pre-compute the prefix to know exactly how many tokens to mask
        answer_prefix = tokenizer("Assistant: ", 
                                   add_special_tokens=False)["input_ids"]

        for messages in dataset["messages"]:
            turns = []
            current_q = None
            
            for msg in messages:
                role, content = msg["role"], msg["content"]
                
                if role == "user":
                    current_q = content
                elif role == "assistant" and current_q is not None:
                    q_tok = tokenizer(f"User: {current_q}", 
                                      add_special_tokens=False)["input_ids"]
                    q_tok = q_tok + [self.EOS] + answer_prefix
                    
                    a_tok = tokenizer(content, 
                                      add_special_tokens=False)["input_ids"]
                    a_tok = a_tok + [self.EOS]
                    
                    turns.append({"q": q_tok, "a": a_tok})
                    current_q = None                                            
                    
            conversations.append(turns)

        return {"conversations": conversations}

    def tokenize(self, dataset):
        cache_file = os.path.join(self.caching_directory + 'tokenized/', "qa_tokens.arrow")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        return dataset.map(
            self._tokenize,
            batched=True,
            num_proc=self.n_processes,
            remove_columns=dataset.column_names,                                # Clean slate
            desc="Tokenizing QA strings",
            load_from_cache_file=True,
            cache_file_name=cache_file 
        )


    # =========================================================================
    # AUTOREGRESSIVE (AR) PIPELINE
    # =========================================================================

    def _group_ar(self, dataset, T):
        """ Chunk the text base on the context window (T) given """
        in_blocks, out_blocks, mask_blocks = [], [], []
        IGNORE = -100

        for turns in dataset['conversations']:
            ids, lbls = [], []                                                  # Initialize
            
            for turn in turns:
                ids.extend(turn["q"] + turn["a"])
                lbls.extend([IGNORE] * len(turn["q"]) + turn["a"])              # Mask out user prompts
                
            i = 0
            while i < len(ids) - 1:                                             # Chunk loop
                chunk_in  = ids[i : i+T]
                chunk_lbl = lbls[i+1 : i+T+1]                                   # Target is shifted right by 1
                
                chunk_in  = [self.BOS] + chunk_in[:-1]                          # Chunk to start with [BOS]
                chunk_lbl = [IGNORE]   + chunk_lbl[:-1]                         # Mask out the [BOS] target
                    
                pad_len_in  = T - len(chunk_in)
                pad_len_lbl = T - len(chunk_lbl)
                
                in_pad   = chunk_in  + [self.tokenizer.pad_token_id] * pad_len_in
                lbl_pad  = chunk_lbl + [IGNORE] * pad_len_lbl
                mask     = [1] * len(chunk_lbl) + [0] * pad_len_lbl
                
                in_blocks.append(in_pad)
                out_blocks.append(lbl_pad)                                      # AR gets pure output_ids (targets)
                mask_blocks.append(mask)
                
                i += T                                                          # Advance directly by T
                
        return {'input_ids': in_blocks, 
                'output_ids': out_blocks, 
                'attention_mask': mask_blocks}

    def group_texts_ar(self, dataset, T):
        group_fn = functools.partial(self._group_ar, T=T)
        cache_file = os.path.join(self.caching_directory + 'grouped/', f"qa_ar_T{T}.arrow")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        return dataset.map(group_fn, 
                           batched=True, 
                           num_proc=self.n_processes, 
                           remove_columns=dataset.column_names,   # <--- ADD THIS
                           load_from_cache_file=True, 
                           cache_file_name=cache_file, 
                           desc='Grouping AR')


    # =========================================================================
    # MASKED DIFFUSION (DiT) PIPELINE
    # =========================================================================
    def _group_dit(self, dataset, T_ctx, T_ans):
        """ Chunk the text base on the context window (T_ctx+T_ans) given 
            T_ctx: is the maximum length for the context token containing
                   the pevious questions and answers
            T_ans: is the context for the answer """
        in_blocks, label_blocks, attn_masks, ans_start_idxs = [], [], [], []
        IGNORE = -100
        PAD = self.tokenizer.pad_token_id

        for turns in dataset["conversations"]:
            for target_idx in range(len(turns)):                                # Unroll: every answer becomes a target once
                target_q = turns[target_idx]["q"]
                target_a = turns[target_idx]["a"]
                
                if len(target_a) > T_ans:
                    target_a = target_a[:T_ans]                                 # Slicing answer to fit maximum allowed window
                    
                ctx_ids = []
                if len(target_q) >= T_ctx - 1:
                    ctx_ids = target_q[-(T_ctx - 1):]                           # Question is massive, take only the end
                else:
                    ctx_ids = target_q
                    hist_idx = target_idx - 1
                    
                    while hist_idx >= 0:                                        # Reverse Assemble History safely using while
                        hist_q = turns[hist_idx]["q"]
                        hist_a = turns[hist_idx]["a"]
                        turn_len = len(hist_q) + len(hist_a)
                        
                        if len(ctx_ids) + turn_len <= T_ctx - 1:
                            ctx_ids = hist_q + hist_a + ctx_ids                 # Prepend entire old turn
                            hist_idx -= 1
                        else:
                            break                                               # Stop if it doesn't fit cleanly
                            
                ctx_ids = [self.BOS] + ctx_ids                                  # Cap the context with [BOS]/[CLS]
                pads = T_ctx + T_ans - len(ctx_ids) - len(target_a)
                
                # Assemble contiguous arrays
                in_ids  = ctx_ids + target_a + [PAD] * pads
                lbl_ids = [IGNORE] * len(ctx_ids) + target_a + [IGNORE] * pads
                a_mask  = [1] * len(ctx_ids) + [1] * len(target_a) + [0] * pads
                
                in_blocks.append(in_ids)
                label_blocks.append(lbl_ids)
                attn_masks.append(a_mask)
                ans_start_idxs.append(len(ctx_ids))                             # Pass the exact boundary index!

        return {
            'input_ids':  in_blocks, 
            'output_ids': label_blocks, 
            'attention_mask': attn_masks, 
            'ans_start_idx': ans_start_idxs
        }


    def group_texts_dit(self, dataset, T_ctx, T_ans):
        group_fn = functools.partial(self._group_dit, T_ctx=T_ctx, T_ans=T_ans)
        cache_file = os.path.join(self.caching_directory + 'grouped/', f"qa_dit_C{T_ctx}_A{T_ans}.arrow")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        return dataset.map(group_fn, 
                           batched=True, 
                           num_proc=self.n_processes, 
                           remove_columns=dataset.column_names,   # <--- ADD THIS
                           load_from_cache_file=True, 
                           cache_file_name=cache_file, 
                           desc='Grouping DiT')




"""
[CLS]User:    [SEP]Assistant: 
[CLS]User: ...[SEP]Assistant: ...[SEP]User: ...[SEP]Assistant: ...[SEP]


[CONTEXT] [ASNWER] [<PAD>, <PAD>, <PAD>, ...]

input_ids 
out_ids

IGNORE_INDEX = -100
"""