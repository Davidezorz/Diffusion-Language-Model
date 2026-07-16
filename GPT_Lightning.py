import torch
import torch.nn.functional as F

import lightning as L

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM





class  GPT(L.LightningModule):

    def __init__(self, backbone, tokenizer, T=None,
                gen_ppl_model_id='gpt2', learning_rate=5e-5,
                warmup_steps=1000,):
        
        super().__init__()
        self.weights_folder = '.weights/'

        self.tokenizer = tokenizer
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.vocab_size

        self.backbone = backbone
        self.learning_rate = learning_rate
        self.warmup_steps=warmup_steps
        """
        # For Generative PPL (External Model)
        self.eval_tokenizer = AutoTokenizer.from_pretrained(gen_ppl_model_id)
        self.eval_model = AutoModelForCausalLM.from_pretrained(gen_ppl_model_id)
        if self.eval_tokenizer.pad_token is None:
            self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token

        self.eval_model.eval()
        for p in self.eval_model.parameters():
            p.requires_grad = False
        """


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr    = self.learning_rate,
            betas =(0.9, 0.999),
            eps   = 1e-8,
            weight_decay = 0)

        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps
        )
        """
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0, 
            end_factor=0.1,
            total_iters=2500
        )
        """
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }

        print(
            f"[OPTIMIZER] lr={self.learning_rate}, "
            f"warmup_steps={self.warmup_steps}"
        )
        return [optimizer], [scheduler_dict]
    

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)


    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)

        self.log(name='trainer/loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True)
        
        return loss
    

    def loss(self, batch):
        inputs  = batch['input_ids']
        outputs = batch['output_ids']
        seqlens = batch.get('attention_mask')

        B, T = inputs.shape

        if seqlens is not None:
            seqlens = seqlens.sum(dim=-1) if seqlens.sum() != B*T else None

        logits = self.backbone(inputs, seqlens)
        B, T, V = logits.shape

        targets = outputs.clone()
        targets[(targets < 0) & (targets != -100)] = -100
        targets[targets >= V] = -100

        valid_targets = targets != -100

        if not valid_targets.any():
            raise RuntimeError(
                "Training batch without valid targets."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=-100
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Training loss not finite: {loss.item()}"
            )

        return loss  


    def loss_qa(self, batch):
        inputs  = batch['input_ids']
        outputs = batch['output_ids']
        seqlens = batch.get('attention_mask')

        B, T = inputs.shape

        if seqlens is not None:
            seqlens = seqlens.sum(dim=-1) if seqlens.sum() != B*T else None

        logits = self.backbone(inputs, seqlens)
        B, T, V = logits.shape

        targets = outputs.clone()

        # ignore everything not valid for CrossEntropy
        targets[(targets < 0) & (targets != -100)] = -100
        targets[targets >= V] = -100

        valid_targets = targets != -100

        if not valid_targets.any():
            raise RuntimeError(
                "Validation batch without valid targets."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=-100
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Validation loss not finite: {loss.item()}"
            )

        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.loss_qa(batch)

        batch_size = batch["input_ids"].shape[0]

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        self.log(
            "val/ppl",
            torch.exp(torch.clamp(loss.detach(), max=20)),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss


    def on_validation_epoch_end(self):
        return
        x = torch.full((4, 1), self.tokenizer.bos_token_id, device=self.device)
        samples = self.generate(x, n_tokens=50)
        decoded_samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
   
        gen_ppl = self.compute_generative_perplexity(decoded_samples)
        self.log('val/gen_ppl', gen_ppl, sync_dist=True)


    @torch.no_grad()
    def compute_generative_perplexity(self, text_samples):
        """ Compute PPL of a text using an external model like GPT2
        Low PPL -> the model generates text that looks natural to GPT2. """
        self.eval_model.to(self.device)
        
        inputs = self.eval_tokenizer(text_samples, return_tensors='pt', 
                                     padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.eval_model(**inputs, labels=inputs['input_ids'])
    
        mask = inputs.get('attention_mask')
        total_tokens = inputs['input_ids'].numel()
        total_tokens = total_tokens if mask is None else mask.sum()
        
        total_nll = outputs.loss * total_tokens  
        return torch.exp(total_nll / total_tokens)


    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        keys = [k for k in state_dict.keys() if k.startswith("eval_model.")]
        
        for k in keys:                                                          # Delete them from the 
            del state_dict[k]                                                   # checkpoint dictionary


    @torch.no_grad()
    def generate_(self, ids, n_tokens, temperature=1):
        for _ in range(n_tokens):
            
            logits = self.backbone(ids)                                         # get the logits
            logits = logits[:, -1, :]                                           # B C

            probs = F.softmax(logits/temperature, dim=-1)                       # apply softmax to get probabilities
            
            id_next = torch.multinomial(probs, num_samples=1)                   # B 1  -> sample from the distribution
            ids = torch.cat((ids, id_next), dim=1)                              # B T+1
        return ids


    @torch.no_grad()
    def generate(self, ids, n_tokens, temperature=1):
        B = ids.shape[0]
        # Track which sequences in the batch are still generating
        unfinished = torch.ones(B, dtype=torch.bool, device=ids.device)

        for _ in range(n_tokens):
            logits = self.backbone(ids)                                         # get the logits
            logits = logits[:, -1, :]                                           # B C

            probs = F.softmax(logits/temperature, dim=-1)                       # apply softmax to get probabilities
            id_next = torch.multinomial(probs, num_samples=1)                   # B 1

            # If a sequence is already finished, force its next token to be EOS
            id_next[~unfinished] = self.tokenizer.eos_token_id

            ids = torch.cat((ids, id_next), dim=1)                              # B T+1

            # Update unfinished mask: turn to False if EOS is generated
            unfinished = unfinished & (id_next.squeeze(-1) != self.tokenizer.eos_token_id)

            # Break early only if ALL sequences in the batch are finished
            if not unfinished.any():
                break

        return ids