import torch
import lightning as L

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch.nn.functional as F




class  GPT(L.LightningModule):

    def __init__(self, tokenizer, backbone, B, 
                 gen_ppl_model_id='gpt2'):
        super().__init__()
        self.weights_folder = '.weights/'

        self.tokenizer = tokenizer
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.vocab_size

        self.backbone = backbone
        self.B = B

        # For Generative PPL (External Model)
        self.eval_tokenizer = AutoTokenizer.from_pretrained(gen_ppl_model_id)
        self.eval_model = AutoModelForCausalLM.from_pretrained(gen_ppl_model_id)
        if self.eval_tokenizer.pad_token is None:
            self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token

        self.eval_model.eval()
        for p in self.eval_model.parameters():
            p.requires_grad = False


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr    = 5e-3,
            betas =(0.9, 0.999),
            eps   = 1e-8,
            weight_decay = 0)
        
        """
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2500
        )
        """
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0, 
            end_factor=0.1,
            total_iters=2500
        )

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
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

        B, T = batch['input_ids'].shape
        if seqlens is not None:
            seqlens = seqlens.sum(dim=-1) if seqlens.sum() != B*T else None
        
        logits = self.backbone(inputs, seqlens)
        B, T, V = logits.shape

        logits  = logits.view(B*T, V)
        targets = outputs.view(B*T)

        loss = F.cross_entropy(logits, 
                               targets,
                               ignore_index=self.tokenizer.pad_token_id)
        return loss        


    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)

        self.log('val/loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log('val/ppl', torch.exp(loss), on_step=False,
                 on_epoch=True, sync_dist=True)
        return loss


    def on_validation_epoch_end(self):
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


    def generate(self, ids, n_tokens, temperature=1):
        for _ in range(n_tokens):
            
            logits = self.backbone(ids)                                         # get the logits
            logits = logits[:, -1, :]                                           # B C

            probs = F.softmax(logits/temperature, dim=-1)                       # apply softmax to get probabilities
            
            id_next = torch.multinomial(probs, num_samples=1)                   # B 1  -> sample from the distribution
            ids = torch.cat((ids, id_next), dim=1)                              # B T+1
        return ids

