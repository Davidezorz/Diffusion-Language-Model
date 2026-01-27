import torch
import lightning as L
import transformers

import torch.nn.functional as F




class  GPT(L.LightningModule):

    def __init__(self, tokenizer, backbone, B):
        super().__init__()
        self.weights_folder = '.weights/'

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        self.backbone = backbone
        self.B = B


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr    = 1e-3,
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
        loss = self.loss(batch['input_ids'], batch['output_ids'])
        return loss
    

    def loss(self, input, output):
        logits = self.backbone(input)
        B, T, V = logits.shape

        logits = logits.view(B*T, V)
        targets = output.view(B*T)
        loss = F.cross_entropy(logits, 
                               targets,
                               ignore_index=self.tokenizer.pad_token_id)

        return loss


    def generate(self, ids, n_tokens, temperature=1):
        for _ in range(n_tokens):
            
            logits = self.backbone(ids)                                         # get the logits
            logits = logits[:, -1, :]                                           # B C

            probs = F.softmax(logits/temperature, dim=-1)                       # apply softmax to get probabilities
            
            id_next = torch.multinomial(probs, num_samples=1)                   # B 1  -> sample from the distribution
            ids = torch.cat((ids, id_next), dim=1)                              # B T+1
        return ids

