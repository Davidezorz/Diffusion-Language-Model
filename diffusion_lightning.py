import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers

import models
import noise_schedule
import utils

LOG2 = math.log(2)


"""
╭ CONVENTIONS ───────────────────────────────────────────────────────────────────╮
│ ├─• B        ▶ batch size                                                      │
│ ├─• T        ▶ number of tokens in a batch i.e. length of a sequence/sentence  │
│ ├─• C        ▶ embedding dimension of each token                               │
│ │                                                                              │
│ ╰─• V        ▶ vocabulary size                                                 │
╰────────────────────────────────────────────────────────────────────────────────╯
"""



# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                                     Loss                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor



class NLL(torchmetrics.aggregation.MeanMetric):
    pass
    

class BPD(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the bits per dimension."""
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value / self.weight)
  




# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ╭──────────────────────────────────────────────────────────────────────────────╮
# │                                  Diffusion                                   │
# ╰──────────────────────────────────────────────────────────────────────────────╯
# ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬

class Diffusion(L.LightningModule):
    def __init__(self,
                 backbone,
                 tokenizer: transformers.PreTrainedTokenizer,
                 B:         int = 16,
                 T:         int = 512
                ):
        super().__init__()
        self.weights_folder = 'weights/' 

        self.T          = T
        self.B          = B
        self.tokenizer  = tokenizer
        self.V          = self.tokenizer.vocab_size
        self.mask_index = self.tokenizer.mask_token_id
        mask_token      = self.tokenizer.mask_token
        
        if (not hasattr(self.tokenizer, 'mask_token') or mask_token is None):   # ◀╮ Define the mask token
            self.mask_index = self.V                                            #  │ if it is not already
            self.V += 1                                                         #  ╰ defined

        self.backbone = backbone                                                # ◀┬ Neural Network
        self.backbone.load(folder=self.weights_folder)                          # ◀╯ initialization


        metrics = torchmetrics.MetricCollection({                               # ◀┬ Metrics 
        'nll': NLL(),                                                           #  │ initialization 
        'bpd': BPD(),                                                           #  │   
        'ppl': Perplexity(),                                                    #  │ 
        })                                                                      #  │

        metrics.set_dtype(torch.float32)                                        #  │ changed from float64 for MPS
        self.train_metrics = metrics.clone(prefix='train/')                     #  │
        self.valid_metrics = metrics.clone(prefix='val/')                       #  │ 
        self.test_metrics  = metrics.clone(prefix='test/')                      #  ╯

        self.noise = noise_schedule.LogLinearNoise()

        self.lr                = 3e-4
        self.sampling_eps      = 1e-3
        self.time_conditioning = True
        self.neg_infinity      = -1000000.0
        
        self.fast_forward_epochs  = None
        self.fast_forward_batches = None

        self.antithetic_sampling = True


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr    = 3e-4,
            betas =(0.9, 0.999),
            eps   = 1e-8,
            weight_decay = 0)
        

        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=2500
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
        attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None

        losses = self._loss(batch['input_ids'], attention_mask)
        loss = losses.loss

        self.train_metrics.update(losses.nlls, losses.token_mask)

        self.log_dict(  self.train_metrics,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True)
        
        self.log(name='trainer/loss',
                value=loss.item(),                                              # The actual loss value
                on_step=True,                                                   # Log at each optimization step
                on_epoch=False,                                                 # Don't compute epoch average
                sync_dist=True)                                                 # Sync across all GPUs/nodes
        return loss
    

    def _loss(self, x0, attention_mask):
        B, T = x0.shape

        if T > self.T:
            assert T == 2 * self.T
            start = np.random.choice(self.T)
            x0 = x0[:, start:start + self.T]

        loss = self._forward_pass_diffusion(x0)
        
        nlls = loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return Loss(loss=token_nll,
                    nlls=nlls,
                    token_mask=attention_mask)


    def _forward_pass_diffusion(self, x0):
        B, T = x0.shape
        t = self._sample_t(B, x0.device)

        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        
        model_output = self.forward(xt, sigma[:, None])
        
        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        
        return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
    

    def _sample_t(self, B, device):
        sample = torch.rand(B, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(B, device=device) / B
            sample = (sample / B + offset) % 1

        t = (1 - self.sampling_eps) * sample + self.sampling_eps
        return t
    


    def q_xt(self, x, p):
        """Computes the noisy sample xt """
        move_indices = torch.rand(* x.shape, device=x.device) < p
        xt = torch.where(move_indices, self.mask_index, x)
        return xt


    def forward(self, x, sigma):
        """Returns log score."""
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)

        logits = self.backbone(x, sigma)
    
        return self._subs_parameterization(logits=logits, xt=x)


    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)                              # ◀─┬ One hot encoding over
        log_probs[unmasked_indices] = self.neg_infinity                         #   │ the not masked tokens
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0                   # ◀─╯
        return log_probs
        

    # ╭ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╮
    # ╰ ─ sampling  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╯
    @torch.no_grad()
    def _sample(self, device='cpu', B = 4, num_steps=1000, eps=1e-5):
        """Generate samples from the model."""
        x = self._sample_prior(B, self.T, device=device)                        # B T   of mask token

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)   # ◀─┬ compute the timestes
        dt = (1 - eps) / num_steps                                              # ◀─╯ and delta timestap

        for i in range(num_steps):                                              #  ╮ 
            print(f"\r{i+1} / {num_steps}", end="   ")                          #  │ Timesteps loop
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)    #  │
            x = self._ddpm_update(x, t, dt)                                     #  ╯

        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)       # ◀─┬ last step, remove all noise 
        x = self.forward(x, self.noise(t)[0]).argmax(dim=-1)                    # ◀─╯ by taking the argmax
        return x 


    def _ddpm_update(self, x, t, dt):                    
        sigma_t, _ = self.noise(t)                                              # B 1
        sigma_s, _ = self.noise(t - dt)                                         # B 1
        
        sigma_t = sigma_t.squeeze(-1)                                           # B
        sigma_s = sigma_s.squeeze(-1)                                           # B

        move_chance_t = 1 - torch.exp(-sigma_t)[:, None, None]                  # B 1 1
        move_chance_s = 1 - torch.exp(-sigma_s)[:, None, None]                  # B 1 1
 
        log_p_x0 = self.forward(x, sigma_t )                                    # B T V
        
        # Technically, this isn't q_xs since there's a division term that
        # is missing. This division term doesn't affect the samples
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)                 # compute q_xs
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]                    # keep a token masked with probability move_chance_s[:, :, 0]

        x_sample = _sample_categorical(q_xs, q_xs.device)                       # sampling



        not_masked = (x != self.mask_index).to(x.dtype)                         #  ╭ where the token are not masked         
        print(f"DEVICE x_sample:   {x_sample.device}")
        print(f"DEVICE not_masked: {not_masked.device}")
        return not_masked * x + (1 - not_masked) * x_sample                     # ◀╯ copy back
     

    def _sample_prior(self, *batch_dims, device):
        return self.mask_index * torch.ones(* batch_dims, dtype=torch.int64,
                                            device=device)    


    def generate(self, x, B = 4, num_steps=10, eps=1e-5):   # TODO: we should use x
        return self._sample(device=x.device, B = B, num_steps=num_steps, 
                            eps=eps)



def _sample_categorical(categorical_probs, device):
  gumbel_norm = ( 1e-10 - (torch.rand_like(categorical_probs, device=device) 
                           + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)