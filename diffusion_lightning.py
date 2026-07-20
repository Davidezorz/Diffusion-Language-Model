import itertools
import math
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers

import models.noise_schedule as noise_schedule
import noise.masking_schedule as masking_schedule


LOG2 = math.log(2)


"""
╭ CONVENTIONS ───────────────────────────────────────────────────────────────────╮
│ ├─• B        ▶ batch size                                                      │
│ ├─• T        ▶ number of tokens in a batch i.e. length of a sequence/sentence  │
│ ├─• T_ans    ▶ number of tokens for the diffusion block (answer section)       │
│ ├─• C        ▶ embedding dimension of each token                               │
│ │                                                                              │
│ ╰─• V        ▶ vocabulary size                                                 │
╰────────────────────────────────────────────────────────────────────────────────╯
"""
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    )

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
                 tokenizer:         transformers.PreTrainedTokenizer,
                 masking_schedule:  masking_schedule.Masking,                   # class in file noise/masking_schedule.py
                 T_ans:             int,
                 learning_rate:     float = 5e-5,
                 warmup_steps:      int   = 1000,
                ):
        super().__init__()
        self.weights_folder = 'weights/' 

        self.tokenizer  = tokenizer
        self.V          = self.tokenizer.vocab_size
        self.mask_index = self.tokenizer.mask_token_id
        mask_token      = self.tokenizer.mask_token
        self.T_ans      = T_ans
        
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

        self.noise = masking_schedule.noise
        self.masking_schedule = masking_schedule

        self.lr                = learning_rate
        self.warmup_steps      = warmup_steps
        self.sampling_eps      = 1e-3
        
        self.fast_forward_epochs  = None
        self.fast_forward_batches = None

        self.antithetic_sampling = True

        self.IGNORE_IDX   = -100
        self.neg_infinity = -1000000.0


    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            checkpoint['sampler_state'] = self.trainer.train_dataloader.sampler.state_dict()


    def on_load_checkpoint(self, checkpoint):
        if 'sampler_state' in checkpoint:
            self.saved_sampler_state = checkpoint['sampler_state']


    def on_train_start(self):
        if hasattr(self, 'saved_sampler_state'):
            self.trainer.train_dataloader.sampler.load_state_dict(self.saved_sampler_state)
            del self.saved_sampler_state


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr    = self.lr,
            betas =(0.9, 0.999),
            eps   = 1e-8,
            weight_decay = 0)
        

        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps
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
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
        ans_start_idx = batch.get('ans_start_idx', None)

        losses = self._loss(
            input_ids     =batch["input_ids"],
            output_ids    =batch["output_ids"],
            attention_mask=attention_mask,
            ans_start_idx =ans_start_idx
        )

        loss = losses.loss

        self.train_metrics.update(losses.nlls, losses.token_mask)

        self.log_dict(
            self.train_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            name="trainer/loss",
            value=loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )

        return loss
    

    def validation_step(self, batch, batch_idx):
        attention_mask = batch.get("attention_mask")
        ans_start_idx = batch.get('ans_start_idx', None)

        losses = self._loss(
            input_ids     =batch["input_ids"],
            output_ids    =batch["output_ids"],
            attention_mask=attention_mask,
            ans_start_idx =ans_start_idx
        )

        self.valid_metrics.update(
            losses.nlls,
            losses.token_mask,
        )

        self.log_dict(
            self.valid_metrics,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val_loss",
            losses.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return losses.loss
    

    def _loss(self, input_ids, output_ids, 
              attention_mask=None, ans_start_idx=None):
        B, T = input_ids.shape

        if ans_start_idx is None:                                              # If we don't know where the answers start, we assume
            ans_start_idx = ans_start_idx = torch.full(                        # the answers are at the end of the tensor
                    (B,),
                    T - self.T_ans, 
                    dtype=torch.long,
                    device=input_ids.device,
                )            

        loss_mask = output_ids != self.IGNORE_IDX                               # We ignore the loss when the tokens_id will be self.IGNORE_IDX

        if attention_mask is not None:
            loss_mask = loss_mask & attention_mask.bool()

        loss = self._forward_pass_diffusion(
            input_ids     =input_ids,
            output_ids    =output_ids,
            attention_mask=attention_mask,
            ans_start_idx =ans_start_idx
        )

        nlls = loss * loss_mask
        count = loss_mask.sum().clamp_min(1)
        token_nll = nlls.sum() / count

        return Loss(
            loss=token_nll,
            nlls=nlls,
            token_mask=loss_mask,
        )


    def _forward_pass_diffusion(self, input_ids, output_ids, 
                                attention_mask=None,
                                ans_start_idx=None):
        B, T = input_ids.shape
        t = self._sample_t(B, input_ids.device)

        sigma, _ = self.noise(t)
        move_chance, loss_weight = self.masking_schedule(t)

        xt = self.q_xt(                                                         # get the corrupted x at time t
            x            =input_ids,
            p            =move_chance,
            ans_start_idx=ans_start_idx,
        )

        seqlens = None                                                          # TODO: could be faster not to use it
        if attention_mask is not None:
            seqlens = attention_mask.sum(dim=-1).long()

        model_output = self.forward(
            xt,
            sigma[:, None],
            seqlens=seqlens,
        )
        safe_output_ids = output_ids.clone()
        safe_output_ids[safe_output_ids == self.IGNORE_IDX] = 0

        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=safe_output_ids[:, :, None],
        ).squeeze(-1)

        loss_weight = self._scatter_to_answer(loss_weight, ans_start_idx,   # we allign the loss weight values
                                              T, fill_value=1)              # with the answer
        return -log_p_theta * loss_weight
    

    def _sample_t(self, B, device):
        sample = torch.rand(B, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(B, device=device) / B
            sample = (sample / B + offset) % 1

        t = (1 - self.sampling_eps) * sample + self.sampling_eps
        return t


    def q_xt(self, x, p, ans_start_idx):
        """Corrupt the answer by putting MASK according to probability p."""
        B, T = x.shape

        move_indices = torch.rand(B, self.T_ans, device=x.device) < p           # B T_ans, actual mask block
        full_mask = self._scatter_to_answer(move_indices, ans_start_idx, T)

        return torch.where(full_mask, self.mask_index, x)                       # returns the text corrupted on the answer            


    def _scatter_to_answer(self, values, ans_start_idx, T, fill_value=0):
        """Scatter a B x T_ans tensor into a B x T tensor at answer positions."""
        B = values.shape[0]

        out = torch.full((B, T), fill_value,                                     # Full mask initialized to `fill_value`
                         dtype=values.dtype,
                         device=values.device,
                        )

        positions = ans_start_idx[:, None] + torch.arange(                      # B T_ans, for QA we corrupt only the part
            self.T_ans, device=values.device, dtype=torch.long                  # where the answer lives, that starts at
        )[None, :]                                                              # ans_start_idx (which has dim B) 

        rows = torch.arange(B, device=values.device)[:, None]                   # B 1
        out[rows, positions] = values                                           # Scatter values to the answer positions

        return out


    def forward(self, x, sigma, seqlens=None):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)

        logits = self.backbone(
            x,
            sigma,
            seqlens=seqlens,
        )

        return self._subs_parameterization(logits=logits, xt=x)


    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity                      # log prob at the mask index = -infinity -> p(mask)=0          
        
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
    def _sample(self, B = 4, num_steps=1000, eps=1e-5):
        """Generate samples from the model."""
        x = self._sample_prior(B, self.T)                                       # B T   of mask token

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)   # ◀─┬ compute the timestes
        dt = (1 - eps) / num_steps                                              # ◀─╯ and delta timestap

        for i in range(num_steps):                                              #  ╮ 
            print(f"\r{i+1} / {num_steps}", end="   ")                          #  │ Timesteps loop
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)    #  │
            x = self._ddpm_update(x, t, dt)                                     #  ╯

        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)       # ◀─┬ last step, remove all noise 
        outputs = self.forward(x, self.noise(t)[0]).argmax(dim=-1)                    # ◀─╯ by taking the argmax
        return outputs 


    def _ddpm_update(self, x, t, dt):                    
        sigma_t, _ = self.noise(t)                                              # B 1
        sigma_s, _ = self.noise(t - dt)                                         # B 1
        
        sigma_t = sigma_t.squeeze(-1)                                           # B
        sigma_s = sigma_s.squeeze(-1)                                           # B

        move_chance_t = 1 - torch.exp(-sigma_t)[:, None, None]                  # B 1 1
        move_chance_s = 1 - torch.exp(-sigma_s)[:, None, None]                  # B 1 1
 
        log_p_x0 = self.forward(x, sigma_t)                                     # B T V
        
        # Technically, this isn't q_xs since there's a division term that
        # is missing. This division term doesn't affect the samples
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)                 # compute q_xs
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]                    # keep a token masked with probability move_chance_s[:, :, 0]

        x_sample = _sample_categorical(q_xs, q_xs.device)                       # sampling

        not_masked = (x != self.mask_index).to(x.dtype)                         #  ╭ where the token are not masked         
        return not_masked * x + (1 - not_masked) * x_sample                     # ◀╯ copy back
     

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(* batch_dims, dtype=torch.int64,
                                            device=self.device)    


    @torch.no_grad()
    def generate(
        self,
        ids,
        ans_start_idx=None,
        num_steps=100,
        eps=1e-5,
        temperature=1.0,
        evaluation_elements=None
    ):
        """
        Generate a response conditioned on a prompt using reverse diffusion.

        The prompt tokens remain fixed throughout the diffusion process, while
        a sequence of [MASK] tokens is appended to represent the answer to be
        generated. At each reverse diffusion step, only the masked answer region
        is updated, whereas the prompt is kept unchanged.

        Since the generated sequence contains no padding, all tokens are valid
        and the sequence length passed to the backbone corresponds to the full
        generated sequence.

        Args:
            ids: prompt tokens of shape (B, T_prompt)
            n_tokens: number of answer tokens to generate
            num_steps: number of reverse diffusion steps
            eps: final diffusion time
            temperature: sampling temperature applied to the predicted logits

        Returns:
            tensor of shape (B, T_prompt + n_tokens) containing the prompt
            followed by the generated answer.
        """
        B, T_prompt = ids.shape
        device = ids.device
        
        T = T_prompt + self.T_ans
        
        if ans_start_idx is None:
            ans_start_idx = torch.full((B,), T_prompt, 
                                       dtype=torch.long, device=device)

        # 1. Base initialization: Fill with PAD and copy the prompt.
        pad_id = self.tokenizer.pad_token_id
        x = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        x[:, :T_prompt] = ids
        
        # 2. Build the boolean mask for the answer positions using the scatter function.
        # This handles shifting the answer block past the variable context length.
        true_block = torch.ones((B, self.T_ans), dtype=torch.bool, device=device)
        gen_mask = self._scatter_to_answer(true_block, ans_start_idx, 
                                           T, fill_value=0)
        
        # 3. Inject [MASK] tokens exactly into the targeted answer block.
        x = torch.where(gen_mask, self.mask_index, x)

        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(B, device=device)
            x_prev = x

            x = self._ddpm_update_conditional(                                  # we run the denoising step by 
                x_t          =x,                                                # keeping the prompt frozen
                t            =t,
                dt           =dt,
                gen_mask     =gen_mask,
                ans_start_idx=ans_start_idx,
                temperature  =temperature,
            )

            if evaluation_elements is not None:
                evaluator, target_labels = evaluation_elements
                last_hidden = self.backbone.last_hidden                         # B T C
                logits      = self.backbone.logits                              # B T V
                evaluator.update_step(i, logits, last_hidden, 
                                      x_prev, target_labels)
            """
            res = tokenizer.decode(
                x[0],
                skip_special_tokens=False,
            )
            print(res, end='\n----\n')"""

        t = timesteps[-1] * torch.ones(B, device=device)                     # final denoise step only on answer positions
        sigma, _ = self.noise(t)

        seqlens = ans_start_idx + self.T_ans
        final_logits = self.forward(x, sigma, seqlens) / temperature

        final_tokens = final_logits.argmax(dim=-1)                              # last stpe unmask everything
        x = torch.where(gen_mask, final_tokens, x)

        return x


    def _ddpm_update_conditional(
            self,
            x_t,
            t,
            dt,
            gen_mask,
            ans_start_idx,
            temperature=1.0,
    ):
        """
        Sample one conditional reverse-diffusion transition from time t to s,
        where s = t - dt. For every currently masked position, the reverse distribution is

        p_theta(x_s | x_t)
            = ((mu_t - mu_s) / mu_t) p_theta(x_0 | x_t)
              + (mu_s / mu_t) delta_MASK,

        where
            mu_t = 1 - exp(-sigma_t)
            mu_s = 1 - exp(-sigma_s)

        Since 1 / mu_t is a common positive normalization factor, the sampler
        uses the equivalent unnormalized categorical weights

            w(x_s)
                = (mu_t - mu_s) p_theta(x_0 | x_t)
                  + mu_s delta_MASK.

        Tokens already revealed remain unchanged according to the SUBS assumptions, positions outside gen_mask are conditioning
        tokens are therefore kept fixed.

        Args:
            x_t: current sequence at diffusion time t, with shape (B, T).
            t: current diffusion time, with shape (B, 1).
            dt: Reverse time-step size. The next time is s = t - dt.
            gen_mask: Boolean tensor identifying positions allowed to evolve.
                In conditional QA generation, these are the answer positions.
            temperature: temperature applied to the predicted clean-token distribution.

        Returns:
            x_s:
                Sequence sampled at the earlier diffusion time s.
        """
        B, T = x_t.shape

        # ------------------------------------------------------------------
        # 1. Define the two diffusion times: current t and earlier s < t
        # ------------------------------------------------------------------
        s = t - dt

        sigma_t, _ = self.noise(t)
        # ------------------------------------------------------------------
        # 2. Forward masking probabilities
        #    mu_t = P(x_t = MASK | x_0)
        #    mu_s = P(x_s = MASK | x_0)
        # ------------------------------------------------------------------
        mu_t, _ = self.masking_schedule(t)
        mu_s, _ = self.masking_schedule(s)

        # Reshape for broadcasting over sequence positions and vocabulary
        mu_t = self._scatter_to_answer(mu_t, ans_start_idx, T, fill_value=0.0)
        mu_s = self._scatter_to_answer(mu_s, ans_start_idx, T, fill_value=0.0)

        # Reshape for broadcasting over vocabulary: [B, T, 1]
        mu_t = mu_t[:, :, None]
        mu_s = mu_s[:, :, None]

        # ------------------------------------------------------------------
        # 3. Predict the clean-token distribution -> log_p_x0 = log p_theta(x_0 | x_t, t)
        # ------------------------------------------------------------------
        seqlens = ans_start_idx + self.T_ans

        log_p_x0 = self.forward(
            x_t,
            sigma_t,
            seqlens=seqlens,
        )

        if temperature != 1.0:
            log_p_x0 = log_p_x0 / temperature

        p_x0 = log_p_x0.exp()

        # ------------------------------------------------------------------
        # 4. Construct the reverse categorical distribution
        # ------------------------------------------------------------------
        reveal_mass = mu_t - mu_s
        remain_masked_mass = mu_s

        reverse_weights = p_x0 * reveal_mass

        reverse_weights[:, :, self.mask_index] = (
            remain_masked_mass[:, :, 0]
        )

        # ------------------------------------------------------------------
        # 5. Sample x_s from the reverse categorical weights
        # ------------------------------------------------------------------
        sampled_x_s = _sample_categorical(
            reverse_weights,
            reverse_weights.device,
        )

        # ------------------------------------------------------------------
        # 6. Enforce the SUBS assumptions:
        #    - if x_t is already visible, then x_s = x_t.
        #    - only currently masked positions may change.
        # ------------------------------------------------------------------
        currently_masked = x_t == self.mask_index

        x_s = torch.where(
            currently_masked,
            sampled_x_s,
            x_t,
        )

        # ------------------------------------------------------------------
        # 7. Enforce conditional generation:
        #    - context positions remain fixed
        #    - only answer positions identified by gen_mask may evolve
        # ------------------------------------------------------------------
        x_s = torch.where(
            gen_mask,
            x_s,
            x_t,
        )

        return x_s


def _sample_categorical(categorical_probs, device):
    gumbel_norm = ( 1e-10 - (torch.rand_like(categorical_probs, device=device) 
                  + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


"""
[T_1, T_2, T_3, PAD, PAD]
[T_1, T_2, T_3, T_4, T_5]

[T_1, T_2, T_3, PAD, PAD] <- [DIFFUSION_BLOCK]
[T_1, T_2, T_3, T_4, T_5] <- [DIFFUSION_BLOCK]

[T_1, T_2, T_3, [DIFFUSION_BLOCK], PAD, PAD]
[T_1, T_2, T_3, T_4, T_5, [DIFFUSION_BLOCK]]

QAQ A
"""


