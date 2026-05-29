import math
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torchmetrics
import transformers

import data_processing as dataloader
from data_processing.samplers import RandomFaultTolerantSampler
import models.noise_schedule as noise_schedule
from models.DiT import DiT

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

"""
Masked Diffusion Language Model (MDLM)

This module implements the training and sampling logic for a masked discrete
diffusion language model; in this file we:
    1. Define the forward corruption process q(x_t | x_0)
       - independent token masking, as in vanilla MDLM
       - optional span masking, used as our modification
       - optional position-dependent masking

    2. Sample diffusion times t and convert them into noise levels sigma(t)

    3. Call the neural denoiser backbone DiT:
           DiT(x_t, sigma) -> logits over vocabulary

    4. Apply the SUBS parameterization:
       - the model cannot predict [MASK] as a clean token
       - unmasked tokens are copied directly

    5. Compute the continuous-time MDLM loss

    6. Implement reverse diffusion sampling from an all-[MASK] prior

What's different wrt DiT.py:
    - this file defines the probabilistic diffusion process and training logic
    - DiT.py defines the neural network p_theta(x_0 | x_t, t) (backward process)
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

class MaskedDiffusionLM(L.LightningModule):
    def __init__(self,
                 config,
                 tokenizer: transformers.PreTrainedTokenizer,
                 B=16,
                 T=512,
                 ):

        super().__init__()
        self.config = config
        self.weights_folder = 'weights/'

        self.T = T
        self.tokenizer = tokenizer
        self.V = len(self.tokenizer)
        self.mask_index = self.tokenizer.mask_token_id
        mask_token = self.tokenizer.mask_token

        if (not hasattr(self.tokenizer, 'mask_token') or mask_token is None):  # ◀╮ Define the mask token
            self.mask_index = self.V  # │ if it is not already
            self.V += 1  # ╰ defined

        self.B = B
        self.antithetic_sampling = False

        # DiT is the neural denoising model
        # It receives corrupted tokens x_t and noise level sigma, and returns vocabulary logits for predicting x_0
        self.denoiser = self.denoiser = DiT(V=self.V)
        self.denoiser.load(folder=self.weights_folder)

        metrics = torchmetrics.MetricCollection({  # ◀┬ Metrics
            'nll': NLL(),  # │ initialization
            'bpd': BPD(),  # │
            'ppl': Perplexity(),  # │
        })  # │

        metrics.set_dtype(torch.float64)  # │
        self.train_metrics = metrics.clone(prefix='train/')  # │
        self.valid_metrics = metrics.clone(prefix='val/')  # │
        self.test_metrics = metrics.clone(prefix='tests/')  # ╯

        self.noise = noise_schedule.LogLinearNoise()

        self.lr = 3e-4
        self.sampling_eps = 1e-3
        self.time_conditioning = True
        self.neg_infinity = -1000000.0

        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        # toggle for the type of corruption (later, span and position)
        self.corruption_type = "independent"  # "independent", "span", "position"
        self.max_span = 5

        # position-dependent noising
        self.position_gamma = 2.0
        self.position_loss_weighting = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)

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

    def on_train_start(self):
        sampler_cls = RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(
                    dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)

            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.B,
                    num_workers=8,  # ◀◀◀◀◀ TODO: should be a variable
                    pin_memory=True,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True)
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)


    def training_step(self, batch, batch_idx):
        attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None

        losses = self._loss(batch['input_ids'], attention_mask)
        loss = losses.loss

        self.train_metrics.update(losses.nlls, losses.token_mask)

        self.log_dict(self.train_metrics,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)

        self.log(name='trainer/loss',
                 value=loss.item(),  # The actual loss value
                 on_step=True,  # Log at each optimization step
                 on_epoch=False,  # Don't compute epoch average
                 sync_dist=True)  # Sync across all GPUs/nodes
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

        if self.corruption_type == "position":
            move_chance, loss_weight = self.position_dependent_noise(
                t=t,
                T=T,
                device=x0.device
            )
        else:
            move_chance = 1 - torch.exp(-sigma[:, None])
            loss_weight = (dsigma / torch.expm1(sigma))[:, None]

        xt = self.q_xt(x0, move_chance)

        model_output = self.forward(xt, sigma[:, None])

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)

        return - log_p_theta * loss_weight


    def _sample_t(self, B, device):
        sample = torch.rand(B, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(B, device=device) / B
            sample = (sample / B + offset) % 1

        t = (1 - self.sampling_eps) * sample + self.sampling_eps
        return t

    def q_xt(self, x, p):
        """
        Compute noisy sample x_t.

        x : (B,T) clean tokens
        p : (B,1) or (B,T) masking probability
        """

        if self.corruption_type == "independent":
            return self.q_xt_independent(x, p)

        elif self.corruption_type == "span":
            return self.q_xt_span(x, p, max_span=self.max_span)

        # since q_xt_independent is equipped of using p as:
        # - different for every position
        # - independent bernoulli using p
        elif self.corruption_type == "position":
            return self.q_xt_independent(x, p)

        else:
            raise ValueError(
                f"Unknown corruption type: {self.corruption_type}"
            )


    # base paper corruption
    def q_xt_independent(self, x, p):
        move_indices = torch.rand(*x.shape, device=x.device) < p
        return torch.where(move_indices, self.mask_index, x)

    # span corruption
    def q_xt_span(self, x, p, max_span=5):

        B, T = x.shape
        device = x.device

        mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        max_span = max_span if max_span > 0 else 4 # we set default span to 4

        for b in range(B):

            # desired number of masked tokens K_t, which we know is approx (1-a_t)L
            target = int((p[b, 0] * T).item())

            masked = 0

            while masked < target:

                span_len = torch.randint(1, max_span + 1, (1,), device=device).item()

                start = torch.randint(0, T, (1,), device=device).item()
                end = min(start + span_len, T)

                before = mask[b].sum().item()
                mask[b, start:end] = True
                after = mask[b].sum().item()

                masked += after - before

                if mask[b].all():
                    break

        xt = torch.where(
            mask,
            self.mask_index,
            x
        )

        return xt

    def position_dependent_noise(self, t, T, device):
        """
        Right-to-left noising:
            - Left positions have lower masking probability
            - Right positions have higher masking probability

        alpha_{t,l} = (1 - t)^{w_l}
        p_mask(t,l) = 1 - alpha_{t,l}
        """

        B = t.shape[0]

        positions = torch.linspace(
            0,
            1,
            T,
            device=device
        )

        weights = 1 + self.position_gamma * positions

        alpha = (1 - t[:, None]).clamp_min(1e-5) ** weights[None, :]

        move_chance = 1 - alpha

        if self.position_loss_weighting:
            # lambda_{t,l} = - alpha'_{t,l} / (1 - alpha_{t,l})
            loss_weight = (
                    weights[None, :]
                    * (1 - t[:, None]).clamp_min(1e-5) ** (weights[None, :] - 1)
                    / (1 - alpha).clamp_min(1e-5)
            )
        else:
            # simpler experimental version:
            # use scalar vanilla MDLM weighting
            sigma, dsigma = self.noise(t)
            loss_weight = (dsigma / torch.expm1(sigma))[:, None]

        return move_chance, loss_weight

    def forward(self, x, sigma):
        """Returns log score."""
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)

        logits = self.denoiser(x, sigma)

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
        unmasked_indices = (xt != self.mask_index)  # ◀─┬ One hot encoding over
        log_probs[unmasked_indices] = self.neg_infinity  # │ the not masked tokens
        log_probs[unmasked_indices, xt[unmasked_indices]] = 0  # ◀─╯
        return log_probs

    # ╭ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╮
    # ╰ ─ sampling  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╯
    @torch.no_grad()
    def _sample(self, B=4, num_steps=1000, eps=1e-5):
        """Generate samples from the model."""
        x = self._sample_prior(B, self.T)  # B T   of mask token

        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)  # ◀─┬ compute the timestes
        dt = (1 - eps) / num_steps  # ◀─╯ and delta timestap

        for i in range(num_steps):  # ╮
            print(f"\r{i + 1} / {num_steps}", end="   ")  # │ Timesteps loop
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)  # │
            x = self._ddpm_update(x, t, dt)  # ╯

        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)  # ◀─┬ last step, remove all noise
        x = self.forward(x, self.noise(t)[0]).argmax(dim=-1)  # ◀─╯ by taking the argmax
        return x

    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)  # B 1
        sigma_s, _ = self.noise(t - dt)  # B 1

        sigma_t = sigma_t.squeeze(-1)  # B
        sigma_s = sigma_s.squeeze(-1)  # B

        move_chance_t = 1 - torch.exp(-sigma_t)[:, None, None]  # B 1 1
        move_chance_s = 1 - torch.exp(-sigma_s)[:, None, None]  # B 1 1

        log_p_x0 = self.forward(x, sigma_t)  # B T V

        # Technically, this isn't q_xs since there's a division term that
        # is missing. This division term doesn't affect the samples
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)  # compute q_xs
        q_xs[:, :, self.mask_index] = move_chance_s[
            :, :, 0]  # keep a token masked with probability move_chance_s[:, :, 0]

        x_sample = _sample_categorical(q_xs)  # sampling

        not_masked = (x != self.mask_index).to(x.dtype)  # ╭ where the token are not masked
        return not_masked * x + (1 - not_masked) * x_sample  # ◀╯ copy back

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)


def _sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)