import math
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
import torchmetrics
import transformers
import models.masking_schedule as masking_schedule
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

        # toggle for the type of corruption (later, position)
        self.corruption_type = "independent"  # "independent", "position", "moving_sigmoid"

        # position-dependent noising
        self.position_gamma = 2.0
        self.position_loss_weighting = False

        self.sigmoid_k = 10.0
        self.calibrated_sigmoid = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)

        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0
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
                    num_workers=0,  # ◀◀◀◀◀ TODO: should be a variable
                    pin_memory=True,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=False)
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def _ensure_batch_tensor(self, x):
        """
        Converts possible batch formats to a tensor of shape (B, T).

        Handles:
        - Tensor already shaped (B, T)
        - list of lists
        - list of tensors shaped (T,)
        - transposed list of tensors shaped (B,)
        """

        if isinstance(x, torch.Tensor):
            return x.to(self.device)

        if isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                x = torch.stack(x)

                # If shape is (T, B), transpose to (B, T)
                if x.ndim == 2 and x.shape[0] != self.B and x.shape[1] == self.B:
                    x = x.T

                return x.to(self.device)

            x = torch.tensor(
                x,
                dtype=torch.long,
                device=self.device
            )

            return x

        raise TypeError(f"Unsupported batch type: {type(x)}")

    def training_step(self, batch, batch_idx):

        input_ids = self._ensure_batch_tensor(batch["input_ids"])
        output_ids = self._ensure_batch_tensor(batch["output_ids"])
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else None

        if attention_mask is not None:
            attention_mask = self._ensure_batch_tensor(attention_mask)

        losses = self._loss(input_ids, output_ids, attention_mask)
        loss = losses.loss

        if not hasattr(self, "_running_losses"):
            self._running_losses = []

        self._running_losses.append(loss.item())

        # if batch_idx == 0:
        #     avg10 = sum(self._running_losses[-10:]) / min(10, len(self._running_losses))
        #     print(
        #         f"[{self.corruption_type}] "
        #         f"batch={batch_idx:03d} "
        #         f"loss={loss.item():.4f} "
        #         f"avg10={avg10:.4f}"
        #     )

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


    def _loss(self, input_ids, output_ids, attention_mask):
        """
        computes the final diffusion training loss for a batch:
        1. crop the sequence if it exceeds the configured context length
        2. build the supervision mask from output_ids
           (output_ids != -100)
        3. use the same mask as the noise mask, so that only prediction
           tokens are corrupted by the forward process
        4. call _forward_pass_diffusion() to obtain the per-token
           diffusion loss
        5. ignore tokens that should not contribute to the objective
           (context and padding)
        6. average the remaining token losses and return the Loss object

        The ingredients are:
            input_ids  -> what the model receives.
            output_ids -> what the model should reconstruct.
            attention_mask -> identifies real tokens (vs padding).
            noise_mask -> identifies which tokens may be corrupted.
            loss_mask -> identifies which tokens contribute to the loss.

        In the conversational setting: noise_mask == loss_mask == (output_ids != -100),
        while attention_mask is only used to ignore padding
        """
        B, T = input_ids.shape

        if T > self.T:
            assert T == 2 * self.T
            start = np.random.choice(self.T)

            input_ids = input_ids[:, start:start + self.T]
            output_ids = output_ids[:, start:start + self.T]

            if attention_mask is not None:
                attention_mask = attention_mask[:, start:start + self.T]

        # we do not compute the loss on PAD and CTX
        loss_mask = output_ids != -100

        if attention_mask is not None:
            loss_mask = loss_mask & attention_mask.bool()

        # noise_mask == loss_mask
        loss = self._forward_pass_diffusion(
            input_ids=input_ids,
            output_ids=output_ids,
            noise_mask=loss_mask,
        )

        nlls = loss * loss_mask
        count = loss_mask.sum().clamp_min(1)

        token_nll = nlls.sum() / count

        return Loss(
            loss=token_nll,
            nlls=nlls,
            token_mask=loss_mask,
        )


    def _forward_pass_diffusion(self, input_ids, output_ids, noise_mask=None):
        """
        performs one forward diffusion training step:

        1. sample a diffusion timestep t
        2. use the masking schedule (independent, position-dependent,
           moving sigmoid) to compute:
              - move_chance: masking probability for each token
              - loss_weight: analytical weighting required by MDLM
        3. apply q_xt() to corrupt the input sequence according to
           move_chance and the provided noise_mask
        4. feed the corrupted sequence to the DiT denoiser together
           with the noise level sigma(t)
        5. gather the log-probability assigned to the target tokens
           (output_ids)
        6. return the per-token weighted negative log-likelihood

        it returns a tensor of shape (B, T), leaving masking and averaging
        to _loss().
        """
        B, T = input_ids.shape
        t = self._sample_t(B, input_ids.device)

        sigma, _ = self.noise(t)

        if self.corruption_type == "independent":
            move_chance, loss_weight = masking_schedule.vanilla_masking(
                t=t,
                T=T,
                device=input_ids.device,
                noise=self.noise,
            )

        elif self.corruption_type == "position":
            move_chance, loss_weight = masking_schedule.position_dependent_masking(
                t=t,
                T=T,
                device=input_ids.device,
                noise=self.noise,
                gamma=self.position_gamma,
                position_loss_weighting=self.position_loss_weighting,
            )

        elif self.corruption_type == "moving_sigmoid":
            move_chance, loss_weight = masking_schedule.moving_sigmoid_masking(
                t=t,
                T=T,
                device=input_ids.device,
                noise=self.noise,
                k=self.sigmoid_k,
                calibrated=self.calibrated_sigmoid,
            )

        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")

        xt = self.q_xt(
            input_ids,
            move_chance,
            noise_mask=noise_mask,
        )

        model_output = self.forward(xt, sigma[:, None])

        # replace ignored targets with a safe dummy index:
        safe_output_ids = output_ids.clone()
        safe_output_ids[safe_output_ids == -100] = 0

        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=safe_output_ids[:, :, None],
        ).squeeze(-1)

        return -log_p_theta * loss_weight


    def _sample_t(self, B, device):
        sample = torch.rand(B, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(B, device=device) / B
            sample = (sample / B + offset) % 1

        t = (1 - self.sampling_eps) * sample + self.sampling_eps
        return t


    def q_xt(self, x, p, noise_mask=None):
        """
        apply masking corruption:
        x:          (B, T) clean token ids
        p:          (B, 1) or (B, T) masking probability
        noise_mask: optional boolean mask (B, T): if provided, only positions where noise_mask=True can be masked (i.e. no CTX tokens)
        """
        move_indices = torch.rand(x.shape, device=x.device) < p

        if noise_mask is not None:
            move_indices = move_indices & noise_mask.bool()

        return torch.where(move_indices, self.mask_index, x)


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