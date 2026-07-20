# Masked Diffusion Language Model

A PyTorch / PyTorch-Lightning implementation of the **MDLM** paper (*Simple and Effective Masked Diffusion Language Models*, Sahoo et al., 2024), extended with **position-dependent** and **moving-sigmoid** noising schedules, and benchmarked against an autoregressive baseline on a conversational QA task.

---

## What is in this repository

The project is split into four layers:

1. **The neural backbone** (`models/`): a small, custom RoPE-based Transformer written from scratch that can be instantiated as an autoregressive (AR), a bidirectional encoder (BERT), or a Diffusion-Transformer (DiT) variant.
2. **The diffusion process** (`diffusion_lightning.py`, `noise/`): the MDLM training objective (weighted CE), the SUBS parameterization, the SUBS-aware reverse diffusion sampler, and three type of noising schedules: 
    - *independent*: the one proposed in the *Simple and Effective Masked Diffusion Language Models* paper
    - *position*: a position dependent noise schedule that assign a weight on each position
    - *moving sigmoid*: a position dependent noise schedule that weights noise according to the sigmoid fuction
3. **Data & training pipeline** (`data_processing/`, `main.py`, `chat.py`): tokenization, QA-style sequence chunking, Lightning training loop, and an interactive chat session.
4. **Evaluation & analysis** (`benchmark.py`, `tests/`): intrinsic trajectory metrics, extrinsic semantic metrics, an LLM-as-a-Judge harness, and sanity tests for the corruption schedules.

---

## How to run it

```bash
# 1. install
pip install -r requirements.txt

# 2. edit config.yaml (mode, corruption_type, gamma, k, etc.)

# 3. train
python main.py

# 4. chat with a trained checkpoint for AR
python chat.py
```

The dataset (`HuggingFaceTB/smoltalk`) is downloaded automatically into `.data/`. Trained checkpoints are saved under `checkpoints/`; fine-tuned weights live under `.weights/`. The chat script reads `config.yaml` and the `checkpoint` path declared there.

---

## File structure

```
.
├── LICENSE
├── README.md
├── project.md                 # project planning notes
├── requirements.txt
├── config.yaml                # single source of truth for every experiment
│
├── main.py                    # entry point: load → tokenize → train → generate
├── chat.py                    # interactive REPL for a trained checkpoint
├── diffusion_lightning.py     # ★ THE Diffusion LightningModule (training + sampling)
├── GPT_Lightning.py           # LightningModule for the AR baseline
├── benchmark.py               # perplexity, BERTScore, FBD, LLM-judge, trajectory
│
├── data_processing/
│   ├── data_manager.py        # tokenization + QA chunking (AR & DiT pipelines)
│   └── samplers.py            # fault-tolerant PyTorch samplers
│
├── noise/
│   ├── noise_schedule.py      # loglinear / cosine / geometric σ(t) schedules
│   └── masking_schedule.py    # ★ independent / position / moving-sigmoid schedules
│
├── models/
│   ├── base_model.py          # shared layers (RoPE, MHA, FFN, Block, LastBlock)
│   ├── AR.py                  # causal baseline
│   ├── BERT.py                # bidirectional encoder variant
│   ├── DiT.py                 # ★ DiT: the MDLM denoiser (timestep-conditioned)
│   ├── mdlm.py                # original MDLM reference implementation (deprecated)
│   ├── noise_schedule.py      # duplicate of /noise/noise_schedule.py (older copy)
│   └── masking_schedule.py    # duplicate of /noise/masking_schedule.py (older copy)
│
├── scripts/
│   ├── smoke_train_mdlm.py
│   ├── smoke_train_reconstruction.py
│   └── smoke_train_toydataset.py
│
├── tests/
│   ├── test.py
│   ├── test_checkpoints.py
│   ├── test_corruption.py
│   ├── test_corruption_diffusion.py    # visual check of the diffusion corruption
│   ├── test_moving_sigmoid.py          # unit tests for the sigmoid schedule
│   ├── debug_visual.py
│   └── cude_test.py
│
├── utils/
│   ├── utils.py
│   ├── transfer_weights.py    # Hugging Face → custom state-dict translation
│   └── layers_checks.py       # layer-by-layer diagnostic vs. the HF reference
│
├── docs/
│   └── mdlm.md                # short mathematical note on the position schedule
│
├── checkpoints/               # Lightning checkpoints (auto-saved)
├── lightning_logs/            # TensorBoard logs
├── .weights/                  # manually-saved raw state-dicts
├── .data/                     # cached datasets & tokenized arrows
├── losses.csv
├── summary.csv
└── plots.ipynb                # visualization / exploratory notebook
```

---

## What the most important files do

### `main.py` — the experiment launcher
Single entry point that:

1. Parses `config.yaml` (`mode ∈ {AR, DiT}`, batch size, layers, heads, learning rate, diffusion hyperparameters).
2. Downloads `HuggingFaceTB/smoltalk` and splits it into train / validation.
3. Builds the tokenizer from `jhu-clsp/ettin-decoder-150m`.
4. Tokenizes the conversations and groups them into `T_ctx` + `T_ans` blocks through `DataManagerQA`.
5. Optionally loads a Hugging Face checkpoint into the custom backbone through `utils/transfer_weights.py`.
6. Trains the model with PyTorch-Lightning (with checkpointing) **or** runs in inference-only mode.
7. At the end, generates qualitative answers for a fixed list of test prompts using the appropriate sampler (autoregressive for `AR`, conditional reverse diffusion for `DiT`).

### `diffusion_lightning.py` — the MDLM training & sampling logic ★
This is the heart of the project. It implements, in one LightningModule:

- **SUBS parameterization** (`_subs_parameterization`): zeroes the logit at the `[MASK]` index (`p(mask) = 0`) and forces the output for already-visible tokens to be a point mass on that token, making their CE contribution zero.
- **The forward corruption** (`q_xt`): given the per-position masking probability `p` returned by the chosen schedule, scatters the mask into the answer block of every sequence in the batch.
- **The continuous-time MDLM loss** (`_forward_pass_diffusion`): samples `t ∼ U(ε, 1)`, looks up `σ(t)` and `move_chance, loss_weight` from the masking schedule, corrupts `x`, gathers the log-probability of the clean answer tokens, and returns the negative weighted log-likelihood.
- **The unconditional reverse-diffusion sampler** (`_sample` / `_ddpm_update`).
- **The conditional QA sampler** (`generate` / `_ddpm_update_conditional`): fixes the prompt tokens, runs the reverse process *only* on the answer positions, and uses the exact MDLM formula `p_θ(x_s | x_t) = (μ_t − μ_s)/μ_t · p_θ(x_0 | x_t) + μ_s/μ_t · δ_MASK` with Gumbel-top-k sampling.

The loss / metrics are tracked through custom `NLL` / `BPD` / `Perplexity` aggregators that accumulate over the union of all masked positions.

### `noise/masking_schedule.py` — the schedule dispatcher
A single `Masking` class that exposes three pluggable schedules, each returning `(move_chance, loss_weight)` for a batch of timesteps:

- **`vanilla_masking`** — the MDLM baseline: every position is masked with the same probability `1 − e^{−σ(t)}`.
- **`position_dependent_masking`** — uses weights `ρ_i = 1 + γ (l_i − ½)` so that the right half of the answer block is masked more aggressively, while the average mask rate stays equal to the vanilla schedule.
- **`moving_sigmoid_masking`** — masks according to `σ(k(l − c_t))`, with the front `c_t` solved analytically so that the *average* mask rate matches `1 − α_t`. This is the "moving front" schedule described in the LaTeX notes.

### `models/DiT.py` — the denoising network ★
The MDLM denoiser. A RoPE-based Transformer that:

- Embeds tokens with a tied input/output embedding.
- Encodes the diffusion time `t` through `TimestepEmbedder` (sinusoidal → MLP) and injects it via **adaptive layer norm (ALN)** in every block (Peebles & Xie, "Scalable Diffusion Models with Transformers").
- Stores `last_hidden` and `logits` on the module so the trajectory evaluator in `benchmark.py` can read them without recomputing the forward pass.

`models/AR.py` and `models/BERT.py` are the same architecture with `is_causal=True` / `False` respectively, used as a baseline and as the encoder weight donor.

### `data_processing/data_manager.py` — QA chunking
`DataManagerQA` converts the smoltalk conversations into `[BOS] context [answer] [PAD...]` blocks of size `T_ctx + T_ans` and records the answer-start index per sample. It exposes two grouping strategies (`group_texts_ar` and `group_texts_dit`); the diffusion model uses the latter, which keeps the question frozen and reserves `T_ans` slots for the answer.

### `GPT_Lightning.py` — the AR baseline
Standard autoregressive LightningModule: shift-based next-token CE loss, top-k/top-p `generate_stream` for the chat demo, and the same AdamW + warmup scheduler used by the diffusion side so the two are directly comparable.

### `benchmark.py` — evaluation suite
A collection of evaluators, not all of which are wired into a single runner:

- `Perplexity` — exact next-token NLL for AR; ELBO-based NLL for the DDM.
- `ChatStructureEvaluator` — length distribution, missing-EOS rate, user-token hallucinations.
- `DiversityEvaluator` — N-gram type/token ratio and Self-BLEU with smoothing for short responses.
- `SemanticEvaluator` — BERTScore (micro) and Fréchet BERT Distance on joint `[Prompt ⊕ Response]` embeddings (macro).
- `LLMJudgeEvaluator` — pairwise A/B judging with a randomized position-swap to neutralize position bias; calls any OpenAI-compatible endpoint (also works with local Ollama / vLLM).
- `DiffusionTrajectoryEvaluator` — records, for every reverse-diffusion step, the **masked-token Shannon entropy**, the **step-perplexity against the ground truth**, and the **cosine similarity of the mean-pooled hidden state to the final state** (semantic convergence). It is the tool that produced the "Trajectory Analysis" plots in the LaTeX write-up.
- `BenchmarkOrchestrator` — high-level runner that ties generation + evaluation + plotting together (most evaluators are scaffolded but the orchestration is left as a starting point).

### `config.yaml` — single source of truth
Everything that varies between experiments lives here: `mode` (AR vs DiT), `corruption_type` (`independent` / `position` / `moving_sigmoid`), `position_gamma`, `sigmoid_k`, `calibrated_sigmoid`, noise schedule, batch size, sequence lengths, learning rate, etc.

---

## Other useful descriptions

- **`models/base_model.py`** — every shared building block: `Rotary` (cached RoPE), `EmbeddingLayer`, `LayerNorm` (parametric scale, no bias), `MultiHeadAttention` (SDPA path on CPU/MPS, FlashAttention path on Ampere+), `FeedForward` (SwiGLU-style gating), `Block`, `LastBlock`, and `BaseModel` (a thin `nn.Module` with `save` / `load` helpers).
- **`utils/transfer_weights.py` + `utils/layers_checks.py`** — port the official `jhu-clsp/ettin-encoder-150m` / `ettin-decoder-150m` weights into the custom architecture and run a layer-by-layer numerical check (used in development to make sure the RoPE, attention, MLP and LayerNorm are correct before any training).
- **`scripts/`** — small smoke-train scripts on tiny datasets (tiny-shakespeare, toy reconstruction) used to iterate on the corruption schedules without paying the cost of the full smoltalk pipeline.
- **`tests/test_moving_sigmoid.py`** — eight focused unit tests covering the analytical `c_{t,k}` formula, the calibrated mean, the right-bias, and the effect of `k`. These are the tests that validate the schedule derivations from the LaTeX notes.
- **`tests/test_corruption_diffusion.py`** — a visual sanity check that prints how a real smoltalk conversation looks at `t = 0.01, 0.5, 0.99` under every corruption mode.
- **`docs/mdlm.md`** — short mathematical note introducing the position-dependent weights `w_t = 1 + γ(p_l − ½)` and `α_{t,l} = (1 − t)^{w_l}`.
- **`noise/noise_schedule.py`** vs **`models/noise_schedule.py`** — both define the same `Noise` class hierarchy (`LogLinearNoise`, `CosineNoise`, `CosineSqrNoise`, `Linear`, `Geometric`); the one under `noise/` is the live one used by `main.py`, the one under `models/` is kept for backward compatibility with the older `mdlm.py` reference implementation.
- **Lightning logs** under `lightning_logs/` (TensorBoard) and raw `losses.csv` / `summary.csv` track every run; `plots.ipynb` is the notebook used to render the figures that go into the report.

---

## Citation

```
@inproceedings{sahoo2024simple,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham and Arriaga, Marianne and Gloeckle, Christophe and Vahdat, Aaron},
  booktitle={NeurIPS},
  year={2024}
}
```
