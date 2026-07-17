
## Person 1 (Paper implementation)
1. Convert AR Transformer to bidirectional attention
2. Implement random t sampling
3. Implement α_t schedule
4. Implement absorbing [MASK] corruption
5. Implement SUBS:
   - no [MASK] prediction
   - copy unmasked tokens
6. Implement MDLM loss
7. Implement basic reverse diffusion sampler

## Person 2 (Modified Paper implementation)
1. Implement vanilla independent token masking first
2. Implement span masking with same expected mask rate
3. Make corruption mode configurable:
   - independent
   - span
4. Verify mask percentage is correct
5. Compare vanilla vs span MDLM
6. Produce plots/tables for the modification


## Person 3
1. Keep AR baseline trainable
2. Add DistilBERT/BERT-tiny baseline
3. Define evaluation dataset split
4. Implement shared metrics:
   - masked token accuracy
   - reconstruction CE
   - sample quality examples
   - runtime
5. Create result tables and plots
6. Maintain README experiment commands

## Models

| Model | Role | Training Objective | Attention Type | Key Details |
|------|------|---------------------|----------------|-------------|
| **AR Transformer** | Baseline | Next-token Cross Entropy | Causal | Existing GPT-style autoregressive model; predicts token `x_t` from `x_<t`; left-to-right generation |
| **Vanilla MDLM** | Main paper reproduction | Diffusion-weighted Masked Cross Entropy | Bidirectional | Implements masked diffusion exactly as paper: absorbing `[MASK]`, random diffusion timestep `t`, α(t) masking schedule, SUBS parameterization, reverse diffusion sampling |
| **Span-MDLM** | Modified research model | Same MDLM objective | Bidirectional | Same MDLM architecture, but corruption process masks contiguous spans instead of independent tokens |
| **DistilBERT / BERT-Tiny** | Pretrained benchmark | MLM / Fine-tuning | Bidirectional | External pretrained comparison model (`distilbert-base-uncased` or `prajjwal1/bert-tiny`) |

---

Span-MDLM: 
Instead of having random masked tokes:
```python
I [MASK] pizza very [MASK]
```

Mask contigous parts:
```python
I [MASK] [MASK] very much
```

to force the model to reconstruct:
- local semantic structure
- syntax continuity

# Benchmarks

| Benchmark | Purpose | Applicable Models | Main Metric |
|----------|---------|------------------|-------------|
| **Masked Reconstruction Cross Entropy** | Measures token reconstruction quality from corruption | MDLM, Span-MDLM, BERT | CE Loss ↓ |
| **Masked Token Accuracy** | % correctly reconstructed masked tokens | MDLM, Span-MDLM, BERT | Accuracy ↑ |
| **Autoregressive Validation CE / PPL** | Measures next-token prediction quality | AR | CE / Perplexity ↓ |
| **Denoising Task** | Corrupted sentence → reconstruct clean sentence | MDLM, Span-MDLM, BERT | Reconstruction Accuracy / BLEU |
| **Cloze QA** | Fill masked answer in question/context | MDLM, Span-MDLM, BERT, optional AR | Accuracy / Exact Match |
| **Generation Samples** | Qualitative language generation | AR, MDLM, Span-MDLM | Human qualitative |
| **Inference Speed** | Runtime efficiency | All | Tokens/sec or sec/sample |
| **Training Stability** | Optimization quality | All trainable models | Loss curves / convergence |

---

# Unified Seamless Workflow

## Shared Batch Format

```python
batch = {
    "input_ids": input_ids,              # model input
    "target_ids": target_ids,            # clean target
    "attention_mask": attention_mask,    # padding visibility
    "corruption_mask": corruption_mask,  # True where corruption happened
    "t": t                               # diffusion timestep (MDLM only)
}
```

All models should expose:
```python
logits = model(input_ids, attention_mask=None, t=None)
```