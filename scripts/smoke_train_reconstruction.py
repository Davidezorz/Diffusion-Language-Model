import time
import torch
import lightning as L
import datasets
from transformers import AutoTokenizer

from data_processing.data_manager import DataManager
from models.mdlm import MaskedDiffusionLM

torch.set_float32_matmul_precision("high")

from datasets import Dataset, DatasetDict

import models.masking_schedule as masking_schedule


def accuracy_by_position(pred, x0, token_mask):
    T = x0.shape[1]
    device = x0.device

    overall = (pred[token_mask] == x0[token_mask]).float().mean().item()

    print(f"Overall masked accuracy: {overall:.4f}")

    positions = torch.arange(T, device=device)
    norm_pos = positions.float() / max(T - 1, 1)

    bins = [
        (0.00, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.01),
    ]

    for low, high in bins:
        pos_mask = ((norm_pos >= low) & (norm_pos < high))[None, :]
        mask = token_mask & pos_mask

        if mask.sum() > 0:
            acc = (pred[mask] == x0[mask]).float().mean().item()
            print(f"{low:.2f}-{min(high, 1.00):.2f}: {acc:.4f}  n={mask.sum().item()}")
        else:
            print(f"{low:.2f}-{min(high, 1.00):.2f}: no masked tokens")


def get_schedule_mask(model, x0, attention_mask, mode, mask_prob=0.3, t_value=0.5):
    """
    mode:
        random
        independent
        position
        moving_sigmoid
        prefix_to_suffix
    """
    device = x0.device
    B, T = x0.shape

    pad_id = model.tokenizer.pad_token_id
    valid = attention_mask.bool() & (x0 != pad_id)

    if mode == "random":
        token_mask = (torch.rand_like(x0.float()) < mask_prob) & valid
        return token_mask

    if mode == "prefix_to_suffix":
        # left-to-right demasking style:
        # keep left prefix visible, mask right suffix
        cutoff = int((1.0 - mask_prob) * T)
        token_mask = torch.zeros_like(x0, dtype=torch.bool)
        token_mask[:, cutoff:] = True
        return token_mask & valid

    t = torch.full((B,), t_value, device=device)

    if mode == "independent":
        move_chance, _ = masking_schedule.vanilla_masking(
            t=t,
            T=T,
            device=device,
            noise=model.noise,
        )

    elif mode == "position":
        move_chance, _ = masking_schedule.position_dependent_masking(
            t=t,
            T=T,
            device=device,
            noise=model.noise,
            gamma=model.position_gamma,
            position_loss_weighting=model.position_loss_weighting,
        )

    elif mode == "moving_sigmoid":
        move_chance, _ = masking_schedule.moving_sigmoid_masking(
            t=t,
            T=T,
            device=device,
            noise=model.noise,
            k=model.sigmoid_k,
            calibrated=model.calibrated_sigmoid,
        )

    else:
        raise ValueError(f"Unknown eval mask mode: {mode}")

    token_mask = (torch.rand_like(x0.float()) < move_chance) & valid
    return token_mask


def evaluate_reconstruction(
    model,
    dm,
    device,
    texts,
    eval_mode,
    mask_prob=0.3,
    t_value=0.5,
    max_length=128,
    print_examples=False,
):
    model.eval()

    enc = dm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    x0 = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device).bool()

    token_mask = get_schedule_mask(
        model=model,
        x0=x0,
        attention_mask=attention_mask,
        mode=eval_mode,
        mask_prob=mask_prob,
        t_value=t_value,
    )

    xt = x0.clone()
    xt[token_mask] = model.mask_index

    with torch.no_grad():
        sigma = torch.ones(x0.shape[0], device=device) * t_value
        log_probs = model(xt, sigma=sigma)
        pred = log_probs.argmax(dim=-1)

    print(f"\n--- Eval mode: {eval_mode} ---")
    print(f"Masked tokens: {token_mask.sum().item()}")

    accuracy_by_position(pred, x0, token_mask)

    if print_examples:
        reconstructed = xt.clone()
        reconstructed[token_mask] = pred[token_mask]

        for i in range(min(2, len(texts))):
            print("\nCLEAN:")
            print(dm.tokenizer.decode(x0[i], skip_special_tokens=True))

            print("\nCORRUPTED:")
            print(dm.tokenizer.decode(xt[i], skip_special_tokens=False))

            print("\nRECONSTRUCTED:")
            print(dm.tokenizer.decode(reconstructed[i], skip_special_tokens=True))


def evaluate_all_masks(model, dm, device, test_texts):
    """
    Evaluates the same trained model under several corruption distributions.
    """
    for mode in [
        "random",
        "independent",
        "position",
        "moving_sigmoid",
        "prefix_to_suffix",
    ]:
        evaluate_reconstruction(
            model=model,
            dm=dm,
            device=device,
            texts=test_texts,
            eval_mode=mode,
            mask_prob=0.3,
            t_value=0.5,
            max_length=128,
            print_examples=False,
        )


def run_smoke(corruption_type, gamma=0.2):
    caching_directory = ".data/"

    B = 8
    T = 32

    dataset = datasets.load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    cache_dir=caching_directory
)
    dataset = dataset["train"].filter(lambda x: len(x["text"].strip()) > 50)
    dataset = dataset.select(range(100))

    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    )

    dm = DataManager(
        caching_directory,
        tokenizer=tokenizer,
        n_processes=4,
    )

    tokens = dm.tokenize(dataset)
    data = dm.group_texts(tokens, T)

    def collate_fn(batch):
        input_ids = torch.tensor(
            [item["input_ids"] for item in batch],
            dtype=torch.long
        )

        attention_mask = torch.tensor(
            [item["attention_mask"] for item in batch],
            dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    loader = torch.utils.data.DataLoader(
        data,
        batch_size=B,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = MaskedDiffusionLM(
        config=None,
        tokenizer=dm.tokenizer,
        B=B,
        T=T
    )

    model.corruption_type = corruption_type
    model.position_gamma = gamma

    if corruption_type == "position":
        model.position_loss_weighting = True
    else:
        model.position_loss_weighting = False

    print("\n==============================")
    print(f"Running: {corruption_type}")
    print(f"Gamma: {gamma}")
    print("==============================\n")

    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        accelerator = "cpu"
        precision = "32-true"
        print("Running on CPU")

    trainer = L.Trainer(
        max_epochs=100,
        limit_train_batches=50,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )

    start = time.time()

    trainer.fit(
        model,
        train_dataloaders=loader
    )

    print(f"\nTraining time: {time.time() - start:.2f}s")

    model.eval()
    device = model.device

    test_texts = [
    dataset[0]["text"],
    dataset[1]["text"],
    dataset[2]["text"],
]
    # print(tokenizer.eos_token_id)
    evaluate_all_masks(model, dm, device, test_texts)


if __name__ == "__main__":
    run_smoke("moving_sigmoid")
    run_smoke("independent")
    run_smoke("position", gamma=0.5)