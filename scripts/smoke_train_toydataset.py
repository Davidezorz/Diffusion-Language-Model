import time
import torch
import lightning as L
import datasets
from transformers import AutoTokenizer

from data_processing.data_manager import DataManager
from models.mdlm import MaskedDiffusionLM

torch.set_float32_matmul_precision("high")

from datasets import Dataset, DatasetDict

def make_toy_dataset(n=512):
    texts = []
    patterns = [
        "the cat sat on the mat .",
        "the dog is running in the park .",
        "he is going to school by bus .",
        "university sucks big time ."
    ]

    for i in range(n):
        texts.append(patterns[i % len(patterns)])

    return DatasetDict({
        "train": Dataset.from_dict({"text": texts})
    })

def reconstruction_test(model, dm, device):
    model.eval()

    texts = [
        "the cat sat on the mat.",
        "the dog is running in the park .",
        "he is going to school by bus .",
        "university sucks big time ."
    ]

    enc = dm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16,
        add_special_tokens=False,
    )

    x0 = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device).bool()

    pad_id = dm.tokenizer.pad_token_id
    valid = attention_mask & (x0 != pad_id)

    token_mask = torch.zeros_like(x0, dtype=torch.bool)

    for b in range(x0.shape[0]):
        valid_positions = torch.where(valid[b])[0]
        pos = valid_positions[len(valid_positions) // 2]
        token_mask[b, pos] = True

    xt = x0.clone()
    xt[token_mask] = model.mask_index

    with torch.no_grad():
        sigma = torch.ones(x0.shape[0], device=device) * 0.5
        log_probs = model(xt, sigma=sigma)
        pred = log_probs.argmax(dim=-1)

    reconstructed = xt.clone()
    reconstructed[token_mask] = pred[token_mask]

    print("\nCLEAN:")
    print(dm.tokenizer.decode(x0[0], skip_special_tokens=True))

    print("CORRUPTED:")
    print(dm.tokenizer.decode(xt[0], skip_special_tokens=False))

    print("RECONSTRUCTED:")
    print(dm.tokenizer.decode(reconstructed[0], skip_special_tokens=True))

    print("target ids:", x0[token_mask].tolist())
    print("pred ids:  ", pred[token_mask].tolist())
    print("target:", dm.tokenizer.decode(x0[token_mask]))
    print("pred:  ", dm.tokenizer.decode(pred[token_mask]))


def reconstruction_test_random(model, dm, device, texts, mask_prob=0.3, max_length=128):
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

    pad_id = dm.tokenizer.pad_token_id
    valid = attention_mask & (x0 != pad_id)

    token_mask = (torch.rand(x0.shape, device=device) < mask_prob) & valid

    xt = x0.clone()
    xt[token_mask] = model.mask_index

    with torch.no_grad():
        sigma = torch.ones(x0.shape[0], device=device) * 0.5
        log_probs = model(xt, sigma=sigma)
        pred = log_probs.argmax(dim=-1)

    reconstructed = xt.clone()
    reconstructed[token_mask] = pred[token_mask]

    acc = (pred[token_mask] == x0[token_mask]).float().mean().item()

    print(f"\nMasked reconstruction accuracy: {acc:.4f}")

    T = x0.shape[1]
    positions = torch.arange(T, device=device)
    norm_pos = positions.float() / (T - 1)

    bins = [
        (0.00, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 1.00),
    ]

    for low, high in bins:

        pos_mask = (
            (norm_pos >= low)
            & (norm_pos < high)
        )[None, :]

        mask = token_mask & pos_mask

        if mask.sum() > 10:
            acc = (
                pred[mask] == x0[mask]
            ).float().mean().item()

            print(
                f"{low:.2f}-{high:.2f}: {acc:.4f}"
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
    reconstruction_test_random(model, dm, device, test_texts)


if __name__ == "__main__":
    run_smoke("moving_sigmoid")
    run_smoke("independent")
    run_smoke("position", gamma=0.5)