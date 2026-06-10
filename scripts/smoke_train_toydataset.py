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
        "the dog ran in the park .",
        "alice likes red apples .",
        "bob likes blue cars ."
    ]

    for i in range(n):
        texts.append(patterns[i % len(patterns)])

    return DatasetDict({
        "train": Dataset.from_dict({"text": texts})
    })

def reconstruction_test(model, dm, device):
    model.eval()

    texts = [
        "the cat sat on the mat .",
        "the dog ran in the park .",
        "alice likes red apples .",
        "bob likes blue cars .",
    ]

    enc = dm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"].to(device)
    clean = input_ids.clone()

    mask_id = dm.tokenizer.mask_token_id

    # mask 50% of non-pad tokens
    pad_id = dm.tokenizer.pad_token_id
    valid = input_ids != pad_id
    rand = torch.rand(input_ids.shape, device=device)
    mask = (rand < 0.5) & valid

    corrupted = input_ids.clone()
    corrupted[mask] = mask_id

    with torch.no_grad():
        sigma = torch.ones(input_ids.shape[0], device=device) * 0.5
        logits = model(corrupted, sigma=sigma)
        pred = logits.argmax(dim=-1)

    reconstructed = corrupted.clone()
    reconstructed[mask] = pred[mask]

    for i in range(len(texts)):
        print("\nCLEAN:")
        print(dm.tokenizer.decode(clean[i], skip_special_tokens=True))

        print("CORRUPTED:")
        print(dm.tokenizer.decode(corrupted[i], skip_special_tokens=False))

        print("RECONSTRUCTED:")
        print(dm.tokenizer.decode(reconstructed[i], skip_special_tokens=True))


def run_smoke(corruption_type, gamma=0.2):
    caching_directory = ".data/"

    B = 8
    T = 32

    dataset = make_toy_dataset(n=512)

    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    )

    dm = DataManager(
        caching_directory,
        tokenizer=tokenizer,
        n_processes=4,
    )

    tokens = dm.tokenize(dataset["train"])
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
        max_epochs=50,
        limit_train_batches=100,
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
    reconstruction_test(model, dm, device)


if __name__ == "__main__":
    run_smoke("independent")
    run_smoke("position", gamma=1.0)