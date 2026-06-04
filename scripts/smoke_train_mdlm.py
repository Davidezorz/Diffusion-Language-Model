import time
import torch
import lightning as L
import datasets

from data_processing.data_manager import DataManager
from models.mdlm import MaskedDiffusionLM


def run_smoke(corruption_type, gamma=0.2):
    caching_directory = ".data/"

    B = 8
    T = 128

    dataset = datasets.load_dataset(
        "Trelis/tiny-shakespeare",
        cache_dir=caching_directory
    )

    dataset = dataset.rename_column("Text", "text")

    dm = DataManager(
        caching_directory,
        n_processes=4
    )

    tokens = dm.tokenize(dataset["train"])
    data = dm.group_texts(tokens, T)
    data = data.with_format("torch")

    loader = dm.getTrainloader(data, B)

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
        max_epochs=3,
        limit_train_batches=200,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    start = time.time()

    trainer.fit(
        model,
        train_dataloaders=loader
    )

    print(f"\nTraining time: {time.time() - start:.2f}s")

    model.eval()

    with torch.no_grad():
        samples = model._sample(
            B=2,
            num_steps=50
        )

    for i, sample in enumerate(samples):
        text = dm.tokenizer.decode(
            sample,
            skip_special_tokens=True
        )

        print(f"\n===== SAMPLE {corruption_type} {i} =====")
        print(text[:500])


if __name__ == "__main__":
    run_smoke("independent")
    run_smoke("position", gamma=0.2)