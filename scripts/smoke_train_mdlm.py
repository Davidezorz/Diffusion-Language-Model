# scripts/smoke_train_mdlm.py

import lightning as L
import datasets

from data_processing.data_manager import DataManager
from models.mdlm import MaskedDiffusionLM


def run_smoke(corruption_type):
    caching_directory = ".data/"
    B, T = 4, 64

    dataset = datasets.load_dataset(
        "Trelis/tiny-shakespeare",
        cache_dir=caching_directory
    )
    dataset = dataset.rename_column("Text", "text")

    dm = DataManager(caching_directory, n_processes=1)
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
    model.position_gamma = 0.5

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=0,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False
    )

    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    run_smoke("independent")
    run_smoke("position")