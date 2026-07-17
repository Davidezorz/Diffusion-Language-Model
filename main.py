import lightning as L
import datasets
from omegaconf import OmegaConf

import models.noise_schedule as noise_schedule
from data_processing.data_manager import DataManagerQA
import utils.utils
from models.AR import AR
from models.BERT import BERT
from GPT_Lightning import GPT
from models.DiT import DiT
from diffusion_lightning import Diffusion
import data_processing.samplers as samplers
from utils.transfer_weights import *
import test
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from lightning.pytorch.callbacks import ModelCheckpoint


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                          Model Loading                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯


def load_ModernBERT():
    print("Downloading ModernBERT weights...")
    hf_model = AutoModelForMaskedLM.from_pretrained(
        "jhu-clsp/ettin-encoder-150m"
        )
    return hf_model

def load_ModernBERTDecoder():
    print("Downloading load_ModernBERTDecoder weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "jhu-clsp/ettin-decoder-150m"
        )
    return hf_model


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                        Dataset Loading                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯

def load_shakespeare(caching_directory,
                     print_first_lines=False):
    dataset = datasets.load_dataset("Trelis/tiny-shakespeare",
                                    cache_dir=caching_directory)
    dataset = dataset.rename_column("Text", "text")

    if print_first_lines == True:
        print('\nFirst 150 element')
        print(repr(dataset['train'][0]['text'][:150]))

    return dataset

def load_smoltalk():
    ds = datasets.load_dataset("HuggingFaceTB/smoltalk",
                               "all",
                               split="train[:10%]",                             # TODO: change it back to 'train'
                               cache_dir=".data"
                              )
    return ds

def load_smoltal_test():
    ds = datasets.load_dataset("HuggingFaceTB/smoltalk",
                               "all",
                               split="test[:10%]",                             # TODO: change it back to 'train'
                               cache_dir=".data"
                              )
    print(ds.cache_files)
    print(len(ds))
    return ds

# ╭───────────────────────────────────────────────────────────────────────────╮
# │                    Dataset diagnostics                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯

def check_ar_targets(dataset,
                     split_name):
    empty_chunks = 0
    total_valid_targets = 0
    min_valid_targets = None
    max_valid_targets = 0

    for example in dataset:
        targets = example["output_ids"]

        valid_targets = sum(
            token != -100
            for token in targets
        )

        total_valid_targets += valid_targets

        if valid_targets == 0:
            empty_chunks += 1

        if min_valid_targets is None:
            min_valid_targets = valid_targets
        else:
            min_valid_targets = min(
                min_valid_targets,
                valid_targets,
            )

        max_valid_targets = max(
            max_valid_targets,
            valid_targets,
        )

    print(f"\n[{split_name} TARGET CHECK]")
    print("Chunks:", len(dataset))
    print("Empty chunks:", empty_chunks)
    print("Minimum valid targets:", min_valid_targets)
    print("Maximum valid targets:", max_valid_targets)
    print("Total valid targets:", total_valid_targets)

def print_token_examples(process_tokens, tokenizer,
                         keys = ['input_ids']):
    print('\nexample of the tokenized dataset (encoded and decoded): \n')
    i = -7
    for key in keys:
        print(f"first ids: {process_tokens[i][key][:20]}")
        print(f"last  ids: {process_tokens[i][key][-20:]}")
        print(f"len:       {process_tokens[i][key].shape}")
        print("Detokenized token: ")
        print(repr(tokenizer.decode(process_tokens[key][i])))

        print()
        print(f"len process_tokens: {len(process_tokens['input_ids'])}\n\n")



def count_pad(process_tokens,
              tokenizer):
    n_pad, tot = 0, 0
    pad_str = tokenizer.pad_token
    pad = torch.tensor(tokenizer.encode(pad_str, add_special_tokens=False))

    for ids in process_tokens['input_ids']:
        are_pad = ids == pad
        n_pad += are_pad.sum()
        tot += len(ids)
    
    print("PAD analysis:")
    print(f"n_pad: {n_pad}")
    print(f"tot:   {tot}")
    print(f"p:     {n_pad/tot*100: .2f}% \n")


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                                   Main                                    │
# ╰───────────────────────────────────────────────────────────────────────────╯

def main():
    """
    Main training and evaluation pipeline.

    Steps
    -----
    1. Load configuration
    2. Load dataset
    3. Tokenize and preprocess
    4. Build dataloaders
    5. Initialize backbone (AR, BERT or DiT)
    6. Load pretrained weights or checkpoint
    7. Train (optional)
    8. Run qualitative generation examples
    """
    torch.set_float32_matmul_precision("high") # for CUDA
    print('Main online\n')

    # -------------------------------------------------------------------------
    # Configuration and runtime setup
    # -------------------------------------------------------------------------
    config = OmegaConf.load("config.yaml")    # get the config
    mode = config.mode

    print(
        f"""
    ======================================================
    Experiment
    ======================================================

    Mode               : {mode}
    Checkpoint         : {config.checkpoint}

    Context length     : {config.backbone.T}
    Embedding dim      : {config.backbone.C}
    Layers             : {config.backbone.N}
    Heads              : {config.backbone.H}

    Learning rate      : {config.training.learning_rate}
    Warmup             : {config.training.warmup_steps}
    Epochs             : {config.training.max_epochs}
    Batch size         : {config.backbone.B}
    Gradient accum.    : {config.training.accumulate_grad_batches}

    ======================================================
    """
    )

    BACKBONES = {"AR": AR,  "BERT": BERT,      "DiT": DiT}
    MODELS    = {"AR": GPT, "BERT": Diffusion, "DiT": Diffusion}

    load_fn   = {"AR":   load_ModernBERTDecoder, 
                 "BERT": load_ModernBERT, 
                 "DiT":  load_ModernBERT}

    n_processes = 4
    device      = utils.utils.getDevice()
    caching_directory = config.caching_directory

    tokenizer = AutoTokenizer.from_pretrained(
        "jhu-clsp/ettin-decoder-150m",
    )

    # -------------------------------------------------------------------------
    # Dataset loading and train/validation split
    # -------------------------------------------------------------------------

    dataset = load_smoltalk()

    print("\n[DATASET SIZE]")
    print("Original conversations:", len(dataset))

    dataset_split = dataset.train_test_split(
        test_size=config.training.validation_fraction,
        seed=config.training.seed,
        shuffle=True,
    )

    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    print("Training conversations:", len(train_dataset))
    print("Validation conversations:", len(val_dataset))

    # -------------------------------------------------------------------------
    # Tokenization and sequence grouping
    # -------------------------------------------------------------------------

    data_manager = DataManagerQA(
        caching_directory,
        tokenizer,
        config.mode,
        n_processes,
    )

    tokenized = {}

    for split_name, split in {
        "train": train_dataset,
        "validation": val_dataset,
    }.items():
        tokenized[split_name] = data_manager.tokenize(
            split,
            split_name=split_name,
        )

    train_tokens = tokenized["train"]
    val_tokens = tokenized["validation"]

    if mode == "AR":
        train_process_tokens = data_manager.group_texts_ar(
            train_tokens,
            config.backbone.T,
            split_name="train",
        )

        val_process_tokens = data_manager.group_texts_ar(
            val_tokens,
            config.backbone.T,
            split_name="validation",
        )

    else:
        train_process_tokens = data_manager.group_texts_dit(
            train_tokens,
            config.backbone.T_ctx,
            config.backbone.T_ans,
            split_name="train",
        )

        val_process_tokens = data_manager.group_texts_dit(
            val_tokens,
            config.backbone.T_ctx,
            config.backbone.T_ans,
            split_name="validation",
        )


    # -------------------------------------------------------------------------
    # Optional dataset subsets for debugging
    # -------------------------------------------------------------------------

    debug_subset_size = config.training.get(
        "debug_subset_size",
        None,
    )

    if debug_subset_size is not None:
        train_subset_size = min(
            debug_subset_size,
            len(train_process_tokens),
        )

        train_process_tokens = train_process_tokens.select(
            range(train_subset_size)
        )

    validation_subset_size = config.training.get(
        "validation_subset_size",
        None,
    )

    if validation_subset_size is not None:
        validation_subset_size = min(
            validation_subset_size,
            len(val_process_tokens),
        )

        val_process_tokens = val_process_tokens.select(
            range(validation_subset_size)
        )

    # -------------------------------------------------------------------------
    # Dataset integrity checks - for AR
    # -------------------------------------------------------------------------

    if mode == "AR":
        check_ar_targets(
            train_process_tokens,
            "TRAIN",
        )

        check_ar_targets(
            val_process_tokens,
            "VALIDATION",
        )

    train_process_tokens = train_process_tokens.with_format("torch")
    val_process_tokens = val_process_tokens.with_format("torch")

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    sampler_cls = samplers.RandomFaultTolerantSampler

    train_loader = data_manager.getTrainloader(
        train_process_tokens,
        config.backbone.B,
        sampler_cls,
    )

    val_loader = torch.utils.data.DataLoader(
        val_process_tokens,
        batch_size=config.backbone.B,
        num_workers=n_processes,
        pin_memory=True,
        shuffle=False,
        persistent_workers=n_processes > 0,
    )

    test.test_train_loader(train_loader)

    # -------------------------------------------------------------------------
    # Defining backbone and loading weights
    # -------------------------------------------------------------------------

    backbone = BACKBONES[mode](
        V=len(data_manager.tokenizer),
        C=config.backbone.C,
        H=config.backbone.H,
        N=config.backbone.N,
    )

    translate_weights_dict = {
        "AR": translate_weights_decoder,
        "BERT": translate_weights_encoder,
        "DiT": translate_weights_encoder,
    }

    # test.test_model(backbone, tokenizer, mode)
    
    # -------------------------------------------------------------------------
        
    print(f"\n{mode} parameters: {utils.utils.numberOfparameters(backbone)}")

    if config.checkpoint is None:
        hf_model = load_fn[mode]()

        trasfer_weights(
            backbone,
            hf_model,
            translate_weights_dict[mode],
            run_validation=True,
            show_layers=False,
        )

        model = MODELS[mode](
            backbone,
            data_manager.tokenizer,
            T=config.backbone.T,
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
        ).to(device)

    else:
        model = MODELS[mode].load_from_checkpoint(
            config.checkpoint,
            backbone=backbone,
            tokenizer=data_manager.tokenizer,
            T=config.backbone.T,
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
        ).to(device)

    # -------------------------------------------------------------------------
    # Diffusion-specific configuration
    # -------------------------------------------------------------------------

    if mode in ["BERT", "DiT"]:
        noise_name = config.diffusion.get("noise_schedule", "loglinear")
        if noise_name == "loglinear":
            model.noise = noise_schedule.LogLinearNoise()
        else:
            raise ValueError(f"Unknown noise schedule: {noise_name}")
        model.corruption_type = config.diffusion.get("corruption_type", "independent")
        model.position_gamma = config.diffusion.get("position_gamma", 2.0)
        model.position_loss_weighting = config.diffusion.get("position_loss_weighting", False)
        model.sigmoid_k = config.diffusion.get("sigmoid_k", 10.0)
        model.calibrated_sigmoid = config.diffusion.get("calibrated_sigmoid", False)

    # -------------------------------------------------------------------------
    # Checkpoint configuration
    # -------------------------------------------------------------------------

    if config.checkpoint is not None:
        print(f"Loaded checkpoint from: {config.checkpoint}")

    print("\n\n")
    if hasattr(model, "noise"):
        print(
            f"Model noise device: "
            f"{model.noise.sigma_max.device}"
        )

    if mode in ["BERT", "DiT"]:
        checkpoint_prefix = (
            f"{mode}-{config.diffusion.corruption_type}"
        )
    else:
        checkpoint_prefix = mode

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=(
            checkpoint_prefix
            + "-{epoch:02d}-{val_loss:.4f}"
        ),
        monitor="val_loss",
        save_top_k=2,
        mode="min",
        save_last=True,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    # allow training only if set in the config
    inference_only = config.get("inference_only", False)

    if not inference_only:
        print("\ntraining:")

        trainer = L.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator=device,
            devices=1,
            enable_progress_bar=True,
            limit_train_batches=config.training.limit_train_batches,
            log_every_n_steps=config.training.log_every_n_steps,
            accumulate_grad_batches=config.training.accumulate_grad_batches,
            callbacks=[checkpoint_callback],
            limit_val_batches=config.training.get(
                "limit_val_batches",
                1.0,
            ),
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

        model.train()

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    # -------------------------------------------------------------------------
    # Qualitative generation test
    # -------------------------------------------------------------------------

    model.to(device)
    model.eval()
    print('\nModel testing:')

    test_prompts = [
        "Can a dog fly?",
        "What is the capital of France?",
        "Explain gravity to a ten-year-old.",
        "Write a Python function that computes the Fibonacci sequence.",
        "Why is the sky blue?",
        "What should someone consider before changing jobs?",
        ]

    for question in test_prompts:
        text1 = f"User: {question}"
        text2 = "Assistant: "

        input1 = tokenizer(
            text1,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        input2 = tokenizer(
            text2,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        bos = torch.tensor([[tokenizer.bos_token_id]])
        eos = torch.tensor([[tokenizer.eos_token_id]])

        inputs = torch.cat([bos, input1, eos, input2], dim=-1).to(device)

        if mode == "AR":
            generated = model.generate(
                inputs,
                n_tokens=256,
                temperature=0.6,
            )

        else:  # DiT 
            generated = model.generate(
                inputs,
                n_tokens=256,
                num_steps=100,
                temperature=0.8,
            )

        print("\nQUESTION:", question)
        print("\n")
        print(tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ))


if __name__ == '__main__':
    main()

