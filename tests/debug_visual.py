import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)



from types import SimpleNamespace

import torch
from data_processing.data_manager import DataManager
from models.Diffusion import Diffusion

import models

class DummyBackbone:
    def __init__(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

models.dit = SimpleNamespace(DIT=DummyBackbone)


def create_model():
    config = SimpleNamespace()

    data_manager = DataManager(
        caching_directory=".data/",
        n_processes=1
    )

    tokenizer = data_manager.tokenizer

    model = Diffusion(
        config=config,
        tokenizer=tokenizer,
        B=16,
        T=100
    )

    model.corruption_type = "span"

    return model


model = create_model()

x = torch.arange(60).reshape(3, 20)

p = torch.tensor([
    [0.2],
    [0.4],
    [0.6]
])

xt = model.q_xt(x, p)

print("\nOriginal")
print(x)

print("\nCorrupted")
print(xt)