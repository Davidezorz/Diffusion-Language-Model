import torch

from types import SimpleNamespace
from transformers import AutoTokenizer
from models.Diffusion import Diffusion
from data_processing.data_manager import DataManager
import models.noise_schedule


class DummyDIT:

    def __init__(self,*args,**kwargs):
        pass

    def load(self,*args,**kwargs):
        pass


models.dit = SimpleNamespace(
    DIT=DummyDIT
)

class DummyBackbone:
    def load(self, *args, **kwargs):
        pass

import models


def create_model():

    # fake config
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

    model.backbone = DummyBackbone()

    model.corruption_type = "span"

    return model

def test_shape():

    model = create_model()

    x = torch.randint(
        0,
        100,
        (8,100)
    )

    p = torch.tensor([[0.3]]*8)

    xt = model.q_xt(x,p)

    assert xt.shape == x.shape

def test_mask_rate():

    model = create_model()

    B=500
    T=100

    x=torch.randint(
        0,
        100,
        (B,T)
    )

    p=torch.tensor([[0.3]]*B)

    xt=model.q_xt(x,p)

    observed=(xt==model.mask_index).float().mean()

    print(
        "Observed:",
        observed.item()
    )

    assert abs(
        observed.item()-0.3
    )<0.05

def test_span_clustering():

    model = create_model()

    B = 100
    T = 100

    x = torch.randint(
        0,
        100,
        (B,T)
    )

    p = torch.tensor([[0.3]]*B)

    xt = model.q_xt(x,p)

    mask = (xt == model.mask_index)

    adjacent_pairs = 0

    for b in range(B):

        adjacent_pairs += (
            mask[b,:-1] &
            mask[b,1:]
        ).sum()

    print(
        "Adjacent masked pairs:",
        adjacent_pairs.item()
    )

    assert adjacent_pairs > 0


def count_mask_segments(mask_row):
    """
    Conta quanti blocchi contigui di True ci sono in una singola sequenza.
    Esempio:
    [False, True, True, False, True] -> 2 segmenti
    """

    if mask_row.sum() == 0:
        return 0

    starts = mask_row & ~torch.cat([
        torch.tensor([False], device=mask_row.device),
        mask_row[:-1]
    ])

    return starts.sum().item()

def test_span_creates_contiguous_blocks():

    model = create_model()
    model.corruption_type = "span"

    B = 200
    T = 100

    x = torch.randint(
        0,
        100,
        (B, T)
    )

    p = torch.tensor([[0.3]] * B)

    xt = model.q_xt(x, p)

    mask = xt == model.mask_index

    total_adjacent_pairs = 0

    for b in range(B):
        total_adjacent_pairs += (
            mask[b, :-1] & mask[b, 1:]
        ).sum().item()

    assert total_adjacent_pairs > 0

    def test_span_has_fewer_segments_than_independent():

        model = create_model()

        B = 200
        T = 100

        x = torch.randint(
            0,
            100,
            (B, T)
        )

        p = torch.tensor([[0.3]] * B)

        model.corruption_type = "span"
        xt_span = model.q_xt(x, p)

        model.corruption_type = "independent"
        xt_ind = model.q_xt(x, p)

        mask_span = xt_span == model.mask_index
        mask_ind = xt_ind == model.mask_index

        span_segments = []
        ind_segments = []

        for b in range(B):
            span_segments.append(
                count_mask_segments(mask_span[b])
            )

            ind_segments.append(
                count_mask_segments(mask_ind[b])
            )

        avg_span_segments = sum(span_segments) / len(span_segments)
        avg_ind_segments = sum(ind_segments) / len(ind_segments)

        print("Avg span segments:", avg_span_segments)
        print("Avg independent segments:", avg_ind_segments)

        assert avg_span_segments < avg_ind_segments