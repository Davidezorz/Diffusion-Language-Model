import torch

from types import SimpleNamespace
from transformers import AutoTokenizer
from models.mdlm import MaskedDiffusionLM
from data_processing.data_manager import DataManager
import models.noise_schedule


class DummyDIT:

    def __init__(self,*args,**kwargs):
        pass

    def load(self,*args,**kwargs):
        pass


models.dit = SimpleNamespace(
    DiT=DummyDIT
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

    model = MaskedDiffusionLM(
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

def test_position_noise_shapes():

    model = create_model()
    model.corruption_type = "position"
    model.position_gamma = 2.0

    B = 8
    T = 100

    t = torch.tensor([0.5] * B)

    move_chance, loss_weight = model.position_dependent_noise(
        t=t,
        T=T,
        device=t.device
    )

    assert move_chance.shape == (B, T)
    assert loss_weight.shape in [(B, T), (B, 1)]

    assert torch.isfinite(move_chance).all()
    assert torch.isfinite(loss_weight).all()

    assert (move_chance >= 0).all()
    assert (move_chance <= 1).all()

def test_position_noise_masks_more_on_right():

    model = create_model()
    model.corruption_type = "position"
    model.position_gamma = 2.0

    B = 512
    T = 100

    x = torch.randint(
        0,
        100,
        (B, T)
    )

    t = torch.tensor([0.5] * B)

    move_chance, _ = model.position_dependent_noise(
        t=t,
        T=T,
        device=x.device
    )

    xt = model.q_xt(x, move_chance)

    mask = xt == model.mask_index

    left_mask_rate = mask[:, :T//3].float().mean().item()
    right_mask_rate = mask[:, -T//3:].float().mean().item()

    print("left mask rate:", left_mask_rate)
    print("right mask rate:", right_mask_rate)

    assert right_mask_rate > left_mask_rate

def test_position_noise_gamma_zero_is_uniform():

    model = create_model()
    model.corruption_type = "position"
    model.position_gamma = 0.0

    B = 512
    T = 100

    x = torch.randint(
        0,
        100,
        (B, T)
    )

    t = torch.tensor([0.5] * B)

    move_chance, _ = model.position_dependent_noise(
        t=t,
        T=T,
        device=x.device
    )

    xt = model.q_xt(x, move_chance)

    mask = xt == model.mask_index

    left_mask_rate = mask[:, :T//3].float().mean().item()
    right_mask_rate = mask[:, -T//3:].float().mean().item()

    print("gamma=0 left:", left_mask_rate)
    print("gamma=0 right:", right_mask_rate)

    assert abs(right_mask_rate - left_mask_rate) < 0.08


def test_position_noise_average_mask_rate_balanced():

    model = create_model()
    model.corruption_type = "position"
    model.position_gamma = 0.2

    B = 512
    T = 100

    t = torch.tensor([0.5] * B)

    move_chance, _ = model.position_dependent_noise(
        t=t,
        T=T,
        device=t.device
    )

    avg_mask_rate = move_chance.mean().item()

    print("position avg mask rate:", avg_mask_rate)

    assert abs(avg_mask_rate - 0.5) < 0.03