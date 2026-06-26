
import torch
import models.noise_schedule as noise_schedule
from models.masking_schedule import moving_sigmoid_masking

# test the shape
def test_shape():
    device = "cpu"

    B = 4
    T = 16
    t = torch.tensor([0.1, 0.3, 0.5, 0.9], device=device)

    noise = noise_schedule.LogLinearNoise()

    move_chance, loss_weight = moving_sigmoid_masking(
        t=t,
        T=T,
        device=device,
        noise=noise,
        k=10.0,
    )

    print("move_chance shape:", move_chance.shape)
    print("loss_weight shape:", loss_weight.shape)

    assert move_chance.shape == (B, T)
    assert loss_weight.shape == (B, 1)

# test that probabilities are between 0 and 1
def test_probabilities_are_valid():
    device = "cpu"
    B = 8
    T = 128

    t = torch.rand(B, device=device)
    noise = noise_schedule.LogLinearNoise()
    move_chance, _ = moving_sigmoid_masking(
        t=t,
        T=T,
        device=device,
        noise=noise,
        k=10.0,
    )

    assert torch.all(move_chance >= 0)
    assert torch.all(move_chance <= 1)

# test that the probabilities of rightmost tokens being masked is higher
def test_right_positions_more_masked_than_left():
    device = "cpu"
    B = 8
    T = 128

    t = torch.rand(B, device=device)
    noise = noise_schedule.LogLinearNoise()

    move_chance, _ = moving_sigmoid_masking(
        t=t,
        T=T,
        device=device,
        noise=noise,
        k=10.0,
    )

    left = move_chance[:, 0]
    right = move_chance[:, -1]

    assert torch.all(right > left)

# test whether masking increases as t increases
def test_masking_increases_with_t():
    device = "cpu"
    T = 128
    noise = noise_schedule.LogLinearNoise()

    t_low = torch.tensor([0.2], device=device)
    t_high = torch.tensor([0.8], device=device)

    p_low, _ = moving_sigmoid_masking(
        t=t_low,
        T=T,
        device=device,
        noise=noise,
        k=10.0,
    )

    p_high, _ = moving_sigmoid_masking(
        t=t_high,
        T=T,
        device=device,
        noise=noise,
        k=10.0,
    )

    assert torch.all(p_high > p_low)