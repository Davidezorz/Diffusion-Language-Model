import torch


def vanilla_masking(t, T, device, noise):
    sigma, dsigma = noise(t)

    move_chance = 1 - torch.exp(-sigma[:, None])
    loss_weight = (dsigma / torch.expm1(sigma))[:, None]

    return move_chance, loss_weight


def position_dependent_masking(
    t,
    T,
    device,
    noise,
    gamma=0.2,
    position_loss_weighting=False,
):
    positions = torch.linspace(0, 1, T, device=device)

    weights = 1 + gamma * (positions - 0.5)
    weights = weights.clamp_min(1e-3)

    sigma, dsigma = noise(t)

    alpha = torch.exp(-sigma[:, None] * weights[None, :])
    move_chance = 1 - alpha

    if position_loss_weighting:
        loss_weight = (
            weights[None, :]
            * dsigma[:, None]
            * alpha
            / (1 - alpha).clamp_min(1e-5)
        )
    else:
        loss_weight = (dsigma / torch.expm1(sigma))[:, None]

    return move_chance, loss_weight


def moving_sigmoid_masking(
    t,
    T,
    device,
    noise,
    k=10.0,
):
    positions = torch.linspace(0, 1, T, device=device)

    logits = k * (positions[None, :] + t[:, None] - 1.0)
    move_chance = torch.sigmoid(logits)

    sigma, dsigma = noise(t)
    loss_weight = (dsigma / torch.expm1(sigma))[:, None]

    return move_chance, loss_weight