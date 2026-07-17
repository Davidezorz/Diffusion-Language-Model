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

def moving_sigmoid_probability(
    t,
    T,
    device,
    k=10.0,
    center=None,
):
    positions = torch.linspace(0, 1, T, device=device)

    if center is None:
        center = 1.0 - t

    logits = k * (positions[None, :] - center[:, None])
    return torch.sigmoid(logits)

def solve_sigmoid_center(
    target_mean,
    T,
    device,
    k=10.0,
    steps=30,
):
    """
    Find c such that:
        mean_l sigmoid(k(l - c)) = target_mean

    target_mean: shape (B,)
    returns center: shape (B,)
    """

    low = torch.full_like(target_mean, -1.0)
    high = torch.full_like(target_mean, 2.0)

    for _ in range(steps):
        mid = (low + high) / 2

        probs = moving_sigmoid_probability(
            t=target_mean,      # unused because center is provided
            T=T,
            device=device,
            k=k,
            center=mid,
        )

        mean = probs.mean(dim=1)

        # f mean is too high, the front is too far left; increase c to move the front right and reduce masking.
        low = torch.where(mean > target_mean, mid, low)
        high = torch.where(mean <= target_mean, mid, high)

    return (low + high) / 2


def moving_sigmoid_masking(
    t,
    T,
    device,
    noise,
    k=10.0,
    calibrated=False,
):
    sigma, dsigma = noise(t)

    vanilla_mean = 1 - torch.exp(-sigma)

    if calibrated:
        center = solve_sigmoid_center(
            target_mean=vanilla_mean,
            T=T,
            device=device,
            k=k,
        )
    else:
        center = 1.0 - t

    move_chance = moving_sigmoid_probability(
        t=t,
        T=T,
        device=device,
        k=k,
        center=center,
    )

    loss_weight = (dsigma / torch.expm1(sigma))[:, None]

    return move_chance, loss_weight