# test the shape
def test_shape(model, device="cuda"):
    B = 4
    T = 16
    t = torch.tensor([0.1, 0.3, 0.5, 0.9], device=device)

    move_chance, loss_weight = model.moving_sigmoid_noise(t, T, device)

    print("move_chance shape:", move_chance.shape)
    print("loss_weight shape:", loss_weight.shape)

    assert move_chance.shape == (B, T)
    assert loss_weight.shape in [(B, 1), (B, T)]

