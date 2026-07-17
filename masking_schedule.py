import torch
import math


class Masking:

    def __init__(self, T, noise, gamma=None, k=None):
        self.T     = T                                                      # Size of the diffusion tonken block
        self.noise = noise                                                  # noise class that provide \sigma and and \sigma'
        
        self.gamma = gamma                                                  # strength of the positional method
        self.k     = k                                                      # strength of the sigmoid method


    def _get_loss_weight(self, dsigma, sigma):
        return (dsigma / torch.expm1(sigma))[:, None]                       # dsigma / {e^{sigma}-1}
    

    def vanilla_masking(self, t, position_loss_weighting=False):
        sigma, dsigma = self.noise(t)

        move_chance = 1 - torch.exp(-sigma[:, None])
        loss_weight = self._get_loss_weight(dsigma, sigma)

        return move_chance, loss_weight
    

    def position_dependent_masking(
        self,
        t,
        position_loss_weighting=False,
    ):
        if self.gamma is None: raise ValueError("gamma not initialized")

        device = t.device
        positions = torch.linspace(0, 1, self.T, device=device)

        weights = 1 + self.gamma * (positions - 0.5)
        weights = weights.clamp_min(1e-3)

        sigma, dsigma = self.noise(t)

        alpha = torch.exp(-sigma[:, None] * weights[None, :])
        move_chance = 1 - alpha

        if position_loss_weighting:
            loss_weight = (
                weights[None, :]
                * dsigma[:, None]
                * alpha / (1 - alpha).clamp_min(1e-5)
            )
        else:
            loss_weight = self._get_loss_weight(dsigma, sigma)

        return move_chance, loss_weight
    

    def moving_sigmoid_masking(self, t, position_loss_weighting=False):
        if self.k is None: raise ValueError("k not initialized")

        device = t.device
        sigma, dsigma = self.noise(t)
        
        alpha = torch.exp(-sigma)
        vanilla_mean = 1 - alpha
        positions = torch.linspace(0, 1, self.T, device=device)

        # Exact integral center
        e_k = math.exp(self.k)
        e_km = torch.exp(self.k * vanilla_mean)
        
        numerator = e_k - e_km
        denominator = e_km - 1  
        center = torch.log(numerator / denominator) / self.k

        if position_loss_weighting:
            # vanilla_mean_prime = d(1 - alpha)/dt = dsigma * alpha
            vanilla_mean_prime = dsigma * alpha
            c_t_prime = -vanilla_mean_prime * e_km * (e_k - 1) / (numerator * denominator)
            
            logits = self.k * (positions[None, :] - center[:, None])
            move_chance = torch.sigmoid(logits)
            loss_weight = -self.k * (1 - move_chance) * c_t_prime[:, None]
            
        else:            
            logits = self.k * (positions[None, :] - center[:, None])
            move_chance = torch.sigmoid(logits)
            loss_weight = self._get_loss_weight(dsigma, sigma)

        return move_chance, loss_weight
    




