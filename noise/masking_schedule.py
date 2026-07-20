import torch
import math
from noise.noise_schedule import Noise


"""
╭ CONVENTIONS ───────────────────────────────────────────────────────────────────╮
│ ├─• B        ▶ batch size                                                      │
│ ├─• T        ▶ number of tokens in a batch i.e. length of a sequence/sentence  │
│ ├─• T_ans    ▶ number of tokens for the diffusion block (answer section)       │
│ ╰─• C        ▶ embedding dimension of each token                               │
╰────────────────────────────────────────────────────────────────────────────────╯
"""



class Masking:

    def __init__(self, T_ans: int, noise: Noise, corruption_type: str, 
                 position_loss_weighting: bool = True, 
                 gamma: float | None = None, k: float | None = None):
        self.corruption_types = {"independent": self.vanilla_masking, 
                                 "position":   self.position_dependent_masking, 
                                 "moving_sigmoid": self.moving_sigmoid_masking}
        self.T_ans = T_ans                                                      # Size of the diffusion tonken block
        self.noise = noise                                                      # noise class that provide \sigma and and \sigma'

        self.position_loss_weighting = position_loss_weighting
        self.gamma = gamma                                                      # strength of the positional method
        self.k     = k                                                          # strength of the sigmoid method

        self.change_corruption_type(corruption_type)


    def change_corruption_type(self, corruption_type):
        if corruption_type not in self.corruption_types.keys():
            raise ValueError("ERROR: corruption type selected not available")
        self.__corruption_type = corruption_type


    def __call__(self, *args, **kwds):
        return self.corruption_types[self.__corruption_type](*args, **kwds)


    def _get_loss_weight(self, dsigma, sigma):
        return (dsigma / torch.expm1(sigma))[:, None]                           # dsigma / {e^{sigma}-1}
    

    def vanilla_masking(self, t, out_sequence=True):
        sigma, dsigma = self.noise(t)                                           # B

        move_chance = 1 - torch.exp(-sigma[:, None])                            # B
        loss_weight = self._get_loss_weight(dsigma, sigma)                      # B

        if out_sequence:                                                        # Force the output to be B T
            move_chance = move_chance.expand(-1, self.T_ans)
            loss_weight = loss_weight.expand(-1, self.T_ans) 

        return move_chance, loss_weight
    

    def position_dependent_masking(self, t, out_sequence=True):
        if self.gamma is None: raise ValueError("masking requires gamma")
        e = 'out_sequence=False not supported in position_dependent_masking'
        if out_sequence==False: raise ValueError(e)

        device = t.device
        positions = torch.linspace(0, 1, self.T_ans, device=device)

        weights = 1 + self.gamma * (positions - 0.5)
        weights = weights.clamp_min(1e-3)

        sigma, dsigma = self.noise(t)

        alpha = torch.exp(-sigma[:, None] * weights[None, :])
        move_chance = 1 - alpha

        if self.position_loss_weighting:
            loss_weight = (
                weights[None, :]
                * dsigma[:, None]
                * alpha / (1 - alpha).clamp_min(1e-5)
            )
        else:
            loss_weight = self._get_loss_weight(dsigma, sigma)

        return move_chance, loss_weight
    

    def moving_sigmoid_masking(self, t, out_sequence=True):
        if self.k is None: raise ValueError("masking requires k")
        e = 'out_sequence=False not supported in moving_sigmoid_masking'
        if out_sequence==False: raise ValueError(e)

        device = t.device
        sigma, dsigma = self.noise(t)
        
        alpha = torch.exp(-sigma)
        vanilla_mean = 1 - alpha
        positions = torch.linspace(0, 1, self.T_ans, device=device)

        # Exact integral center
        e_k = math.exp(self.k)
        e_km = torch.exp(self.k * vanilla_mean)
        
        numerator = e_k - e_km
        denominator = e_km - 1  
        center = torch.log(numerator / denominator) / self.k

        if self.position_loss_weighting:
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