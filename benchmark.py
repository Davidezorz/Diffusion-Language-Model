import torch
from torchmetrics import MeanMetric
# Benchmarking methods that compare performance between the Discrete Diffusion Model and the AutoRegressive Model
class NNL(MeanMetric):
    pass
class Perplexity(NNL):
    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value/self.weight)

def perplexity(generated_text, begin_token_index, end_token_index):
    token_mask = torch.ones_like(generated_text, dtype=torch.bool)
    token_mask[:begin_token_index] = 0
    token_mask[end_token_index+1:] = 0
    masked_text = [-100 if generated_text[i] == "[PAD]" else generated_text[i] for i in range(len(generated_text))]


def turn_length():
    pass

def marker_correctness():
    pass

def self_bleu():
    pass

def conversational_diversity():
    pass

def bert_score():
    pass

def frechet_bert_distance():
    pass

def llm_as_judge():
    pass

# Benchmarking methods that track the performance through steps of unmasking process for the Discrete Diffusion Model, for different unmasking policies.
def perplexity_through_steps():
    pass

def entropy_through_steps():
    pass

def inter_step_bert_score():
    pass

def part_of_speech_emergence():
    pass

def prompt_adherence_through_steps():
    pass