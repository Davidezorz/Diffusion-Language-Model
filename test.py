
import numpy as np
import torch
import matplotlib.pyplot as plt


def test_smoltalk(ds, tokenizer):
    """
    for conversation in ds["messages"]:
        for msg in conversation:
            print(f'----{msg["role"]}:----\n{msg["content"]}')
        
    return
    """
    """
    if "capital of Italy" in msg["content"]:
        print(f'----{msg["role"]}:----\n{msg["content"]}')
        return
    """ 
    

    BOS = tokenizer.bos_token
    EOS = tokenizer.eos_token

    lengths = []
    for conversation in ds["messages"]:
        # Convert the conversation into plain text
        text = BOS + "\n".join(
            f"User: {msg["role"]}{EOS}Assistant: {msg["content"]}{EOS}"
            for msg in conversation
        )

        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        lengths.append(n_tokens)

    lengths = np.asarray(lengths)

    print()
    print(f"Min:             {lengths.min()}")
    print(f"Max:             {lengths.max()}")
    print(f"Mean:            {lengths.mean():.1f}\n")
    print(f"Median:          {np.median(lengths):.1f}")
    for q in [50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
        print(f"{q}th percentile: {np.percentile(lengths, q):.1f}")
    print("\n\n")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(lengths, bins=100, edgecolor="black", alpha=0.7)
    ax1.axvline(np.median(lengths), color="red", ls="--", label="Median")
    ax1.axvline(np.percentile(lengths, 95), color="orange", ls="--", label="95th percentile")
    ax1.set_xlabel("Number of tokens")
    ax1.set_ylabel("Number of conversations")
    ax1.set_title("Length Distribution")
    ax1.legend()

    # CDF
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    ax2.plot(sorted_lengths, cdf, lw=2)
    ax2.set_xlabel("Conversation length (tokens)")
    ax2.set_ylabel("Fraction of conversations")
    ax2.set_title("CDF")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2048*2)

    fig.tight_layout()
    plt.show()





def test_train_loader(train_loader):
    print(f"\nTesting grainloader:")
    for batch in train_loader:
        seqlens = batch.get('attention_mask')
        print(f"input_ids:  {batch['input_ids'].shape}")
        print(f"output_ids: {batch['output_ids'].shape}")
        print(f"seqlens:    {seqlens.shape}\n")
        break
    




def test_model(model, tokenizer, mode):
    if mode == 'AR':
        text1 = "I love the italian cities!"
        text2 = "The capital of Italy is"
    elif mode in ['BERT', 'DiT']:
        text1 = "I love the italian cities!"
        text2 = "The capital of Italy is [MASK][MASK]"
    else:
        raise KeyError
        
    tokenizer_kwargs = {'return_tensors': "pt", 'add_special_tokens': False}
    input1 = tokenizer(text1, **tokenizer_kwargs)['input_ids']
    input2 = tokenizer(text2, **tokenizer_kwargs)['input_ids']

    BOS = torch.tensor([[tokenizer.bos_token_id]]).expand(1, -1)
    EOS = torch.tensor([[tokenizer.eos_token_id]]).expand(1, -1)

    inputs = torch.cat([BOS, input1, EOS, BOS, input2], dim=-1)
    print(inputs.shape)
    print(inputs)
    print(tokenizer.decode(inputs))
    print("\n\n")

    if mode == "AR":
        output = model.generate(inputs, n_tokens=25, 
                                temperature=0.5, tokenizer=tokenizer)
    else:
        output = model.generate(inputs, mask_id=tokenizer.mask_token_id)
    print(tokenizer.decode(output))



from masking_schedule import Masking
from noise_schedule import LogLinearNoise
from utils.utils import getDevice

def test_masking_schedule():
    device = getDevice()
    noise = LogLinearNoise()
    B, T = 2, 10
    masking = Masking(T=T, noise=noise, k=10, gamma=0.2)

    t = torch.zeros(B, device=device) + 0.5
    vanilla, vanilla_w  = masking.vanilla_masking(t, True)
    print(f"vanilla:      \n{vanilla}")
    print(f"vanilla_w:    \n{vanilla_w}\n\n")

    positional, positional_w = masking.position_dependent_masking(t, True)
    print(f"positional:   \n{positional}")
    print(f"positional_w: \n{positional_w}\n\n")
    
    sigmoid, sigmoid_w = masking.moving_sigmoid_masking(t, True)
    print(f"sigmoid:      \n{sigmoid}")
    print(f"sigmoid_w:    \n{sigmoid_w}\n\n")
    
