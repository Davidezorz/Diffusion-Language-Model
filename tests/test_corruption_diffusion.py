import torch
import datasets
from transformers import AutoTokenizer

# Import your custom modules
from data_processing.data_manager import DataManagerQA
import noise.noise_schedule as noise_schedule
import noise.masking_schedule as masking_schedule
from diffusion_lightning import Diffusion

import lightning

class DummyBackbone(torch.nn.Module):
    """A lightweight mock backbone to initialize the Diffusion model."""
    def __init__(self):
        super().__init__()
    def load(self, *args, **kwargs):
        pass
    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]

def test_dit_corruption():
    print("Initializing tokenizer...")

    lightning.seed_everything(0)
    tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
    
    # -------------------------------------------------------------------------
    # 1. Configuration for the test
    # -------------------------------------------------------------------------
    T_ctx = 64
    T_ans = 32
    B = 2
    
    print("\nLoading a tiny subset of smoltalk...")
    ds = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="train[:10]", cache_dir=".data")
    
    # -------------------------------------------------------------------------
    # 2. Data Processing (using your exact DiT pipeline)
    # -------------------------------------------------------------------------
    dm = DataManagerQA(
        caching_directory=".data/", 
        tokenizer=tokenizer, 
        mode='DiT', 
        n_processes=1
    )
    
    print("\nTokenizing and Grouping...")
    tokenized = dm.tokenize(ds, split_name="test_corr")
    grouped = dm.group_texts_dit(tokenized, T_ctx=T_ctx, T_ans=T_ans, split_name="test_corr")
    grouped = grouped.with_format("torch")
    
    loader = torch.utils.data.DataLoader(grouped, batch_size=B)
    batch = next(iter(loader))
    
    # -------------------------------------------------------------------------
    # 3. Initialize the Diffusion Module
    # -------------------------------------------------------------------------
    noise = noise_schedule.get_noise("loglinear")
    masking = masking_schedule.Masking(
        T_ans=T_ans,
        noise=noise,
        corruption_type="moving_sigmoid",
        position_loss_weighting=False,
        k=15,
        gamma=0.2
    )
    
    model = Diffusion(
        backbone=DummyBackbone(),
        tokenizer=tokenizer,
        masking_schedule=masking,
        T_ans=T_ans
    )
    
    input_ids = batch["input_ids"]
    ans_start_idx = batch["ans_start_idx"]
    
    print("\n" + "="*60)
    print(" BATCH INFORMATION")
    print("="*60)
    print(f"Context Length (T_ctx) : {T_ctx}")
    print(f"Answer Length (T_ans)  : {T_ans}")
    print(f"Total Sequence Length  : {input_ids.shape[1]}")
    
    # -------------------------------------------------------------------------
    # 4. Run Corruption at Different Timesteps in PARALLEL
    # -------------------------------------------------------------------------
    # t ≈ 0.0 (mostly clean) -> t ≈ 0.5 (half masked) -> t ≈ 0.99 (fully masked)
    timesteps_to_test = [0.01, 0.5, 0.99] 
    
    # We will store the full batched outputs here
    batch_results = []
    
    for t_val in timesteps_to_test:
        # Create a batched tensor for timestep t (Shape: B)
        t_tensor = torch.full((B,), t_val)
        
        # move_chance will be (B, T_ans) for moving_sigmoid
        move_chance, _ = masking(t_tensor)
        
        # Corrupt the whole batch at once
        q_xt_out = model.q_xt(
            x=input_ids, 
            p=move_chance, 
            ans_start_idx=ans_start_idx
        )
        
        batch_results.append((t_val, move_chance, q_xt_out))
        
    # -------------------------------------------------------------------------
    # 5. Print out the results sample by sample
    # -------------------------------------------------------------------------
    for b in range(B):
        start_idx = ans_start_idx[b].item()
        
        print("\n" + "═"*60)
        print(f" SAMPLE {b} (Answer starts at token index: {start_idx})")
        print("═"*60)
        
        # Decode the pure context (Question)
        context_text = tokenizer.decode(input_ids[b, :start_idx], skip_special_tokens=False)
        print(f"\n[CONTEXT ONLY (0 to {start_idx})]:\n{context_text}")
        print(f"actual context len {input_ids[b, :start_idx].shape}")

        # Decode the pure target answer area
        answer_text = tokenizer.decode(input_ids[b, start_idx:start_idx+T_ans], skip_special_tokens=False)
        print(f"\n[TARGET ANSWER AREA ({start_idx} to {start_idx+T_ans})]:\n{answer_text}")
        print(f"actual answer len {input_ids[b, start_idx:start_idx+T_ans].shape}")
        print(f"actual total len {input_ids[b].shape}")
        
        print("\n" + "-"*40)
        print(" CORRUPTION PROGRESSION")
        print("-"*40)
        
        for t_val, move_chance_batch, q_xt_batch in batch_results:
            # Since move_chance_batch is (B, T_ans), we take the mean for this specific sample 
            # to show an "average" mask probability across the sequence.
            avg_mask_prob = move_chance_batch[b].mean().item()
            
            # Decode the corrupted sequence for this specific sample
            corrupted_text = tokenizer.decode(q_xt_batch[b], skip_special_tokens=False)
            
            print(f"\n▶ t = {t_val} (Avg Mask Probability: {avg_mask_prob:.2%})")
            print(corrupted_text)

if __name__ == '__main__':
    test_dit_corruption()