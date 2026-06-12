import numpy as np
import torch
import torch.nn.functional as F

class Perplexity:
    """
    This class contains methods to calculate the perplexity for both the Autoregressive model and the Diffusion model.
    The main method is 'calculate_benchmark_perplexities' that loops through the validation set and computes the final perplexity for both models.
    """
    def __init__(self, model, input_ids, labels, model_type='ar', mask_token_id=103, vocab_size=30522):
        """
        Initialize the Perplexity class with the model, input data, and configuration.
         - model: the language model to evaluate (can be either AR or DDM)
         - input_ids: the tokenized input sentences (batch of token ids)
         - labels: the tokenized labels (with -100 for prompt tokens and real token ids for answer tokens)
         - model_type: 'ar' for Autoregressive, 'ddm' for Diffusion Model
         - mask_token_id: the token id used for masking in the Diffusion Model (default is 103)
         - vocab_size: the size of the vocabulary (needed for the DDM loss calculation, default is 30522 for BERT-like models)
        """
        self.model = model
        self.input_ids = input_ids
        self.labels = labels
        self.model_type = model_type
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        # Automatically understand if the model is on CPU or GPU
        self.device = next(model.parameters()).device

    # ==========================================
    # 1. Function for autoregressive function
    # ==========================================
    def _evaluate_ar_batch(self, input_ids, labels):
        """
        Compute the sum of the Cross-Entropy loss for a batch in the Autoregressive model.
        """
        # Do we not want to calculate the gradient
        with torch.no_grad():
            # Forward pass: the AR look the token and predict the following one.
            logits = self.model(input_ids)
            
            # Shift: the logits in position 'i' should predict the label in position 'i+1'
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute the summed loss on all the batch (we will ignore the prompt setting the value to -100)
            # Let's we use reduction='sum' to accumulate the total value
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100, 
                reduction='sum'
                )   
            # Let's count how many token of REAL answer that are in this batch
            valid_tokens = (shift_labels != -100).sum().item()
        return loss.item(), valid_tokens
    
    # ==========================================
    # 2. Function for Diffusion Model
    # ==========================================
    def _evaluate_ddm_batch(self, input_ids, labels):
        """
        Compute the sum of the weighted Loss (ELBO) for a batch in the Diffusion Model.
        Do we use a linear schedule just for start.  
        """
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            # Let sample a casual 't' time from 0 to 1 for EVERY sentence in the batch 
            t = torch.rand(batch_size, 1, device=self.device)
            # For a linear schedule, the percentage  of masked tocken is simply t
            # (In a more complex schedules, mask_ratio = compute_schedule(t))
            mask_ratio = t 
            # For the linear schedule, the mathematical derived value that works as the ELBO weight is constant (1.0)
            # If in the future we would use a different schedule, this weight will be changed in function of t
            weight_t = 1.0 
            
            # Let's create a casual mask
            rand_matrix = torch.rand(batch_size, seq_len, device=self.device)
            
            # Let's mask ONLY the token of the answer (labels != -100) 
            # with a probability of mask_ratio
            is_response_token = (labels != -100)
            mask_bool = is_response_token & (rand_matrix < mask_ratio)
            
            # We apply the mask to the input
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_bool] = self.mask_token_id
            
            # We create the labels for the Loss: let we isolate ONLY the maskered token
            ddm_labels = labels.clone()
            ddm_labels[~mask_bool] = -100 
            
            # Forward Pass
            logits = self.model(masked_input_ids)
            
            # Compute the Loss NOT reducted, to be able to apply the weight for the sentence
            loss_per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size), 
                ddm_labels.view(-1), 
                ignore_index=-100, 
                reduction='none' # Mantain the unroll tensor
            )
            
            # Reroll the tensor to (batch_size, seq_len) and sum the loss for every sentences
            loss_per_seq = loss_per_token.view(batch_size, seq_len).sum(dim=1)
            # Multiply for the ELBO weight of the 't' step of that specific sentence
            weighted_loss_per_seq = loss_per_seq * weight_t
            # Sum all to obtain the total loss of the batch
            batch_total_loss = weighted_loss_per_seq.sum().item()
            # Count how many token has been ACTUALLY masked and valuated in this batch
            valid_masked_tokens = (ddm_labels != -100).sum().item()
            
        return batch_total_loss, valid_masked_tokens
    # ==========================================
    # 3. Main validation loop
    # ==========================================
    def calculate(self, val_dataloader):
        """
        Loop thru the entire dataset and calculate the final Perplexity.
        """
        self.model.eval()
        
        if self.model_type == 'ar':
            print("🚀 Starting Autoregressive Perplexity Evaluation...")
            ar_total_loss = 0.0
            ar_total_tokens = 0
            # Let's iterate over the validation set
            for batch in val_dataloader:
                # Spostiamo i dati sulla GPU (se disponibile)
                input_ids = batch['input_ids'].to(self.device) 
                labels = batch['labels'].to(self.device)
                # AR evaluation
                ar_loss, ar_tokens = self._evaluate_ar_batch(input_ids, labels)
                ar_total_loss += ar_loss
                ar_total_tokens += ar_tokens
            # --- FINAL COMPUTATION ---
            # AR: Exponential of the mathematical average
            ar_mean_loss = ar_total_loss / max(ar_total_tokens, 1) # max per evitare divisioni per zero
            perplexity = torch.exp(torch.tensor(ar_mean_loss))

        elif self.model_type == 'ddm':
            print("🚀 Starting Diffusion Model (ELBO) Perplexity Evaluation...")
            ddm_total_loss = 0.0
            ddm_total_tokens = 0
            # Let's iterate over the validation set
            for batch in val_dataloader:
                # Spostiamo i dati sulla GPU (se disponibile)
                input_ids = batch['input_ids'].to(self.device) 
                labels = batch['labels'].to(self.device)
                # DDM evaluation
                ddm_loss, ddm_tokens = self._evaluate_ddm_batch(input_ids, labels)
                ddm_total_loss += ddm_loss
                ddm_total_tokens += ddm_tokens
            # --- FINAL COMPUTATION ---
            # DDM: Exponential of the weighted average (ELBO)
            ddm_mean_loss = ddm_total_loss / max(ddm_total_tokens, 1)
            perplexity = torch.exp(torch.tensor(ddm_mean_loss))
            
        print(f"✅ Evaluation Completed!")
        print(f"📊 Perplexity: {perplexity.item():.2f}")
        
        return perplexity.item()

class ChatStructureEvaluator:
    """
    Evaluates the structural correctness and length of generated responses 
    from text models in chat-based scenarios.
    """
    def __init__(self, eos_token_id="<eos>", user_token_id=None):
        """
        Initializes the evaluator.
        - eos_token_id: the ID of the end-of-generation token (e.g., <|endoftext|> or <eos>, default: <eos>).
        - user_token_id: (Optional) the ID of the user marker (e.g., <user>). 
                         Used to catch if the model hallucinates and pretends to be the user.
        """
        self.eos_token_id = eos_token_id
        self.user_token_id = user_token_id
        
        # Dictionary to accumulate evaluation results
        self.results = {
            'AR': {'lengths': [], 'missing_eos': 0, 'user_hallucinations': 0},
            'DDM': {'lengths': [], 'missing_eos': 0, 'user_hallucinations': 0},
            'REAL': {'lengths': []} # Ground truth lengths for comparison
        }

    def _evaluate_single_sequence(self, generated_tokens, prompt_length):
        """
        Analyzes a single generated sequence.
        - generated_tokens: the list (or 1D tensor) of tokens (including the prompt).
        - prompt_length: the length of the prompt, used to isolate only the response.
        """
        # Isolate ONLY the response generated by the model
        response_tokens = generated_tokens[prompt_length:]
        
        # If it's a PyTorch tensor, convert it to a Python list for easier processing
        if hasattr(response_tokens, 'tolist'):
            response_tokens = response_tokens.tolist()
            
        metrics = {
            'length': 0,
            'missing_eos': True,
            'user_hallucination': False
        }
        
        # 1. Find where the model stopped (or where it SHOULD have stopped)
        try:
            # Look for the first appearance of the EOS token
            eos_index = response_tokens.index(self.eos_token_id)
            metrics['missing_eos'] = False
            # The actual generated response length is up to the EOS (excluded)
            actual_response = response_tokens[:eos_index]
        except ValueError:
            # The model never generated EOS (it reached max_length)
            actual_response = response_tokens
            
        metrics['length'] = len(actual_response)
        
        # 2. Check for User Token Hallucination
        if self.user_token_id is not None and self.user_token_id in actual_response:
            metrics['user_hallucination'] = True
            
        return metrics

    def add_batch_results(self, model_name, batch_generated_ids, prompt_lengths):
        """
        Processes a batch of generated sequences and adds them to the statistics.
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not recognized. Use 'AR', 'DDM' or 'REAL'.")
            
        for gen_seq, p_len in zip(batch_generated_ids, prompt_lengths):
            metrics = self._evaluate_single_sequence(gen_seq, p_len)
            
            self.results[model_name]['lengths'].append(metrics['length'])
            
            # 'REAL' is only used for length distributions, ignore structural errors for ground truth
            if model_name != 'REAL':
                if metrics['missing_eos']:
                    self.results[model_name]['missing_eos'] += 1
                if metrics['user_hallucination']:
                    self.results[model_name]['user_hallucinations'] += 1

    def print_report(self):
        """
        Prints a formatted statistical report.
        """
        print("\n" + "="*50)
        print("📊 STRUCTURAL AND TURN LENGTH REPORT")
        print("="*50)
        
        for model in ['REAL', 'AR', 'DDM']:
            data = self.results[model]
            lengths = data['lengths']
            
            if not lengths:
                continue
                
            total_seqs = len(lengths)
            avg_len = np.mean(lengths)
            std_len = np.std(lengths)
            
            print(f"\n[{model} MODEL] - Evaluated {total_seqs} sequences")
            print(f"  • Average Length:   {avg_len:.2f} tokens (± {std_len:.2f})")
            
            if model != 'REAL':
                err_eos = (data['missing_eos'] / total_seqs) * 100
                err_usr = (data['user_hallucinations'] / total_seqs) * 100
                print(f"  • Missing EOS:      {err_eos:.1f}% ({data['missing_eos']} occurrences)")
                print(f"  • User Hallucination:{err_usr:.1f}% ({data['user_hallucinations']} occurrences)")

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