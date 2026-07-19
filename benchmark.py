import datasets, torch, os, json, random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy import linalg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score_calc
from openai import OpenAI
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.utils import getDevice
from data_processing.data_manager import DataManagerQA
import models.masking_schedule as masking_schedule

from models.AR import AR
from models.DiT import DiT
from GPT_Lightning import GPT
from diffusion_lightning import Diffusion

class Perplexity:
    """
    This class contains methods to calculate the perplexity for both the Autoregressive model and the Diffusion model.
    The main method is 'calculate_benchmark_perplexities' that loops through the validation set and computes the final perplexity for both models.
    """
    def __init__(self, model, model_type='ar', mask_token_id=103, vocab_size=30522):
        """
        Initialize the Perplexity class with the model, input data, and configuration.
         - model: the language model to evaluate (can be either AR or DDM)
         - model_type: 'ar' for Autoregressive, 'ddm' for Diffusion Model
         - mask_token_id: the token id used for masking in the Diffusion Model (default is 103)
         - vocab_size: the size of the vocabulary (needed for the DDM loss calculation, default is 30522 for BERT-like models)
        """
        self.model = model
        self.model_type = model_type
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        # Automatically understand if the model is on CPU or GPU
        self.device = "cuda" #getDevice()

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
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            # 1. Sample a random 't' time from 0 to 1 for EVERY sentence in the batch
            t = torch.rand(batch_size, device=self.device)
            
            # --- NEW DYNAMIC MASKING LOGIC (Standalone Functions) ---
            if self.model.corruption_type == "independent":
                move_chance, loss_weight = masking_schedule.vanilla_masking(
                    t=t,
                    T=seq_len,
                    device=self.device,
                    noise=self.model.noise
                )
            elif self.model.corruption_type == "position":
                move_chance, loss_weight = masking_schedule.position_dependent_masking(
                    t=t,
                    T=seq_len,
                    device=self.device,
                    noise=self.model.noise,
                    gamma=self.model.position_gamma,
                    position_loss_weighting=self.model.position_loss_weighting
                )
            elif self.model.corruption_type == "moving_sigmoid":
                move_chance, loss_weight = masking_schedule.moving_sigmoid_masking(
                    t=t,
                    T=seq_len,
                    device=self.device,
                    noise=self.model.noise,
                    k=self.model.sigmoid_k,
                    calibrated=self.model.calibrated_sigmoid
                )
            else:
                raise ValueError(f"Unknown corruption type: {self.model.corruption_type}")
            # ---------------------------------
            
            # 2. Create the mask
            rand_matrix = torch.rand(batch_size, seq_len, device=self.device)
            is_response_token = (labels != -100)
            
            # move_chance adapts automatically based on the schedule shape
            mask_bool = is_response_token & (rand_matrix < move_chance)
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_bool] = self.mask_token_id
            
            ddm_labels = labels.clone()
            ddm_labels[~mask_bool] = -100 
            
            # 3. Forward Pass (Providing the required sigma)
            sigma, _ = self.model.noise(t)
            logits = self.model(masked_input_ids, sigma=sigma)
            
            # 4. Compute the unreduced Loss
            loss_per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size), 
                ddm_labels.view(-1), 
                ignore_index=-100, 
                reduction='none' 
            )
            
            # 5. Apply the ELBO Math
            loss_per_seq = loss_per_token.view(batch_size, seq_len)
            
            # Multiply by the dynamically calculated loss_weight
            weighted_loss_per_seq = loss_per_seq * loss_weight
            
            batch_total_loss = weighted_loss_per_seq.sum().item()
            valid_masked_tokens = (ddm_labels != -100).sum().item()
            
        return batch_total_loss, valid_masked_tokens
    # ==========================================
    # 3. Main validation loop
    # ==========================================
    def calculate(self, val_dataloader):
        """
        Loop thru the entire dataset using optimized batching and 
        mixed precision to calculate the final global Perplexity.
        """
        self.model.eval()
        
        # Optional: Optimize execution graph if not already compiled
        # if not hasattr(self.model, '_compiled'):
        #     self.model = torch.compile(self.model)
        #     self.model._compiled = True

        if self.model_type == 'ar':
            print("🚀 Starting Autoregressive Perplexity Evaluation...")
            ar_total_loss = 0.0
            ar_total_tokens = 0
            
            # Use tqdm progress bar to monitor mini-batch throughput
            for batch in tqdm(val_dataloader, desc="Processing AR Batches"):
                input_ids = batch['input_ids'].to(self.device) 
                labels = batch['labels'].to(self.device)
                
                # Execute the forward pass using accelerated half-precision
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    ar_loss, ar_tokens = self._evaluate_ar_batch(input_ids, labels)
                    
                ar_total_loss += ar_loss
                ar_total_tokens += ar_tokens
                
            # Compute mathematically sound global average across all mini-batches
            ar_mean_loss = ar_total_loss / max(ar_total_tokens, 1)
            perplexity = torch.exp(torch.tensor(ar_mean_loss))

        elif self.model_type == 'ddm':
            print("🚀 Starting Diffusion Model (ELBO) Perplexity Evaluation...")
            ddm_total_loss = 0.0
            ddm_total_tokens = 0
            
            for batch in tqdm(val_dataloader, desc="Processing DDM Batches"):
                input_ids = batch['input_ids'].to(self.device) 
                labels = batch['labels'].to(self.device)
                
                # Execute the reverse-process blank filling pass in half-precision
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    ddm_loss, ddm_tokens = self._evaluate_ddm_batch(input_ids, labels)
                    
                ddm_total_loss += ddm_loss
                ddm_total_tokens += ddm_tokens
                
            ddm_mean_loss = ddm_total_loss / max(ddm_total_tokens, 1)
            perplexity = torch.exp(torch.tensor(ddm_mean_loss))
            
        print(f"✅ Evaluation Completed!")
        print(f"📊 Global Dataset Perplexity: {perplexity.item():.2f}")
        
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

class DiversityEvaluator:
    """
    Evaluates the lexical and structural diversity of generated texts.
    Combines Unique Token Ratio (Vocabulary richness) and Self-BLEU (Structural repetitiveness).
    """
    def __init__(self, sample_size_for_bleu=1000, n_gram_size=2):
        """
        - sample_size_for_bleu: Max number of sentences to evaluate for Self-BLEU (prevents O(N^2) explosion).
        - n_gram_size: The N-gram size for the Unique Token Ratio (1 = unique words, 2 = unique bigrams).
        """
        self.sample_size_for_bleu = sample_size_for_bleu
        self.n_gram_size = n_gram_size
        
        # Smoothing function is critical for short chat responses!
        # Without it, responses shorter than 4 words will crash the BLEU calculation.
        self.smoother = SmoothingFunction().method1

    def _get_ngrams(self, sequence, n):
        """Helper to extract n-grams from a list of tokens."""
        return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

    def calculate_unique_ratio(self, list_of_token_sequences):
        """
        Calculates the Type-Token Ratio (TTR) based on N-grams.
        Higher is better (more diverse vocabulary).
        """
        all_ngrams = []
        for seq in list_of_token_sequences:
            all_ngrams.extend(self._get_ngrams(seq, self.n_gram_size))
            
        if not all_ngrams:
            return 0.0
            
        unique_ngrams = set(all_ngrams)
        ratio = len(unique_ngrams) / len(all_ngrams)
        return ratio * 100 # Return as percentage

    def calculate_self_bleu(self, list_of_token_sequences):
        """
        Calculates Self-BLEU by comparing each sentence against the rest of the generated corpus.
        LOWER is better (less repetitiveness).
        """
        # Filter out extremely short sequences (e.g., less than 2 tokens) to avoid noisy scores
        valid_sequences = [seq for seq in list_of_token_sequences if len(seq) > 1]
        
        # Subsample to avoid infinite loop
        if len(valid_sequences) > self.sample_size_for_bleu:
            valid_sequences = torch.random.sample(valid_sequences, self.sample_size_for_bleu, replace=False)
            
        total_sentences = len(valid_sequences)
        if total_sentences < 2:
            return 0.0

        bleu_scores = []
        
        # O(N^2) operation on the sampled subset
        for i in range(total_sentences):
            hypothesis = valid_sequences[i]
            
            # The references are ALL OTHER generated sentences
            # We slice the list to exclude the current hypothesis
            references = valid_sequences[:i] + valid_sequences[i+1:]
            
            # Calculate BLEU score (using tokens directly, no string decoding needed!)
            score = sentence_bleu(references, hypothesis, smoothing_function=self.smoother)
            bleu_scores.append(score)
            
        # Return average Self-BLEU (0 to 100)
        return np.mean(bleu_scores) * 100

    def evaluate_models(self, results_dict):
        """
        Main runner. Expects a dictionary like: 
        {'AR': [[12, 45, ...], [..]], 'DDM': [...], 'REAL': [...]}
        Note: Sequences must be JUST THE RESPONSES, prompt must be removed!
        """
        print("\n" + "="*50)
        print("🌍 LEXICAL & CONVERSATIONAL DIVERSITY REPORT")
        print("="*50)
        
        for model_name, sequences in results_dict.items():
            if not sequences:
                continue
                
            print(f"\n[{model_name} MODEL]")
            
            # 1. Unique Token Ratio
            unique_ratio = self.calculate_unique_ratio(sequences)
            print(f"  • Unique {self.n_gram_size}-gram Ratio: {unique_ratio:.2f}% (Higher is better)")
            
            # 2. Self-BLEU
            self_bleu = self.calculate_self_bleu(sequences)
            print(f"  • Average Self-BLEU:   {self_bleu:.2f} (LOWER is better)")

class SemanticEvaluator:
    """
    Evaluates deep semantic similarity and distribution distances using pre-trained BERT models.
    Implements BERTScore and Prompt-based Fréchet BERT Distance (FBD).
    """
    def __init__(self, bert_model_name="bert-base-uncased", device="cuda"):
        """
        Initializes the evaluation models ONCE to save I/O overhead.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.bert_model_name = bert_model_name
        
        print(f"Loading BERT evaluator ({bert_model_name}) into {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.model.eval()

    def calculate_bertscore(self, references, hypotheses):
        """
        Calculates the BERTScore between generated responses (hypotheses) 
        and ground-truth responses (references).
        Returns the F1 scores mean.
        """
        print("Calculating BERTScore...")
        # We use the official bert_score library for robust calculation
        P, R, F1 = bert_score_calc(
            cands=hypotheses, 
            refs=references, 
            lang="en", # Change to "it" if your chat dataset is in Italian
            model_type=self.bert_model_name,
            device=self.device,
            verbose=False
        )
        return F1.mean().item() * 100 # Return as percentage

    def _get_sentence_embeddings(self, texts, batch_size=32):
        """
        Helper method to extract the [CLS] embedding from BERT for a list of texts.
        Uses batching to prevent Out Of Memory errors on large datasets.
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Extract the [CLS] token representation (the first token) 
                # which acts as the aggregated sentence embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())
                
        return np.concatenate(all_embeddings, axis=0)

    def calculate_frechet_bert_distance(self, prompts, real_responses, generated_responses):
        """
        Calculates FBD on the joint distribution of [Prompt + Response].
        Measures if the generated conversational flow matches the real human flow.
        LOWER distance is better.
        """
        print("Extracting embeddings for Fréchet Distance...")
        
        # 1. Create the joint conversational strings
        real_conversations = [f"{p} [SEP] {r}" for p, r in zip(prompts, real_responses)]
        generated_conversations = [f"{p} [SEP] {g}" for p, g in zip(prompts, generated_responses)]
        
        # 2. Extract embeddings
        real_embs = self._get_sentence_embeddings(real_conversations)
        gen_embs = self._get_sentence_embeddings(generated_conversations)
        
        # 3. Calculate Mean and Covariance matrices for both distributions
        mu_real = np.mean(real_embs, axis=0)
        sigma_real = np.cov(real_embs, rowvar=False)
        
        mu_gen = np.mean(gen_embs, axis=0)
        sigma_gen = np.cov(gen_embs, rowvar=False)
        
        print("Computing Fréchet Math...")
        
        # 4. The Fréchet Math: d^2 = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
        diff = mu_real - mu_gen
        
        # Matrix square root of the product of covariances
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        # Prevent complex numbers caused by floating point inaccuracies
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        frechet_distance = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
        
        return frechet_distance

    def evaluate_models(self, dataset_dict):
        """
        Main runner. Expects a dictionary with lists of raw strings:
        {
          'prompts': ["Hi!", "How are you?"],
          'real': ["Hello!", "I'm fine."],
          'ar': ["Hey there.", "I am good."],
          'ddm': ["Hi.", "Doing okay."]
        }
        """
        print("\n" + "="*50)
        print("🧠 SEMANTIC & DISTRIBUTION REPORT (BERT-BASED)")
        print("="*50)
        
        prompts = dataset_dict['prompts']
        real_resp = dataset_dict['real']
        
        for model in ['ar', 'ddm']:
            gen_resp = dataset_dict[model]
            
            print(f"\n[{model.upper()} MODEL]")
            
            b_score = self.calculate_bertscore(references=real_resp, hypotheses=gen_resp)
            print(f"  • BERTScore (F1):           {b_score:.2f}% (Higher is better)")
            
            fbd_score = self.calculate_frechet_bert_distance(prompts, real_resp, gen_resp)
            print(f"  • Fréchet BERT Dist (FBD):  {fbd_score:.2f} (LOWER is better)")

class LLMJudgeEvaluator:
    """
    Evaluates conversational models using an LLM-as-a-Judge in a Pairwise A/B test setup.
    Requires an OpenAI-compatible API (can be OpenAI, Groq, Ollama, vLLM).
    """
    def __init__(self, api_key=None, base_url=None, model_name="gpt-4o-mini"):
        """
        Initializes the LLM Client.
        If you want to use local Llama-3 via Ollama:
        - download and run Ollama server: https://ollama.com/docs/installation
        - download Llama-3 model (8B parameters): `ollama pull llama3`
        - launch the server: `ollama serve`
        - call the class as: LLMJudgeEvaluator(base_url="http://localhost:11434/v1", model_name="llama3")
        """
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy-key"),
            base_url=base_url
        )
        self.model_name = model_name
        
        # Results tracker
        self.results = {
            'AR_wins': 0,
            'DDM_wins': 0,
            'Ties': 0,
            'Errors': 0
        }

    def _get_system_prompt(self):
        """
        The Rubric. This is the most critical part of the LLM-as-a-Judge.
        It defines the criteria and forces JSON output.
        """
        return """You are an impartial, expert evaluator of AI conversational models.
Your task is to compare two responses (Model A and Model B) to a given User Prompt.

Evaluate based on the following criteria:
1. Relevance: Does it answer the prompt directly and accurately?
2. Fluency: Is the language natural and grammatically correct?
3. Coherence: Does the text flow logically without structural hallucinations?
4. Conciseness: Penalize responses that are unnecessarily verbose or repetitive.

You must output ONLY a valid JSON object with the following structure:
{
    "reasoning": "A brief step-by-step explanation comparing both models.",
    "winner": "A" | "B" | "Tie"
}
Do not include markdown blocks or any other text outside the JSON.
"""

    def _evaluate_pair(self, prompt, ar_response, ddm_response):
        """
        Evaluates a single pair of responses. 
        Includes 'Position Bias' mitigation by randomly swapping A and B.
        """
        # Randomize positions to avoid "Position Bias" (LLMs often favor Model A)
        swap = random.choice([True, False])
        
        if swap:
            model_a, model_b = ddm_response, ar_response
            ar_label, ddm_label = "B", "A"
        else:
            model_a, model_b = ar_response, ddm_response
            ar_label, ddm_label = "A", "B"

        # Construct the user message
        user_message = f"""[USER PROMPT]
                            {prompt}

                            [MODEL A RESPONSE]
                            {model_a}

                            [MODEL B RESPONSE]
                            {model_b}
                        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": user_message}
                ],
                # Force JSON mode for easy parsing
                response_format={ "type": "json_object" }, 
                temperature=0.0 # Temperature 0 for deterministic evaluation
            )
            
            # Parse the JSON output
            result_json = json.loads(response.choices[0].message.content)
            winner = result_json.get("winner")
            reasoning = result_json.get("reasoning")
            
            # De-swap to figure out who actually won
            if winner == "Tie":
                actual_winner = "Tie"
            elif winner == ar_label:
                actual_winner = "AR"
            elif winner == ddm_label:
                actual_winner = "DDM"
            else:
                actual_winner = "Error"
                
            return actual_winner, reasoning
            
        except Exception as e:
            print(f"API Error: {e}")
            return "Error", str(e)

    def run_benchmark(self, dataset_dict):
        """
        Runs the pairwise evaluation over the whole dataset.
        Expects: {'prompts': [...], 'ar': [...], 'ddm': [...]}
        """
        print(f"🚀 Starting LLM-as-a-Judge Evaluation using {self.model_name}...")
        
        prompts = dataset_dict['prompts']
        ar_resps = dataset_dict['ar']
        ddm_resps = dataset_dict['ddm']
        
        # Save detailed reasoning for the final report
        detailed_logs = []
        
        for p, ar, ddm in tqdm(zip(prompts, ar_resps, ddm_resps), total=len(prompts)):
            winner, reasoning = self._evaluate_pair(p, ar, ddm)
            
            if winner == "AR":
                self.results['AR_wins'] += 1
            elif winner == "DDM":
                self.results['DDM_wins'] += 1
            elif winner == "Tie":
                self.results['Ties'] += 1
            else:
                self.results['Errors'] += 1
                
            detailed_logs.append({
                "prompt": p,
                "winner": winner,
                "reasoning": reasoning
            })
            
        return self.results, detailed_logs

    def print_report(self):
        """Prints the win-rate statistics."""
        total = sum(self.results.values()) - self.results['Errors']
        if total == 0:
            print("No successful evaluations.")
            return
            
        ar_winrate = (self.results['AR_wins'] / total) * 100
        ddm_winrate = (self.results['DDM_wins'] / total) * 100
        tie_rate = (self.results['Ties'] / total) * 100
        
        print("\n" + "="*50)
        print("⚖️ LLM-AS-A-JUDGE WIN-RATE REPORT")
        print("="*50)
        print(f"  🏆 Autoregressive (AR) Win Rate: {ar_winrate:.1f}%")
        print(f"  🏆 Diffusion (DDM) Win Rate:     {ddm_winrate:.1f}%")
        print(f"  🤝 Tie Rate:                     {tie_rate:.1f}%")
        print(f"  (Errors/Failures: {self.results['Errors']})")

# Benchmarking methods that track the performance through steps of unmasking process for the Discrete Diffusion Model, for different unmasking policies.
class DiffusionTrajectoryEvaluator:
    """
    Evaluates the generation trajectory of a Discrete Diffusion Model step-by-step.
    Tracks Entropy, Step-Perplexity, and Internal Semantic Convergence (Self-Similarity).
    """
    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id
        
        # Global history across the entire dataset
        self.history = {
            'entropy': {},
            'step_ppl': {},
            'semantic_convergence': {} # Cosine similarity to the final step
        }
        
        # Temporary storage for the current batch being generated
        self._temp_batch_data = {}

    def start_new_batch(self):
        """
        Must be called right before starting the step-by-step loop for a new batch.
        """
        self._temp_batch_data = {
            'entropy': {},
            'step_ppl': {},
            'hidden_states': {}
        }

    def _calculate_entropy(self, logits, masked_indices):
        """Calculates Shannon Entropy only for currently masked tokens."""
        masked_logits = logits[masked_indices]
        if masked_logits.numel() == 0:
            return 0.0
            
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean().item()

    def _calculate_step_ppl(self, logits, masked_indices, target_labels):
        """Calculates pseudo-perplexity only for currently masked tokens."""
        masked_logits = logits[masked_indices]
        masked_targets = target_labels[masked_indices]
        
        if masked_logits.numel() == 0 or masked_targets.numel() == 0:
            return 1.0
            
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        return torch.exp(loss).item()

    def update_step(self, step, logits, last_hidden_state, current_input_ids, target_labels):
        """
        Runs inside the DDM inference loop at each step.
        """
        masked_indices = (current_input_ids == self.mask_token_id)
        
        # 1. Track Fast Math Metrics
        ent = self._calculate_entropy(logits, masked_indices)
        self._temp_batch_data['entropy'][step] = ent
        
        ppl = self._calculate_step_ppl(logits, masked_indices, target_labels)
        self._temp_batch_data['step_ppl'][step] = ppl
        
        # 2. Track Semantic State (Hidden State)
        # Average pooling across the sequence length to get a single vector per sentence.
        # Shape goes from (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        # .detach().cpu() is CRUCIAL to prevent GPU Out Of Memory!
        pooled_state = last_hidden_state.mean(dim=1).detach().cpu()
        self._temp_batch_data['hidden_states'][step] = pooled_state

    def finalize_batch(self):
        """
        Must be called AFTER the step-by-step loop finishes for the current batch.
        It computes the semantic similarity backwards.
        """
        steps = sorted(self._temp_batch_data['hidden_states'].keys())
        if not steps:
            return
            
        # The last step represents the "Final Grounded Meaning" of the generation
        final_step = steps[-1]
        final_states = self._temp_batch_data['hidden_states'][final_step]
        
        for step in steps:
            # Initialize global dictionary lists if they don't exist
            if step not in self.history['entropy']:
                self.history['entropy'][step] = []
                self.history['step_ppl'][step] = []
                self.history['semantic_convergence'][step] = []
                
            # Append fast metrics
            self.history['entropy'][step].append(self._temp_batch_data['entropy'][step])
            self.history['step_ppl'][step].append(self._temp_batch_data['step_ppl'][step])
            
            # Calculate Semantic Convergence (Cosine Similarity)
            current_states = self._temp_batch_data['hidden_states'][step]
            
            # cosine_similarity returns a tensor of shape (batch_size,)
            # We take the mean to get a single number for this batch at this step
            cos_sim = F.cosine_similarity(current_states, final_states, dim=-1)
            self.history['semantic_convergence'][step].append(cos_sim.mean().item())

    def get_aggregated_trajectory(self):
        """Returns the final averaged trajectory arrays ready for plotting."""
        trajectory = {}
        for metric, steps_dict in self.history.items():
            if not steps_dict:
                continue
            sorted_steps = sorted(steps_dict.keys())
            trajectory[metric] = {
                'steps': sorted_steps,
                'values': [np.mean(steps_dict[s]) for s in sorted_steps]
            }
        return trajectory

class BenchmarkManager:
    """
    Manages the entire benchmarking process for AR and DDM models.
    Handles dataset loading, tokenization, DataLoader creation, and evaluation.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = OmegaConf.load(config_path)

        self.caching_dir = self.config.caching_directory 
        self.device = getDevice()
        self.test_ds = self.load_smoltal_test()
        self.T = self.config.backbone.T
        self.T_context = self.config.backbone.T_ctx
        self.T_answer = self.config.backbone.T_ans
        self.B = self.config.backbone.B
        self.C = self.config.backbone.C
        self.H = self.config.backbone.H  
        self.N = self.config.backbone.N
        self.lr = self.config.training.learning_rate
        self.wup_steps = self.config.training.warmup_steps

        self.tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
        self.vocab_size = len(self.tokenizer)
        self.dm_qa = DataManagerQA(caching_directory=self.caching_dir, 
                                   tokenizer=self.tokenizer,  
                                   n_processes=0)
        
        self.ar_model_ckpt = "checkpoints/AR-epoch=07-val_loss=1.3041.ckpt/AR-epoch=07-val_loss=1.3041.ckpt"
        self.ddm_model_unif_ckpt = "checkpoints/DiT-independent-epoch=09-val_loss=1.7500.ckpt/DiT-independent-epoch=09-val_loss=1.7500.ckpt"
        self.ddm_model_sigm_ckpt = "checkpoints/DiT-moving_sigmoid-epoch=07-val_loss=1.8796.ckpt/DiT-moving_sigmoid-epoch=07-val_loss=1.8796.ckpt"
        self.ddm_model_posdip_ckpt = "checkpoints/DiT-position-epoch=06-val_loss=1.6808.ckpt/DiT-position-epoch=06-val_loss=1.6808.ckpt"

    def load_smoltal_test(self):
        ds = datasets.load_dataset("HuggingFaceTB/smoltalk",
                                "all",
                                split="test[:10%]",                          
                                cache_dir=".data"
                                )
        print(ds.cache_files)
        print(len(ds))
        return ds
    
    def tokenize_and_group(self):
        # 4. Tokenize and Group the Dataset
        print("Tokenizing and grouping dataset...")
        tokenized_ds = self.dm_qa.tokenize(self.test_ds, split_name="test")
        
        # Choose a context window size (T) that fits your GPU memory
        grouped_ds = self.dm_qa.group_texts_ar(tokenized_ds, T=self.T, split_name="test")
        grouped_ds = self.dm_qa.group_texts_dit(tokenized_ds, T_ctx=self.T_context, T_ans=self.T_answer, split_name="test")

        
        # 5. Format for PyTorch
        # Hugging Face and your benchmark expect 'labels', but DataManagerQA outputs 'output_ids'
        grouped_ds_ar = grouped_ds.rename_column("output_ids", "labels")
        grouped_ds_ar.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
        grouped_ds_dit = grouped_ds.rename_column("output_ids", "labels")
        grouped_ds_dit.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
        
        # 6. Create the DataLoader
        # B is the batch size. We use the built-in getTrainloader method.
        self.val_dataloader_ar = self.dm_qa.getTrainloader(grouped_ds_ar, B=self.B)
        self.val_dataloader_dit = self.dm_qa.getTrainloader(grouped_ds_dit, B=self.B)

    def model(self, model_to_test = "ar", model_variant="unif"):
        models = {
            "ar": self._upload_models_ar(),
            "ddm": self._upload_models_ddm(model_variant=model_variant)
        }
        return models[model_to_test]
    
    def _upload_models_ar(self):
        print("Loading AR model...")
        # Initialize Autoregressive backbone and load Lightning checkpoint
        backbone_ar = AR(V=self.vocab_size, C=self.C, H=self.H, N=self.N)
        model_ar = GPT.load_from_checkpoint(self.ar_model_ckpt,
                                            backbone=backbone_ar,
                                            tokenizer=self.tokenizer,
                                            T=self.T,
                                            learning_rate=self.lr,
                                            warmup_steps=self.wup_steps,
                                            strict=False).to(self.device)
        return model_ar

    def _upload_models_ddm(self, model_variant="unif"):
        print("Loading DDM model...")
        model_ckpt_map = {
            "unif": self.ddm_model_unif_ckpt,
            "sigm": self.ddm_model_sigm_ckpt,
            "posdip": self.ddm_model_posdip_ckpt
        }
        # Initialize DiT/DDM backbone and load Lightning checkpoint
        backbone_ddm = DiT(V=self.vocab_size, C=self.C, H=self.H, N=self.N)
        self.model_ddm = Diffusion.load_from_checkpoint(model_ckpt_map[model_variant],
                                            backbone=backbone_ddm,
                                            tokenizer=self.tokenizer,
                                            T_ctx=self.T_context,
                                            T_ans=self.T_answer,
                                            learning_rate=self.lr,
                                            warmup_steps=self.wup_steps,
                                            strict=False).to(self.device)
        return self.model_ddm

    def run_perplexity_benchmark(self):
        """This method runs the perplexity benchmark for all models (AR and all 3 DDM variants)."""
        self.ppx_values = {
            "AR": None,
            "DDM_uniform": None,
            "DDM_sigmoid": None,
            "DDM_pos_dip": None
        }

        print("Calculating Perplexity for AR model...")
        ppx = Perplexity(model=self.model("ar").backbone, 
                        model_type='ar', 
                        mask_token_id=self.tokenizer.mask_token_id, 
                        vocab_size=self.vocab_size)
        self.ppx_values['AR'] = ppx.calculate(self.val_dataloader_ar)

        print("Calculating Perplexity for DDM model, uniform...")
        ppx = Perplexity(model=self.model("ddm", model_variant="unif"), 
                        model_type='ddm', 
                        mask_token_id=self.tokenizer.mask_token_id, 
                        vocab_size=self.vocab_size)
        self.ppx_values['DDM_uniform'] = ppx.calculate(self.val_dataloader_dit)

        print("Calculating Perplexity for DDM model, sigmoid...")
        ppx = Perplexity(model=self.model("ddm", model_variant="sigm"), 
                        model_type='ddm', 
                        mask_token_id=self.tokenizer.mask_token_id, 
                        vocab_size=self.vocab_size)
        self.ppx_values['DDM_sigmoid'] = ppx.calculate(self.val_dataloader_dit)

        print("Calculating Perplexity for DDM model, positional dipendent...")
        ppx = Perplexity(model=self.model("ddm", model_variant="posdip"), 
                        model_type='ddm', 
                        mask_token_id=self.tokenizer.mask_token_id, 
                        vocab_size=self.vocab_size)
        self.ppx_values['DDM_pos_dip'] = ppx.calculate(self.val_dataloader_dit)

    def generate_token_answers(self, model_type = "ar", model_variant="unif"):
        """
        Generates answers for a given model type (AR or DDM).
        Returns a list of generated responses.
        """
        # given the dataset test_ds, we will generate answers for each prompt
        current_model = self.model(model_type, model_variant=model_variant)
        current_model.eval()
        dataloader = self.val_dataloader_ar if model_type == "ar" else self.val_dataloader_dit
        generated_answers = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating answers for {model_type.upper()}"):
                prompts = batch['input_ids'].to(self.device)
                if model_type == "ddm":
                    # For DDM, we need to run the unmasking loop
                    outputs = current_model.generate(prompts, n_tokens=100, num_steps=20)
                else:
                    outputs = current_model.generate(prompts, max_length=100)
                generated_answers.extend(outputs.cpu().numpy())
        return generated_answers

    def decode_tokens_to_text(self, token_list, prompt_length=None):
        """
        Transforms a list of token arrays into human-readable text.
        If prompt_length is provided, it slices off the prompt and returns only the generated answer.
        """
        text_answers = []
        
        for tokens in tqdm(token_list, desc="Decoding tokens to text"):
            # If a prompt length is provided, slice the array to get only the new tokens
            if prompt_length is not None:
                response_tokens = tokens[prompt_length:]
            else:
                response_tokens = tokens
                
            # Decode the tokens back into a string, ignoring padding and special tokens
            text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            text_answers.append(text)
            
        return text_answers






#==========================================
# MAIN BENCHMARK ORCHESTRATOR
#==========================================
class BenchmarkOrchestrator:
    """
    Main Launcher to handle Generation, Evaluation, and Plotting for the AR vs DDM benchmark.

    /project_root
    ├── run_benchmarks.py      <-- the Launcher
    ├── /results
    │   ├── /datasets          <-- Here we save the generated samples (prompts, ar, ddm)
    │   ├── /metrics           <-- Here we save the raw JSON scores
    │   └── /plots             <-- Here the PDF/PNG plots end up
    """
    def __init__(self, model_ar=None, model_ddm=None, dataloader=None, results_dir="./results"):
        self.model_ar = model_ar
        self.model_ddm = model_ddm
        self.dataloader = dataloader
        self.results_dir = results_dir
        
        # Create directory structure
        self.dirs = {
            'datasets': os.path.join(results_dir, "datasets"),
            'metrics': os.path.join(results_dir, "metrics"),
            'plots': os.path.join(results_dir, "plots")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    # ==========================================
    # 1. GENERATION PHASE
    # ==========================================

    def generate_and_save_dataset(self, filename="eval_dataset.json", max_samples=None):
        """
        Generates text using both models and saves the prompt, real response, AR, and DDM responses.
        If the file already exists, it can skip generation to save time.
        """
        filepath = os.path.join(self.dirs['datasets'], filename)
        
        if os.path.exists(filepath):
            print(f"📦 Dataset found at {filepath}. Skipping generation.")
            return filepath
            
        print("🚀 Starting Generation Phase...")
        self.model_ar.eval()
        self.model_ddm.eval()
        
        dataset = {
            'prompts': [],
            'real_responses': [],
            'ar_responses': [],
            'ddm_responses': []
        }
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Generating Texts"):
                if max_samples and samples_processed >= max_samples:
                    break
                    
                prompts = batch['input_ids'].cuda()
                
                # 1. Get AR Generation (Replace with your specific AR generation function)
                ar_gen = self.model_ar.generate(prompts, max_length=100)
                
                # 2. Get DDM Generation (Replace with your specific DDM unmasking loop)
                ddm_gen = self.model_ddm.generate(prompts, steps=20)
                
                # Note: Decode the tensors to strings here before saving to JSON!
                # dataset['prompts'].extend(decoded_prompts)
                # dataset['real_responses'].extend(decoded_reals)
                # dataset['ar_responses'].extend(decoded_ar)
                # dataset['ddm_responses'].extend(decoded_ddm)
                
                samples_processed += prompts.size(0)
                
        # Save to disk
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4)
        print(f"✅ Generation complete. Saved {samples_processed} samples to {filepath}")
        return filepath

    # ==========================================
    # 2. EVALUATION PHASE
    # ==========================================
    def load_dataset(self, filename="eval_dataset.json"):
        filepath = os.path.join(self.dirs['datasets'], filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_all_benchmarks(self, dataset_filename="eval_dataset.json"):
        """
        Loads the generated text and runs the requested evaluators.
        """
        print("🧪 Starting Evaluation Phase...")
        data = self.load_dataset(dataset_filename)
        
        results = {}
        
        # --- 2.1 Diversity Benchmark (Unique Tokens & Self-BLEU) ---
        print("\n--> Running Diversity Evaluator...")
        # evaluator_div = DiversityEvaluator()
        # results['diversity'] = evaluator_div.evaluate_models(data)
        
        # --- 2.2 Semantic Benchmark (BERTScore & Fréchet) ---
        print("\n--> Running Semantic Evaluator...")
        # evaluator_sem = SemanticEvaluator()
        # results['semantic'] = evaluator_sem.evaluate_models(data)
        
        # --- 2.3 LLM Judge Benchmark ---
        print("\n--> Running LLM Judge Evaluator...")
        # evaluator_judge = LLMJudgeEvaluator(model_name="llama3") # Local Ollama
        # results['judge'] = evaluator_judge.run_benchmark(data)
        
        # Save aggregated metrics
        metrics_file = os.path.join(self.dirs['metrics'], "final_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ All evaluations complete. Metrics saved to {metrics_file}")
        
        return results

    # ==========================================
    # 3. PLOTTING PHASE (Matplotlib)
    # ==========================================
    def generate_plots(self, metrics_filename="final_metrics.json", trajectory_data=None):
        """
        Reads the saved metrics and generates professional Matplotlib charts.
        """
        print("📊 Generating Plots...")
        metrics_path = os.path.join(self.dirs['metrics'], metrics_filename)
        
        # 1. Comparative Bar Charts (AR vs DDM)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            models = ['AR', 'DDM']
            
            # Example Bar: LLM Judge Win Rate
            if 'judge' in metrics:
                wins = [metrics['judge']['AR_wins'], metrics['judge']['DDM_wins']]
                axs[0].bar(models, wins, color=['#4C72B0', '#DD8452'])
                axs[0].set_title('LLM Judge Wins')
                axs[0].set_ylabel('Number of Wins')
                
            # Example Bar: Diversity (Self-BLEU)
            if 'diversity' in metrics:
                bleu = [metrics['diversity']['ar']['self_bleu'], metrics['diversity']['ddm']['self_bleu']]
                axs[1].bar(models, bleu, color=['#4C72B0', '#DD8452'])
                axs[1].set_title('Self-BLEU (Lower is better)')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots'], 'comparative_bars.pdf'))
            plt.close()

        # 2. Line Charts for DDM Trajectory (Step-by-Step)
        # This requires the data from the DiffusionTrajectoryEvaluator
        if trajectory_data:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            steps = trajectory_data['entropy']['steps']
            entropy_vals = trajectory_data['entropy']['values']
            sim_vals = trajectory_data['semantic_convergence']['values']
            
            # Plot Entropy on left Y-axis
            color = 'tab:red'
            ax1.set_xlabel('Generative Steps (t)')
            ax1.set_ylabel('Masked Entropy', color=color)
            ax1.plot(steps, entropy_vals, color=color, marker='o', linewidth=2, label='Entropy')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Plot Semantic Convergence on right Y-axis
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Cosine Similarity (to final state)', color=color)
            ax2.plot(steps, sim_vals, color=color, marker='s', linewidth=2, label='Semantic Convergence')
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('DDM Generation Trajectory (Unmasking Policy Analysis)')
            fig.tight_layout()
            plt.savefig(os.path.join(self.dirs['plots'], 'ddm_trajectory.pdf'))
            plt.close()
            
        print(f"✅ Plots generated and saved to {self.dirs['plots']}")
