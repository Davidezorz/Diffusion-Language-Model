import datasets, torch, os, json, random, gc
import numpy as np
import pandas as pd
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
    from text models, focusing strictly on End-Of-Sequence (EOS) token generation.
    """
    def __init__(self, eos_token_id=None):
        if eos_token_id is None:
            raise ValueError("You must provide an integer eos_token_id from your tokenizer!")
            
        self.eos_token_id = eos_token_id
        
        self.results = {
            'REAL': {'lengths': []},
            'AR answer': {'lengths': [], 'missing_eos': 0},
            'DiT uniform': {'lengths': [], 'missing_eos': 0},
            'DiT sigmoid': {'lengths': [], 'missing_eos': 0},
            'DiT positional dependent': {'lengths': [], 'missing_eos': 0}
        }

    def _evaluate_single_sequence(self, response_tokens):
        # response_tokens is already sliced from the prompt in the generation step
        if hasattr(response_tokens, 'tolist'):
            response_tokens = response_tokens.tolist()
            
        metrics = {'length': 0, 'missing_eos': True}
        
        try:
            eos_index = response_tokens.index(self.eos_token_id)
            metrics['missing_eos'] = False
            actual_response = response_tokens[:eos_index]
        except ValueError:
            actual_response = response_tokens
            
        metrics['length'] = len(actual_response)
        return metrics

    def evaluate_dataframe(self, df_tokens):
        """Take in the DataFrame containing the tokens and analyze them."""
        print("\n🔍 Structural and Length Analysis (EOS detection only)...")
        
        for col in self.results.keys():
            if col in df_tokens.columns:
                for tokens in df_tokens[col]:
                    metrics = self._evaluate_single_sequence(tokens)
                    
                    self.results[col]['lengths'].append(metrics['length'])
                    
                    if col != 'REAL':
                        if metrics['missing_eos']:
                            self.results[col]['missing_eos'] += 1

    def print_report(self):
        print("\n" + "="*50)
        print("📊 LENGTH AND STRUCTURE (EOS) REPORT")
        print("="*50)

        collected_data = []
        for col, data in self.results.items():
            lengths = data.get('lengths', [])
            if not lengths:
                continue
                
            total_seqs = len(lengths)
            avg_len = np.mean(lengths)
            std_len = np.std(lengths)
            
            print(f"\n[{col.upper()}] - Evaluated {total_seqs} sequences")
            print(f"  • Average Length: {avg_len:.2f} tokens (± {std_len:.2f})")
            
            if col != 'REAL':
                err_eos = (data['missing_eos'] / total_seqs) * 100
                print(f"  • Missing EOS:    {err_eos:.1f}% ({data['missing_eos']} occurrences)")
            else:
                err_eos = 0.0
            
            collected_data.append({
                'model': col,
                'avg_length': avg_len,
                'std_length': std_len,
                'missing_eos': err_eos
            })

        collected_df = pd.DataFrame(collected_data)
        return collected_df

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

    def evaluate_dataframe(self, df_tokens):
        """Use the benchmarking functions to evaluate diversity in a DataFrame."""
        print("\n" + "="*50)
        print("🌍 LEXICAL & CONVERSATIONAL DIVERSITY REPORT")
        print("="*50)
        
        cols_to_evaluate = ['AR answer', 'DiT uniform', 'DiT sigmoid', 'DiT positional dependent']
        results = {'model': [], 'unique_ngram_ratio': [], 'self_bleu': []}
        
        for col in cols_to_evaluate:
            if col in df_tokens.columns:
                print(f"\n[{col.upper()}]")
                sequences = df_tokens[col].tolist()
                
                unique_ratio = self.calculate_unique_ratio(sequences)
                print(f"  • Unique {self.n_gram_size}-gram Ratio: {unique_ratio:.2f}% (Higher is better)")
                
                self_bleu = self.calculate_self_bleu(sequences)
                print(f"  • Average Self-BLEU:   {self_bleu:.2f} (LOWER is better)")
                
                results['model'].append(col)
                results['unique_ngram_ratio'].append(unique_ratio)
                results['self_bleu'].append(self_bleu)
                
        return pd.DataFrame(results)

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

    def evaluate_dataframe(self, df_text):
        """Legge le stringhe testuali dal DataFrame per calcolare le similarità semantiche."""
        print("\n" + "="*50)
        print("🧠 SEMANTIC & DISTRIBUTION REPORT (BERT-BASED)")
        print("="*50)
        
        prompts = df_text['context'].tolist()
        real_resp = df_text['true answer'].tolist()
        
        cols_to_evaluate = ['AR answer', 'DiT uniform', 'DiT sigmoid', 'DiT positional dependent']
        results = {
            'model': [],
            'bertscore': [],
            'fbd_score': []
        }

        for col in cols_to_evaluate:
            if col in df_text.columns:
                print(f"\n[{col.upper()}]")
                gen_resp = df_text[col].astype(str).tolist()
                
                b_score = self.calculate_bertscore(references=real_resp, hypotheses=gen_resp)
                print(f"  • BERTScore (F1):           {b_score:.2f}% (Higher is better)")
                
                fbd_score = self.calculate_frechet_bert_distance(prompts, real_resp, gen_resp)
                print(f"  • Fréchet BERT Dist (FBD):  {fbd_score:.2f} (LOWER is better)")
                results['model'].append(col)
                results['bertscore'].append(b_score)
                results['fbd_score'].append(fbd_score)
        return pd.DataFrame(results)

class LLMJudgeEvaluator:
    """
    Evaluates 4 conversational models simultaneously using a Borda Count ranking system (3, 2, 1, 0 points)
    with a local LLM-as-a-Judge via Ollama.
    """
    def __init__(self, model_name="llama3", base_url="http://localhost:11434/v1"):
        # Setup for Local Ollama
        self.client = OpenAI(
            api_key="ollama", # Ignored by Ollama, required by OpenAI client
            base_url=base_url
        )
        self.model_name = model_name
        
        # Point accumulators for each model
        self.results = {
            'AR answer': 0,
            'DiT uniform': 0,
            'DiT positional dependent': 0,
            'DiT sigmoid': 0,
            'Errors': 0
        }

    def _get_system_prompt(self):
        return """You are an impartial, expert evaluator of AI conversational models.
Your task is to rank four different responses (Model A, Model B, Model C, and Model D) to a given User Prompt from best to worst.
You MUST provide a strict ordering with NO ties allowed.

Evaluate based on:
1. Relevance: Answers the prompt directly and accurately.
2. Fluency: Natural, grammatically correct language.
3. Coherence: Logical flow without structural artifacts or repetition.

You must output ONLY a valid JSON object with this exact structure:
{
    "reasoning": "A brief explanation justifying your full ranking.",
    "ranking": ["1st_letter", "2nd_letter", "3rd_letter", "4th_letter"]
}
Where "ranking" is an array of letters ["A", "B", "C", "D"] ordered from best (1st place) to worst (4th place).
Do not include markdown codeblocks or extra text outside the JSON.
"""

    def evaluate_dataframe(self, df_text):
        """
        Runs the 4-way ordinal ranking evaluation over the entire text DataFrame.
        """
        print(f"\n🚀 Starting 4-Way Ranked LLM-as-a-Judge with Ollama ({self.model_name})...")
        
        models_cols = ['AR answer', 'DiT uniform', 'DiT positional dependent', 'DiT sigmoid']
        points_map = [3, 2, 1, 0] # 1st -> 3 pts, 2nd -> 2 pts, 3rd -> 1 pt, 4th -> 0 pts
        
        for idx, row in tqdm(df_text.iterrows(), total=len(df_text)):
            prompt = str(row['context'])
            responses = {col: str(row[col]) for col in models_cols}
            
            # Shuffle models to eliminate position bias
            shuffled_cols = list(responses.keys())
            random.shuffle(shuffled_cols)
            
            # Map letters A, B, C, D to shuffled model column names
            letter_to_model = {
                "A": shuffled_cols[0],
                "B": shuffled_cols[1],
                "C": shuffled_cols[2],
                "D": shuffled_cols[3]
            }

            user_message = f"[USER PROMPT]\n{prompt}\n\n"
            for letter, col_name in letter_to_model.items():
                user_message += f"[MODEL {letter} RESPONSE]\n{responses[col_name]}\n\n"

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"}, 
                    temperature=0.0 # Deterministic
                )
                
                result_json = json.loads(response.choices[0].message.content)
                ranking = result_json.get("ranking", [])
                
                # Validation: ensure we got an array of 4 unique valid letters
                if len(ranking) == 4 and set(ranking) == {"A", "B", "C", "D"}:
                    for rank_idx, letter in enumerate(ranking):
                        actual_model = letter_to_model[letter]
                        pts = points_map[rank_idx]
                        self.results[actual_model] += pts
                else:
                    self.results['Errors'] += 1
                    
            except Exception as e:
                self.results['Errors'] += 1
                
        return self.print_report(len(df_text))

    def print_report(self, total_evaluated):
        """Prints the Borda count scores and average points per prompt."""
        total_valid = total_evaluated - self.results['Errors']
        max_possible_pts = total_valid * 3
        
        print("\n" + "="*50)
        print("⚖️ LLM-AS-A-JUDGE (BORDA COUNT RANKING REPORT)")
        print("="*50)
        
        if total_valid <= 0:
            print("No valid evaluations completed.")
            return pd.DataFrame()
            
        collected_data = []
        models_cols = ['AR answer', 'DiT uniform', 'DiT positional dependent', 'DiT sigmoid']
        
        # Sort models by total accumulated points descending
        sorted_models = sorted(models_cols, key=lambda m: self.results[m], reverse=True)
        
        for rank, col in enumerate(sorted_models, 1):
            total_pts = self.results[col]
            avg_pts = total_pts / total_valid
            pct_max = (total_pts / max_possible_pts) * 100 if max_possible_pts > 0 else 0.0
            
            print(f"  #{rank} {col}: {total_pts} pts (Avg: {avg_pts:.2f} pts/prompt | {pct_max:.1f}% of max)")
            collected_data.append({
                'rank': rank,
                'model': col,
                'total_points': total_pts,
                'avg_points_per_prompt': avg_pts,
                'pct_max_points': pct_max
            })
            
        print(f"  (Errors/Failures: {self.results['Errors']})")
        return pd.DataFrame(collected_data)

# Benchmarking methods that track the performance through steps of unmasking process for the Discrete Diffusion Model, for different unmasking policies.
class DiffusionTrajectoryEvaluator:
    """
    Evaluates the generation trajectory of a Discrete Diffusion Model step-by-step.
    Tracks Entropy, Step-Perplexity, and Internal Semantic Convergence (Self-Similarity).
    """
    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id
        
        # Global history across batches
        self.history = {
            'entropy': {},
            'step_ppl': {},
            'semantic_convergence': {}
        }
        
        self._reset_batch_data()

    def _reset_batch_data(self):
        """Resets temporary batch storage."""
        self._temp_batch_data = {
            'entropy': {},
            'step_ppl': {},
            'hidden_states': {}
        }

    def _calculate_entropy(self, logits, masked_indices):
        masked_logits = logits[masked_indices]
        if masked_logits.numel() == 0:
            return 0.0
            
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean().item()

    def _calculate_step_ppl(self, logits, masked_indices, target_labels):
        masked_logits = logits[masked_indices]
        masked_targets = target_labels[masked_indices]
        
        if masked_logits.numel() == 0 or masked_targets.numel() == 0:
            return 1.0
            
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        return torch.exp(loss).item()

    def update_step(self, step, logits, last_hidden_state, current_input_ids, target_labels):
        """Runs inside the DDM inference loop at each unmasking step."""
        masked_indices = (current_input_ids == self.mask_token_id)
        
        # 1. Track metrics
        self._temp_batch_data['entropy'][step] = self._calculate_entropy(logits, masked_indices)
        self._temp_batch_data['step_ppl'][step] = self._calculate_step_ppl(logits, masked_indices, target_labels)
        
        # 2. Store pooled hidden states
        pooled_state = last_hidden_state.mean(dim=1).detach().cpu()
        self._temp_batch_data['hidden_states'][step] = pooled_state

    def finalize_batch(self):
        """Computes backward semantic convergence once a batch completes."""
        steps = sorted(self._temp_batch_data['hidden_states'].keys())
        if not steps:
            return
            
        final_step = steps[-1]
        final_states = self._temp_batch_data['hidden_states'][final_step]
        
        for step in steps:
            if step not in self.history['entropy']:
                self.history['entropy'][step] = []
                self.history['step_ppl'][step] = []
                self.history['semantic_convergence'][step] = []
                
            self.history['entropy'][step].append(self._temp_batch_data['entropy'][step])
            self.history['step_ppl'][step].append(self._temp_batch_data['step_ppl'][step])
            
            current_states = self._temp_batch_data['hidden_states'][step]
            cos_sim = F.cosine_similarity(current_states, final_states, dim=-1)
            self.history['semantic_convergence'][step].append(cos_sim.mean().item())
            
        self._reset_batch_data()

    def export_dataframe(self, variant_name="unif"):
        """Compiles trajectory metrics into a Pandas DataFrame and saves to CSV."""
        aggregated_data = {
            'step': [],
            'avg_entropy': [],
            'avg_step_ppl': [],
            'avg_semantic_convergence': []
        }
        
        for step in sorted(self.history['entropy'].keys()):
            aggregated_data['step'].append(step)
            aggregated_data['avg_entropy'].append(np.mean(self.history['entropy'][step]))
            aggregated_data['avg_step_ppl'].append(np.mean(self.history['step_ppl'][step]))
            aggregated_data['avg_semantic_convergence'].append(np.mean(self.history['semantic_convergence'][step]))
        
        df_traj = pd.DataFrame(aggregated_data)
        filename = f"trajectory_report_{variant_name}.csv"
        df_traj.to_csv(filename, index=False)
        print(f"✅ Saved step trajectory report to {filename}")
        return df_traj

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
        self.eos_token_id = self.tokenizer.eos_token_id
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
        prompt_lengths = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating answers for {model_type.upper()}"):
                prompts = batch['input_ids'].to(self.device)
                batch_prompt_lengths = (batch['attention_mask'].sum(dim=1) - 1).tolist()  # Exclude EOS token
                if model_type == "ddm":
                    # For DDM, we need to run the unmasking loop
                    outputs = current_model.generate(prompts, n_tokens=100, num_steps=20)
                else:
                    outputs = current_model.generate(prompts, max_length=100)
                generated_answers.extend(outputs.cpu().numpy())
                prompt_lengths.extend(batch_prompt_lengths)
        gc.collect()
        torch.cuda.empty_cache()
        return generated_answers, prompt_lengths

    def decode_tokens_to_text(self, token_list, prompt_lengths=None):
        """
        Transforms a list of token arrays into human-readable text.
        If prompt_lengths (list) is provided, it slices off the prompt for each sequence.
        """
        text_answers = []
        
        # Enumerate gives us the index 'i' to match the tokens with their specific prompt length
        for i, tokens in enumerate(tqdm(token_list, desc="Decoding tokens to text")):
            if prompt_lengths is not None:
                p_len = prompt_lengths[i]
                response_tokens = tokens[p_len:]
            else:
                response_tokens = tokens
                
            # Decode the tokens back into a string, ignoring padding and special tokens
            text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            text_answers.append(text)
            
        return text_answers

    def build_token_dataframe(self, save_path="benchmark_tokens.pkl"):
        """
        Generates responses across all models and stores the raw token arrays 
        (without the prompts) in a Pandas DataFrame.
        """
        print("\n👩‍💻 Building the Token-based Pandas Dataset...")
        
        contexts = []
        true_answers = []
        
        # 1. Extraction from dataloader
        print("📝 Extracting Context and True Answers...")
        for batch in self.val_dataloader_ar:
            input_ids = batch['input_ids'].tolist()
            prompt_lengths = (batch['attention_mask'].sum(dim=1) - 1).tolist() 
            
            for seq, p_len in zip(input_ids, prompt_lengths):
                # Separate context tokens from the true answer tokens
                contexts.append(seq[:p_len])
                true_answers.append(seq[p_len:])

        # 2. Token generation and slicing
        print("\n🤖 Generating tokens for AR...")
        ar_tokens, ar_lengths = self.generate_token_answers(model_type="ar")
        ar_answers = [seq[p:] for seq, p in zip(ar_tokens, ar_lengths)]
        
        print("\n🤖 Generating tokens for DDM Uniform...")
        ddm_unif_tokens, ddm_unif_lengths = self.generate_token_answers(model_type="ddm", model_variant="unif")
        ddm_unif_answers = [seq[p:] for seq, p in zip(ddm_unif_tokens, ddm_unif_lengths)]
        
        print("\n🤖 Generating tokens for DDM Positional Dependent...")
        ddm_posdip_tokens, ddm_posdip_lengths = self.generate_token_answers(model_type="ddm", model_variant="posdip")
        ddm_posdip_answers = [seq[p:] for seq, p in zip(ddm_posdip_tokens, ddm_posdip_lengths)]
        
        print("\n🤖 Generating tokens for DDM Sigmoid...")
        ddm_sigm_tokens, ddm_sigm_lengths = self.generate_token_answers(model_type="ddm", model_variant="sigm")
        ddm_sigm_answers = [seq[p:] for seq, p in zip(ddm_sigm_tokens, ddm_sigm_lengths)]

        # 3. Aligning and saving
        min_len = min(len(contexts), len(ar_answers), len(ddm_unif_answers), len(ddm_posdip_answers), len(ddm_sigm_answers))
        
        data = {
            "context": contexts[:min_len],
            "true answer": true_answers[:min_len],
            "AR answer": ar_answers[:min_len],
            "DiT uniform": ddm_unif_answers[:min_len],
            "DiT positional dependent": ddm_posdip_answers[:min_len],
            "DiT sigmoid": ddm_sigm_answers[:min_len]
        }

        df_tokens = pd.DataFrame(data)
        
        # Use to_pickle to preserve the list structure of tokens
        df_tokens.to_pickle(save_path)
        print(f"\n✅ Token Pandas Dataset successfully saved to: {save_path}")
        
        return df_tokens
    
    def build_text_dataframe(self, df_tokens=None, load_path="benchmark_tokens.pkl", save_path="benchmark_text.csv"):
        """
        Converts a token-based Pandas DataFrame into human-readable text and saves it as a CSV.
        """
        print("\n🔤 Converting Token Dataset to Text Dataset...")
        
        # Load the dataframe from disk if not passed directly
        if df_tokens is None:
            print(f"📂 Loading token dataframe from {load_path}...")
            df_tokens = pd.read_pickle(load_path)
            
        df_text = pd.DataFrame()
        
        for column in df_tokens.columns:
            print(f"Decoding column: '{column}'...")
            # Since we already sliced away the prompts in the previous method, 
            # we can pass prompt_lengths=None to the decoding function
            col_tokens = df_tokens[column].tolist()
            df_text[column] = self.decode_tokens_to_text(col_tokens, prompt_lengths=None)
            
        df_text.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\n✅ Text Pandas Dataset successfully saved to: {save_path}")
        
        return df_text

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

        df_ppx = pd.DataFrame(list(self.ppx_values.items()), columns=['Model', 'Perplexity'])
        df_ppx.to_csv("perplexity_results.csv", index=False)

    def run_structure_evaluation(self, df_tokens_path="benchmark_tokens.pkl"):
        """Runs the ChatStructureEvaluator on the offline token DataFrame."""
        print("\n🔍 Initializing Structure Evaluator...")
        try:
            df_tokens = pd.read_pickle(df_tokens_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_tokens_path}. Generate tokens first!")
            return
            
        evaluator = ChatStructureEvaluator(eos_token_id=self.eos_token_id)
        evaluator.evaluate_dataframe(df_tokens)
        
        collector_df = evaluator.print_report()
        collector_df.to_csv("structure_evaluation_results.csv", index=False)

    def run_diversity_evaluation(self, df_tokens_path="benchmark_tokens.pkl"):
        """Evaluates diversity using the Token DataFrame (requires token arrays)."""
        try:
            df = pd.read_pickle(df_tokens_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_tokens_path}.")
            return
            
        de = DiversityEvaluator()
        df_results = de.evaluate_dataframe(df_tokens=df)  
        df_results.to_csv("diversity_evaluation_results.csv", index=False)

    def run_semantic_evaluation(self, df_text_path="benchmark_text.csv"):
        """Evaluates semantic distribution using the Text DataFrame."""
        try:
            df = pd.read_csv(df_text_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_text_path}.")
            return
            
        se = SemanticEvaluator(bert_model_name="bert-base-uncased", device=self.device)
        df_results = se.evaluate_dataframe(df_text=df)  
        df_results.to_csv("semantic_evaluation_results.csv", index=False)
        
    def run_llm_judge_evaluation(self, df_text_path="benchmark_text.csv"):
        """Runs the 4-way Ollama LLM Judge on the Text DataFrame."""
        try:
            df = pd.read_csv(df_text_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_text_path}.")
            return
            
        judge = LLMJudgeEvaluator(model_name="llama3", base_url="http://localhost:11434/v1")
        df_results = judge.evaluate_dataframe(df_text=df)
        df_results.to_csv("llm_judge_results.csv", index=False)

    def run_trajectory_evaluation(self, num_steps=20):
        """
        Evaluates step-by-step unmasking trajectories (Entropy, Step-PPL, Semantic Convergence)
        for all DDM variants while models are loaded on GPU.
        """
        print("\n" + "="*50)
        print("📈 RUNNING DDM TRAJECTORY EVALUATION")
        print("="*50)
        
        ddm_variants = ["unif", "sigm", "posdip"]
        
        for variant in ddm_variants:
            print(f"\n🔄 Tracking trajectory for DDM variant: {variant}...")
            model = self.model("ddm", model_variant=variant)
            model.eval()
            
            evaluator = DiffusionTrajectoryEvaluator(mask_token_id=self.tokenizer.mask_token_id)
            
            with torch.no_grad():
                for batch in tqdm(self.val_dataloader_dit, desc=f"Evaluating Trajectory ({variant})"):
                    prompts = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Run generation while feeding step metrics to the evaluator
                    model.generate(
                        prompts, 
                        token_start_idx=batch['token_start_idx'], 
                        num_steps=num_steps, 
                        evaluator=evaluator, 
                        target_labels=labels
                    )
                    evaluator.finalize_batch()
            
            evaluator.export_dataframe(variant_name=variant)
            
            # Memory Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()

class GraphsGenerator:
    """
    Generates visualizations for the benchmark results.
    """
    def __init__(self):
        pass

    def generate_perplexity_graph(self, csv_path="perplexity_results.csv"):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['Model'], df['Perplexity'], color=['blue', 'orange', 'green', 'red'])
        plt.title("Perplexity Comparison Across Models")
        plt.ylabel("Perplexity")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("perplexity_comparison.png")
        print("✅ Perplexity graph saved as perplexity_comparison.png")

    def generate_structure_evaluation_graph(self, csv_path="structure_evaluation_results.csv"):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['avg_points_per_prompt'], color=['blue', 'orange', 'green', 'red'])
        plt.title("Structure Evaluation Scores Across Models")
        plt.ylabel("Average Points per Prompt")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("structure_evaluation_comparison.png")
        print("✅ Structure evaluation graph saved as structure_evaluation_comparison.png")

    def generate_diversity_evaluation_graph(self, csv_path="diversity_evaluation_results.csv"):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['diversity_score'], color=['blue', 'orange', 'green', 'red'])
        plt.title("Diversity Evaluation Scores Across Models")
        plt.ylabel("Diversity Score")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("diversity_evaluation_comparison.png")
        print("✅ Diversity evaluation graph saved as diversity_evaluation_comparison.png")

    def generate_semantic_evaluation_graph(self, csv_path="semantic_evaluation_results.csv"):
        # Bert Score
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['bertscore'], color=['blue', 'orange', 'green', 'red'])
        plt.title("Semantic Evaluation (BERTScore) Across Models")
        plt.ylabel("BERTScore (F1)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("semantic_evaluation_comparison.png")
        print("✅ Semantic evaluation graph saved as semantic_evaluation_comparison.png")
        # Frechet BERT Distance
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['fbd_score'], color=['blue', 'orange', 'green', 'red'])
        plt.title("Semantic Evaluation (Fréchet BERT Distance) Across Models")
        plt.ylabel("Fréchet BERT Distance (Lower is Better)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("semantic_evaluation_fbd_comparison.png")
        print("✅ Semantic evaluation (FBD) graph saved as semantic_evaluation_fbd_comparison.png")

    def generate_llm_judge_graph(self, csv_path="llm_judge_results.csv"):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['total_points'], color=['blue', 'orange', 'green', 'red'])
        plt.title("LLM Judge Evaluation (Borda Count) Across Models")
        plt.ylabel("Total Points")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("llm_judge_evaluation_comparison.png")
        print("✅ LLM Judge evaluation graph saved as llm_judge_evaluation_comparison.png")

    def generate_trajectory_graphs(self):
        """
        Reads the trajectory CSVs for the 3 DDM variants and generates 
        3 separate Matplotlib figures for Entropy, Step-Perplexity, and Semantic Convergence.
        """
        print("\n📊 Generating Separate Trajectory Graphs...")

        # Define the exact filenames created by our trajectory evaluator
        files = {
            "Uniform": "trajectory_report_unif.csv",
            "Sigmoid": "trajectory_report_sigm.csv",
            "Positional Dependent": "trajectory_report_posdip.csv"
        }

        # Setup professional colors and markers for the 3 lines
        styles = {
            "Uniform": {"color": "#4C72B0", "marker": "o"},             # Deep Blue
            "Sigmoid": {"color": "#DD8452", "marker": "s"},             # Warm Orange
            "Positional Dependent": {"color": "#55A868", "marker": "^"} # Soft Green
        }

        dataframes = {}
        for name, filename in files.items():
            if os.path.exists(filename):
                dataframes[name] = pd.read_csv(filename)
            else:
                print(f"❌ Error: Could not find {filename}. Run run_trajectory_evaluation first!")
                return

        # Define what we are plotting in each of the 3 separate graphs
        metrics = [
            {
                "column": "avg_entropy",
                "title": "Masked Entropy Trajectory",
                "ylabel": "Shannon Entropy (Lower = More Confident)",
                "filename": "ddm_trajectory_entropy.pdf"
            },
            {
                "column": "avg_step_ppl",
                "title": "Pseudo-Perplexity Trajectory",
                "ylabel": "Step-Perplexity (Lower = Better)",
                "filename": "ddm_trajectory_perplexity.pdf"
            },
            {
                "column": "avg_semantic_convergence",
                "title": "Semantic Convergence Trajectory",
                "ylabel": "Cosine Similarity (Closer to 1.0 = Better)",
                "filename": "ddm_trajectory_semantic.pdf"
            }
        ]

        # Generate and save a separate graph for each metric
        for metric in metrics:
            plt.figure(figsize=(8, 6)) # Perfect size for individual slides or paper columns
            
            for variant_name, df in dataframes.items():
                plt.plot(
                    df['step'], 
                    df[metric["column"]], 
                    label=variant_name,
                    color=styles[variant_name]["color"], 
                    marker=styles[variant_name]["marker"], 
                    linewidth=2.5,
                    markersize=7
                )

            # Dress up the graph
            plt.title(metric["title"], fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Generative Steps (t)", fontsize=12)
            plt.ylabel(metric["ylabel"], fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=11)

            # Ensure the layout is tight so labels don't overlap, then save
            plt.tight_layout()
            plt.savefig(metric["filename"], format='pdf', bbox_inches='tight')
            plt.close() # Keep matplotlib memory clean
            
            print(f"✅ Saved graph: {metric['filename']}")
            
        print("🎉 All 3 trajectory graphs successfully generated!")


if __name__ == "__main__":
    manager = BenchmarkManager(config_path="config.yaml")

    manager.tokenize_and_group()
    manager.build_token_dataframe(save_path="benchmark_tokens.pkl")
    manager.build_text_dataframe(df_tokens="benchmark_tokens.pkl", save_path="benchmark_text.csv")
    manager.run_perplexity_benchmark()
    manager.run_structure_evaluation(df_tokens_path="benchmark_tokens.pkl")  # Set to None to generate tokens
    manager.run_diversity_evaluation(df_tokens_path="benchmark_tokens.pkl")
    manager.run_semantic_evaluation(df_text_path="benchmark_text.csv")
    manager.run_llm_judge_evaluation(df_text_path="benchmark_text.csv")
    manager.run_trajectory_evaluation(num_steps=20)

    gg = GraphsGenerator()

    gg.generate_perplexity_graph()
    gg.generate_structure_evaluation_graph()
    gg.generate_diversity_evaluation_graph()
    gg.generate_semantic_evaluation_graph()
    gg.generate_llm_judge_graph()
    gg.generate_trajectory_graphs()