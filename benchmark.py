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
import noise.masking_schedule as masking_schedule
import noise.noise_schedule as noise_schedule

from models.AR import AR
from models.DiT import DiT
from GPT_Lightning import GPT
from diffusion_lightning import Diffusion

class Perplexity:
    def __init__(self, model, model_type='ar', mask_token_id=103, vocab_size=30522):
        self.model = model
        self.model_type = model_type
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.device = getDevice()

    def _evaluate_ar_batch(self, input_ids, labels):
        with torch.no_grad():
            logits = self.model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100, 
                reduction='sum'
            )   
            valid_tokens = (shift_labels != -100).sum().item()
        return loss.item(), valid_tokens
    
    def _scatter_to_answer(self, values, ans_start_idx, T, T_ans, fill_value=0.0):
        """A self-contained helper to scatter values into the answer block of a tensor."""
        B = values.shape[0]
        out = torch.full((B, T), fill_value, dtype=values.dtype, device=values.device)
        positions = ans_start_idx[:, None] + torch.arange(T_ans, device=values.device, dtype=torch.long)[None, :]
        rows = torch.arange(B, device=values.device)[:, None]
        out[rows, positions] = values
        return out

    def _evaluate_ddm_batch(self, input_ids, labels, ans_start_idx):
        batch_size, seq_len = input_ids.shape
        with torch.no_grad():
            t = torch.rand(batch_size, device=self.device)
            move_chance, loss_weight = self.model.masking_schedule(t)

            # --- CORRECTED LOGIC ---
            # Use a scatter operation to correctly align probabilities with the answer block.
            T_ans = self.model.T_ans
            full_move_chance = self._scatter_to_answer(move_chance, ans_start_idx, seq_len, T_ans, fill_value=0.0)

            # Handle both (B, 1) and (B, T_ans) shapes for loss_weight
            if loss_weight.dim() == 1 or loss_weight.shape[1] == 1:
                expanded_weight = loss_weight.expand(-1, T_ans)
                full_loss_weight = self._scatter_to_answer(expanded_weight, ans_start_idx, seq_len, T_ans, fill_value=1.0)
            else: # Assumes shape is (B, T_ans)
                full_loss_weight = self._scatter_to_answer(loss_weight, ans_start_idx, seq_len, T_ans, fill_value=1.0)
            # --- END CORRECTION ---

            is_response_token = (labels != -100)
            rand_matrix = torch.rand(batch_size, seq_len, device=self.device)
            
            mask_bool = is_response_token & (rand_matrix < full_move_chance)
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_bool] = self.mask_token_id
            
            ddm_labels = labels.clone()
            ddm_labels[~mask_bool] = -100 
            
            sigma, _ = self.model.noise(t)
            logits = self.model(masked_input_ids, sigma=sigma)
            
            # CORRECTED: Use nll_loss because the model's forward pass already returns log-probabilities.
            # Using cross_entropy would incorrectly apply log_softmax a second time.
            loss_per_token = F.nll_loss(logits.view(-1, self.vocab_size), ddm_labels.view(-1), ignore_index=-100, reduction='none')
            raw_batch_loss = loss_per_token.sum().item()
            
            loss_per_seq = loss_per_token.view(batch_size, seq_len)
            weighted_loss_per_seq = loss_per_seq * full_loss_weight
            elbo_batch_loss = weighted_loss_per_seq.sum().item()
            
            valid_masked_tokens = (ddm_labels != -100).sum().item()
            
        return elbo_batch_loss, raw_batch_loss, valid_masked_tokens

    def calculate(self, val_dataloader):
        self.model.eval()
        if self.model_type == 'ar':
            print("🚀 Starting Autoregressive Perplexity Evaluation...")
            ar_total_loss, ar_total_tokens = 0.0, 0
            for batch in tqdm(val_dataloader, desc="Processing AR Batches"):
                input_ids, labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)

                ar_loss, ar_tokens = self._evaluate_ar_batch(input_ids, labels)
                ar_total_loss += ar_loss
                ar_total_tokens += ar_tokens
            ar_mean_loss = ar_total_loss / max(ar_total_tokens, 1)
            perplexity = torch.exp(torch.tensor(ar_mean_loss))

        elif self.model_type == 'ddm':
            print("🚀 Starting Diffusion Model Perplexity Evaluation...")
            ddm_total_elbo = 0.0
            ddm_total_raw = 0.0
            ddm_total_tokens = 0
            
            for batch in tqdm(val_dataloader, desc="Processing DDM Batches"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                ans_start_idx = batch['ans_start_idx'].to(self.device)


                elbo_loss, raw_loss, ddm_tokens = self._evaluate_ddm_batch(input_ids, labels, ans_start_idx)
                    
                ddm_total_elbo += elbo_loss
                ddm_total_raw += raw_loss
                ddm_total_tokens += ddm_tokens
                
            # The highly compressed schedule metric (will be ~1.01)
            elbo_mean_loss = ddm_total_elbo / max(ddm_total_tokens, 1)
            elbo_perplexity = torch.exp(torch.tensor(elbo_mean_loss))
            
            # The FAIR comparison metric (will be comparable to your AR model's 3.76)
            raw_mean_loss = ddm_total_raw / max(ddm_total_tokens, 1)
            raw_perplexity = torch.exp(torch.tensor(raw_mean_loss))
            
            print(f"✅ DDM Evaluation Completed!")
            print(f"📊 DDM ELBO Perplexity (Schedule Health): {elbo_perplexity.item():.2f}")
            print(f"📊 DDM Raw Perplexity (Fair Comparison):  {raw_perplexity.item():.2f}")
            perplexity = raw_perplexity
            
        print(f"✅ Evaluation Completed! Global Dataset Perplexity: {perplexity.item():.2f}")
        return perplexity.item()

class ChatStructureEvaluator:
    def __init__(self, eos_token_id=50282):
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

    def _evaluate_single_sequence(self, response_tokens, max_gen_len=256, pad_token_id=50283):
        eos_flag = False
        length = 0
        gen_token = 0
        for idx in range(len(response_tokens)):
            if response_tokens[idx] == self.eos_token_id:  # EOS token
                eos_flag = True
                length = idx + 1
                break
            elif idx == max_gen_len - 1:
                length = max_gen_len
            elif response_tokens[idx] != pad_token_id:  # Padding token
                idx += 1
        return {'length': length, 'missing_eos': not eos_flag}

    def evaluate_dataframe(self, df_tokens):
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
        lengths_dict = {} # <-- NEW: We will collect the raw lengths here
        
        for col, data in self.results.items():
            lengths = data.get('lengths', [])
            if not lengths:
                continue
                
            # Save the raw lengths for our new dataframe
            lengths_dict[col] = lengths
            
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
            
        df_lengths = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lengths_dict.items()]))
        
        return pd.DataFrame(collected_data), df_lengths

class DiversityEvaluator:
    def __init__(self, sample_size_for_bleu=1000, n_gram_size=2):
        self.sample_size_for_bleu = sample_size_for_bleu
        self.n_gram_size = n_gram_size
        self.smoother = SmoothingFunction().method1

    def _get_ngrams(self, sequence, n):
        return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

    def calculate_unique_ratio(self, list_of_token_sequences):
        all_ngrams = []
        for seq in list_of_token_sequences:
            all_ngrams.extend(self._get_ngrams(seq, self.n_gram_size))
        if not all_ngrams:
            return 0.0
        unique_ngrams = set(all_ngrams)
        ratio = len(unique_ngrams) / len(all_ngrams)
        return ratio * 100

    def calculate_self_bleu(self, list_of_token_sequences):
        valid_sequences = [seq for seq in list_of_token_sequences if len(seq) > 1]
        if len(valid_sequences) > self.sample_size_for_bleu:
            valid_sequences = random.sample(valid_sequences, self.sample_size_for_bleu)
        total_sentences = len(valid_sequences)
        if total_sentences < 2:
            return 0.0
        bleu_scores = []
        for i in tqdm(range(total_sentences), desc="Calculating Self-BLEU"):
            hypothesis = valid_sequences[i]
            references = valid_sequences[:i] + valid_sequences[i+1:]
            score = sentence_bleu(references, hypothesis, smoothing_function=self.smoother)
            bleu_scores.append(score)
        return np.mean(bleu_scores) * 100

    def evaluate_dataframe(self, df_tokens):
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
    def __init__(self, bert_model_name="bert-base-uncased", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.bert_model_name = bert_model_name
        print(f"Loading BERT evaluator ({bert_model_name}) into {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.model.eval()

    def calculate_bertscore(self, references, hypotheses):
        print("Calculating BERTScore...")
        P, R, F1 = bert_score_calc(cands=hypotheses, refs=references, lang="en", model_type=self.bert_model_name, device=self.device, verbose=False)
        return F1.mean().item() * 100

    def _get_sentence_embeddings(self, texts, batch_size=32):
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def calculate_frechet_bert_distance(self, prompts, real_responses, generated_responses):
        print("Extracting embeddings for Fréchet Distance...")
        real_conversations = [f"{p} [SEP] {r}" for p, r in zip(prompts, real_responses)]
        generated_conversations = [f"{p} [SEP] {g}" for p, g in zip(prompts, generated_responses)]
        real_embs = self._get_sentence_embeddings(real_conversations)
        gen_embs = self._get_sentence_embeddings(generated_conversations)
        mu_real, sigma_real = np.mean(real_embs, axis=0), np.cov(real_embs, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_embs, axis=0), np.cov(gen_embs, rowvar=False)
        print("Computing Fréchet Math...")
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2.0 * covmean)

    def evaluate_dataframe(self, df_text):
        print("\n" + "="*50)
        print("🧠 SEMANTIC & DISTRIBUTION REPORT (BERT-BASED)")
        print("="*50)
        prompts = df_text['context'].tolist()
        real_resp = df_text['true answer'].tolist()
        cols_to_evaluate = ['AR answer', 'DiT uniform', 'DiT sigmoid', 'DiT positional dependent']
        results = {'model': [], 'bertscore': [], 'fbd_score': []}
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
    def __init__(self, model_name="gemma4", base_url="http://localhost:11434/v1"):
        self.client = OpenAI(api_key="gemma4", base_url=base_url)
        self.model_name = model_name
        self.results = {'AR answer': 0, 'DiT uniform': 0, 'DiT positional dependent': 0, 'DiT sigmoid': 0, 'Errors': 0}

    def _get_system_prompt(self):
        return """You are an impartial, expert and very fast evaluator of AI conversational models.
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
        print(f"\n🚀 Starting 4-Way Ranked LLM-as-a-Judge with Ollama ({self.model_name})...")
        models_cols = ['AR answer', 'DiT uniform', 'DiT positional dependent', 'DiT sigmoid']
        points_map = [3, 2, 1, 0]
        n_samples = 0
        MAX_SAMPLES = 100
        for idx, row in tqdm(df_text.iterrows(), total=MAX_SAMPLES, desc="Evaluating Rows"):
            prompt = str(row['context'])
            responses = {col: str(row[col]) for col in models_cols}
            shuffled_cols = list(responses.keys())
            random.shuffle(shuffled_cols)
            letter_to_model = {"A": shuffled_cols[0], "B": shuffled_cols[1], "C": shuffled_cols[2], "D": shuffled_cols[3]}
            user_message = f"[USER PROMPT]\n{prompt}\n\n"
            for letter, col_name in letter_to_model.items():
                user_message += f"[MODEL {letter} RESPONSE]\n{responses[col_name]}\n\n"
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=[{"role": "system", "content": self._get_system_prompt()}, {"role": "user", "content": user_message}],
                    response_format={"type": "json_object"}, temperature=0.0
                )
                result_json = json.loads(response.choices[0].message.content)
                ranking = result_json.get("ranking", [])
                if len(ranking) == 4 and set(ranking) == {"A", "B", "C", "D"}:
                    for rank_idx, letter in enumerate(ranking):
                        actual_model = letter_to_model[letter]
                        self.results[actual_model] += points_map[rank_idx]
                else:
                    self.results['Errors'] += 1
            except Exception as e:
                self.results['Errors'] += 1
            n_samples += 1
            if n_samples >= MAX_SAMPLES+1:  # Limit to MAX_SAMPLES samples for evaluation
                break
        return self.print_report(len(df_text))

    def print_report(self, total_evaluated):
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
        sorted_models = sorted(models_cols, key=lambda m: self.results[m], reverse=True)
        for rank, col in enumerate(sorted_models, 1):
            total_pts = self.results[col]
            avg_pts = total_pts / total_valid
            pct_max = (total_pts / max_possible_pts) * 100 if max_possible_pts > 0 else 0.0
            print(f"  #{rank} {col}: {total_pts} pts (Avg: {avg_pts:.2f} pts/prompt | {pct_max:.1f}% of max)")
            collected_data.append({'rank': rank, 'model': col, 'total_points': total_pts, 'avg_points_per_prompt': avg_pts, 'pct_max_points': pct_max})
        print(f"  (Errors/Failures: {self.results['Errors']})")
        return pd.DataFrame(collected_data)

class DiffusionTrajectoryEvaluator:
    def __init__(self, mask_token_id):
        self.mask_token_id = mask_token_id
        self.history = {'entropy': {}, 'step_ppl': {}, 'semantic_convergence': {}}
        self._reset_batch_data()

    def _reset_batch_data(self):
        self._temp_batch_data = {'entropy': {}, 'step_ppl': {}, 'hidden_states': {}}

    def _calculate_entropy(self, logits, masked_indices):
        masked_logits = logits[masked_indices]
        if masked_logits.numel() == 0:
            return 0.0
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.mean().item()

    def _calculate_step_ppl(self, logits, masked_indices, target_labels):
        import torch
        import torch.nn.functional as F

        # To align with the model's sampling behavior, we must explicitly forbid
        # predicting the [MASK] token. The SUBS parameterization does this by
        # setting the logit for the [MASK] token to negative infinity. We
        # replicate that logic here on a clone of the raw logits.
        eval_logits = logits.clone()
        eval_logits[:, :, self.mask_token_id] = -torch.inf

        # 1. Extract predictions for the exact masked positions.
        masked_logits = eval_logits[masked_indices]

        # 2. If no tokens were masked, PPL is undefined. Return 1.0 (loss=0).
        if masked_logits.numel() == 0:
            return 1.0
            
        # 3. Extract the ground truth labels for *those same masked positions*.
        #    This ensures perfect alignment between predictions and targets.
        masked_targets = target_labels[masked_indices]
        
        # 4. Calculate cross-entropy loss. `ignore_index` handles non-answer tokens.
        loss = F.cross_entropy(
            # Promote to float32 for numerical stability during loss calculation.
            # This prevents overflow issues when logits have a large dynamic range,
            # which is common in early diffusion steps.
            masked_logits.float(),
            masked_targets,
            ignore_index=-100,
            reduction='mean'
        )

        # 5. If loss is not finite (e.g., all targets were ignored), return 1.0.
        if not torch.isfinite(loss):
            return 1.0

        return torch.exp(loss).item()

    def update_step(self, step, logits, last_hidden_state, current_input_ids, target_labels):
        """
        Update the evaluator with metrics for a specific diffusion step.
        arguments:
            step (int): The current diffusion step.
            logits (torch.Tensor): The model's output logits for the current step.
            last_hidden_state (torch.Tensor): The last hidden state from the model for the current step.
            current_input_ids (torch.Tensor): The input IDs at the current step, including masked tokens.
            target_labels (torch.Tensor): The true labels corresponding to the input IDs, with -100 for non-target positions.
        """
        masked_indices = (current_input_ids == self.mask_token_id)
        self._temp_batch_data['entropy'][step] = self._calculate_entropy(logits, masked_indices)
        self._temp_batch_data['step_ppl'][step] = self._calculate_step_ppl(logits, masked_indices, target_labels)
        pooled_state = last_hidden_state.mean(dim=1).detach().cpu()
        self._temp_batch_data['hidden_states'][step] = pooled_state

    def finalize_batch(self):
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

    def export_dataframe(self, variant_name="unif", save_dir="."):
        aggregated_data = {'step': [], 'avg_entropy': [], 'avg_step_ppl': [], 'avg_semantic_convergence': []}
        for step in sorted(self.history['entropy'].keys()):
            aggregated_data['step'].append(step)
            aggregated_data['avg_entropy'].append(np.mean(self.history['entropy'][step]))
            aggregated_data['avg_step_ppl'].append(np.mean(self.history['step_ppl'][step]))
            aggregated_data['avg_semantic_convergence'].append(np.mean(self.history['semantic_convergence'][step]))
        df_traj = pd.DataFrame(aggregated_data)
        
        # Save dynamically to the provided directory
        full_path = os.path.join(save_dir, f"trajectory_report_{variant_name}.csv")
        df_traj.to_csv(full_path, index=False)
        print(f"✅ Saved step trajectory report to {full_path}")
        return df_traj

class BenchmarkManager:
    def __init__(self, config_path="config.yaml"):
        self.config = OmegaConf.load(config_path)
        self.caching_dir = self.config.caching_directory 
        self.device = getDevice()
        
        # --- NEW: Automated Directory Management ---
        self.results_dir = "benchmark_results"
        self.dirs = {
            'datasets': os.path.join(self.results_dir, "datasets"),
            'metrics': os.path.join(self.results_dir, "metrics")
        }
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)
            
        self.test_ds = self.load_smoltal_test()
        self.T, self.T_context, self.T_answer = self.config.backbone.T, self.config.backbone.T_ctx, self.config.backbone.T_ans
        self.B, self.C, self.H, self.N = self.config.backbone.B, self.config.backbone.C, self.config.backbone.H, self.config.backbone.N
        self.lr, self.wup_steps = self.config.training.learning_rate, self.config.training.warmup_steps

        self.tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-150m")
        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.dm_qa = DataManagerQA(caching_directory=self.caching_dir, tokenizer=self.tokenizer, n_processes=4)
        
        self.ar_model_ckpt = "checkpoints/AR-epoch=07-val_loss=1.3041.ckpt/AR-epoch=07-val_loss=1.3041.ckpt"
        self.ddm_model_unif_ckpt = "checkpoints/DiT-independent-epoch=09-val_loss=1.7500.ckpt/DiT-independent-epoch=09-val_loss=1.7500.ckpt"
        self.ddm_model_sigm_ckpt = "checkpoints/DiT-moving_sigmoid-epoch=07-val_loss=1.8796.ckpt/DiT-moving_sigmoid-epoch=09-val_loss=0.2141.ckpt"
        self.ddm_model_posdip_ckpt = "checkpoints/DiT-position-epoch=06-val_loss=1.6808.ckpt/DiT-position-epoch=07-val_loss=1.6972.ckpt"   
    
    def load_smoltal_test(self):
        ds = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="test[:1%]", cache_dir=".data")
        return ds
    
    def tokenize_and_group(self):
        print("Tokenizing and grouping dataset...")
        tokenized_ds = self.dm_qa.tokenize(self.test_ds, split_name="test")
        grouped_ds = self.dm_qa.group_texts_ar(tokenized_ds, T=self.T, split_name="test")
        grouped_ds = self.dm_qa.group_texts_dit(tokenized_ds, T_ctx=self.T_context, T_ans=self.T_answer, split_name="test")
        grouped_ds_ar = grouped_ds.rename_column("output_ids", "labels")
        grouped_ds_ar.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
        grouped_ds_dit = grouped_ds.rename_column("output_ids", "labels")
        grouped_ds_dit.set_format(type="torch", columns=["input_ids", "labels", "attention_mask", "ans_start_idx"])
        self.val_dataloader_ar = self.dm_qa.getTrainloader(grouped_ds_ar, B=self.B)
        self.val_dataloader_dit = self.dm_qa.getTrainloader(grouped_ds_dit, B=self.B)

    def model(self, model_to_test="ar", model_variant="unif"):
        models = {"ar": self._upload_models_ar(), "ddm": self._upload_models_ddm(model_variant=model_variant)}
        return models[model_to_test]
    
    def _upload_models_ar(self):
        print("Loading AR model...")
        backbone_ar = AR(V=self.vocab_size, C=self.C, H=self.H, N=self.N)
        model_ar = GPT.load_from_checkpoint(self.ar_model_ckpt, backbone=backbone_ar, tokenizer=self.tokenizer, learning_rate=self.lr, warmup_steps=self.wup_steps, strict=False).to(self.device)
        return model_ar
    
    def _upload_models_ddm(self, model_variant="unif"):
        print("Loading DDM model...")
        model_ckpt_map = {"unif": self.ddm_model_unif_ckpt, "sigm": self.ddm_model_sigm_ckpt, "posdip": self.ddm_model_posdip_ckpt}
        backbone_ddm = DiT(V=self.vocab_size, C=self.C, H=self.H, N=self.N)
    


        noise_name = self.config.diffusion.get("noise_schedule", "loglinear")        # for now wel is supported onlt loglinear
        noise = noise_schedule.get_noise(noise_name)

        corruption_type = self.config.diffusion.get("corruption_type","independent") # independent | position | moving_sigmoid
        pos_weighting   = self.config.diffusion.get("position_loss_weighting", False)
        masking = masking_schedule.Masking(                                     # define the masking strategy
            T_ans                  =self.T_answer,
            noise                  =noise,
            corruption_type        =corruption_type,
            position_loss_weighting=pos_weighting,
            gamma                  =self.config.diffusion.get("position_gamma", None),
            k                      =self.config.diffusion.get("sigmoid_k", None)
        )

        model_kwargs = {'backbone':         backbone_ddm,
                        'tokenizer':        self.tokenizer,
                        'learning_rate':    self.lr,
                        'warmup_steps':     self.wup_steps,
                        'masking_schedule': masking,
                        'T_ans':            self.T_answer}

        self.model_ddm = Diffusion.load_from_checkpoint(
                model_ckpt_map[model_variant],
                strict=False,
                **model_kwargs
            ).to(self.device)

        return self.model_ddm

    def generate_token_answers(self, model_type="ar", model_variant="unif"):
        current_model = self.model(model_type, model_variant=model_variant)
        current_model.eval()
        dataloader = self.val_dataloader_ar if model_type == "ar" else self.val_dataloader_dit
        
        all_outputs = []
        all_prompt_lengths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating answers for {model_type.upper()}"):
                prompts = batch['input_ids'].to(self.device)

                if model_type == "ddm":
                    # Correctly get ans_start_idx from the DDM dataloader
                    ans_start_idx = batch.get('ans_start_idx')
                    if ans_start_idx is not None:
                        ans_start_idx = ans_start_idx.to(self.device)

                    outputs = current_model.generate(prompts, ans_start_idx=ans_start_idx, num_steps=20)
                    
                    # For DDM, prompt length is ans_start_idx. Fallback to shape if not present.
                    if ans_start_idx is None:
                        batch_prompt_lengths = [prompts.shape[1]] * prompts.shape[0]
                    else:
                        batch_prompt_lengths = ans_start_idx.tolist()
                else: # AR model
                    outputs = current_model.generate(prompts, n_tokens=256)
                    # For AR, prompt length is the number of non-pad tokens from the attention mask.
                    batch_prompt_lengths = batch['attention_mask'].sum(dim=1).tolist()
                        
                all_outputs.extend(outputs.cpu().tolist())
                all_prompt_lengths.extend(batch_prompt_lengths)
                gc.collect()
                torch.cuda.empty_cache()
        
        return all_outputs, all_prompt_lengths

    def decode_tokens_to_text(self, token_list, prompt_lengths=None):
        text_answers = []
        for i, tokens in enumerate(tqdm(token_list, desc="Decoding tokens to text")):
            response_tokens = tokens[prompt_lengths[i]:] if prompt_lengths is not None else tokens
            text_answers.append(self.tokenizer.decode(response_tokens, skip_special_tokens=True))
        return text_answers

    def build_dataframe_from_partials(self, save_filename="benchmark_tokens.pkl"):
        import ast
        print("\n👩‍💻 Merging partial datasets from .pkl files...")
        
        contexts = []
        true_answers = []
        
        # 1. Correctly extract Context and True Answer using the -100 label boundary
        for batch in tqdm(self.val_dataloader_ar, desc="Extracting Contexts & Answers"):
            input_ids = batch['input_ids'].tolist()
            labels = batch['labels'].tolist()
            
            for seq, label_seq in zip(input_ids, labels):
                # Find all positions where labels are NOT -100 (this marks the answer)
                non_pad_indices = [i for i, lbl in enumerate(label_seq) if lbl != -100]
                
                if non_pad_indices:
                    ans_start = non_pad_indices[0]
                    ans_end = non_pad_indices[-1] + 1
                    
                    context = seq[:ans_start]
                    true_ans = seq[ans_start:ans_end]
                else:
                    context = seq
                    true_ans = []
                    
                contexts.append(context)
                true_answers.append(true_ans)

        # 2. Define the partial files and load them
        partial_files = {
            "AR answer": "ar_answers.pkl",
            "DiT uniform": "ddm_unif_answers.pkl",
            "DiT positional dependent": "ddm_posdip_answers.pkl",
            "DiT sigmoid": "ddm_sigm_answers.pkl"
        }
        
        loaded_outputs = {}
        for col_name, file_name in partial_files.items():
            file_path = os.path.join(self.dirs['datasets'], file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Missing partial dataset: {file_path}. Please run `build_token_dataframe` first.")
            print(f"📂 Loading {file_name}...")
            # pd.read_pickle returns the DataFrame we saved. We extract the first (and only) column.
            loaded_outputs[col_name] = pd.read_pickle(file_path).iloc[:, 0].tolist()

        # 3. Create context/answer DataFrame and merge everything
        all_data = {
            "context": contexts,
            "true answer": true_answers,
            **loaded_outputs
        }

        # 4. Align all list lengths to the shortest one to create a valid DataFrame
        min_len = min(len(v) for v in all_data.values())
        print(f"📊 Aligning all datasets to the shortest length: {min_len} samples.")
        
        aligned_data = {k: v[:min_len] for k, v in all_data.items()}
        
        df_tokens = pd.DataFrame(aligned_data)

        # 5. Save the final merged dataframe
        full_save_path = os.path.join(self.dirs['datasets'], save_filename)
        df_tokens.to_pickle(full_save_path)
        print(f"✅ Merged token dataset saved to: {full_save_path}")
        
        return df_tokens

    def build_token_dataframe(self, save_filename="benchmark_tokens.pkl"):
        print("\n👩‍💻 Building the Token-based Pandas Dataset...")
        contexts, true_answers = [], []
        for batch in tqdm(self.val_dataloader_ar, desc="Extracting Contexts & True Answers"):
            input_ids = batch['input_ids'].tolist()
            labels = batch['labels'].tolist()
            
            for seq, label_seq in zip(input_ids, labels):
                try:
                    # Find all positions where labels are NOT -100 (this marks the answer)
                    non_pad_indices = [i for i, lbl in enumerate(label_seq) if lbl != -100]
                    ans_start = non_pad_indices[0]
                    ans_end = non_pad_indices[-1] + 1
                    
                    contexts.append(seq[:ans_start])
                    true_answers.append(seq[ans_start:ans_end])
                except StopIteration:
                    # Safe fallback in case of a completely padded sequence
                    contexts.append(seq)
                    true_answers.append([])

        ddm_unif_tokens, ddm_unif_lengths = self.generate_token_answers(model_type="ddm", model_variant="unif")
        ddm_unif_answers = [seq[p:] for seq, p in zip(ddm_unif_tokens, ddm_unif_lengths)]
        ddm_unif_answers_df = pd.DataFrame({"DiT uniform": ddm_unif_answers})
        ddm_unif_answers_df.to_pickle(os.path.join(self.dirs['datasets'], "ddm_unif_answers.pkl"))
        
        ddm_posdip_tokens, ddm_posdip_lengths = self.generate_token_answers(model_type="ddm", model_variant="posdip")
        ddm_posdip_answers = [seq[p:] for seq, p in zip(ddm_posdip_tokens, ddm_posdip_lengths)]
        ddm_posdip_answers_df = pd.DataFrame({"DiT positional dependent": ddm_posdip_answers})
        ddm_posdip_answers_df.to_pickle(os.path.join(self.dirs['datasets'], "ddm_posdip_answers.pkl"))
        
        ddm_sigm_tokens, ddm_sigm_lengths = self.generate_token_answers(model_type="ddm", model_variant="sigm")
        ddm_sigm_answers = [seq[p:] for seq, p in zip(ddm_sigm_tokens, ddm_sigm_lengths)]
        ddm_sigm_answers_df = pd.DataFrame({"DiT sigmoid": ddm_sigm_answers})
        ddm_sigm_answers_df.to_pickle(os.path.join(self.dirs['datasets'], "ddm_sigm_answers.pkl"))

        ar_tokens, ar_lengths = self.generate_token_answers(model_type="ar")
        ar_answers = [seq[p:] for seq, p in zip(ar_tokens, ar_lengths)]
        ar_answers_df = pd.DataFrame({"AR answer": ar_answers})
        ar_answers_df.to_pickle(os.path.join(self.dirs['datasets'], "ar_answers.pkl"))

        min_len = min(len(contexts), len(ar_answers), len(ddm_unif_answers), len(ddm_posdip_answers), len(ddm_sigm_answers))
        data = {
            "context": contexts[:min_len], "true answer": true_answers[:min_len], "AR answer": ar_answers[:min_len],
            "DiT uniform": ddm_unif_answers[:min_len], "DiT positional dependent": ddm_posdip_answers[:min_len], "DiT sigmoid": ddm_sigm_answers[:min_len]
        }
        df_tokens = pd.DataFrame(data)
        full_save_path = os.path.join(self.dirs['datasets'], save_filename)
        df_tokens.to_pickle(full_save_path)
        print(f"\n✅ Token Pandas Dataset successfully saved to: {full_save_path}")
        return df_tokens
    
    def build_text_dataframe(self, load_filename="benchmark_tokens.pkl", save_filename="benchmark_text.csv"):
        print("\n🔤 Converting Token Dataset to Text Dataset...")
        full_load_path = os.path.join(self.dirs['datasets'], load_filename)
        print(f"📂 Loading token dataframe from {full_load_path}...")
        df_tokens = pd.read_pickle(full_load_path)
            
        df_text = pd.DataFrame()
        for column in df_tokens.columns:
            print(f"Decoding column: '{column}'...")
            col_tokens = df_tokens[column].tolist()
            df_text[column] = self.decode_tokens_to_text(col_tokens, prompt_lengths=None)
            
        full_save_path = os.path.join(self.dirs['datasets'], save_filename)
        df_text.to_csv(full_save_path, index=False, encoding='utf-8')
        print(f"\n✅ Text Pandas Dataset successfully saved to: {full_save_path}")
        return df_text

    def run_perplexity_benchmark(self, save_filename="perplexity_results.csv"):
        self.ppx_values = {"AR": None, "DDM_uniform": None, "DDM_sigmoid": None, "DDM_pos_dip": None}
        
        print("Calculating Perplexity for AR model...")
        ppx = Perplexity(model=self.model("ar").backbone, model_type='ar', mask_token_id=self.tokenizer.mask_token_id, vocab_size=self.vocab_size)
        self.ppx_values['AR'] = ppx.calculate(self.val_dataloader_ar)

        print("Calculating Perplexity for DDM model, uniform...")
        ppx = Perplexity(model=self.model("ddm", model_variant="unif"), model_type='ddm', mask_token_id=self.tokenizer.mask_token_id, vocab_size=self.vocab_size)
        self.ppx_values['DDM_uniform'] = ppx.calculate(self.val_dataloader_dit)

        print("Calculating Perplexity for DDM model, sigmoid...")
        ppx = Perplexity(model=self.model("ddm", model_variant="sigm"), model_type='ddm', mask_token_id=self.tokenizer.mask_token_id, vocab_size=self.vocab_size)
        self.ppx_values['DDM_sigmoid'] = ppx.calculate(self.val_dataloader_dit)

        print("Calculating Perplexity for DDM model, positional dependent...")
        ppx = Perplexity(model=self.model("ddm", model_variant="posdip"), model_type='ddm', mask_token_id=self.tokenizer.mask_token_id, vocab_size=self.vocab_size)
        self.ppx_values['DDM_pos_dip'] = ppx.calculate(self.val_dataloader_dit)

        df_ppx = pd.DataFrame(list(self.ppx_values.items()), columns=['Model', 'Perplexity'])
        full_save_path = os.path.join(self.dirs['metrics'], save_filename)
        df_ppx.to_csv(full_save_path, index=False)
        print(f"✅ Saved perplexity results to {full_save_path}")

    def run_structure_evaluation(self, load_filename="benchmark_tokens.pkl", save_filename="structure_evaluation_results.csv"):
        print("\n🔍 Initializing Structure Evaluator...")
        full_load_path = os.path.join(self.dirs['datasets'], load_filename)
        try:
            df_tokens = pd.read_pickle(full_load_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {full_load_path}. Generate tokens first!")
            return
            
        evaluator = ChatStructureEvaluator(eos_token_id=self.eos_token_id)
        evaluator.evaluate_dataframe(df_tokens)
        collector_df, lengths_df = evaluator.print_report()
        
        full_save_path = os.path.join(self.dirs['metrics'], save_filename)
        full_save_path2 = os.path.join(self.dirs['metrics'], "structure_evaluation_lengths.csv")
        collector_df.to_csv(full_save_path, index=False)
        lengths_df.to_csv(full_save_path2, index=False)
        print(f"✅ Saved structure results to {full_save_path}")
        print(f"✅ Saved structure lengths to {full_save_path2}")

    def run_diversity_evaluation(self, load_filename="benchmark_tokens.pkl", save_filename="diversity_evaluation_results.csv"):
        full_load_path = os.path.join(self.dirs['datasets'], load_filename)
        try:
            df = pd.read_pickle(full_load_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {full_load_path}.")
            return
            
        de = DiversityEvaluator()
        df_results = de.evaluate_dataframe(df_tokens=df)
        
        full_save_path = os.path.join(self.dirs['metrics'], save_filename)
        df_results.to_csv(full_save_path, index=False)
        print(f"✅ Saved diversity results to {full_save_path}")

    def run_semantic_evaluation(self, load_filename="benchmark_text.csv", save_filename="semantic_evaluation_results.csv"):
        full_load_path = os.path.join(self.dirs['datasets'], load_filename)
        try:
            df = pd.read_csv(full_load_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {full_load_path}.")
            return

        df.fillna("", inplace=True)
            
        se = SemanticEvaluator(bert_model_name="bert-base-uncased", device=self.device)
        df_results = se.evaluate_dataframe(df_text=df)
        
        full_save_path = os.path.join(self.dirs['metrics'], save_filename)
        df_results.to_csv(full_save_path, index=False)
        print(f"✅ Saved semantic results to {full_save_path}")
        
    def run_llm_judge_evaluation(self, load_filename="benchmark_text.csv", save_filename="llm_judge_results.csv"):
        full_load_path = os.path.join(self.dirs['datasets'], load_filename)
        try:
            df = pd.read_csv(full_load_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {full_load_path}.")
            return

        import socket, subprocess, time

        # --- Automated Server Management ---
        server_process = None
        
        # First, check if the server is already running.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_server_running = s.connect_ex(('localhost', 11434)) == 0

        if not is_server_running:
            print("LLM Judge server not found. Starting it automatically...")
            command = "ollama run gemma4"
            
            # Use Popen to run the command in the background, hiding its output.
            server_process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Server process started with PID: {server_process.pid}. Waiting for it to become ready...")

            # Poll the server until it's ready or we time out.
            max_wait_seconds = 30
            wait_start_time = time.time()
            server_ready = False
            while time.time() - wait_start_time < max_wait_seconds:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_check:
                    if s_check.connect_ex(('localhost', 11434)) == 0:
                        print("✅ Server is ready.")
                        server_ready = True
                        break
                time.sleep(1)
            
            if not server_ready:
                print(f"❌ Server failed to start within {max_wait_seconds} seconds. Aborting.")
                if server_process:
                    server_process.terminate()
                    server_process.wait()
                return
        else:
            print("LLM Judge server is already running. Proceeding with evaluation.")

        try:
            # This is the main evaluation logic.
            judge = LLMJudgeEvaluator(model_name="gemma4", base_url="http://localhost:11434/v1")
            df_results = judge.evaluate_dataframe(df_text=df)
            
            full_save_path = os.path.join(self.dirs['metrics'], save_filename)
            df_results.to_csv(full_save_path, index=False)
            print(f"✅ Saved LLM Judge results to {full_save_path}")

        finally:
            # This block guarantees that if we started the server, we also stop it.
            if server_process:
                print("\nShutting down the automatically started LLM Judge server...")
                server_process.terminate()
                try:
                    # Wait for a few seconds for the process to terminate gracefully
                    server_process.wait(timeout=5)
                    print("Server shut down successfully.")
                except subprocess.TimeoutExpired:
                    print("Server did not terminate gracefully. Forcing shutdown...")
                    server_process.kill()
                    print("Server process killed.")

    def run_trajectory_evaluation(self, num_steps=20, max_samples=100):
        print("\n" + "="*50)
        print(f"📈 RUNNING DDM TRAJECTORY EVALUATION (LIMIT: {max_samples} SAMPLES)")
        print("="*50)
        
        ddm_variants = ["unif", "sigm", "posdip"]
        for variant in ddm_variants:
            print(f"\n🔄 Tracking trajectory for DDM variant: {variant}...")
            model = self.model("ddm", model_variant=variant)
            model.eval()
            
            evaluator = DiffusionTrajectoryEvaluator(mask_token_id=self.tokenizer.mask_token_id)
            
            samples_processed = 0
            
            with torch.no_grad():
                for batch in tqdm(self.val_dataloader_dit, total=max_samples, desc=f"Evaluating Trajectory ({variant})"):
                    
                    # 1. Determine how many samples are needed to reach exactly max_samples
                    needed = max_samples - samples_processed
                    
                    # 2. Slice the tensors so we don't overshoot the limit on the final batch
                    prompts = batch['input_ids'][:needed].to(self.device)
                    labels = batch['labels'][:needed].to(self.device)

                    # --- PATCH FOR SHAPE MISMATCH ---
                    # The generate() function in diffusion_lightning.py incorrectly extends the
                    # sequence length from 1024 to 1280. To prevent an IndexError during
                    # evaluation, we pad the labels tensor here to match that bugged length.
                    bugged_total_len = prompts.shape[1] + model.T_ans
                    padding_needed = bugged_total_len - labels.shape[1]
                    if padding_needed > 0:
                        padded_labels = F.pad(labels, (0, padding_needed), 'constant', -100)
                    else:
                        padded_labels = labels
                    # --- END PATCH ---

                    ans_start_idx = batch.get('ans_start_idx', None)
                    if ans_start_idx is not None:
                        ans_start_idx = ans_start_idx[:needed].to(self.device)
                        
                    # 3. Generate and evaluate
                    model.generate(
                        prompts, 
                        ans_start_idx=ans_start_idx,
                        num_steps=num_steps, 
                        evaluation_elements=[evaluator, padded_labels]
                    )
                    
                    evaluator.finalize_batch()
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # 4. Increment the counter and break the loop if the limit is reached
                    samples_processed += prompts.size(0)
                    if samples_processed >= max_samples:
                        break
            
            # Save into the metrics directory directly
            evaluator.export_dataframe(variant_name=variant, save_dir=self.dirs['metrics'])
            del model
            
class GraphsGenerator:
    """
    Generates visualizations for the benchmark results.
    """
    def __init__(self):
        # --- NEW: Automated Directory Management ---
        self.results_dir = "benchmark_results"
        self.dirs = {
            'metrics': os.path.join(self.results_dir, "metrics"),
            'plots': os.path.join(self.results_dir, "plots")
        }
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)

    def generate_perplexity_graph(self, csv_filename="perplexity_results.csv"):
        full_load_path = os.path.join(self.dirs['metrics'], csv_filename)
        df = pd.read_csv(full_load_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['Model'], df['Perplexity'], color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
        plt.title("Perplexity Comparison Across Models")
        plt.ylabel("Perplexity")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        full_save_path = os.path.join(self.dirs['plots'], "perplexity_comparison.png")
        plt.savefig(full_save_path)
        print(f"✅ Perplexity graph saved as {full_save_path}")

    def generate_structure_evaluation_graph(self, csv_filename="structure_evaluation_results.csv", csv_lengths_filename="structure_evaluation_lengths.csv"):
        full_load_path = os.path.join(self.dirs['metrics'], csv_filename)
        full_lengths_load_path = os.path.join(self.dirs['metrics'], csv_lengths_filename)

        df = pd.read_csv(full_load_path)
        lengths_df = pd.read_csv(full_lengths_load_path)
        
        # ---------------------------------------------------------
        # 1. BOXPLOT: Distribution of Sequence Lengths
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        
        # We extract each column, dropping the NaNs (since models might have slightly different sample counts)
        data_to_plot = [(lengths_df[col].dropna()) for col in lengths_df.columns]
        
        # Create the boxplot (patch_artist=True allows us to fill the boxes with color)
        plt.bar(df['model'],256, color='#4C72B0')
        plt.title("Distribution of Generated Sequence Lengths (Tokens)")
        plt.ylabel("Length (Tokens)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        full_save_path_box = os.path.join(self.dirs['plots'], "structure_evaluation_comparison.png")
        plt.savefig(full_save_path_box)
        plt.close() # Always keep the figure chiuso when done!
        print(f"✅ Structure evaluation graph saved as {full_save_path_box}")

        # ---------------------------------------------------------
        # 2. BAR CHART: Missing EOS Percentages
        # ---------------------------------------------------------
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['missing_eos'], color='#C44E52')
        plt.title("Missing EOS Percentage (Lower is Better)")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        full_save_path_eos = os.path.join(self.dirs['plots'], "missing_eos_comparison.png")
        plt.savefig(full_save_path_eos)
        plt.close() # Keep the figure chiuso
        print(f"✅ Missing EOS graph saved as {full_save_path_eos}")

    def generate_diversity_evaluation_graph(self, csv_filename="diversity_evaluation_results.csv"):
        full_load_path = os.path.join(self.dirs['metrics'], csv_filename)
        df = pd.read_csv(full_load_path)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        axs[0].bar(df['model'], df['unique_ngram_ratio'], color='#4C72B0')
        axs[0].set_title("Unique N-Gram Ratio (Higher is Better)")
        axs[0].set_ylabel("Percentage (%)")
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].bar(df['model'], df['self_bleu'], color='#DD8452')
        axs[1].set_title("Self-BLEU (Lower is Better)")
        axs[1].set_ylabel("Score")
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        full_save_path = os.path.join(self.dirs['plots'], "diversity_evaluation_comparison.png")
        plt.savefig(full_save_path)
        print(f"✅ Diversity evaluation graph saved as {full_save_path}")

    def generate_semantic_evaluation_graph(self, csv_filename="semantic_evaluation_results.csv"):
        full_load_path = os.path.join(self.dirs['metrics'], csv_filename)
        df = pd.read_csv(full_load_path)
        
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['bertscore'], color='#4C72B0')
        plt.title("Semantic Evaluation (BERTScore) Across Models")
        plt.ylabel("BERTScore (F1 %)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path_bert = os.path.join(self.dirs['plots'], "semantic_evaluation_comparison.png")
        plt.savefig(save_path_bert)
        print(f"✅ Semantic evaluation graph saved as {save_path_bert}")
        
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['fbd_score'], color='#DD8452')
        plt.title("Semantic Evaluation (Fréchet BERT Distance) Across Models")
        plt.ylabel("Fréchet BERT Distance (Lower is Better)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path_fbd = os.path.join(self.dirs['plots'], "semantic_evaluation_fbd_comparison.png")
        plt.savefig(save_path_fbd)
        print(f"✅ Semantic evaluation (FBD) graph saved as {save_path_fbd}")

    def generate_llm_judge_graph(self, csv_filename="llm_judge_results.csv"):
        full_load_path = os.path.join(self.dirs['metrics'], csv_filename)
        df = pd.read_csv(full_load_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['total_points'], color='#55A868')
        plt.title("LLM Judge Evaluation (Borda Count) Across Models")
        plt.ylabel("Total Points")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        full_save_path = os.path.join(self.dirs['plots'], "llm_judge_evaluation_comparison.png")
        plt.savefig(full_save_path)
        print(f"✅ LLM Judge evaluation graph saved as {full_save_path}")

    def generate_trajectory_graphs(self):
        print("\n📊 Generating Separate Trajectory Graphs...")
        files = {
            "Uniform": "trajectory_report_unif.csv",
            "Sigmoid": "trajectory_report_sigm.csv",
            "Positional Dependent": "trajectory_report_posdip.csv"
        }
        styles = {
            "Uniform": {"color": "#4C72B0", "marker": "o"},
            "Sigmoid": {"color": "#DD8452", "marker": "s"},
            "Positional Dependent": {"color": "#55A868", "marker": "^"}
        }

        dataframes = {}
        for name, filename in files.items():
            full_load_path = os.path.join(self.dirs['metrics'], filename)
            if os.path.exists(full_load_path):
                dataframes[name] = pd.read_csv(full_load_path)
            else:
                print(f"❌ Error: Could not find {full_load_path}. Run run_trajectory_evaluation first!")
                return

        metrics = [
            {
                "column": "avg_entropy",
                "title": "Masked Entropy Trajectory",
                "ylabel": "Shannon Entropy (Lower = More Confident)",
                "filename": "ddm_trajectory_entropy.png"
            },
            {
                "column": "avg_step_ppl",
                "title": "Pseudo-Perplexity Trajectory",
                "ylabel": "Step-Perplexity (Lower = Better)",
                "filename": "ddm_trajectory_perplexity.png"
            },
            {
                "column": "avg_semantic_convergence",
                "title": "Semantic Convergence Trajectory",
                "ylabel": "Cosine Similarity (Closer to 1.0 = Better)",
                "filename": "ddm_trajectory_semantic.png"
            }
        ]

        for metric in metrics:
            plt.figure(figsize=(8, 6))
            for variant_name, df in dataframes.items():
                plt.plot(
                    df['step'], df[metric["column"]], label=variant_name,
                    color=styles[variant_name]["color"], marker=styles[variant_name]["marker"], 
                    linewidth=2.5, markersize=7
                )

            ylabel = metric["ylabel"]
            if metric["column"] == "avg_step_ppl":
                plt.yscale('log')
                ylabel += " (log scale)"

            plt.title(metric["title"], fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Generative Steps (t)", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            full_save_path = os.path.join(self.dirs['plots'], metric["filename"])
            plt.savefig(full_save_path, format='png', bbox_inches='tight')
            plt.close() 
            print(f"✅ Saved graph: {full_save_path}")
            
        print("🎉 All 3 trajectory graphs successfully generated!")


if __name__ == "__main__":
    manager = BenchmarkManager(config_path="config.yaml")

    manager.tokenize_and_group()
    #manager.build_token_dataframe(save_filename="benchmark_tokens.pkl")
    #manager.build_dataframe_from_partials(save_filename="benchmark_tokens.pkl")
    #manager.run_perplexity_benchmark(save_filename="perplexity_results.csv")
    #manager.build_text_dataframe(load_filename="benchmark_tokens.pkl", save_filename="benchmark_text.csv")
    #manager.run_structure_evaluation(load_filename="benchmark_tokens.pkl", save_filename="structure_evaluation_results.csv")
    #manager.run_diversity_evaluation(load_filename="benchmark_tokens.pkl", save_filename="diversity_evaluation_results.csv")
    #manager.run_semantic_evaluation(load_filename="benchmark_text.csv", save_filename="semantic_evaluation_results.csv")
    #manager.run_trajectory_evaluation(num_steps=20)
    #manager.run_llm_judge_evaluation(load_filename="benchmark_text.csv", save_filename="llm_judge_results.csv")

    gg = GraphsGenerator()

    #gg.generate_perplexity_graph(csv_filename="perplexity_results.csv")
    #gg.generate_structure_evaluation_graph(csv_filename="structure_evaluation_results.csv")
    #gg.generate_diversity_evaluation_graph(csv_filename="diversity_evaluation_results.csv")
    #gg.generate_semantic_evaluation_graph(csv_filename="semantic_evaluation_results.csv")
    gg.generate_trajectory_graphs()
    #gg.generate_llm_judge_graph(csv_filename="llm_judge_results.csv")
