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
    Calculates perplexity for Autoregressive and Diffusion models.
    """
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
    
    def _evaluate_ddm_batch(self, input_ids, labels):
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            t = torch.rand(batch_size, device=self.device)
            
            if self.model.corruption_type == "independent":
                move_chance, loss_weight = masking_schedule.vanilla_masking(
                    t=t, T=seq_len, device=self.device, noise=self.model.noise
                )
            elif self.model.corruption_type == "position":
                move_chance, loss_weight = masking_schedule.position_dependent_masking(
                    t=t, T=seq_len, device=self.device, noise=self.model.noise,
                    gamma=self.model.position_gamma, position_loss_weighting=self.model.position_loss_weighting
                )
            elif self.model.corruption_type == "moving_sigmoid":
                move_chance, loss_weight = masking_schedule.moving_sigmoid_masking(
                    t=t, T=seq_len, device=self.device, noise=self.model.noise,
                    k=self.model.sigmoid_k, calibrated=self.model.calibrated_sigmoid
                )
            else:
                raise ValueError(f"Unknown corruption type: {self.model.corruption_type}")
            
            rand_matrix = torch.rand(batch_size, seq_len, device=self.device)
            is_response_token = (labels != -100)
            
            mask_bool = is_response_token & (rand_matrix < move_chance)
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_bool] = self.mask_token_id
            
            ddm_labels = labels.clone()
            ddm_labels[~mask_bool] = -100 
            
            sigma, _ = self.model.noise(t)
            logits = self.model(masked_input_ids, sigma=sigma)
            
            loss_per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size), 
                ddm_labels.view(-1), 
                ignore_index=-100, 
                reduction='none' 
            )
            
            loss_per_seq = loss_per_token.view(batch_size, seq_len)
            weighted_loss_per_seq = loss_per_seq * loss_weight
            
            batch_total_loss = weighted_loss_per_seq.sum().item()
            valid_masked_tokens = (ddm_labels != -100).sum().item()
            
        return batch_total_loss, valid_masked_tokens

    def calculate(self, val_dataloader):
        self.model.eval()

        if self.model_type == 'ar':
            print("🚀 Starting Autoregressive Perplexity Evaluation...")
            ar_total_loss = 0.0
            ar_total_tokens = 0
            
            for batch in tqdm(val_dataloader, desc="Processing AR Batches"):
                input_ids = batch['input_ids'].to(self.device) 
                labels = batch['labels'].to(self.device)
                
                with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
                    ar_loss, ar_tokens = self._evaluate_ar_batch(input_ids, labels)
                    
                ar_total_loss += ar_loss
                ar_total_tokens += ar_tokens
                
            ar_mean_loss = ar_total_loss / max(ar_total_tokens, 1)
            perplexity = torch.exp(torch.tensor(ar_mean_loss))

        elif self.model_type == 'ddm':
            print("🚀 Starting Diffusion Model (ELBO) Perplexity Evaluation...")
            ddm_total_loss = 0.0
            ddm_total_tokens = 0
            
            for batch in tqdm(val_dataloader, desc="Processing DDM Batches"):
                input_ids = batch['input_ids'].to(self.device) 
                labels = batch['labels'].to(self.device)
                
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
    Evaluates lexical and structural diversity of generated texts.
    """
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
        
        # FIX: Replaced invalid torch.random.sample with standard random.sample
        if len(valid_sequences) > self.sample_size_for_bleu:
            valid_sequences = random.sample(valid_sequences, self.sample_size_for_bleu)
            
        total_sentences = len(valid_sequences)
        if total_sentences < 2:
            return 0.0

        bleu_scores = []
        for i in range(total_sentences):
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
    """
    Evaluates semantic similarity and distribution distances using BERT.
    """
    def __init__(self, bert_model_name="bert-base-uncased", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.bert_model_name = bert_model_name
        
        print(f"Loading BERT evaluator ({bert_model_name}) into {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.model.eval()

    def calculate_bertscore(self, references, hypotheses):
        print("Calculating BERTScore...")
        P, R, F1 = bert_score_calc(
            cands=hypotheses, 
            refs=references, 
            lang="en",
            model_type=self.bert_model_name,
            device=self.device,
            verbose=False
        )
        return F1.mean().item() * 100

    def _get_sentence_embeddings(self, texts, batch_size=32):
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
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
        
        mu_real = np.mean(real_embs, axis=0)
        sigma_real = np.cov(real_embs, rowvar=False)
        mu_gen = np.mean(gen_embs, axis=0)
        sigma_gen = np.cov(gen_embs, rowvar=False)
        
        print("Computing Fréchet Math...")
        diff = mu_real - mu_gen
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        frechet_distance = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
        return frechet_distance

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
    """
    Evaluates 4 conversational models simultaneously using a Borda Count ranking system (3, 2, 1, 0 points)
    with a local LLM-as-a-Judge via Ollama.
    """
    def __init__(self, model_name="llama3", base_url="http://localhost:11434/v1"):
        self.client = OpenAI(
            api_key="ollama", 
            base_url=base_url
        )
        self.model_name = model_name
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
        print(f"\n🚀 Starting 4-Way Ranked LLM-as-a-Judge with Ollama ({self.model_name})...")
        
        models_cols = ['AR answer', 'DiT uniform', 'DiT positional dependent', 'DiT sigmoid']
        points_map = [3, 2, 1, 0]
        
        for idx, row in tqdm(df_text.iterrows(), total=len(df_text)):
            prompt = str(row['context'])
            responses = {col: str(row[col]) for col in models_cols}
            
            shuffled_cols = list(responses.keys())
            random.shuffle(shuffled_cols)
            
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
                    temperature=0.0
                )
                
                result_json = json.loads(response.choices[0].message.content)
                ranking = result_json.get("ranking", [])
                
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
            collected_data.append({
                'rank': rank,
                'model': col,
                'total_points': total_pts,
                'avg_points_per_prompt': avg_pts,
                'pct_max_points': pct_max
            })
            
        print(f"  (Errors/Failures: {self.results['Errors']})")
        return pd.DataFrame(collected_data)

class DiffusionTrajectoryEvaluator:
    """
    Evaluates generation trajectory of DDM step-by-step.
    """
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
        masked_logits = logits[masked_indices]
        masked_targets = target_labels[masked_indices]
        if masked_logits.numel() == 0 or masked_targets.numel() == 0:
            return 1.0
        loss = F.cross_entropy(masked_logits, masked_targets, reduction='mean')
        return torch.exp(loss).item()

    def update_step(self, step, logits, last_hidden_state, current_input_ids, target_labels):
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

    def export_dataframe(self, variant_name="unif"):
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
    Manages benchmarking pipeline for AR and DDM models.
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
        ds = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="test[:10%]", cache_dir=".data")
        return ds
    
    def tokenize_and_group(self):
        print("Tokenizing and grouping dataset...")
        tokenized_ds = self.dm_qa.tokenize(self.test_ds, split_name="test")
        
        grouped_ds = self.dm_qa.group_texts_ar(tokenized_ds, T=self.T, split_name="test")
        grouped_ds = self.dm_qa.group_texts_dit(tokenized_ds, T_ctx=self.T_context, T_ans=self.T_answer, split_name="test")

        grouped_ds_ar = grouped_ds.rename_column("output_ids", "labels")
        grouped_ds_ar.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
        grouped_ds_dit = grouped_ds.rename_column("output_ids", "labels")
        grouped_ds_dit.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
        
        self.val_dataloader_ar = self.dm_qa.getTrainloader(grouped_ds_ar, B=self.B)
        self.val_dataloader_dit = self.dm_qa.getTrainloader(grouped_ds_dit, B=self.B)

    def model(self, model_to_test="ar", model_variant="unif"):
        models = {
            "ar": self._upload_models_ar(),
            "ddm": self._upload_models_ddm(model_variant=model_variant)
        }
        return models[model_to_test]
    
    def _upload_models_ar(self):
        print("Loading AR model...")
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

    def generate_token_answers(self, model_type="ar", model_variant="unif"):
        current_model = self.model(model_type, model_variant=model_variant)
        current_model.eval()
        dataloader = self.val_dataloader_ar if model_type == "ar" else self.val_dataloader_dit
        generated_answers = []
        prompt_lengths = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating answers for {model_type.upper()}"):
                prompts = batch['input_ids'].to(self.device)
                batch_prompt_lengths = (batch['attention_mask'].sum(dim=1) - 1).tolist()
                if model_type == "ddm":
                    outputs = current_model.generate(prompts, n_tokens=100, num_steps=20)
                else:
                    outputs = current_model.generate(prompts, max_length=100)
                generated_answers.extend(outputs.cpu().numpy())
                prompt_lengths.extend(batch_prompt_lengths)
        gc.collect()
        torch.cuda.empty_cache()
        return generated_answers, prompt_lengths

    def decode_tokens_to_text(self, token_list, prompt_lengths=None):
        text_answers = []
        for i, tokens in enumerate(tqdm(token_list, desc="Decoding tokens to text")):
            if prompt_lengths is not None:
                p_len = prompt_lengths[i]
                response_tokens = tokens[p_len:]
            else:
                response_tokens = tokens
                
            text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            text_answers.append(text)
            
        return text_answers

    def build_token_dataframe(self, save_path="benchmark_tokens.pkl"):
        print("\n👩‍💻 Building the Token-based Pandas Dataset...")
        contexts, true_answers = [], []
        
        for batch in self.val_dataloader_ar:
            input_ids = batch['input_ids'].tolist()
            prompt_lengths = (batch['attention_mask'].sum(dim=1) - 1).tolist() 
            
            for seq, p_len in zip(input_ids, prompt_lengths):
                contexts.append(seq[:p_len])
                true_answers.append(seq[p_len:])

        ar_tokens, ar_lengths = self.generate_token_answers(model_type="ar")
        ar_answers = [seq[p:] for seq, p in zip(ar_tokens, ar_lengths)]
        
        ddm_unif_tokens, ddm_unif_lengths = self.generate_token_answers(model_type="ddm", model_variant="unif")
        ddm_unif_answers = [seq[p:] for seq, p in zip(ddm_unif_tokens, ddm_unif_lengths)]
        
        ddm_posdip_tokens, ddm_posdip_lengths = self.generate_token_answers(model_type="ddm", model_variant="posdip")
        ddm_posdip_answers = [seq[p:] for seq, p in zip(ddm_posdip_tokens, ddm_posdip_lengths)]
        
        ddm_sigm_tokens, ddm_sigm_lengths = self.generate_token_answers(model_type="ddm", model_variant="sigm")
        ddm_sigm_answers = [seq[p:] for seq, p in zip(ddm_sigm_tokens, ddm_sigm_lengths)]

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
        df_tokens.to_pickle(save_path)
        print(f"\n✅ Token Pandas Dataset successfully saved to: {save_path}")
        return df_tokens
    
    def build_text_dataframe(self, df_tokens=None, load_path="benchmark_tokens.pkl", save_path="benchmark_text.csv"):
        print("\n🔤 Converting Token Dataset to Text Dataset...")
        
        # FIX: Handles both path strings or DataFrame objects seamlessly
        if isinstance(df_tokens, str):
            load_path = df_tokens
            df_tokens = None
            
        if df_tokens is None:
            print(f"📂 Loading token dataframe from {load_path}...")
            df_tokens = pd.read_pickle(load_path)
            
        df_text = pd.DataFrame()
        for column in df_tokens.columns:
            print(f"Decoding column: '{column}'...")
            col_tokens = df_tokens[column].tolist()
            df_text[column] = self.decode_tokens_to_text(col_tokens, prompt_lengths=None)
            
        df_text.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\n✅ Text Pandas Dataset successfully saved to: {save_path}")
        return df_text

    def run_perplexity_benchmark(self):
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
        df_ppx.to_csv("perplexity_results.csv", index=False)

    def run_structure_evaluation(self, df_tokens_path="benchmark_tokens.pkl"):
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
        try:
            df = pd.read_pickle(df_tokens_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_tokens_path}.")
            return
            
        de = DiversityEvaluator()
        df_results = de.evaluate_dataframe(df_tokens=df)  
        df_results.to_csv("diversity_evaluation_results.csv", index=False)

    def run_semantic_evaluation(self, df_text_path="benchmark_text.csv"):
        try:
            df = pd.read_csv(df_text_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_text_path}.")
            return
            
        se = SemanticEvaluator(bert_model_name="bert-base-uncased", device=self.device)
        df_results = se.evaluate_dataframe(df_text=df)  
        df_results.to_csv("semantic_evaluation_results.csv", index=False)
        
    def run_llm_judge_evaluation(self, df_text_path="benchmark_text.csv"):
        try:
            df = pd.read_csv(df_text_path)
        except FileNotFoundError:
            print(f"❌ Error: Could not find {df_text_path}.")
            return
            
        judge = LLMJudgeEvaluator(model_name="llama3", base_url="http://localhost:11434/v1")
        df_results = judge.evaluate_dataframe(df_text=df)
        df_results.to_csv("llm_judge_results.csv", index=False)

    def run_trajectory_evaluation(self, num_steps=20):
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
                    
                    model.generate(
                        prompts, 
                        token_start_idx=batch.get('token_start_idx', None), 
                        num_steps=num_steps, 
                        evaluator=evaluator, 
                        target_labels=labels
                    )
                    evaluator.finalize_batch()
            
            evaluator.export_dataframe(variant_name=variant)
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
        plt.bar(df['Model'], df['Perplexity'], color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'])
        plt.title("Perplexity Comparison Across Models")
        plt.ylabel("Perplexity")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("perplexity_comparison.png")
        print("✅ Perplexity graph saved as perplexity_comparison.png")

    def generate_structure_evaluation_graph(self, csv_path="structure_evaluation_results.csv"):
        # FIX: Corrected column access to 'avg_length' and dynamic bar coloring
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['avg_length'], color='#4C72B0')
        plt.title("Average Generated Response Length (Tokens)")
        plt.ylabel("Average Length (Tokens)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("structure_evaluation_comparison.png")
        print("✅ Structure evaluation graph saved as structure_evaluation_comparison.png")

    def generate_diversity_evaluation_graph(self, csv_path="diversity_evaluation_results.csv"):
        # FIX: Replaced non-existent 'diversity_score' with 'unique_ngram_ratio' and 'self_bleu'
        df = pd.read_csv(csv_path)
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
        plt.savefig("diversity_evaluation_comparison.png")
        print("✅ Diversity evaluation graph saved as diversity_evaluation_comparison.png")

    def generate_semantic_evaluation_graph(self, csv_path="semantic_evaluation_results.csv"):
        df = pd.read_csv(csv_path)
        
        # BERTScore
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['bertscore'], color='#4C72B0')
        plt.title("Semantic Evaluation (BERTScore) Across Models")
        plt.ylabel("BERTScore (F1 %)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("semantic_evaluation_comparison.png")
        print("✅ Semantic evaluation graph saved as semantic_evaluation_comparison.png")
        
        # Frechet BERT Distance
        plt.figure(figsize=(8, 5))
        plt.bar(df['model'], df['fbd_score'], color='#DD8452')
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
        plt.bar(df['model'], df['total_points'], color='#55A868')
        plt.title("LLM Judge Evaluation (Borda Count) Across Models")
        plt.ylabel("Total Points")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("llm_judge_evaluation_comparison.png")
        print("✅ LLM Judge evaluation graph saved as llm_judge_evaluation_comparison.png")

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
            if os.path.exists(filename):
                dataframes[name] = pd.read_csv(filename)
            else:
                print(f"❌ Error: Could not find {filename}. Run run_trajectory_evaluation first!")
                return

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

        for metric in metrics:
            plt.figure(figsize=(8, 6))
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

            plt.title(metric["title"], fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Generative Steps (t)", fontsize=12)
            plt.ylabel(metric["ylabel"], fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=11)

            plt.tight_layout()
            plt.savefig(metric["filename"], format='pdf', bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved graph: {metric['filename']}")
            
        print("🎉 All 3 trajectory graphs successfully generated!")


if __name__ == "__main__":
    manager = BenchmarkManager(config_path="config.yaml")

    manager.tokenize_and_group()
    manager.build_token_dataframe(save_path="benchmark_tokens.pkl")
    manager.build_text_dataframe(load_path="benchmark_tokens.pkl", save_path="benchmark_text.csv")
    manager.run_perplexity_benchmark()
    manager.run_structure_evaluation(df_tokens_path="benchmark_tokens.pkl")
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