import torch
import torch.nn.functional as F

# ==========================================
# 1. FUNZIONE PER IL MODELLO AUTOREGRESSIVO
# ==========================================
def evaluate_ar_batch(model_ar, input_ids, labels):
    """
    Calcola la somma della Cross-Entropy loss per un batch nel modello Autoregressivo.
    """
    # Assicuriamoci di non calcolare gradienti
    with torch.no_grad():
        # Forward pass: l'AR vede i token e prevede il successivo
        logits = model_ar(input_ids)
        
        # Shift: i logits in posizione 'i' devono prevedere la label in posizione 'i+1'
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calcoliamo la loss sommata su tutto il batch (ignorando il prompt a -100)
        # Usiamo reduction='sum' per accumulare il valore totale
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1), 
            ignore_index=-100, 
            reduction='sum'
        )
        
        # Contiamo quanti token di VERA risposta c'erano in questo batch
        valid_tokens = (shift_labels != -100).sum().item()
        
    return loss.item(), valid_tokens


# ==========================================
# 2. FUNZIONE PER IL MODELLO A DIFFUSIONE
# ==========================================
def evaluate_ddm_batch(model_ddm, input_ids, labels, mask_token_id, vocab_size):
    """
    Calcola la somma della Loss pesata (ELBO) per un batch nel modello a Diffusione.
    Usa una schedule lineare di base.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    with torch.no_grad():
        # 1. Campioniamo un tempo 't' casuale tra 0 e 1 per OGNI frase nel batch
        t = torch.rand(batch_size, 1, device=device)
        
        # Per una schedule lineare, la percentuale di maschera è semplicemente t
        # (In schedule più complesse, mask_ratio = compute_schedule(t))
        mask_ratio = t 
        
        # Per la schedule lineare, la derivata matematica che fa da peso per l'ELBO è costante (1.0)
        # Se in futuro userete schedule a coseno, questo peso cambierà in funzione di t!
        weight_t = 1.0 
        
        # 2. Creiamo la maschera casuale
        rand_matrix = torch.rand(batch_size, seq_len, device=device)
        
        # Mascheriamo SOLO i token della risposta (labels != -100) 
        # con una probabilità pari a mask_ratio
        is_response_token = (labels != -100)
        mask_bool = is_response_token & (rand_matrix < mask_ratio)
        
        # 3. Applichiamo la maschera agli input
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_bool] = mask_token_id
        
        # 4. Creiamo le labels per la Loss: isoliamo SOLO i token mascherati
        ddm_labels = labels.clone()
        ddm_labels[~mask_bool] = -100 
        
        # 5. Forward Pass
        logits = model_ddm(masked_input_ids)
        
        # 6. Calcoliamo la Loss NON ridotta, per poter applicare il peso per singola frase
        loss_per_token = F.cross_entropy(
            logits.view(-1, vocab_size), 
            ddm_labels.view(-1), 
            ignore_index=-100, 
            reduction='none' # Manteniamo il tensore srotolato
        )
        
        # Riarrotoliamo il tensore a (batch_size, seq_len) e sommiamo le loss per ogni frase
        loss_per_seq = loss_per_token.view(batch_size, seq_len).sum(dim=1)
        
        # Moltiplichiamo per il peso ELBO dello step 't' di quella specifica frase
        weighted_loss_per_seq = loss_per_seq * weight_t
        
        # Sommiamo tutto per ottenere la loss totale del batch
        batch_total_loss = weighted_loss_per_seq.sum().item()
        
        # Contiamo quanti token sono stati EFFETTIVAMENTE mascherati e valutati in questo batch
        valid_masked_tokens = (ddm_labels != -100).sum().item()
        
    return batch_total_loss, valid_masked_tokens


# ==========================================
# 3. IL CICLO DI VALIDAZIONE PRINCIPALE
# ==========================================
def calculate_benchmark_perplexities(val_dataloader, model_ar, model_ddm, mask_token_id, vocab_size):
    """
    Esegue il loop sull'intero dataset e calcola le Perplexity finali.
    """
    model_ar.eval()
    model_ddm.eval()
    
    ar_total_loss = 0.0
    ar_total_tokens = 0
    
    ddm_total_loss = 0.0
    ddm_total_tokens = 0
    
    # Iteriamo su tutto il validation set
    for batch in val_dataloader:
        # Spostiamo i dati sulla GPU (se disponibile)
        input_ids = batch['input_ids'].cuda() 
        labels = batch['labels'].cuda()
        
        # Valutazione AR
        ar_loss, ar_tokens = evaluate_ar_batch(model_ar, input_ids, labels)
        ar_total_loss += ar_loss
        ar_total_tokens += ar_tokens
        
        # Valutazione DDM
        ddm_loss, ddm_tokens = evaluate_ddm_batch(model_ddm, input_ids, labels, mask_token_id, vocab_size)
        ddm_total_loss += ddm_loss
        ddm_total_tokens += ddm_tokens
        
    # --- CALCOLO FINALE ---
    # AR: Esponenziale della media matematica
    ar_mean_loss = ar_total_loss / max(ar_total_tokens, 1) # max per evitare divisioni per zero
    ar_perplexity = torch.exp(torch.tensor(ar_mean_loss))
    
    # DDM: Esponenziale della media pesata (ELBO)
    ddm_mean_loss = ddm_total_loss / max(ddm_total_tokens, 1)
    ddm_perplexity = torch.exp(torch.tensor(ddm_mean_loss))
    
    print(f"✅ Valutazione Completata!")
    print(f"📊 Autoregressive Perplexity: {ar_perplexity.item():.2f}")
    print(f"📊 Diffusion Model (ELBO) Perplexity: {ddm_perplexity.item():.2f}")
    
    return ar_perplexity.item(), ddm_perplexity.item()



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