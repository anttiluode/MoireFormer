"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MOIRÉ CONVERSATIONAL TRAINER v3 (Advanced Curriculums)                      ║
║                                                                              ║
║  Added new high-quality dataset loaders (Guanaco, TinyStories, FineWeb)      ║
║  to expand the semantic phase-space and cure hallucinations.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
from typing import Optional
from dataclasses import dataclass

# ============================================================================
# 1. ARCHITECTURE 
# ============================================================================

@dataclass
class MoireGPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    gamma_slots: int = 8
    dropout: float = 0.1
    bias: bool = False
    use_theta_gating: bool = True
    
    @property
    def head_dim(self):
        return self.n_embd // self.n_head

class MoireAttention(nn.Module):
    def __init__(self, config: MoireGPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.gamma_slots = config.gamma_slots
        
        self.q_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        if config.use_theta_gating:
            self.theta_offset = nn.Parameter(torch.randn(config.n_head) * 0.1)
        
        self.scale = 1.0 / math.sqrt(config.head_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        
        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        q_amp, q_phase = q_raw.chunk(2, dim=-1)
        k_amp, k_phase = k_raw.chunk(2, dim=-1)
        
        q_amp = F.softplus(q_amp.view(B, T, self.n_head, self.head_dim).transpose(1, 2))
        q_phase = q_phase.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_amp = F.softplus(k_amp.view(B, T, self.n_head, self.head_dim).transpose(1, 2))
        k_phase = k_phase.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Optimized Interference
        q_real = q_amp * torch.cos(q_phase)
        q_imag = q_amp * torch.sin(q_phase)
        k_real = k_amp * torch.cos(k_phase)
        k_imag = k_amp * torch.sin(k_phase)
        
        real_scores = torch.matmul(q_real, k_real.transpose(-1, -2))
        imag_scores = torch.matmul(q_imag, k_imag.transpose(-1, -2))
        scores = (real_scores + imag_scores) * self.scale
        
        if self.config.use_theta_gating and T > self.gamma_slots:
            positions = torch.arange(T, device=x.device, dtype=torch.float32)
            cycle_ids = positions / self.gamma_slots
            cycle_dist = cycle_ids.unsqueeze(0) - cycle_ids.unsqueeze(1)
            theta_off = self.theta_offset.view(self.n_head, 1, 1)
            theta_gate = torch.cos(theta_off * cycle_dist.unsqueeze(0))
            scores = scores * theta_gate.unsqueeze(0)
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = self.resid_dropout(
            self.out_proj(
                torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, C)
            )
        )
        return out

class MoireBlock(nn.Module):
    def __init__(self, config: MoireGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MoireAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class MoireGPT(nn.Module):
    def __init__(self, config: MoireGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MoireBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Moiré GPT] {n_params/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, attention_mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ============================================================================
# 2. DATASET LOADERS (NEW CURRICULUMS ADDED)
# ============================================================================


def load_dataset_ultimate_mix(tokenizer, seq_len: int, max_chars_per=15_000_000):
    """The Ultimate Curriculum: 1/3 Conversation, 1/3 Logic, 1/3 Facts"""
    print("Loading Ultimate Mix (Guanaco + TinyStories + FineWeb)...")
    from datasets import load_dataset
    
    all_texts = []
    
    # 1. Guanaco (Conversational / Persona)
    print("  -> Fetching Guanaco...")
    ds_g = load_dataset("timdettmers/openassistant-guanaco", split="train")
    chars = 0
    for row in ds_g:
        text = row['text'].replace("### Human:", "User:").replace("### Assistant:", "Bot:")
        all_texts.append(text)
        chars += len(text)
        if chars > max_chars_per: break

    # 2. TinyStories (Grammar / Narrative Logic)
    print("  -> Fetching TinyStories...")
    ds_t = load_dataset("roneneldan/TinyStories", split="train")
    chars = 0
    for row in ds_t:
        all_texts.append(row['text'])
        chars += len(row['text'])
        if chars > max_chars_per: break

    # 3. FineWeb (Math / Science / Facts)
    print("  -> Fetching FineWeb-Edu...")
    ds_f = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    chars = 0
    for row in ds_f:
        all_texts.append(row['text'])
        chars += len(row['text'])
        if chars > max_chars_per: break

    # CRITICAL: Shuffle the documents so the wave-field learns everything simultaneously!
    print("  -> Shuffling the multiverse...")
    random.shuffle(all_texts)
    
    # Join with an end-of-text token so thoughts don't bleed into each other
    full_text = "\n\n<|endoftext|>\n\n".join(all_texts)
    print(f"Total Mixed Corpus: {len(full_text):,} chars")
    
    return _tokenize_text(full_text, tokenizer, seq_len)

def _tokenize_text(text: str, tokenizer, seq_len: int):
    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    chunk_size = 1_000_000 
    tokens = []
    print("Tokenizing data...")
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        tokens.extend(tokenizer.encode(chunk, add_special_tokens=False))
    tokenizer.model_max_length = old_max
    stride = seq_len // 2
    sequences = []
    for i in range(0, len(tokens) - seq_len, stride):
        sequences.append(tokens[i:i + seq_len])
    print(f"Created {len(sequences):,} training sequences.")
    return torch.tensor(sequences, dtype=torch.long)

def load_dataset_guanaco(tokenizer, seq_len: int):
    """High quality conversational flow."""
    print("Loading OpenAssistant-Guanaco...")
    from datasets import load_dataset
    ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
    text_chunks = []
    for row in ds:
        text = row['text']
        # Convert tags so the model builds on what it learned in Dolly
        text = text.replace("### Human:", "User:")
        text = text.replace("### Assistant:", "Bot:")
        text_chunks.append(text)
    full_text = "\n\n".join(text_chunks)
    print(f"Total: {len(full_text):,} chars")
    return _tokenize_text(full_text, tokenizer, seq_len)

def load_dataset_tinystories(tokenizer, seq_len: int, max_chars: int = 15_000_000):
    """Logic, object permanence, and grammar."""
    print("Loading TinyStories...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = []
    current_chars = 0
    for row in ds:
        texts.append(row['text'])
        current_chars += len(row['text'])
        if current_chars > max_chars:
            break
    full_text = "\n\n<|endoftext|>\n\n".join(texts)
    print(f"Total: {len(full_text):,} chars")
    return _tokenize_text(full_text, tokenizer, seq_len)

def load_dataset_fineweb(tokenizer, seq_len: int, max_chars: int = 15_000_000):
    """Hard factual data to separate phase-clumps."""
    print("Loading FineWeb-Edu (Sample)...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    texts = []
    current_chars = 0
    for row in ds:
        texts.append(row['text'])
        current_chars += len(row['text'])
        if current_chars > max_chars:
            break
    full_text = "\n\n".join(texts)
    print(f"Total: {len(full_text):,} chars")
    return _tokenize_text(full_text, tokenizer, seq_len)

def load_dataset_mixed(tokenizer, seq_len: int):
    # Keep the old mixed loader for legacy support
    print("Loading mixed (Dolly + Wiki)...")
    from datasets import load_dataset
    all_text = []
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for row in ds:
        user_text = row['instruction'].strip()
        if row['context'].strip(): user_text += "\n" + row['context'].strip()
        all_text.append(f"User: {user_text}\nBot: {row['response'].strip()}\n")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_text = "\n".join([t for t in wiki['text'] if len(t.strip()) > 50])
    all_text.append(wiki_text[:5_000_000])
    return _tokenize_text("\n".join(all_text), tokenizer, seq_len)


# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train(model, train_data, config, args):
    device = args.device
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # ADD THIS LINE: Initialize the AMP GradScaler
    scaler = torch.amp.GradScaler('cuda')
    
    n_batches = len(train_data) // args.batch_size
    total_steps = args.epochs * n_batches
    warmup_steps = min(200, total_steps // 10)
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    start_epoch = 0
    global_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming weights from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # If we switch datasets, the optimizer momentum might be bad for the new data.
            # We will load the weights, but NOT the optimizer/step state so it trains fresh 
            # on the new data curriculum!
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                # ADD THIS: Load the optimizer momentum so it doesn't start from scratch!
                if 'optimizer_state' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("  -> Optimizer momentum restored.")
            else:
                model.load_state_dict(checkpoint)
            
            print(f"  Weights loaded. Starting Phase 2 curriculum at Epoch 1.")
        else:
            print(f"  Checkpoint {args.resume} not found, starting fresh.")
            
    loss_history = []
    t_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        perm = torch.randperm(len(train_data))
        train_data_shuffled = train_data[perm]
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for i in range(0, len(train_data_shuffled) - args.batch_size, args.batch_size):
            batch = train_data_shuffled[i:i + args.batch_size].to(device)
            
            optimizer.zero_grad()
            
            # 2. Wrap the forward pass in BFloat16 Autocast
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(batch[:, :-1], batch[:, 1:])
            
            # 3. Scale the loss and backpropagate
            scaler.scale(loss).backward()
            
            # Unscale before clipping gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 4. Step optimizer and scaler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            loss_history.append(loss.item())
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            if global_step % args.log_every == 0:
                elapsed = time.time() - t_start
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {global_step:6d} | "
                      f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
        
        avg_epoch = epoch_loss / max(epoch_steps, 1)
        print(f"=== Epoch {epoch+1} Complete | Avg Loss: {avg_epoch:.4f} ===")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = f'moire_phase2_ep{epoch+1}.pt'
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': {
                    'n_layer': config.n_layer, 'n_head': config.n_head,
                    'n_embd': config.n_embd, 'max_seq_len': config.max_seq_len,
                }
            }, ckpt_path)
            
            weights_path = f'moire_phase2_weights_ep{epoch+1}.pt'
            torch.save(model.state_dict(), weights_path)
            print(f"  Saved: {weights_path}")
            
    torch.save(model.state_dict(), 'moire_phase2_weights_final.pt')
    print(f"Training complete! Final weights saved.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='large', choices=['small', 'medium', 'large', 'xlarge'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4) # Lower LR for finetuning
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='ultimate',
                        choices=['mixed', 'guanaco', 'tinystories', 'fineweb', 'ultimate'])
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Model size presets
    SIZE_PRESETS = {
        'small': {'n_layer': 4, 'n_head': 8, 'n_embd': 256},
        'medium': {'n_layer': 6, 'n_head': 8, 'n_embd': 512},
        'large': {'n_layer': 8, 'n_head': 8, 'n_embd': 768},      # 104.9M params
        'xlarge': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},   # ~151M params (Tad bigger!)
    }
    p = SIZE_PRESETS[args.size]
    config = MoireGPTConfig(n_layer=p['n_layer'], n_head=p['n_head'], n_embd=p['n_embd'])
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    LOADERS = {
        'mixed': load_dataset_mixed,
        'guanaco': load_dataset_guanaco,
        'tinystories': load_dataset_tinystories,
        'fineweb': load_dataset_fineweb,
        'ultimate': load_dataset_ultimate_mix,
    }
    train_data = LOADERS[args.dataset](tokenizer, config.max_seq_len)
    
    model = MoireGPT(config)
    train(model, train_data, config, args)

if __name__ == "__main__":
    main()