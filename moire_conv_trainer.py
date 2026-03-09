"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MOIRÉ CONVERSATIONAL TRAINER v2                                             ║
║                                                                              ║
║  Trains the phase-interference model on conversational data.                 ║
║  Multiple dataset options, checkpoint resume, bigger model configs.          ║
║                                                                              ║
║  Recommended run on RTX 3060 (12GB):                                         ║
║    python moire_conv_trainer.py --size medium --epochs 15 --dataset mixed    ║
║                                                                              ║
║  Antti Luode — PerceptionLab | Claude (Opus 4.6) — Architecture              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

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
# 1. ARCHITECTURE (unchanged from the proven design)
# ============================================================================

@dataclass
class MoireGPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 257      # 256 + 1 for target offset
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
        
        # --- OPTIMIZED MOIRÉ INTERFERENCE ---
        # Convert Polar (Amp, Phase) to Cartesian (Real, Imaginary)
        q_real = q_amp * torch.cos(q_phase)
        q_imag = q_amp * torch.sin(q_phase)
        
        k_real = k_amp * torch.cos(k_phase)
        k_imag = k_amp * torch.sin(k_phase)
        
        # Re(Q * K*) = (Q_real @ K_real^T) + (Q_imag @ K_imag^T)
        # This replaces the 5D broadcast with standard 4D Matrix Multiplication!
        real_scores = torch.matmul(q_real, k_real.transpose(-1, -2))
        imag_scores = torch.matmul(q_imag, k_imag.transpose(-1, -2))
        
        scores = (real_scores + imag_scores) * self.scale
        # ------------------------------------
        
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
        assert T <= self.config.max_seq_len, f"Seq len {T} > max {self.config.max_seq_len}"
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, attention_mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
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
# 2. DATASET LOADING — Multiple options
# ============================================================================

def load_dataset_mixed(tokenizer, seq_len: int, max_wiki_chars: int = 5_000_000):
    """
    Load a mix of conversational + general knowledge data.
    
    Conversational: databricks-dolly-15k (instruction following)
    General knowledge: wikitext-2 (language structure)
    
    The mix ensures the model learns both conversation format AND 
    general language competence.
    """
    from datasets import load_dataset
    
    all_text = []
    
    # --- Dolly 15k (conversational) ---
    print("Loading databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    for row in ds:
        user_text = row['instruction'].strip()
        if row['context'].strip():
            user_text += "\n" + row['context'].strip()
        bot_text = row['response'].strip()
        all_text.append(f"User: {user_text}\nBot: {bot_text}\n")
    print(f"  Dolly: {len(ds):,} conversations")

    # --- WikiText-2 (general knowledge) ---
    print("Loading wikitext-2 for general language knowledge...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_text = "\n".join([t for t in wiki['text'] if len(t.strip()) > 50])
    if len(wiki_text) > max_wiki_chars:
        wiki_text = wiki_text[:max_wiki_chars]
    all_text.append(wiki_text)
    print(f"  WikiText: {len(wiki_text):,} chars")

    full_text = "\n".join(all_text)
    print(f"Total text: {len(full_text):,} chars")
    
    return _tokenize_text(full_text, tokenizer, seq_len)


def load_dataset_dolly(tokenizer, seq_len: int):
    """Dolly-15k only — pure conversational."""
    from datasets import load_dataset
    
    print("Loading databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    text_chunks = []
    for row in ds:
        user_text = row['instruction'].strip()
        if row['context'].strip():
            user_text += "\n" + row['context'].strip()
        bot_text = row['response'].strip()
        text_chunks.append(f"User: {user_text}\nBot: {bot_text}\n")
    
    full_text = "\n".join(text_chunks)
    print(f"Total: {len(full_text):,} chars from {len(ds):,} conversations")
    return _tokenize_text(full_text, tokenizer, seq_len)


def load_dataset_wiki(tokenizer, seq_len: int):
    """WikiText-2 only — general language (the proven benchmark)."""
    from datasets import load_dataset
    
    print("Loading wikitext-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join([t for t in ds['text'] if len(t.strip()) > 50])
    print(f"Total: {len(text):,} chars")
    return _tokenize_text(text, tokenizer, seq_len)


def load_dataset_openassistant(tokenizer, seq_len: int):
    """OpenAssistant oasst1 — real human multi-turn conversations."""
    from datasets import load_dataset
    
    print("Loading OpenAssistant/oasst1...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  Failed to load oasst1: {e}")
        print("  Falling back to dolly...")
        return load_dataset_dolly(tokenizer, seq_len)
    
    # oasst1 has a tree structure. We'll flatten: group by parent_id
    # For simplicity, just use each message as a standalone text
    text_chunks = []
    for row in ds:
        role = row.get('role', 'unknown')
        text = row.get('text', '').strip()
        if not text:
            continue
        if role == 'prompter':
            text_chunks.append(f"User: {text}")
        elif role == 'assistant':
            text_chunks.append(f"Bot: {text}")
    
    full_text = "\n".join(text_chunks)
    print(f"Total: {len(full_text):,} chars from {len(text_chunks):,} messages")
    return _tokenize_text(full_text, tokenizer, seq_len)


def _tokenize_text(text: str, tokenizer, seq_len: int):
    """Convert text to overlapping token sequences, avoiding tokenizer memory limits."""
    # Temporarily disable the tokenizer's max_length warning for the bulk encode
    old_model_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30) # Set to infinity essentially
    
    # We tokenize in chunks if the text is truly massive to prevent RAM spikes
    chunk_size = 1_000_000 
    tokens = []
    
    print("Tokenizing data...")
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        tokens.extend(tokenizer.encode(chunk, add_special_tokens=False))
        
    # Restore the original max length
    tokenizer.model_max_length = old_model_max_len

    stride = seq_len // 2
    sequences = []
    
    # Now build the actual training sequences safely
    for i in range(0, len(tokens) - seq_len, stride):
        sequences.append(tokens[i:i + seq_len])
    
    data = torch.tensor(sequences, dtype=torch.long)
    print(f"Training sequences: {len(sequences):,} × {seq_len} tokens")
    return data


# ============================================================================
# 3. TRAINING
# ============================================================================

def train(model, train_data, config, args):
    device = args.device
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    n_batches = len(train_data) // args.batch_size
    total_steps = args.epochs * n_batches
    warmup_steps = min(200, total_steps // 10)
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('step', 0)
            # Fast-forward scheduler
            for _ in range(global_step):
                scheduler.step()
            print(f"  Resumed at epoch {start_epoch}, step {global_step}")
        else:
            print(f"  Checkpoint {args.resume} not found, starting fresh.")
    
    loss_history = []
    t_start = time.time()
    
    print(f"\n{'='*72}")
    print(f"TRAINING: Moiré Attention Conversational Model")
    print(f"{'='*72}")
    print(f"  Model: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params")
    print(f"  Data: {len(train_data):,} sequences")
    print(f"  Epochs: {start_epoch+1}–{args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  Total steps: ~{total_steps:,} | Warmup: {warmup_steps}")
    print(f"  Checkpoint every: {args.save_every} epochs")
    print()
    
    for epoch in range(start_epoch, args.epochs):
        perm = torch.randperm(len(train_data))
        train_data_shuffled = train_data[perm]
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for i in range(0, len(train_data_shuffled) - args.batch_size, args.batch_size):
            batch = train_data_shuffled[i:i + args.batch_size].to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits, loss = model(input_ids, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            loss_val = loss.item()
            loss_history.append(loss_val)
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1
            
            if global_step % args.log_every == 0:
                elapsed = time.time() - t_start
                lr_now = scheduler.get_last_lr()[0]
                avg = epoch_loss / epoch_steps
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {global_step:6d} | "
                      f"Loss: {loss_val:.4f} | Avg: {avg:.4f} | "
                      f"LR: {lr_now:.2e} | {elapsed:.0f}s")
        
        avg_epoch = epoch_loss / max(epoch_steps, 1)
        print(f"=== Epoch {epoch+1} Complete | Avg Loss: {avg_epoch:.4f} ===")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = f'moire_conv_ep{epoch+1}.pt'
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch + 1,
                'step': global_step,
                'loss': avg_epoch,
                'config': {
                    'n_layer': config.n_layer,
                    'n_head': config.n_head,
                    'n_embd': config.n_embd,
                    'max_seq_len': config.max_seq_len,
                    'gamma_slots': config.gamma_slots,
                }
            }, ckpt_path)
            
            # Also save just the model weights for the chat interface
            weights_path = f'moire_conv_weights_ep{epoch+1}.pt'
            torch.save(model.state_dict(), weights_path)
            print(f"  Saved: {ckpt_path} (full) + {weights_path} (weights only)")
    
    # Final save
    torch.save(model.state_dict(), 'moire_conv_weights_final.pt')
    
    # Save loss curve
    with open('moire_conv_losses.json', 'w') as f:
        json.dump({'losses': loss_history, 'epochs': args.epochs}, f)
    
    print(f"\nTraining complete! Final weights: moire_conv_weights_final.pt")
    print(f"Loss curve: moire_conv_losses.json")
    
    return loss_history


# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Moiré Conversational Trainer")
    
    # Model size presets
    parser.add_argument('--size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Model size preset')
    
    # Override individual config values
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--seq_len', type=int, default=256,
                        help='Context length (tokens)')
    parser.add_argument('--gamma_slots', type=int, default=8)
    
    # Training
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Data
    parser.add_argument('--dataset', type=str, default='mixed',
                        choices=['mixed', 'dolly', 'wiki', 'oasst'],
                        help='Dataset to train on')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # ── Model size presets ──
    # Designed for RTX 3060 12GB with the given batch sizes
    SIZE_PRESETS = {
        'small': {   # ~16M params, 64MB weights — the proven config
            'n_layer': 4, 'n_head': 8, 'n_embd': 256,
            'batch_size': 8,
        },
        'medium': {  # ~60M params, ~240MB weights — sweet spot for 12GB
            'n_layer': 6, 'n_head': 8, 'n_embd': 512,
            'batch_size': 4,
        },
        'large': {   # ~120M params, ~480MB — pushes 12GB, use batch 2
            'n_layer': 8, 'n_head': 8, 'n_embd': 768,
            'batch_size': 2,
        },
    }
    
    preset = SIZE_PRESETS[args.size]
    n_layer = args.n_layer or preset['n_layer']
    n_head = args.n_head or preset['n_head']
    n_embd = args.n_embd or preset['n_embd']
    if args.batch_size == 4 and args.size != 'medium':
        # Only override batch_size if user didn't explicitly set it
        args.batch_size = preset['batch_size']
    
    print(f"Device: {args.device}")
    print(f"Model size: {args.size} (n_layer={n_layer}, n_head={n_head}, n_embd={n_embd})")
    print()
    
    # ── Tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # ── Config ──
    config = MoireGPTConfig(
        max_seq_len=args.seq_len + 1,  # +1 for target offset
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        gamma_slots=args.gamma_slots,
    )
    
    # ── Dataset ──
    LOADERS = {
        'mixed': load_dataset_mixed,
        'dolly': load_dataset_dolly,
        'wiki': load_dataset_wiki,
        'oasst': load_dataset_openassistant,
    }
    train_data = LOADERS[args.dataset](tokenizer, config.max_seq_len)
    
    if train_data is None or len(train_data) == 0:
        print("ERROR: No training data loaded!")
        return
    
    # ── Model ──
    model = MoireGPT(config)
    
    # ── Train ──
    train(model, train_data, config, args)
    
    # ── Sample generation ──
    print()
    print("─" * 72)
    print("GENERATION SAMPLES")
    print("─" * 72)
    
    model.eval()
    model.to(args.device)
    prompts = [
        "User: What is the capital of France?\nBot:",
        "User: Tell me a story about a cat.\nBot:",
        "User: How does gravity work?\nBot:",
    ]
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(args.device)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=60, temperature=0.7)
        text_out = tokenizer.decode(generated[0].tolist())
        # Show just the bot response
        if "Bot:" in text_out:
            response = text_out.split("Bot:")[-1].strip()
        else:
            response = text_out[len(prompt):]
        print(f"  Q: {prompt.split(chr(10))[0].replace('User: ', '')}")
        print(f"  A: {response[:200]}")
        print()


if __name__ == "__main__":
    main()
