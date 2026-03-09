"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MOIRÉ ATTENTION: Phase-Interference Language Model                          ║
║                                                                              ║
║  Architecture: GPT-2 Small with attention replaced by wave interference.     ║
║                                                                              ║
║  The core mechanism:                                                         ║
║  - Standard attention: score = Q·K^T / sqrt(d)                               ║
║  - Moiré attention:    score = Re[Q_c · conj(K_c)]                           ║
║                                                                              ║
║  Where Q_c and K_c are complex-valued projections:                           ║
║    Q_c = Q_amp · exp(i · Q_phase)                                            ║
║    K_c = K_amp · exp(i · K_phase)                                            ║
║                                                                              ║
║  This IS the proven phase coherence metric Re[φ · exp(-iθ)] from the        ║
║  wave memory experiments (30/30 retrieval), now applied to token             ║
║  embeddings instead of spatial solitons.                                     ║
║                                                                              ║
║  Theta-Gamma Multiplexing:                                                   ║
║  - Context is divided into "theta cycles" of G tokens each                   ║
║  - Within a cycle: full phase-interference attention (gamma binding)         ║
║  - Across cycles: gated by learned theta phase offset per head               ║
║  - This creates nested temporal structure like biological PAC                ║
║                                                                              ║
║  The "static snapshot" principle: we do NOT evolve a wave equation.          ║
║  We compute the interference pattern directly and use it as attention.       ║
║  This is differentiable and trains with standard cross-entropy.              ║
║                                                                              ║
║  Antti Luode — PerceptionLab | Claude (Anthropic, Opus 4.6) — Design        ║
║  March 2026                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import time
import os
from typing import Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class MoireGPTConfig:
    """Configuration for the Moiré Attention GPT model."""
    vocab_size: int = 50257       # GPT-2 tokenizer vocab
    max_seq_len: int = 256        # Context length
    n_layer: int = 6              # Transformer layers (small model for testing)
    n_head: int = 8               # Attention heads
    n_embd: int = 512             # Embedding dimension
    gamma_slots: int = 8          # Tokens per theta cycle (the "gamma window")
    dropout: float = 0.1
    bias: bool = False            # No bias in linear layers (cleaner)
    
    # Moiré-specific
    phase_dim: int = 64           # Dimension of phase space per head
    theta_gate_temp: float = 1.0  # Temperature for theta gating softmax
    use_theta_gating: bool = True # Enable/disable theta-gamma multiplexing
    
    @property
    def head_dim(self):
        return self.n_embd // self.n_head


# ============================================================================
# 2. MOIRÉ ATTENTION — The Core Mechanism
# ============================================================================

class MoireAttention(nn.Module):
    """
    Phase-interference attention mechanism.
    
    Instead of score = Q·K^T / sqrt(d), we compute:
    
    1. Project hidden states to amplitude and phase:
       Q_amp, Q_phase = linear(x)   →  Q_c = Q_amp · exp(i · Q_phase)
       K_amp, K_phase = linear(x)   →  K_c = K_amp · exp(i · K_phase)
    
    2. Interference score = Re[Q_c · conj(K_c)]
       = Q_amp · K_amp · cos(Q_phase - K_phase)
       
       This is EXACTLY Re[φ · exp(-iθ_probe)] from the wave memory,
       where Q is the probe and K is the stored memory.
    
    3. Theta-gamma multiplexing:
       - Within each gamma window: full interference scoring
       - Across gamma windows: multiply by theta gate factor
         gate = cos(theta_offset_head · cycle_distance)
       - This creates periodic modulation of long-range attention
         exactly like biological phase-amplitude coupling
    
    The value projection remains standard (real-valued).
    """
    
    def __init__(self, config: MoireGPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.gamma_slots = config.gamma_slots
        
        # Q projection: amplitude + phase (2x head_dim per head)
        self.q_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # K projection: amplitude + phase
        self.k_proj = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # V projection: standard real-valued
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Theta gate: one learned phase offset per head
        # This controls how attention decays across theta cycles
        if config.use_theta_gating:
            self.theta_offset = nn.Parameter(
                torch.randn(config.n_head) * 0.1
            )
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(config.head_dim)
        
        # For logging / analysis
        self.last_phase_diff = None
    
    def forward(
        self,
        x: torch.Tensor,              # [B, T, C]
        attention_mask: Optional[torch.Tensor] = None,  # [B, 1, T, T] or None
    ) -> torch.Tensor:
        B, T, C = x.shape
        
        # ── 1. Project to amplitude and phase ──
        q_raw = self.q_proj(x)  # [B, T, 2*C]
        k_raw = self.k_proj(x)  # [B, T, 2*C]
        v = self.v_proj(x)      # [B, T, C]
        
        # Split into amplitude and phase components
        q_amp, q_phase = q_raw.chunk(2, dim=-1)   # each [B, T, C]
        k_amp, k_phase = k_raw.chunk(2, dim=-1)
        
        # Reshape for multi-head: [B, n_head, T, head_dim]
        q_amp = q_amp.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q_phase = q_phase.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_amp = k_amp.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_phase = k_phase.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Amplitude is positive (softplus to keep it well-behaved)
        q_amp = F.softplus(q_amp)
        k_amp = F.softplus(k_amp)
        
        # Phase is unconstrained (wraps naturally via cos)
        # No need to clamp — cos handles any real number
        
        # ── 2. Moiré interference score ──
        # score[b,h,i,j] = sum_d( q_amp[b,h,i,d] * k_amp[b,h,j,d] * cos(q_phase[b,h,i,d] - k_phase[b,h,j,d]) )
        #
        # This is Re[Q_c · conj(K_c)] summed over the head dimension.
        # Equivalent to the wave memory's phase coherence metric.
        
        # Compute phase difference: [B, n_head, T_q, T_k, head_dim]
        phase_diff = q_phase.unsqueeze(3) - k_phase.unsqueeze(2)  # [B, H, T, 1, D] - [B, H, 1, T, D]
        cos_phase = torch.cos(phase_diff)  # [B, H, T_q, T_k, D]
        
        # Amplitude product
        amp_product = q_amp.unsqueeze(3) * k_amp.unsqueeze(2)  # [B, H, T_q, T_k, D]
        
        # Interference score: sum over head_dim
        scores = (amp_product * cos_phase).sum(dim=-1)  # [B, H, T_q, T_k]
        scores = scores * self.scale
        
        # Store for analysis
        self.last_phase_diff = phase_diff.detach().mean(dim=-1)  # [B, H, T, T]
        
        # ── 3. Theta-gamma multiplexing ──
        if self.config.use_theta_gating and T > self.gamma_slots:
            # Compute which theta cycle each token belongs to
            # cycle_id[t] = t // gamma_slots
            positions = torch.arange(T, device=x.device, dtype=torch.float32)
            cycle_ids = positions / self.gamma_slots  # continuous cycle position
            
            # Cycle distance between each pair of tokens
            cycle_dist = cycle_ids.unsqueeze(0) - cycle_ids.unsqueeze(1)  # [T, T]
            
            # Theta gate: cos(theta_offset * cycle_distance)
            # For each head, different theta_offset → different periodic modulation
            # Shape: [n_head, 1, 1] * [1, T, T] → [n_head, T, T]
            theta_off = self.theta_offset.view(self.n_head, 1, 1)
            theta_gate = torch.cos(theta_off * cycle_dist.unsqueeze(0))  # [H, T, T]
            
            # Within-cycle tokens (cycle_dist < 1) get full attention (gate ≈ 1)
            # Cross-cycle tokens get modulated by cos(theta * distance)
            # This naturally creates the nested structure:
            #   - nearby tokens: strong binding (gamma)
            #   - distant tokens: periodic resonance (theta modulation)
            
            scores = scores * theta_gate.unsqueeze(0)  # [B, H, T, T]
        
        # ── 4. Causal mask ──
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # ── 5. Softmax and aggregate ──
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out


# ============================================================================
# 3. STANDARD ATTENTION (for baseline comparison)
# ============================================================================

class StandardAttention(nn.Module):
    """Vanilla scaled dot-product attention for fair comparison."""
    
    def __init__(self, config: MoireGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(config.head_dim)
    
    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


# ============================================================================
# 4. TRANSFORMER BLOCK
# ============================================================================

class MoireBlock(nn.Module):
    """Transformer block with Moiré or Standard attention."""
    
    def __init__(self, config: MoireGPTConfig, use_moire: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        if use_moire:
            self.attn = MoireAttention(config)
        else:
            self.attn = StandardAttention(config)
        
        # MLP
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


# ============================================================================
# 5. FULL MODEL
# ============================================================================

class MoireGPT(nn.Module):
    """
    Small GPT with Moiré Attention.
    
    This is NOT a pretrained model — it's trained from scratch to test
    whether phase-interference attention can learn language structure.
    
    The test: train Moiré GPT and Standard GPT on the same data with
    the same architecture (everything identical except the attention
    mechanism). Compare loss curves.
    """
    
    def __init__(self, config: MoireGPTConfig, use_moire: bool = True):
        super().__init__()
        self.config = config
        self.use_moire = use_moire
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            MoireBlock(config, use_moire=use_moire)
            for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_type = "Moiré" if use_moire else "Standard"
        print(f"[{model_type} GPT] {n_params/1e6:.1f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None, attention_mask=None):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} > max {self.config.max_seq_len}"
        
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
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
# 6. TRAINING LOOP — Self-contained, no HuggingFace Trainer
# ============================================================================

def create_dataset_from_text(text: str, tokenizer, seq_len: int, stride: int = None):
    """Create overlapping sequences from a text corpus."""
    if stride is None:
        stride = seq_len // 2
    
    tokens = tokenizer.encode(text)
    sequences = []
    for i in range(0, len(tokens) - seq_len, stride):
        sequences.append(tokens[i:i + seq_len])
    
    return torch.tensor(sequences, dtype=torch.long)


def train_model(model, train_data, config, n_epochs=3, batch_size=8, lr=3e-4, 
                device='cuda', log_every=50, model_name="model"):
    """
    Simple training loop. Returns loss history for comparison.
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate warmup + cosine decay
    n_batches = len(train_data) // batch_size
    total_steps = n_epochs * n_batches
    warmup_steps = min(100, total_steps // 10)
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    loss_history = []
    step = 0
    t_start = time.time()
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(train_data))
        train_data_shuffled = train_data[perm]
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        for i in range(0, len(train_data_shuffled) - batch_size, batch_size):
            batch = train_data_shuffled[i:i+batch_size].to(device)
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
            step += 1
            
            if step % log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - t_start
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [{model_name}] Step {step:5d} | Loss: {loss_val:.4f} | "
                      f"Avg: {avg_loss:.4f} | LR: {lr_now:.2e} | {elapsed:.0f}s")
        
        avg_epoch = epoch_loss / max(epoch_steps, 1)
        print(f"  [{model_name}] Epoch {epoch+1}/{n_epochs} complete | Avg loss: {avg_epoch:.4f}")
    
    return loss_history


# ============================================================================
# 7. PHASE ANALYSIS — Diagnostic tools
# ============================================================================

def analyze_phase_structure(model, sample_input, device='cuda'):
    """
    After training, examine what the Moiré heads learned:
    - Do different heads learn different theta offsets?
    - Is the phase structure non-trivial?
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        sample = sample_input.unsqueeze(0).to(device)
        _ = model(sample[:, :-1])
    
    analysis = {}
    
    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, 'theta_offset'):
            theta = block.attn.theta_offset.detach().cpu().numpy()
            analysis[f'layer_{i}_theta_offsets'] = theta.tolist()
            
        if hasattr(block.attn, 'last_phase_diff') and block.attn.last_phase_diff is not None:
            pd = block.attn.last_phase_diff.cpu().numpy()
            # Mean absolute phase difference per head
            mean_pd = np.mean(np.abs(pd), axis=(0, 2, 3))  # [n_head]
            analysis[f'layer_{i}_mean_phase_diff'] = mean_pd.tolist()
    
    return analysis


# ============================================================================
# 8. MAIN — The Experiment
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Moiré Attention vs Standard Attention")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--gamma_slots', type=int, default=8)
    parser.add_argument('--data_source', type=str, default='wikitext',
                        choices=['wikitext', 'tiny_shakespeare', 'synthetic'])
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip standard attention baseline')
    parser.add_argument('--log_every', type=int, default=25)
    args = parser.parse_args()
    
    device = args.device
    print(f"Device: {device}")
    print()
    
    # ── Configuration ──
    config = MoireGPTConfig(
        max_seq_len=args.seq_len + 1,  # +1 for target offset
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        gamma_slots=args.gamma_slots,
        dropout=0.1,
    )
    
    # ── Data ──
    print("=" * 72)
    print("MOIRÉ ATTENTION — Phase-Interference Language Model Experiment")
    print("=" * 72)
    print()
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if args.data_source == 'wikitext':
        print("Loading WikiText-2 dataset...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join([t for t in ds['text'] if len(t.strip()) > 50])
    elif args.data_source == 'tiny_shakespeare':
        print("Loading Tiny Shakespeare...")
        from datasets import load_dataset
        ds = load_dataset("tiny_shakespeare", split="train")
        text = ds['text'][0]
    else:
        print("Generating synthetic data...")
        # Repeat patterns that should be easy for phase-interference to detect
        text = ("The cat sat on the mat. " * 200 + 
                "The dog ran in the park. " * 200 +
                "A bird flew over the tree. " * 200)
    
    print(f"Text length: {len(text):,} chars")
    train_data = create_dataset_from_text(text, tokenizer, config.max_seq_len, 
                                          stride=config.max_seq_len // 2)
    print(f"Training sequences: {len(train_data):,} × {config.max_seq_len} tokens")
    print()
    
    # ── Train Moiré model ──
    print("─" * 72)
    print("TRAINING: Moiré Attention GPT")
    print("─" * 72)
    print(f"Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd, "
          f"gamma_slots={config.gamma_slots}")
    print()
    
    moire_model = MoireGPT(config, use_moire=True)
    moire_losses = train_model(
        moire_model, train_data, config,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, log_every=args.log_every, model_name="Moiré"
    )
    
    # ── Train Standard baseline ──
    if not args.skip_baseline:
        print()
        print("─" * 72)
        print("TRAINING: Standard Attention GPT (Baseline)")
        print("─" * 72)
        print()
        
        std_model = MoireGPT(config, use_moire=False)
        std_losses = train_model(
            std_model, train_data, config,
            n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, log_every=args.log_every, model_name="Standard"
        )
    
    # ── Results ──
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    
    # Final loss comparison (average of last 10% of training)
    tail = max(1, len(moire_losses) // 10)
    moire_final = np.mean(moire_losses[-tail:])
    print(f"Moiré Attention  — Final avg loss: {moire_final:.4f}")
    
    if not args.skip_baseline:
        std_final = np.mean(std_losses[-tail:])
        print(f"Standard Attention — Final avg loss: {std_final:.4f}")
        
        delta = moire_final - std_final
        pct = 100 * delta / std_final
        print()
        if delta < 0:
            print(f"Moiré is BETTER by {abs(delta):.4f} ({abs(pct):.1f}%)")
        elif delta > 0:
            print(f"Standard is BETTER by {delta:.4f} ({pct:.1f}%)")
        else:
            print("Identical performance")
    
    # ── Phase analysis ──
    print()
    print("─" * 72)
    print("PHASE STRUCTURE ANALYSIS")
    print("─" * 72)
    
    sample = train_data[0]
    analysis = analyze_phase_structure(moire_model, sample, device)
    
    for key, val in analysis.items():
        if 'theta' in key:
            vals = [f"{v:+.3f}" for v in val]
            print(f"  {key}: [{', '.join(vals)}]")
        elif 'phase_diff' in key:
            vals = [f"{v:.3f}" for v in val]
            print(f"  {key}: [{', '.join(vals)}]")
    
    # ── Generation samples ──
    print()
    print("─" * 72)
    print("GENERATION SAMPLES")
    print("─" * 72)
    
    moire_model.eval()
    prompts = ["The cat", "In the beginning", "Once upon a"]
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        with torch.no_grad():
            generated = moire_model.generate(input_ids, max_new_tokens=30, temperature=0.8)
        text_out = tokenizer.decode(generated[0].tolist())
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: {text_out[:120]}...")
        print()
    
    # ── Save results ──
    results = {
        'config': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'gamma_slots': config.gamma_slots,
            'max_seq_len': config.max_seq_len,
            'use_theta_gating': config.use_theta_gating,
        },
        'moire_losses': moire_losses,
        'moire_final_loss': float(moire_final),
        'phase_analysis': analysis,
    }
    
    if not args.skip_baseline:
        results['std_losses'] = std_losses
        results['std_final_loss'] = float(std_final)
        results['delta'] = float(delta)
    
    out_path = 'moire_attention_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    
    # ── Save model ──
    torch.save(moire_model.state_dict(), 'moire_gpt_weights.pt')
    print(f"Moiré model saved to moire_gpt_weights.pt")
    
    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print()
    print("What to look for:")
    print("  1. Does Moiré loss converge? (If yes: phase interference can learn language)")
    print("  2. Is Moiré loss ≤ Standard loss? (If yes: interference is competitive)")
    print("  3. Are theta offsets diverse? (If yes: heads learned different periodicities)")
    print("  4. Are phase diffs non-trivial? (If yes: the model uses phase structure)")
    print()
    print("What this means for the Deerskin Architecture:")
    print("  If the model learns, then Re[Q_c · conj(K_c)] — the SAME formula that gave")
    print("  30/30 retrieval in the wave memory — can also serve as the similarity metric")
    print("  for language. The 'static snapshot' of wave interference is sufficient for")
    print("  attention without requiring field evolution.")


if __name__ == "__main__":
    main()
