"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MOIRÉ CHAT — Interactive inference for any trained Moiré model              ║
║                                                                              ║
║  Auto-detects model config from checkpoint, or specify manually.             ║
║                                                                              ║
║  Usage:                                                                      ║
║    python moire_chat.py                                    # uses defaults   ║
║    python moire_chat.py --weights moire_phase2_weights_ep4.pt --size xlarge  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn.functional as F
import sys
import os
import argparse

# Import architecture — try both trainer versions
try:
    from moire_conv_trainer_v5 import MoireGPT, MoireGPTConfig
except ImportError:
    try:
        from moire_conv_trainer_v3 import MoireGPT, MoireGPTConfig
    except ImportError:
        print("Error: Could not import MoireGPT.")
        print("Make sure moire_conv_trainer_v4.py or v3 is in the same folder.")
        sys.exit(1)


def load_model(args):
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Try to load config from full checkpoint
    config_dict = None
    weights_path = args.weights
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if 'config' in ckpt:
            config_dict = ckpt['config']
            print(f"  Config from checkpoint: {config_dict}")
        weights_path = args.checkpoint  # Will extract model_state below
    
    # Build config
    if config_dict:
        config = MoireGPTConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=config_dict.get('n_layer', 4),
            n_head=config_dict.get('n_head', 8),
            n_embd=config_dict.get('n_embd', 256),
            max_seq_len=config_dict.get('max_seq_len', 257),
            gamma_slots=config_dict.get('gamma_slots', 8),
            use_theta_gating=True,
        )
    else:
        # Use size preset (Added xlarge!)
        PRESETS = {
            'small':  {'n_layer': 4,  'n_head': 8,  'n_embd': 256, 'max_seq_len': 129},
            'medium': {'n_layer': 6,  'n_head': 8,  'n_embd': 512, 'max_seq_len': 257},
            'large':  {'n_layer': 8,  'n_head': 8,  'n_embd': 768, 'max_seq_len': 257},
            'xlarge': {'n_layer': 12, 'n_head': 12, 'n_embd': 768, 'max_seq_len': 257},
        }
        p = PRESETS[args.size]
        config = MoireGPTConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=p['n_layer'], n_head=p['n_head'], n_embd=p['n_embd'],
            max_seq_len=p['max_seq_len'], gamma_slots=8, use_theta_gating=True,
        )
    
    print(f"Initializing Moiré model ({config.n_layer}L, {config.n_head}H, {config.n_embd}E)...")
    model = MoireGPT(config)
    
    # Load weights
    print(f"Loading weights from {weights_path}...")
    try:
        state = torch.load(weights_path, map_location=args.device, weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            model.load_state_dict(state)
    except FileNotFoundError:
        print(f"Error: {weights_path} not found!")
        sys.exit(1)
    
    model.to(args.device)
    
#   # Only compress to bfloat16 if we are using the GPU!
#   if args.device == 'cuda':
#       model.bfloat16()
        
    model.eval()
    
    return model, tokenizer, config


def generate(model, tokenizer, config, prompt, max_tokens=80, temperature=0.7, 
             top_k=40, top_p=0.9, device='cuda'):
    """Generate with top-k AND top-p (nucleus) sampling for better quality."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print("Moiré: ", end="", flush=True)
    
    for _ in range(max_tokens):
        idx_cond = input_ids[:, -(config.max_seq_len - 1):]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        word = tokenizer.decode(next_token[0].tolist())
        print(word, end="", flush=True)
        
        # Stop at newline after "Bot:" response to prevent rambling
        decoded_so_far = tokenizer.decode(input_ids[0].tolist())
        if decoded_so_far.count('\n') > prompt.count('\n') + 2:
            break
    
    print()
    return input_ids


def main():
    parser = argparse.ArgumentParser(description="Moiré Chat Interface")
    parser.add_argument('--weights', type=str, default='moire_conv_weights_final.pt',
                        help='Path to model weights (.pt)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to full checkpoint (auto-detects config)')
    parser.add_argument('--size', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'xlarge'],
                        help='Model size if no checkpoint config available')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=80)
    parser.add_argument('--mode', type=str, default='chat',
                        choices=['chat', 'complete'],
                        help='chat: formats as User/Bot. complete: raw completion')
    args = parser.parse_args()
    
    print(f"=== Moiré Attention Chat ===")
    print(f"Device: {args.device.upper()}")
    print()
    
    model, tokenizer, config = load_model(args)
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'='*50}")
    print(f"Moiré field ready. {n_params:.1f}M parameters.")
    if args.mode == 'chat':
        print(f"Chat mode: your input becomes 'User: ...' and model generates 'Bot: ...'")
    else:
        print(f"Completion mode: model continues your text directly.")
    print(f"Temperature: {args.temperature} | Max tokens: {args.max_tokens}")
    print(f"Type 'quit' to exit.")
    print(f"{'='*50}\n")
    
    while True:
        try:
            user_input = input("You: " if args.mode == 'chat' else "Prompt: ")
            if user_input.lower().strip() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
            
            if args.mode == 'chat':
                prompt = f"User: {user_input}\nBot:"
            else:
                prompt = user_input
            
            generate(model, tokenizer, config, prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=args.device)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()