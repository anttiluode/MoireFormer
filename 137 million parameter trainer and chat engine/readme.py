# 🌊 MoiréFormer Advanced Scaling (v5)

This subfolder contains the next-generation scaling scripts for the MoiréFormer architecture. While the main repository hosts the 104.9M proof-of-concept, these scripts are designed to train and interact with the **137.9M parameter (`xlarge`) wave-field** using our highly curated "Ultimate Mix" curriculum.

### 📂 Contents

* **`moire_conv_trainer_v5.py`**: The advanced high-performance trainer.
* **`moire_chat3.py`**: The upgraded interactive inference engine.

---

### 🚀 What's New in v5?

**1. The `xlarge` Architecture (138M Params):** Scaled up to 12 Layers and 12 Heads (768 embedding dimension) for deeper phase-interference complexity and wider context resonance.

**2. The "Ultimate Mix" Curriculum:** A meticulously balanced dataloader designed to cure hallucinations and expand the semantic phase-space by combining:
* *Guanaco:* For conversational persona and instruction following.
* *TinyStories:* For narrative logic, object permanence, and grammar cohesion.
* *FineWeb-Edu:* For factual grounding and separating phase-clumps.

**3. Hardware Optimization:** Native support for PyTorch AMP (Automatic Mixed Precision) via `GradScaler` to maximize iteration speed on cloud hardware (like Kaggle P100/T4x2 GPUs) while preventing gradient underflow.

**4. Advanced Inference (`moire_chat3.py`):** * **Real-time streaming:** Tokens are printed word-by-word as the wave-field collapses.
* **Nucleus Sampling:** Added Top-P filtering alongside Top-K for highly coherent, non-repetitive text generation.
* **Smart Loading:** Auto-detects model configurations (Layers, Heads) directly from saved full-state `.pt` checkpoints.

---

### 🛠️ How to Use

#### 1. Training the Ultimate Mix
To fire up the trainer using the new curriculum and mixed precision:

    python moire_conv_trainer_v5.py --size xlarge --dataset ultimate --epochs 10 --batch_size 2

*Note: The trainer automatically saves both full-state checkpoints (for resuming) and raw weights (for pure inference).*

#### 2. Chatting with the Model
Place your trained checkpoint in this folder and run the upgraded chat engine:

    python moire_chat3.py --checkpoint moire_phase2_ep1.pt --size xlarge --temperature 0.7