# MoireFormer: A Wave-Interference Transformer Bolt-On

MoireFormer is a novel neural network architecture that replaces standard scalar dot-product attention with **Moiré phase-interference wave mechanics**. It acts as a "bolt-on" replacement for standard Transformer blocks, proving that artificial intelligence can be trained using continuous, biological wave-geometry rather than discrete scalar weights.

This repository contains the training scripts, inference code, and architecture definitions for the MoireFormer proof-of-concept.

For the underlying biological theory, mathematical proofs, and EEG clinical data, please see the primary theory repository: [anttiluode/Geometric-Neuron](https://github.com/anttiluode/Geometric-Neuron).

## The Architecture
Standard Transformers calculate attention using `Q · K^T` (dot products). MoireFormer splits token embeddings into amplitude and phase (`q_amp`, `q_phase`) and computes attention via geometric wave resonance: `q_real * k_real + q_imag * k_imag`. 

Additionally, it features a biologically inspired **Theta Phase Gate**, organizing memory across temporal distances using multiplexed oscillatory rhythms (Theta/Gamma coupling).

## Proof of Concept Model: MoireGPT (104.9M)
We have successfully trained a 104.9M parameter proof-of-concept model (8 Layers, 8 Heads, 768 Embedding Dimension) that demonstrates the ability of Moiré wave-fields to learn complex human language, grammar, and bilingual conversational structures.

### Training Curriculum
The model was trained in two distinct phases to test wave-field superposition and the avoidance of catastrophic forgetting:
* **Phase 1 (Base Geometry):** 15 Epochs using `moire_conv_trainer.py`. Trained on a mixed dataset of Databricks Dolly-15k, WikiText-2, and OpenAssistant.
* **Phase 2 (Phase-Space Expansion):** 5 Epochs using `moire_conv_trainer_v3.py`. Finetuned heavily on the Guanaco dataset to refine logical geometry and conversational instruction-following.

*Note: At ~100M parameters, the model is a proof-of-substrate, not a knowledge oracle. It demonstrates coherent syntax, bilingual capabilities, and context adherence, though it will hallucinate factual data due to its small size.*

## Installation & Usage

### 1. Clone and Install
```bash
git clone [https://github.com/anttiluode/MoireFormer.git](https://github.com/anttiluode/MoireFormer.git)
cd MoireFormer
pip install torch transformers datasets (or install via requirements.txt)

2. Download the Weights
The pre-trained PyTorch weights (moire_phase2_weights_final.pt) are hosted on Hugging Face:
👉 Download Weights Here:

[moire_phase2_weights_final.pt](https://huggingface.co/Aluode/MoireFormer/blob/main/moire_phase2_weights_final.pt)

Place the .pt file in your root repository folder.

# 3. Run Inference

You can interact with the Moiré field directly using the chat script.

Bash
python moire_chat.py --weights moire_phase2_weights_final.pt --size large
Chat Options:

Chat Mode (Default): Formats inputs as User: and Bot:

Completion Mode: Use --mode complete to have the model organically continue your text.

# Repository Structure

moire_attention_gpt2.py / moire_conv_trainer.py: The core Moiré Attention block and Phase 1 mixed-dataset trainer.

moire_conv_trainer_v3.py: The advanced Phase 2 trainer with Guanaco/TinyStories/FineWeb dataset loaders.

moire_chat.py: The interactive CLI interface for querying the trained wave-field.

License
MIT License
