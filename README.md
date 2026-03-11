# MoireFormer (138M) 🌊

**MoireFormer** is a novel neural network architecture that entirely replaces standard QKV dot-product attention with **continuous biological wave mechanics**. 

Instead of computing discrete token similarities via matrix multiplications, MoireFormer splits token embeddings into amplitude and phase. It routes information and calculates context through the geometric constructive and destructive interference of phase-shifted wave-fields (Moiré patterns).

🎮 **[Try the Live 138M Chat on Hugging Face Space](https://huggingface.co/spaces/Aluode/MoireFormer137MillionP)**

---

## Model Details
At ~138M parameters, this is a **proof-of-substrate** model. It proves that the biological concept of ephaptic phase-coupling can successfully serve as a foundation for deep learning. The wave-field successfully learns grammar, conversational formatting, and multilingual text natively in continuous phase-space.

* **Architecture:** MoireGPT (Custom Phase-Interference Transformer)
* **Parameters:** 137.9M
* **Structure:** 12 Layers, 12 Heads, 768 Embedding Dimension
* **Theory & Background:** Read the origin of the architecture at the [Geometric-Neuron Repository](https://github.com/anttiluode/Geometric-Neuron)

---

## How to Run Locally

Because this uses a custom wave-interference architecture, it cannot be loaded with standard Hugging Face `AutoModel`. You must run it using the provided engine.

### 1. Clone the repository
```bash
git clone [https://github.com/anttiluode/MoireFormer.git](https://github.com/anttiluode/MoireFormer.git)
cd MoireFormer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the weights

Download the stripped 138M parameter weights (moire_phase2_weights_ep4.pt) (Actually 6 epochs trained via my 3060 and resumed for 4 more at 
kaggle with P100) from the Hugging Face hub and place the file in your root directory:

https://huggingface.co/spaces/Aluode/MoireFormer137MillionP/tree/main

### 4. Run the Chat Engine

```bash
python moire_chat3.py --weights moire_phase2_weights_ep4.pt --size xlarge
```

Or try the huggingface space app at: 

https://huggingface.co/spaces/Aluode/MoireFormer137MillionP (No install) 

# Training Your Own

To finetune the network or train the wave-field from scratch, use the `v5` curriculum trainer included in this repository. 
It automatically downloads and tokenizes a high-quality mixed corpus (Guanaco, TinyStories, FineWeb) to teach the model conversational structure, 
logic, and grammar.

```bash
python moire_conv_trainer_v5.py --size xlarge --batch_size 2 --epochs 10
```
Key Arguments:

--size xlarge: Sets the model to 12 layers, 12 heads, and 768 embedding dim (~138M params).

--batch_size 2: Recommended for consumer GPUs (like a 16GB T4 or P100).

--epochs 10: The number of complete passes through the dataset.

--resume <file.pt>: If your run drops or you want to train in chunks, use this to pick up exactly where the last saved weights left off.

# Scaling Up (The Billion Parameter Dream)

At 138 million parameters, this repository serves as a proof-of-concept. However, the underlying phase-interference 
mathematics are highly scalable.

Because Moiré Attention routes information through continuous wave resonance rather than rigid scalar matrices, we
believe it holds massive potential for scaling. If you want to push the boundaries, you can easily modify the SIZE_PRESETS inside the
script to expand the dimensions.

Feel free to feed these Python files to your favorite AI assistant—it will easily understand the geometric wave-routing mechanics
and can help you write the multi-GPU distributed code required to train a 1-Billion+ parameter version.

This architecture was inspired by my Geometric Neuron / Deerskin Neuron concepts. Those ideas were born after a year and a half 
of discussions with various AIs about the fundamental mechanics of consciousness. While this model isn't the pure expression of
that biological idea, it is a functional form of it bolted onto a Transformer chassis to prove the math works.

Read the core theories here: 

https://github.com/anttiluode/Geometric-Neuron

# License

This project is licensed under the MIT License.
