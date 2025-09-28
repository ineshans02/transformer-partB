# Tiny Transformer — Assignment-1 (Part B)

**Author:** Ines Hans Capitan  
**Course:** Embedded Systems for AI  
**Task:** Train a tiny Transformer (character-level) on Tiny Shakespeare and generate 200 tokens from prompts like `ROMEO:` or `JULIET:`.

---

## Repo Structure

transformer-partB/
├─ model.py # Transformer model (MHA, MLP, LayerNorm, residuals)
├─ train.py # training loop, logs CSV per epoch, checkpoint
├─ generate.py # loads checkpoint and writes 2 samples (Romeo/Juliet)
├─ tinyshakespeare.txt # dataset (put here)
├─ vocab.json # saved by train.py
├─ config.json # saved by train.py
├─ checkpoints/ # weights (ignored by git)
├─ logs/ # training_log.csv (ignored by git)
└─ samples/ # romeo.txt, juliet.txt (ignored by git)


> `.gitignore` excludes `checkpoints/`, `logs/`, `samples/`, and local venvs.

---

## Instructions

### 0) Open the project folder
```bash
cd "/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/transformer-partB"

### 1)
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/pip" install --upgrade pip
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/pip" install torch torchvision torchaudio

###2
cp logs/training_log.csv ./training_log.csv
cp samples/romeo.txt ./romeo.txt
cp samples/juliet.txt ./juliet.txt

## explanation
I trained the tiny Transformer for 3000 steps on my MacBook Air using MPS (Apple GPU). The model has ~3.22M parameters (4 layers, d_model=256, 4 heads, d_ff=1024, dropout 0.1; context 128, batch 64). The validation loss decreased smoothly from ~4.38 at the start to ~1.571 at the end, showing steady learning. The generated samples from prompts “ROMEO:” and “JULIET:” look Shakespeare-like in structure (character names, line breaks, punctuation) but still invent words at this size and training length — which is expected for a char-level model. With more steps (e.g., 10k–20k) or a larger hidden size, fluency and coherence should improve further.

# go to repo
cd "/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/transformer-partB"

# ensure env tools
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/pip" install --upgrade pip
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/pip" install torch torchvision torchaudio

# train
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/python" train.py

# generate (after checkpoint exists)
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/python" generate.py

