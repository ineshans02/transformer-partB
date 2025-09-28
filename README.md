cat > README.md << 'EOF'
# Assignment-1: Understanding & Implementing Transformers
**Author:** Ines Hans Capitan

This repository contains the implementation and outputs for Assignment-1 (Parts B and C).

## Files
- `model.py` — Transformer model (multi-head self-attention, MLP, LayerNorm, residuals).
- `train.py` — Training script (saves checkpoint and CSV logs).
- `generate.py` — Text generation script (writes two 200-token samples).
- `tinyshakespeare.txt` — Dataset (character-level).
- `logs/training_log.csv` — Training and validation loss per epoch.
- `samples/romeo.txt`, `samples/juliet.txt` — Generated samples (200 tokens each).
- `part_c.tex` (and/or `PART_C_Report.md`) — System analysis for Part C.

## How to Run (same environment used for the assignment)
Use the existing project virtual environment `.esAI`.

```bash
# activate (if not already active)
source "/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/activate"

# ensure PyTorch is available
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/pip" install --upgrade pip
"/Users/ineshans/Desktop/UTSA Fall 2025/Embedded Systems AI/Assignment 1/.esAI/bin/pip" install torch torchvision torchaudio

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

