import os, json, torch
from model import TinyTransformer

def load_vocab():
    with open("vocab.json", "r", encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return chars, stoi, itos

def encode(s, stoi): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t, itos): return "".join([itos[int(i)] for i in t])

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    chars, stoi, itos = load_vocab()
    model = TinyTransformer(
        vocab_size=len(chars),
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    ).to(device)

    state = torch.load(os.path.join("checkpoints", "tiny_transformer.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs("samples", exist_ok=True)

    # ROMEO
    ctx = encode("ROMEO:", stoi).unsqueeze(0).to(device)
    out = model.generate(ctx, max_new_tokens=200, temperature=0.9, top_k=40)
    text1 = decode(out[0], itos)
    with open("samples/romeo.txt", "w", encoding="utf-8") as f:
        f.write(text1)
    print("\n=== ROMEO SAMPLE ===\n" + text1 + "\n====================")

    # JULIET
    ctx = encode("JULIET:", stoi).unsqueeze(0).to(device)
    out = model.generate(ctx, max_new_tokens=200, temperature=0.9, top_k=40)
    text2 = decode(out[0], itos)
    with open("samples/juliet.txt", "w", encoding="utf-8") as f:
        f.write(text2)
    print("\n=== JULIET SAMPLE ===\n" + text2 + "\n====================")

if __name__ == "__main__":
    main()
