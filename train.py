import os, json, time, urllib.request
import torch
from model import TinyTransformer

# ---- Config ----
cfg = {
    "batch_size": 64,
    "block_size": 128,
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "lr": 3e-4,
    "epochs": 15,              # 15 * 200 = 3000 steps
    "iters_per_epoch": 200,
    "eval_iters": 100,
    "grad_clip": 1.0,
    "seed": 42,
    "data_path": "tinyshakespeare.txt",
    "data_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

torch.manual_seed(cfg["seed"])
device = cfg["device"]
print(f"Using device: {device}")

# ---- Data ----
if not os.path.exists(cfg["data_path"]):
    try:
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(cfg["data_url"], cfg["data_path"])
    except Exception as e:
        raise RuntimeError(f"Place the file at {os.path.abspath(cfg['data_path'])}. Error: {e}")

with open(cfg["data_path"], "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump({"chars": chars}, f, ensure_ascii=False, indent=2)
with open("config.json", "w", encoding="utf-8") as f:
    json.dump({
        "vocab_size": vocab_size,
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "d_ff": cfg["d_ff"],
        "block_size": cfg["block_size"],
        "dropout": cfg["dropout"],
    }, f, indent=2)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join([itos[int(i)] for i in t])

data = encode(text)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - cfg["block_size"] - 1, (cfg["batch_size"],))
    x = torch.stack([source[i:i+cfg["block_size"]] for i in ix])
    y = torch.stack([source[i+1:i+cfg["block_size"]+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(iters=100):
    model.eval()
    out = {}
    total = 0.0
    for split in ["train", "val"]:
        total = 0.0
        for _ in range(iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total += loss.item()
        out[split] = total / iters
    model.train()
    return out

# ---- Model & Opt ----
model = TinyTransformer(
    vocab_size=vocab_size,
    d_model=cfg["d_model"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    d_ff=cfg["d_ff"],
    block_size=cfg["block_size"],
    dropout=cfg["dropout"],
).to(device)

print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

# ---- Train ----
csv_path = os.path.join("logs", "training_log.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("epoch,train_loss,val_loss,elapsed_s\n")

start = time.time()
for epoch in range(1, cfg["epochs"] + 1):
    for _ in range(cfg["iters_per_epoch"]):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

    losses = estimate_loss(iters=cfg["eval_iters"])
    elapsed = time.time() - start
    print(f"epoch {epoch:3d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f} | {elapsed:.1f}s")
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{losses['train']:.3f},{losses['val']:.3f},{elapsed:.1f}\n")

ckpt_path = os.path.join("checkpoints", "tiny_transformer.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"Saved model to {ckpt_path}")
