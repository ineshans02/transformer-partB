# bench_flops.py
import time, torch
device = "mps" if torch.backends.mps.is_available() else "cpu"

# size for a decent matmul (adjust if OOM)
N = 4096
a = torch.randn(N, N, device=device, dtype=torch.float32)
b = torch.randn(N, N, device=device, dtype=torch.float32)

# warmup
for _ in range(3):
    (a @ b).sum().item()

# timed
torch.mps.synchronize() if device=="mps" else None
t0 = time.time()
c = a @ b
torch.mps.synchronize() if device=="mps" else None
dt = time.time() - t0

# FLOPs for matmul: 2*N^3
flops = 2*(N**3)
tflops = flops / dt / 1e12
print(f"Device={device} N={N}  time={dt:.3f}s  ~{tflops:.2f} TFLOP/s (FP32, rough)")
