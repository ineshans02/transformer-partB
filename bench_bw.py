# bench_bw.py
import time, torch
device = "mps" if torch.backends.mps.is_available() else "cpu"

Nbytes = 512 * 1024 * 1024  # 512MB
x = torch.empty(Nbytes, dtype=torch.uint8, device=device)
y = torch.empty_like(x)

torch.mps.synchronize() if device=="mps" else None
t0 = time.time()
y.copy_(x)  # device-device copy
torch.mps.synchronize() if device=="mps" else None
dt = time.time() - t0
gbps = (Nbytes / dt) / 1e9
print(f"Device={device} copy  size={Nbytes/1e6:.1f}MB  time={dt:.3f}s  ~{gbps:.1f} GB/s")
