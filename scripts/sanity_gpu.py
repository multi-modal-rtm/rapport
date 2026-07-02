import time

import torch
import torch.nn as nn

print(f"torch.__version__ = {torch.__version__}")
print(f"torch.version.cuda = {torch.version.cuda}")
print(f"cuda available = {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device name = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# 4096x4096 bf16 matmul, timed over 100 iterations
size = 4096
a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
b = torch.randn(size, size, device=device, dtype=torch.bfloat16)

# warmup
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()

n_iters = 100
start = time.perf_counter()
for _ in range(n_iters):
    c = a @ b
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

flops_per_matmul = 2 * size**3
total_flops = flops_per_matmul * n_iters
tflops = total_flops / elapsed / 1e12

print(f"\n4096x4096 bf16 matmul x{n_iters} iters: {elapsed:.4f}s total, "
      f"{elapsed / n_iters * 1000:.4f} ms/iter, {tflops:.2f} TFLOPS")

# Tiny forward+backward through MultiheadAttention on GPU
embed_dim = 64
num_heads = 4
seq_len = 16
batch_size = 8

mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(device)
x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)

out, _ = mha(x, x, x)
loss = out.sum()
loss.backward()
torch.cuda.synchronize()

print(f"\nMultiheadAttention forward+backward OK: out.shape={tuple(out.shape)}, "
      f"x.grad is not None = {x.grad is not None}")
print("\nSANITY CHECK PASSED")
