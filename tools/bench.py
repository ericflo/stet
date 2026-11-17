#!/usr/bin/env python3
"""Throughput of candidate student configs on this machine (random tokens)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.nn.functional as F

from stet.model import GPT

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = [  # name, layers, heads, d, ctx, bs, vocab
    ("nano-50k", 4, 4, 192, 256, 16, 50257),
    ("nano-8k", 4, 4, 192, 256, 16, 8192),
    ("micro-50k", 6, 6, 384, 512, 16, 50257),
    ("micro-8k", 6, 6, 384, 512, 16, 8192),
    ("micro-8k-b32", 6, 6, 384, 512, 32, 8192),
    ("micro-16k", 6, 6, 384, 512, 16, 16384),
    ("std-8k", 8, 8, 512, 512, 32, 8192),
    ("std-50k", 8, 8, 512, 512, 32, 50257),
]
only = sys.argv[1:]
for name, L, H, D, C, B, V in configs:
    if only and name not in only:
        continue
    torch.manual_seed(0)
    m = GPT(L, H, D, C, V).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=dev.type == "cuda")
    x = torch.randint(0, V, (B, C), device=dev); y = torch.randint(0, V, (B, C), device=dev)
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = m(x)
        loss = F.cross_entropy(logits.float().view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for _ in range(5): step()
    if dev.type == "cuda": torch.cuda.synchronize()
    n = 30; t = time.time()
    for _ in range(n): step()
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = time.time() - t
    print(f"{name:14s} params {m.n_params()/1e6:6.1f}M  {n*B*C/dt:9,.0f} tok/s  {dt/n*1000:6.1f} ms/step", flush=True)
