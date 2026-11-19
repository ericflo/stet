"""The fixed training run and the exam.

Training is budgeted in tokens, never in seconds: every machine does exactly the same
work, in exactly the same order, on exactly the tokens the edit script produced. There is
no shuffling here. The order of the stream is the curator's to choose.
"""

from __future__ import annotations

import math
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import GPT
from .spec import Scale, Season


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_info(device: torch.device) -> dict:
    info = {
        "device": device.type,
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 2**30, 1)
        info["backend"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return info


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def precision(device: torch.device) -> str:
    """The arithmetic the student trains in. Official grades must all use the same one."""
    return "bf16" if device.type == "cuda" else "fp32"


def weights_sha256(model: torch.nn.Module) -> str:
    """Hash of a model's weights. Two machines that start a round agree on this exactly."""
    import hashlib

    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()


def build_model(scale: Scale, season: Season, init_seed: int, device: torch.device) -> GPT:
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    # Initialise on the CPU so the starting weights are bit-identical on every machine.
    model = GPT(scale.n_layer, scale.n_head, scale.d_model, scale.ctx, season.vocab_size)
    return model.to(device)


def lr_at(step: int, steps: int, peak: float, warmup_frac: float, final_frac: float) -> float:
    warm = max(1, int(steps * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def train(
    tokens: np.ndarray,
    scale: Scale,
    season: Season,
    init_seed: int,
    device: torch.device,
    progress=None,
    max_steps: int | None = None,
) -> tuple[GPT, dict]:
    """Train the student for exactly the budget. Row b of `tokens` is lane b; step t reads
    columns [t * ctx, (t + 1) * ctx] of every lane. Nothing is shuffled."""
    cfg = season.train_cfg
    torch.set_float32_matmul_precision("high")
    model = build_model(scale, season, init_seed, device)
    init_sha256 = weights_sha256(model)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=scale.lr,
        betas=(cfg["beta1"], cfg["beta2"]),
        fused=device.type == "cuda",
    )
    steps = scale.steps if max_steps is None else min(scale.steps, max_steps)
    per_step = scale.batch_size * scale.ctx
    assert tokens.shape == (scale.batch_size, scale.steps * scale.ctx + 1), "tokens do not match the training budget"
    log_every = max(1, steps // 200)
    curve: list[list[float]] = []
    acc = torch.zeros((), device=device)
    acc_n = 0
    model.train()
    t0 = time.time()
    for step in range(steps):
        a = step * scale.ctx
        chunk = torch.from_numpy(tokens[:, a : a + scale.ctx + 1].astype(np.int64)).to(device, non_blocking=True)
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        for g in opt.param_groups:
            g["lr"] = lr_at(step, scale.steps, scale.lr, cfg["warmup_frac"], cfg["final_lr_frac"])
        with _autocast(device):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        acc += loss.detach()
        acc_n += 1
        if (step + 1) % log_every == 0 or step + 1 == steps:
            mean = float(acc / acc_n)
            if not math.isfinite(mean):
                raise FloatingPointError(f"training diverged at step {step + 1}")
            curve.append([step + 1, round(mean, 4)])
            acc.zero_()
            acc_n = 0
            if progress:
                progress(step + 1, steps, mean, time.time() - t0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    info = {
        "steps": steps,
        "tokens": steps * per_step,
        "seconds": round(seconds, 2),
        "tokens_per_second": round(steps * per_step / max(seconds, 1e-9)),
        "final_train_loss": curve[-1][1] if curve else None,
        "curve": curve,
        "params": model.n_params(),
        "init_sha256": init_sha256,
        "precision": precision(device),
    }
    return model, info


