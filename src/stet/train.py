"""The fixed training run and post-pretraining SFT probe.

Training is budgeted in tokens, never in seconds: every machine does exactly the same
work, in exactly the same order, on exactly the tokens the edit script produced. There is
no shuffling here. The order of the stream is the curator's to choose.

Season 1-2 train a small GPT from random weights. Season 3 starts from a pinned
multilingual base model (Qwen/Qwen3-0.6B-Base), deterministically re-initializes the upper
half of its transformer layers from the round's seed on CPU, trains unshuffled on the
curator's stream using fixed micro-batch gradient accumulation (so 16 GB+ and 32 GB+ GPUs
execute identical math), and runs a short standardized disjoint multilingual SFT probe
before sitting the exam.
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
from .tok import _hf_model_dir, encode_one


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


def _n_params(model: torch.nn.Module) -> int:
    if hasattr(model, "n_params"):
        return int(model.n_params())
    return sum(p.numel() for p in model.parameters())


def _fetch_hf_student(cfg: dict):
    """Fetch pinned HF student files into the local cache, verifying every file's sha256."""
    from .data import fetch

    model_dir = _hf_model_dir(cfg["repo"], cfg["revision"])
    model_dir.mkdir(parents=True, exist_ok=True)
    for f in cfg["files"]:
        dest = model_dir / f["path"]
        fetch(
            [f["url"]],
            f["sha256"],
            dest,
            label=f"{cfg['repo']}/{f['path']}",
            max_bytes=int(f["size"]) + (1 << 20),
        )
    return model_dir


def _reset_top_layers(model: torch.nn.Module, reset_frac: float, init_seed: int) -> None:
    """Deterministically re-initialize the upper `reset_frac` transformer blocks on CPU."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or reset_frac <= 0:
        return
    n = len(layers)
    start = n - int(round(n * reset_frac))
    gen = torch.Generator(device="cpu").manual_seed(int(init_seed))
    with torch.no_grad():
        for layer in layers[start:]:
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.normal_(mean=0.0, std=0.02, generator=gen)
                    if m.bias is not None:
                        m.bias.zero_()
                elif hasattr(m, "weight") and m.weight is not None and m.weight.ndim == 1:
                    m.weight.fill_(1.0)
                    if getattr(m, "bias", None) is not None:
                        m.bias.zero_()


def build_model(scale: Scale, season: Season, init_seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    student = season.raw.get("student") or {}
    if student.get("kind") == "hf_topreset":
        from transformers import AutoModelForCausalLM

        model_dir = _fetch_hf_student(student)
        # Load on CPU in fp32/bf16 so top-half reset is bit-identical across machines.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        _reset_top_layers(model, float(student.get("reset_frac", 0.5)), init_seed)
        return model.to(device)
    # Season 1-2: Initialise on the CPU so the starting weights are bit-identical on every machine.
    model = GPT(scale.n_layer, scale.n_head, scale.d_model, scale.ctx, season.vocab_size)
    return model.to(device)


def lr_at(step: int, steps: int, peak: float, warmup_frac: float, final_frac: float) -> float:
    warm = max(1, min(8, steps // 4), int(steps * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _sft_lr_at(step: int, steps: int, peak: float, warmup_steps: int, final_frac: float) -> float:
    warm = max(1, warmup_steps)
    if step < warm:
        return peak * (step + 1) / warm
    t = (step - warm) / max(1, steps - warm)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def _forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def run_sft_probe(
    model: torch.nn.Module,
    season: Season,
    scale: Scale,
    enc,
    sft_seed: bytes | int,
    device: torch.device,
) -> dict:
    """Run the standardized disjoint multilingual SFT probe before sitting the exam."""
    from .exam.sft import build_sft_examples

    sft_cfg = season.raw.get("sft") or {}
    steps = int(scale.sft_steps)
    if not sft_cfg or steps <= 0:
        return {}
    seed_bytes = sft_seed if isinstance(sft_seed, bytes) else str(sft_seed).encode()
    n_ex = int(sft_cfg.get("examples", 256))
    bs = int(sft_cfg.get("batch_size", 16))
    mb = min(bs, int(sft_cfg.get("micro_batch", 4)))
    accum = max(1, bs // mb)
    max_ctx = min(scale.ctx, int(sft_cfg.get("ctx", 512)))
    peak_lr = float(sft_cfg.get("lr", 1e-4))
    final_frac = float(sft_cfg.get("final_lr_frac", 0.1))
    warm = int(sft_cfg.get("warmup_steps", 4))
    wd = float(sft_cfg.get("weight_decay", 0.01))
    grad_clip = float(season.train_cfg.get("grad_clip", 1.0))
    eot = season.eot

    examples = build_sft_examples(seed_bytes, n=n_ex)
    encoded: list[tuple[list[int], int]] = []
    for prompt, comp in examples:
        a = [eot] + encode_one(enc, prompt)
        b = encode_one(enc, comp) + [eot]
        if not b:
            continue
        room = max_ctx + 1
        keep = len(a) if len(a) + len(b) <= room else min(len(a), max(room - len(b), room // 2))
        ids = a[len(a) - keep :] + b[: room - keep]
        if len(ids) > 1 and keep < len(ids):
            encoded.append((ids, keep))
    if not encoded:
        return {}

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr,
        betas=(float(season.train_cfg["beta1"]), float(season.train_cfg["beta2"])),
        fused=device.type == "cuda",
    )

    model.train()
    t0 = time.time()
    last_loss = 0.0
    n_pool = len(encoded)
    for step in range(steps):
        lr = _sft_lr_at(step, steps, peak_lr, warm, final_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for m in range(accum):
            start = (step * bs + m * mb) % n_pool
            batch_items = [encoded[(start + k) % n_pool] for k in range(mb)]
            T = max(len(ids) for ids, _ in batch_items) - 1
            x = torch.full((mb, T), eot, dtype=torch.long)
            y = torch.full((mb, T), -100, dtype=torch.long)
            for r, (ids, first) in enumerate(batch_items):
                x[r, : len(ids) - 1] = torch.tensor(ids[:-1])
                tgt = torch.tensor(ids[1:])
                tgt[: max(0, first - 1)] = -100
                y[r, : len(ids) - 1] = tgt
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _autocast(device):
                logits = _forward_logits(model, x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100) / accum
            loss.backward()
            step_loss += float(loss.detach())
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"SFT probe diverged at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = step_loss
    if device.type == "cuda":
        torch.cuda.synchronize()
    del opt
    return {"steps": steps, "examples": n_pool, "seconds": round(time.time() - t0, 2), "final_loss": round(last_loss, 4)}


def train(
    tokens: np.ndarray,
    scale: Scale,
    season: Season,
    init_seed: int,
    device: torch.device,
    progress=None,
    max_steps: int | None = None,
    sft_seed: bytes | None = None,
    enc=None,
) -> tuple[torch.nn.Module, dict]:
    """Train the student for exactly the budget. Row b of `tokens` is lane b; step t reads
    columns [t * ctx, (t + 1) * ctx] of every lane. Nothing is shuffled."""
    cfg = season.train_cfg
    torch.set_float32_matmul_precision("high")
    model = build_model(scale, season, init_seed, device)
    init_sha256 = weights_sha256(model)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=scale.lr,
        betas=(cfg["beta1"], cfg["beta2"]),
        fused=device.type == "cuda",
    )
    steps = scale.steps if max_steps is None else min(scale.steps, max_steps)
    per_step = scale.batch_size * scale.ctx
    assert tokens.shape == (scale.batch_size, scale.steps * scale.ctx + 1), "tokens do not match the training budget"
    mb = scale.micro_batch if scale.micro_batch > 0 else scale.batch_size
    assert scale.batch_size % mb == 0, f"batch_size {scale.batch_size} must be divisible by micro_batch {mb}"
    accum = scale.batch_size // mb
    log_every = max(1, steps // 200)
    curve: list[list[float]] = []
    acc = torch.zeros((), device=device)
    acc_n = 0
    model.train()
    t0 = time.time()
    for step in range(steps):
        a = step * scale.ctx
        for g in opt.param_groups:
            g["lr"] = lr_at(step, scale.steps, scale.lr, cfg["warmup_frac"], cfg["final_lr_frac"])
        opt.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for m in range(accum):
            lo, hi = m * mb, (m + 1) * mb
            chunk = torch.from_numpy(tokens[lo:hi, a : a + scale.ctx + 1].astype(np.int64)).to(device, non_blocking=True)
            x = chunk[:, :-1]
            y = chunk[:, 1:]
            with _autocast(device):
                logits = _forward_logits(model, x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)) / accum
            loss.backward()
            step_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        acc += step_loss
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
    del opt
    pretrain_seconds = time.time() - t0

    sft_info = {}
    if max_steps is None and scale.sft_steps > 0 and season.raw.get("sft"):
        if enc is None:
            from .tok import load_encoding

            enc = load_encoding(season)
        sft_info = run_sft_probe(model, season, scale, enc, sft_seed if sft_seed is not None else init_seed, device)

    seconds = time.time() - t0
    info = {
        "steps": steps,
        "tokens": steps * per_step,
        "seconds": round(seconds, 2),
        "pretrain_seconds": round(pretrain_seconds, 2),
        "tokens_per_second": round(steps * per_step / max(pretrain_seconds, 1e-9)),
        "final_train_loss": curve[-1][1] if curve else None,
        "curve": curve,
        "params": _n_params(model),
        "init_sha256": init_sha256,
        "precision": precision(device),
    }
    if sft_info:
        info["sft"] = sft_info
    return model, info
