"""The fixed training run and post-pretraining SFT probe.

Training is budgeted in tokens, never in seconds: every machine does exactly the same
work, in exactly the same order, on exactly the tokens the edit script produced. There is
no shuffling here. The order of the stream is the curator's to choose.

Season 1-2 train a small GPT from random weights. Season 3 starts from a pinned
multilingual base model (baidu/ERNIE-4.5-0.3B-Base-PT), freezes the embeddings and lower
half of its transformer layers, deterministically re-initializes the upper half from the
round's seed on CPU, trains unshuffled on the curator's stream using fixed micro-batch
gradient accumulation with row-chunked lm_head projection (so 12 GB+ and 32 GB+ GPUs
execute identical math under 10 GiB VRAM), and runs a brief standardized disjoint
multilingual SFT probe before sitting the exam.
"""

from __future__ import annotations

import contextlib
import math
import os
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def device_backend(device: torch.device) -> str:
    """Concrete grading backend identifier separating NVIDIA CUDA from AMD ROCm."""
    dev_type = getattr(device, "type", str(device).split(":", 1)[0])
    if dev_type == "cuda":
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return dev_type


from .spec import hardware_backend as hardware_backend  # noqa: E402


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
        info["backend"] = device_backend(device)
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        if isinstance(major, int) and isinstance(minor, int):
            info["cuda_capability"] = f"{major}.{minor}"
        if info["backend"] == "cuda":
            if isinstance(major, int):
                info["bf16_native"] = major >= 8
            else:
                try:
                    info["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                except TypeError:
                    try:
                        info["bf16_native"] = bool(torch.cuda.is_bf16_supported())
                    except Exception:
                        pass
                except Exception:
                    pass
    return info


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@contextlib.contextmanager
def _deterministic_cuda(enabled: bool):
    """Run Season 3 CUDA training with deterministic FlashAttention/cuBLAS kernels so
    identical inputs on the same GPU model produce bit-for-bit identical weights."""
    if not enabled:
        yield
        return
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True, warn_only=bool(getattr(torch.version, "hip", None)))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench


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
    """Fetch pinned HF student files into the local cache, verifying every file's sha256 on every load."""
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
            verify=True,
        )
    return model_dir


def _set_activation_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    if enabled:
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        if hasattr(model, "activation_checkpointing_disable"):
            model.activation_checkpointing_disable()
        elif hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()


def _ce_loss(logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    """FP32 cross-entropy that avoids materialising a multi-gigabyte FP32 logits tensor on
    large-vocabulary models by computing row-by-row over the micro-batch."""
    if logits.ndim == 3 and logits.size(-1) >= 65536 and logits.size(0) > 1:
        if ignore_index == -100 and (targets == ignore_index).any():
            num = logits.new_zeros((), dtype=torch.float32)
            den = 0
            for r in range(logits.size(0)):
                valid = int((targets[r] != ignore_index).sum().item())
                if valid > 0:
                    num = num + F.cross_entropy(logits[r].float(), targets[r], ignore_index=ignore_index, reduction="sum")
                    den += valid
            return num / max(1, den)
        return sum(F.cross_entropy(logits[r].float(), targets[r], ignore_index=ignore_index) for r in range(logits.size(0))) / logits.size(0)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)


def _backward_chunked_head(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, scale_div: float, device: torch.device
) -> torch.Tensor:
    """Exact FP32 cross-entropy with row-chunked lm_head projection so large-vocab models
    (e.g. ERNIE-4.5's 103,424-token head) never materialize a multi-gigabyte logits tensor.

    Mathematically equivalent to computing full-batch logits and FP32 cross-entropy (including
    when `lm_head.weight` is tied to `model.embed_tokens.weight`), while keeping peak VRAM
    under 10 GiB on 12 GB+ volunteer GPUs.
    """
    inner = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if inner is not None and lm_head is not None and vocab >= 65536 and x.size(0) > 1:
        with _autocast(device):
            h = inner(x).last_hidden_state
        h_leaf = h.detach().requires_grad_(True)
        mb = h.size(0)
        total_loss = h.new_zeros((), dtype=torch.float32)
        for r in range(mb):
            with _autocast(device):
                logits_r = lm_head(h_leaf[r])
            loss_r = F.cross_entropy(logits_r.float(), y[r]) / (mb * scale_div)
            loss_r.backward()
            total_loss = total_loss + loss_r.detach()
        h.backward(h_leaf.grad)
        return total_loss
    with _autocast(device):
        logits = _forward_logits(model, x)
    loss = _ce_loss(logits, y) / scale_div
    loss.backward()
    return loss.detach()


def _reset_top_layers(
    model: torch.nn.Module, reset_frac: float, init_seed: int, *, freeze_bottom: bool = False
) -> None:
    """Deterministically re-initialize the upper `reset_frac` transformer blocks and final norm on CPU.

    When `freeze_bottom=True`, also freeze embeddings, tied `lm_head`, and the lower pretrained
    layers so autograd skips activation storage and backward passes for the frozen prefix (1.8x
    faster training and <=10 GiB peak VRAM without activation checkpointing).
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
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
        # Reset the final RMSNorm too so pretrained norm gains don't amplify initial random logits.
        final_norm = getattr(inner, "norm", None)
        if final_norm is not None and getattr(final_norm, "weight", None) is not None:
            final_norm.weight.fill_(1.0)
            if getattr(final_norm, "bias", None) is not None:
                final_norm.bias.zero_()
        if freeze_bottom:
            emb = getattr(inner, "embed_tokens", None)
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad_(False)
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None:
                for p in lm_head.parameters():
                    p.requires_grad_(False)
            for layer in layers[:start]:
                for p in layer.parameters():
                    p.requires_grad_(False)


def build_model(scale: Scale, season: Season, init_seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    student = season.raw.get("student") or {}
    if student.get("kind") == "hf_topreset":
        from transformers import AutoModelForCausalLM

        model_dir = _fetch_hf_student(student)
        # Keep fp32 master weights on every device so AdamW, RMSNorm gains, weight decay, and
        # the low-LR SFT probe update cleanly without bf16 rounding loss; bf16 is used only in
        # autocast during forward/backward.
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        freeze_bottom = bool(student.get("freeze_bottom", False))
        _reset_top_layers(model, float(student.get("reset_frac", 0.5)), init_seed, freeze_bottom=freeze_bottom)
        if device.type == "cuda" and not freeze_bottom:
            _set_activation_checkpointing(model, True)
        return model.to(device)
    # Season 1-2: Initialise on the CPU so the starting weights are bit-identical on every machine.
    model = GPT(scale.n_layer, scale.n_head, scale.d_model, scale.ctx, season.vocab_size)
    return model.to(device)


def lr_at(step: int, steps: int, peak: float, warmup_frac: float, final_frac: float, warmup_steps: int = 0) -> float:
    warm = int(warmup_steps) if warmup_steps > 0 else max(1, min(8, steps // 4), int(steps * warmup_frac))
    warm = max(1, min(warm, steps))
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

    examples = build_sft_examples(seed_bytes, n=n_ex, season=season)
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
            inner = getattr(model, "model", None)
            lm_head = getattr(model, "lm_head", None)
            vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
            if inner is not None and lm_head is not None and vocab >= 65536:
                # Project only valid completion tokens through lm_head so SFT never materializes
                # full (mb, T, V) logits (keeps peak reserved VRAM <= 10 GiB on 12 GB GPUs).
                with _autocast(device):
                    out = inner(x)
                    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                mask = y != -100
                den = int(mask.sum().item())
                if den <= 0:
                    continue
                with _autocast(device):
                    logits_sel = lm_head(h[mask])
                loss = F.cross_entropy(logits_sel.float(), y[mask], reduction="sum") / (den * accum)
            else:
                with _autocast(device):
                    logits = _forward_logits(model, x)
                loss = _ce_loss(logits, y, ignore_index=-100) / accum
            loss.backward()
            step_loss += float(loss.detach())
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"SFT probe diverged at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = step_loss
    model.zero_grad(set_to_none=True)
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
    use_det = (season.raw.get("student") or {}).get("kind") == "hf_topreset" and device.type == "cuda"
    with _deterministic_cuda(use_det):
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
                g["lr"] = lr_at(step, scale.steps, scale.lr, cfg["warmup_frac"], cfg["final_lr_frac"], scale.warmup_steps)
            opt.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for m in range(accum):
                lo, hi = m * mb, (m + 1) * mb
                chunk = torch.from_numpy(tokens[lo:hi, a : a + scale.ctx + 1].astype(np.int64)).to(device, non_blocking=True)
                x = chunk[:, :-1]
                y = chunk[:, 1:]
                step_loss += _backward_chunked_head(model, x, y, accum, device)
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
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pretrain_seconds = time.time() - t0

        sft_info = {}
        if max_steps is None and scale.sft_steps > 0 and season.raw.get("sft"):
            if enc is None:
                from .tok import load_encoding

                enc = load_encoding(season)
            sft_info = run_sft_probe(model, season, scale, enc, sft_seed if sft_seed is not None else init_seed, device)
        _set_activation_checkpointing(model, False)
        model.zero_grad(set_to_none=True)
        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
