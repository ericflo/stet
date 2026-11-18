"""Sitting the exam: one number per item, higher is better, all in bits per byte.

  passage  minus the bits per byte of the passage
  answer   minus the bits per byte of the right answer, given its context
  pair     the bits by which the good sentence is preferred, per byte of the good sentence

Nothing is decided by picking a choice: at the official scale a student's choices are near
chance, while the likelihood of the right answer moves a long way.
"""

from __future__ import annotations

import math

from .items import Item

LN2 = math.log(2)


def sequences(items: list[Item], enc, eot: int, ctx: int) -> tuple[list[tuple[list[int], int]], list[tuple[int, int]]]:
    """Token sequences to score, each (ids, first scored position), and for each item the slice
    of sequences that belong to it. A sequence longer than the window is cut from the left."""
    seqs: list[tuple[list[int], int]] = []
    spans: list[tuple[int, int]] = []
    for it in items:
        start = len(seqs)
        if it.kind == "passage":
            ids = [eot] + enc.encode_ordinary(it.text)
            for i in range(0, len(ids) - 1, ctx):
                seqs.append((ids[i : i + ctx + 1], 1))
        elif it.kind == "answer":
            a, b = [eot] + enc.encode_ordinary(it.ctx), enc.encode_ordinary(it.cont)
            ids, first = a + b, len(a)
            if len(ids) > ctx + 1:
                cut = len(ids) - (ctx + 1)
                ids, first = ids[cut:], max(1, first - cut)
            seqs.append((ids, first))
        else:
            for s in (it.good, it.bad):
                seqs.append(([eot] + enc.encode_ordinary(s)[:ctx], 1))
        spans.append((start, len(seqs)))
    return seqs, spans


def nll(model, seqs: list[tuple[list[int], int]], device, batch: int = 64) -> list[float]:
    """Summed negative log-likelihood (nats) of each sequence's scored tokens."""
    import torch
    import torch.nn.functional as F

    out = [0.0] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    model.eval()
    for j in range(0, len(order), batch):
        idx = order[j : j + batch]
        T = max(len(seqs[i][0]) for i in idx) - 1
        x = torch.zeros((len(idx), T), dtype=torch.long)
        y = torch.full((len(idx), T), -100, dtype=torch.long)
        for r, i in enumerate(idx):
            ids, first = seqs[i]
            x[r, : len(ids) - 1] = torch.tensor(ids[:-1])
            tgt = torch.tensor(ids[1:])
            tgt[: first - 1] = -100
            y[r, : len(ids) - 1] = tgt
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
            else:
                logits = model(x)
        loss = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum(1)
        for r, i in enumerate(idx):
            out[i] = float(loss[r])
    return out


def sit(model, items: list[Item], enc, eot: int, ctx: int, device, batch: int = 64) -> list[float]:
    """The student's score on every item, in order. Higher is better."""
    seqs, spans = sequences(items, enc, eot, ctx)
    losses = nll(model, seqs, device, batch)
    scores = []
    for it, (a, b) in zip(items, spans, strict=True):
        if it.kind == "passage":
            scores.append(-sum(losses[a:b]) / LN2 / max(1, len(it.text.encode())))
        elif it.kind == "answer":
            scores.append(-losses[a] / LN2 / max(1, len(it.cont.encode())))
        else:
            scores.append((losses[a + 1] - losses[a]) / LN2 / max(1, len(it.good.encode())))
    return scores
