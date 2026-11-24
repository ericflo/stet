"""Sitting the exam: one number per item, higher is better.

  passage  minus the bits per byte of the passage
  answer   minus the bits per byte of the right answer, given its context (with floor bytes)
  pair     the bits by which the good sentence is preferred, per byte of the good sentence
  choice   discriminative per-token log-probability margin (bits per token):
           log2 P_tok(right | ctx) - max_{w in wrongs} log2 P_tok(w | ctx).
           Surface style matching lifts right and wrong options together, giving zero gain.

The bytes are never taken to be fewer than the season's floor (16). An answer of one
or two tokens is one or two draws of the student's luck; put over its three or four bytes it
would swing as far as a whole sentence does.

All of an item is scored, whatever its length. A passage longer than the student's window is
read a window at a time. So is an answer or choice continuation: its context is cut from the
left to leave it room (never to less than half a window, when there is that much), and what
does not fit is read on in further windows.
"""

from __future__ import annotations

import math

from ..tok import encode_one
from .items import Item

LN2 = math.log(2)


def _append_conditioned(seqs: list[tuple[list[int], int]], a: list[int], b: list[int], ctx: int) -> None:
    """Append windowed sequences scoring every token of `b` given context `a`."""
    if not b:
        seqs.append((a[: ctx + 1], len(a[: ctx + 1])))
        return
    room = ctx + 1
    keep = len(a) if len(a) + len(b) <= room else min(len(a), max(room - len(b), room // 2))
    head = b[: room - keep]
    seqs.append((a[len(a) - keep :] + head, keep))
    for i in range(len(head) - 1, len(b) - 1, ctx):  # the rest of a continuation too long for one window
        seqs.append((b[i : i + ctx + 1], 1))


def sequences(items: list[Item], enc, eot: int, ctx: int) -> tuple[list[tuple[list[int], int]], list[tuple[int, int]]]:
    """Token sequences to score, each (ids, first scored position), and for each item the slice
    of sequences that belong to it. Every token of a passage, answer, or choice is scored once."""
    seqs: list[tuple[list[int], int]] = []
    spans: list[tuple[int, int]] = []
    for it in items:
        start = len(seqs)
        if it.kind == "passage":
            ids = [eot] + encode_one(enc, it.text)
            for i in range(0, len(ids) - 1, ctx):
                seqs.append((ids[i : i + ctx + 1], 1))
        elif it.kind == "answer":
            a, b = [eot] + encode_one(enc, it.ctx), encode_one(enc, it.cont)
            _append_conditioned(seqs, a, b, ctx)
        elif it.kind == "choice":
            a = [eot] + encode_one(enc, it.ctx)
            # One or more windows for each option (right first, then every wrong); record each
            # option's boundary implicitly via single-window cap for choices (options are <= 400 chars).
            for opt in (it.cont, *it.wrongs):
                b = encode_one(enc, opt)
                room = ctx + 1
                keep = len(a) if len(a) + len(b) <= room else min(len(a), max(room - len(b), room // 2))
                head = b[: max(1, room - keep)] if b else []
                seqs.append((a[len(a) - keep :] + head, keep))
        else:
            for s in (it.good, it.bad):
                seqs.append(([eot] + encode_one(enc, s)[:ctx], 1))
        spans.append((start, len(seqs)))
    return seqs, spans


def nll(model, seqs: list[tuple[list[int], int]], device, batch: int = 32) -> list[float]:
    """Summed negative log-likelihood (nats) of each sequence's scored tokens."""
    import torch
    import torch.nn.functional as F

    out = [0.0] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    model.eval()
    # Keep activation memory bounded when evaluating large-vocab HF models (e.g. 151k vocab at ctx=1024).
    vocab = getattr(getattr(model, "config", None), "vocab_size", 0) or 0
    if vocab >= 65536:
        batch = min(batch, 8)
    for j in range(0, len(order), batch):
        idx = order[j : j + batch]
        T = max(len(seqs[i][0]) for i in idx) - 1
        if T <= 0:
            continue
        x = torch.zeros((len(idx), T), dtype=torch.long)
        y = torch.full((len(idx), T), -100, dtype=torch.long)
        for r, i in enumerate(idx):
            ids, first = seqs[i]
            if len(ids) <= 1:
                continue
            x[r, : len(ids) - 1] = torch.tensor(ids[:-1])
            tgt = torch.tensor(ids[1:])
            tgt[: max(0, first - 1)] = -100
            y[r, : len(ids) - 1] = tgt
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw = model(x)
            else:
                raw = model(x)
            logits = raw.logits if hasattr(raw, "logits") else raw
        loss = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum(1)
        for r, i in enumerate(idx):
            out[i] = float(loss[r])
    return out


def sit(model, items: list[Item], enc, eot: int, ctx: int, device, batch: int = 64, floor: int = 1) -> list[float]:
    """The student's score on every item, in order. Higher is better. `floor` is the season's
    answer_floor_bytes: the fewest bytes an answer or pair's bits are put over. FloatingPointError
    if a score is not a finite number."""
    seqs, spans = sequences(items, enc, eot, ctx)
    losses = nll(model, seqs, device, batch)
    floor = max(1, floor)
    scores = []
    for it, (a, b) in zip(items, spans, strict=True):
        if it.kind == "passage":
            scores.append(-sum(losses[a:b]) / LN2 / max(floor, len(it.text.encode())))
        elif it.kind == "answer":
            scores.append(-sum(losses[a:b]) / LN2 / max(floor, len(it.cont.encode())))
        elif it.kind == "choice":
            # Per-token log2 probability margin: right minus best wrong distractor.
            lps = []
            for k in range(a, b):
                ids, first = seqs[k]
                ntok = max(1, len(ids) - first)
                lps.append(-losses[k] / LN2 / ntok)
            scores.append(lps[0] - max(lps[1:]))
        else:
            scores.append((losses[a + 1] - losses[a]) / LN2 / max(floor, len(it.good.encode())))
    # A student whose arithmetic overflowed has no score: said here, so that no grade is sent with NaN in it.
    for n, s in enumerate(scores):
        if not math.isfinite(s):
            raise FloatingPointError(f"the student's score on exam item {n + 1} of {len(scores)} is not a number ({s})")
    return scores
