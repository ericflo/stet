"""Sitting the exam: one number per item, in bits per byte, higher is better.

  passage  minus the bits per byte of the passage
  answer   minus the bits per byte of the right answer, given its context (with floor bytes)
  pair     the bits by which the good sentence is preferred, per byte of the good sentence
  context  twin-context comprehension (bits per byte of the continuation):
           (mean_{w in wrongs} nll(cont | w) - nll(cont | ctx)) / LN2 / max(floor, bytes(cont)).
           Scores the exact same continuation under its real opening vs matched wrong openings
           from other documents of the same source and language, so unconditional passage ease,
           surface style matching, and document-start bias cancel identically, while a uniform
           model scores 0 (no extreme-value penalty).

The bytes are never taken to be fewer than the season's floor (16). An answer of one
or two tokens is one or two draws of the student's luck; put over its three or four bytes it
would swing as far as a whole sentence does.

All of an item is scored, whatever its length. A passage longer than the student's window is
read a window at a time. So is an answer: its context is cut from the left to leave it room
(never to less than half a window, when there is that much), and what does not fit is read on
in further windows.
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
        elif it.kind == "context":
            b = encode_one(enc, it.cont)
            # Cap continuation to at most half a window so the real context and every wrong
            # context always score the exact same continuation tokens.
            b_scored = b[: max(1, (ctx + 1) // 2)] if b else []
            max_ctx = max(1, ctx + 1 - len(b_scored))
            for c_str in (it.ctx, *it.wrongs):
                a = [eot] + encode_one(enc, c_str)
                keep = min(len(a), max_ctx)
                seqs.append((a[len(a) - keep :] + b_scored, keep))
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
    # Keep activation and logits memory well below the 16 GiB grader floor for large-vocab
    # models (e.g. Qwen3's 151,936 vocab at ctx=1024): pack batches by padded token count
    # (matching micro_batch=2 * ctx=1024 = 2,048 tokens) and compute FP32 CE only over scored
    # positions in bounded chunks.
    vocab = getattr(getattr(model, "config", None), "vocab_size", 0) or 0
    large_vocab = vocab >= 65536
    max_padded_tokens = 2048
    ce_chunk = 512

    batches: list[list[int]] = []
    if large_vocab:
        cur: list[int] = []
        for i in order:
            seq_t = max(1, len(seqs[i][0]) - 1)
            if cur and (len(cur) >= batch or (len(cur) + 1) * seq_t > max_padded_tokens):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    else:
        for j in range(0, len(order), batch):
            batches.append(order[j : j + batch])

    for idx in batches:
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
            if large_vocab or logits.size(-1) >= 65536:
                row_idx, pos_idx = (y != -100).nonzero(as_tuple=True)
                loss = torch.zeros(len(idx), dtype=torch.float32, device=device)
                n_scored = int(row_idx.numel())
                for start in range(0, n_scored, ce_chunk):
                    r_sl = row_idx[start : start + ce_chunk]
                    p_sl = pos_idx[start : start + ce_chunk]
                    sub_loss = F.cross_entropy(logits[r_sl, p_sl].float(), y[r_sl, p_sl], reduction="none")
                    loss.index_add_(0, r_sl, sub_loss)
            else:
                loss = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum(1)
            for r, i in enumerate(idx):
                out[i] = float(loss[r])
            del raw, logits, loss
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
        elif it.kind == "context":
            # Extra bits per byte saved on the exact same continuation by its real context vs
            # the mean of matched wrong contexts from the same source and language.
            mean_wrong = sum(losses[a + 1 : b]) / max(1, b - a - 1)
            scores.append((mean_wrong - losses[a]) / LN2 / max(floor, len(it.cont.encode())))
        else:
            scores.append((losses[a + 1] - losses[a]) / LN2 / max(floor, len(it.good.encode())))
    # A student whose arithmetic overflowed has no score: said here, so that no grade is sent with NaN in it.
    for n, s in enumerate(scores):
        if not math.isfinite(s):
            raise FloatingPointError(f"the student's score on exam item {n + 1} of {len(scores)} is not a number ({s})")
    return scores
