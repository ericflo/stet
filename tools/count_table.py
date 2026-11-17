"""A trigram count table, as a check on the student.

The student is the measuring instrument of the competition. A cheap test that it is
measuring something: train a plain trigram count model (absolute discounting, backing off to
bigrams and unigrams) on exactly the tokens the student trains on, and score it on the same
exam. A student worth using should do clearly better than a lookup table of its own data.

CPU only. Not part of scoring.
"""
import numpy as np


def fit(tokens: np.ndarray, vocab: int, discount: float = 0.75, alpha: float = 0.5):
    """Return score(ids) -> per-token negative log likelihood under the trigram model."""
    B = np.uint64(vocab)
    t = tokens.reshape(-1).astype(np.uint64)
    c1 = np.bincount(t.astype(np.int64), minlength=vocab).astype(np.float64)
    p1 = (c1 + alpha) / (c1.sum() + alpha * vocab)
    k2, n2 = np.unique(t[:-1] * B + t[1:], return_counts=True)
    ctx1 = np.bincount(t[:-1].astype(np.int64), minlength=vocab).astype(np.float64)
    typ1 = np.bincount((k2 // B).astype(np.int64), minlength=vocab).astype(np.float64)
    k3, n3 = np.unique((t[:-2] * B + t[1:-1]) * B + t[2:], return_counts=True)
    c2k, c2n = np.unique(t[:-2] * B + t[1:-1], return_counts=True)
    t2k, t2n = np.unique(k3 // B, return_counts=True)

    def look(keys, vals, q):
        pos = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return np.where(keys[pos] == q, vals[pos], 0).astype(np.float64)

    def score(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.uint64)
        w, v = a[1:], a[:-1]
        wi, vi = w.astype(np.int64), v.astype(np.int64)
        pu = p1[wi]
        c = look(k2, n2, v * B + w)
        pb = np.where(ctx1[vi] > 0, (np.maximum(c - discount, 0) + discount * typ1[vi] * pu) / np.maximum(ctx1[vi], 1), pu)
        pt = pb.copy()
        if len(a) > 2:
            ctx = a[:-2] * B + v[1:]
            c3, cc, tt = look(k3, n3, ctx * B + w[1:]), look(c2k, c2n, ctx), look(t2k, t2n, ctx)
            pt[1:] = np.where(cc > 0, (np.maximum(c3 - discount, 0) + discount * tt * pb[1:]) / np.maximum(cc, 1), pb[1:])
        return -np.log(pt)

    return score


def exam_loss(score, exam: dict, enc, eot: int) -> float:
    """The same macro-average over sources that the student is scored with."""
    per = []
    for source in sorted(exam):
        nll, n = 0.0, 0
        for toks in enc.encode_ordinary_batch([it["text"] for it in exam[source]]):
            a = np.asarray([eot] + toks, dtype=np.uint64)
            nll += float(score(a).sum())
            n += len(a) - 1
        per.append(nll / max(n, 1))
    return float(np.mean(per))
