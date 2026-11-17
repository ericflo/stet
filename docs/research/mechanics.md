# How comparable efforts work

Research date 2026-11-16. Every claim was read on the linked page unless marked UNVERIFIED.
This is the evidence behind the choices in [DESIGN.md](../DESIGN.md).

## 1. Speedrun-style competitions

**modded-nanogpt** ([README](https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/README.md))
- Fixed: FineWeb token streams, target (≤3.28 val loss), hardware (8×H100). Rule 1 forbids
  touching the data pipeline — Stet is the inverse.
- 92 records (17 in 2024, 39 in 2025, 36 in 2026), ~55 named contributors, 5 co-credited to AI systems.
- Submission is a PR with logs that embed the full source. Entrants show p<0.01 that mean loss ≤ 3.28.
- Record #92 shipped 18 runs interleaved with 9 runs of the previous record on the same machine.
- Optimization track ([rules](https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/master/records/track_3_optimization/README.md)):
  one-sided z-test with assumed σ=0.0013; per-run early stopping banned; the maintainer does
  not reproduce results himself — "skeptical results" that refute an entry are first-class
  and can earn a ban. 46 results from 30 contributors in two months.
- Sticky because: post a record, get amplified. 64 of 92 records link to an X post. Every
  record adds a named line to a cumulative technique list.

**nanochat** ([leaderboard](https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LEADERBOARD.md)):
the largest single gain was a dataset swap (2.76 h → 2.02 h). Seven identical runs gave CORE
0.2512–0.2677; merely reshuffling shard order moves the result.

**autoresearch** ([README](https://raw.githubusercontent.com/karpathy/autoresearch/master/README.md)):
fixed 5-minute wall-clock budget; results "not comparable to other people running on other
compute platforms".

**AlgoPerf** ([docs](https://raw.githubusercontent.com/mlcommons/algorithmic-efficiency/main/docs/DOCUMENTATION.md)):
organisers rerun everything; first round drew 18 submissions from 10 teams. Heavy,
committee-run scoring limits turnout.

**GPU-kernel competitions**
- GPU MODE / KernelBot ([README](https://raw.githubusercontent.com/gpu-mode/kernelbot/main/README.md)):
  CLI submission, platform-run. ~532k submissions. Some on-prem timings were unreliable.
- KernelGuard ([ICML 2026](https://icml.cc/virtual/2026/67922)): with a public evaluator,
  entrants found exploits "from timer manipulation to physics-impossible scoring". Static
  rules, physics-floor checks and an LLM judge cut hacked submissions from 3.45% to 0.37%.
- KernelArena ([case study](https://www.wafer.ai/blog/reward-hacking)): a kernel reported
  104.89× while reading uninitialised memory and passed a 1e-3 tolerance check.

**Take for Stet:** a numbered record table with a one-line technique each; logs that embed
the source; a published σ and a one-line significance rule; refutations as first-class
results; budgets in tokens, never wall-clock; expect a steady stream of exploits.

## 2. Volunteer and distributed verification

**BOINC**
- Quorum of equivalent replicas; the project defines equivalence (floats within a tolerance).
  ([replication](https://raw.githubusercontent.com/wiki/BOINC/boinc/JobReplication.md))
- **Adaptive replication**: full replication costs ≥50% of compute. Instead track
  consecutive valid results per host; below 10 never trusted; above that trusted with
  probability 1 − 1/N; zeroed on any invalid result. Target overhead 5–10%.
  ([doc](https://raw.githubusercontent.com/wiki/BOINC/boinc/Adaptive-Replication.md))
- Credit: claimed runtime is assumed falsifiable; grant a fixed amount per validated job.
  ([doc](https://raw.githubusercontent.com/wiki/BOINC/boinc/CreditNew.md))

**Folding@home** ([FAQ](https://foldingathome.org/faq/stats)): bonus points only for
identities with a track record (≥10 returned units, 80% return rate). Teams pool points.

**GIMPS** ([maths](https://www.mersenne.org/various/math.php)): double-checks must match;
error rate ~1.5% on clean first tests.

**Bittensor**: validators copy the public consensus instead of evaluating ("weight
copying"); fixed with commit–reveal timed to a drand round
([doc](https://taostats.io/docs/concepts/validators/weight-copying)). The Gradients subnet's
documented exploits: training on test data after locating the original datasets, cloning
winners under new accounts, self-mining
([write-up](https://taodaily.io/from-exploits-to-excellence-how-gradients-reinvented-ai-competitions)).

**Proof-of-learning**: forged at lower cost; "current PoL verification is not robust to
adversaries" ([Fang et al.](https://arxiv.org/abs/2208.03567)). Bitwise-reproducible
training (Gensyn Verde/RepOps) costs 58–374% overhead ([paper](https://arxiv.org/html/2502.19405v1)).
Cheap cryptographic proofs of training do not exist.

**lichess fishnet** ([README](https://raw.githubusercontent.com/lichess-org/fishnet/master/README.md)):
personal keys trace work to an account. Deliberately no contributor leaderboard: "It would
incentivize gaming the metrics."

**Lc0** ([site](https://training.lczero.org/)): daily, monthly and all-time top-50 boards.
9 users produced 57,069 games in one day: volunteer supply is very concentrated.

**Take for Stet:** pair new verifiers until they have a streak of agreements, then
spot-check; assign jobs server-side, never by self-selection; hide other results until a
verifier has reported; credit per validated job, not per claimed hour; rank only validated
work; do not make latency depend on a crowd.

## 3. Floating-point reproducibility

- PyTorch: results are "not guaranteed across PyTorch releases, individual commits, or
  different platforms" ([notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)).
- modded-nanogpt: 26 unseeded runs gave 3.2756–3.2805 (sd 0.0013). With a **pinned seed**,
  two machines of the same GPU model gave 3.2763 and 3.2789 — not bitwise.
- ResNet-50/CIFAR-10: same-seed σ 3.0e-3 vs different-seed 3.5e-3; GPU nondeterminism is
  most of the variance ([paper](https://ar5iv.labs.arxiv.org/html/2001.11396)).
- No published fixed-seed variance across different consumer GPUs for a small transformer
  was found. Treat as unmeasured; measure it.

**Take for Stet:** a single run is never the score; publish σ per hardware class; a record
must be pairwise-significant against the incumbent on the same rounds.

## 4. Sandboxing untrusted code with a GPU

| CVE | Severity | Fixed in NVIDIA Container Toolkit | Source |
|---|---|---|---|
| CVE-2024-0132 | 9.0 | 1.16.2 | [bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5582) |
| CVE-2025-23266 | 9.0 | 1.17.8 | [bulletin](https://nvidia.custhelp.com/app/answers/detail/a_id/5659) |
| CVE-2026-24260 | 8.5 | 1.19.1 | [bulletin](https://raw.githubusercontent.com/NVIDIA/product-security/main/2026/5850/5850.md) |

- The 2024 and 2025 escapes need an **attacker-controlled image**. "Containers are not a
  strong security barrier" ([Wiz](https://www.wiz.io/blog/nvidia-ai-vulnerability-cve-2025-23266-nvidiascape)).
- gVisor `--nvproxy` supports a handful of datacentre GPUs with strict driver matching
  ([guide](https://gvisor.dev/docs/user_guide/gpu/)). Kata needs a whole GPU via VFIO.
- Baseline flags ([Docker](https://docs.docker.com/reference/cli/docker/container/run/),
  [OWASP](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)):
  `--network none`, `--read-only` + `--tmpfs`, `--cap-drop all`,
  `--security-opt no-new-privileges`, default seccomp, `--pids-limit`, `--memory`, `--cpus`,
  non-root `--user`, no Docker socket, never `--privileged`.

**Take for Stet:** never run entrant-supplied images — one organiser-pinned image, entrant
code as a read-only mount; refuse to start on Container Toolkit < 1.19.1; only the editor
stage is untrusted, and it hands back plain data; tell volunteers plainly this is not
VM-grade isolation.

## 5. What gets people to take part

- Clout and attribution: named, numbered records the maintainer amplifies.
- BOINC: nearly every participant wants a number for their contribution; team rivalry was
  "viral marketing"; show total *and* recent-average credit so newcomers can rank
  ([paper](https://boinc.berkeley.edu/boinc_papers/credit/text.php)).
- Openness as a condition: ARC Prize required permissively licensed, reproducible code.
- Many ways to matter: records, per-method bests, refutations.
- Pitfalls: raw-volume leaderboards invite gaming; competitive ranking can silence beginners.

## 6. Public randomness: drand

| Chain | Period | Genesis (unix) | Chain hash |
|---|---|---|---|
| `quicknet` | 3 s | 1692803367 | `52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971` |
| `default` | 30 s | 1595431050 | `8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce` |

- Relays: `https://api.drand.sh`, `https://api2.drand.sh`, `https://api3.drand.sh`, `https://drand.cloudflare.com`.
- Endpoint: `https://api.drand.sh/{chain-hash}/public/{round}` → `{"round", "randomness", "signature"}`.
- Round for unix time T: `floor((T − genesis) / period) + 1`.
- `randomness == sha256(signature)`; the signature is BLS12-381 over `sha256(round as 8-byte big-endian)`.
- quicknet is unchained: a round verifies on its own.

**Take for Stet:** commit editors before the round's drand round exists; store round,
randomness and signature with every round so anyone can re-derive it; query two relays.
