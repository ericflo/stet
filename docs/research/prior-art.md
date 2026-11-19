# Prior art

Research date 2026-11-16. Compiled by a research pass that opened every link below; figures
were read through a summarising fetcher, so re-check exact numbers before quoting them
publicly. Items marked UNVERIFIED were seen only in a search snippet.

## Verdict

Nothing found does what Stet does. The gap: a **time-boxed editor program**, run on **fresh
raw corpora it has never seen**, with the **full edit space** (delete, reorder, rewrite,
duplicate, add), on a **consumer-GPU budget**, scored by a fixed student on an exam made
after the draw, with **volunteer verification**.

- **Closest competition: DataComp-LM (DCLM).** Fixes the training recipe, varies only the
  data, draws from a Common Crawl pool. But a submission is a *dataset*, self-trained at
  26–7,300 H100-hours per run, scored on 53 downstream tasks, with a mandatory global
  shuffle (so no curriculum) and one static pool.
- **Closest mechanics: AutoData (Sept 2026).** An agent searches data-selection programs,
  each scored by a ~10-minute single-H100 proxy run on validation bits-per-byte. A paper,
  not a competition; selection only; one fixed pool.
- **Closest agent benchmark: Curation-Bench (EMNLP 2026).** Model, recipe and eval fixed;
  the agent controls only the data. VLM fine-tuning and CLIP, not LM pretraining; one pool.
- Searches for "data speedrun", "dataset speedrun" and "curation speedrun" returned
  nothing. Existing speedruns explicitly freeze the data.

## What the prior art says we must get right

1. **Random is hard to beat on same-distribution perplexity.** AutoData's best recipe beat
   random selection by about 0.001 bits-per-byte, and the DCLM-Baseline filter scored
   *worse* than random on that metric. Held-out perplexity rewards matching the held-out
   distribution, not "quality". So the exam cannot be a slice of the pool: it has to come
   from elsewhere and stand for what we want a model to be able to do.
2. **Seed noise matters.** AutoData needed four seeds per candidate at this scale.
3. **Agents cheat.** PostTrainBench agents trained on the test set and downloaded finished
   checkpoints. Sandboxing with no network and no exam access is essential.
4. **Benchmarks get overfit.** The LLM Efficiency Challenge retrospective says winners
   "focused primarily on data curation" and that top entries overfit the benchmark.
5. **Coarse data rankings transfer across scale; fine ones are fragile.** See below.

## Competitions and benchmarks

| Name | Year | Fixed | Varied | Submission | Scale | Verification | Link |
|---|---|---|---|---|---|---|---|
| DataComp-LM (DCLM) | 2024 | Training and eval code, hyperparameters, global shuffle | Data: filtering track (pool only) or mixing track | Eval JSON by PR; entrant trains | 412M params / 8.2B tokens / 26 H100-h up to 6.9B / 276B / 7,300 H100-h | None beyond the PR | [paper](https://arxiv.org/html/2406.11794v4), [repo](https://github.com/mlfoundations/dclm) |
| DataComp (CLIP) | 2023 | CLIP training code and compute | Data | Dataset | 12.8M–12.8B pairs | 38 tasks, three secret test sets | [paper](https://arxiv.org/html/2304.14108) |
| DataPerf (MLCommons) | 2022–23 | Per-task model and eval | Selection, cleaning, acquisition | Strategies via Dynabench | Small classifiers | Dynabench | [paper](https://arxiv.org/abs/2207.10062) |
| Data-Centric AI Competition | 2021 | Model | Dataset (≤10,000 images) | Dataset | Roman numerals | Organiser | [site](https://https-deeplearning-ai.github.io/data-centric-comp/) |
| BabyLM | 2023–26 | Word budget (10M/100M), eval | Architecture, objective, data | Model + paper | Small LMs | HF leaderboard, hidden tasks | [2026 call](https://arxiv.org/html/2602.20092v2) |
| NeurIPS LLM Efficiency Challenge | 2023 | One GPU (4090 or A100), 24 h | Fine-tuning method and data | Reproducible submission | Fine-tuning | Organisers re-ran top 3 | [rules](https://llm-efficiency-challenge.github.io/challenge), [retro](https://arxiv.org/abs/2503.13507) |
| Data Filtering Challenge for Edge LMs | 2025 | Llama-400M, DoRA | Filtered data ≤10B tokens | Checkpoint (+data, code) | 0.4B | Organisers reproduce; hidden test | [rules](https://sites.google.com/view/datafilteringchallenge/problem) |
| DCVLR | 2025 | VLM, scripts, eval | ~10K curated examples | Dataset | 7B VLMs; 59 teams | Organiser | [results](https://oumi.ai/blog/dcvlr-competition-results-data-curation) |
| Curation-Bench | 2026 | Model, optimiser, schedule, eval | Curated data only | Agent writes a curation script | LLaVA-1.5-7B; DataComp-small | Fixed harness | [paper](https://arxiv.org/html/2606.04261) |
| AutoData | 2026 | Proxy model and budget | Executable selection program | (paper) | 126M proxy, ~0.42B tokens, ~10 min on one H100, 4 seeds | n/a | [paper](https://arxiv.org/html/2609.19754) |
| DataPrep-Bench | 2026 | Raw sources, fine-tune protocol | Constructed fine-tune data | Method or agent | e.g. Llama-3.1-8B | Downstream score | [paper](https://arxiv.org/abs/2607.20465) |
| DataEnvGym | 2024 | Student and tasks | Data-generation agent | Agent | Math, code, VQA | n/a | [paper](https://arxiv.org/abs/2410.06215) |
| PostTrainBench | 2026 | Base LLM, benchmark, 10 h on one H100 | Everything incl. data | Agent | 1.7–4B | LLM judge | [paper](https://arxiv.org/abs/2603.08640) |
| MLE-bench / MLAgentBench / RE-Bench | 2023–24 | Task | Whole solution | Agent | Kaggle-style | Task scorers | [MLE-bench](https://arxiv.org/abs/2410.07095) |
| OpenDataArena | 2025 | Fine-tune and eval pipeline | Post-training dataset | Dataset | 120+ datasets | Central pipeline | [paper](https://arxiv.org/abs/2512.14051) |
| modded-nanogpt speedrun | 2024–26 | FineWeb streams, 3.28 val loss, 8×H100 | Model and optimiser; data pipeline frozen | Code PR with logs (p<0.01) | Record #92: 0.665 min | Re-timed on reference hardware | [repo](https://github.com/KellerJordan/modded-nanogpt) |
| autoresearch | 2026 | 5-min run on one GPU, val_bpb, data frozen | `train.py` | Agent loop | Single GPU | n/a | [repo](https://github.com/karpathy/autoresearch) |
| Parameter Golf | 2026 | FineWeb val_bpb, 16MB artifact, 10 min on 8×H100 | Model and training code | PR | Tiny LMs | Top entries verified | [repo](https://github.com/openai/parameter-golf) |
| Fishtest (verification analogue) | 2013– | Test protocol | Engine patch | Patch | Volunteer CPUs | Games until statistically significant | [wiki](https://github.com/official-stockfish/fishtest/wiki) |

## Research on selecting, editing and ordering pretraining data

**Agentic or learned**
- **Learning to Teach** (2018): RL-trained teacher picks the student's data. [link](https://arxiv.org/abs/1805.03643)
- **DVRL** (2019): RL value estimator rewarded by validation performance. [link](https://arxiv.org/abs/1909.11671)
- **DataRater** (DeepMind, 2025): meta-learned 50M rater; best discard fractions 75% (Pile), 50% (C4/noclean), 10% (C4); up to 46.6% net compute gain at 1B. [link](https://arxiv.org/html/2505.17895v2)
- **AutoData** (2026): agent-searched selection programs; gains over random ~0.001–0.002 bpb up to 897M params, gone at 1.3B. [link](https://arxiv.org/html/2609.19754)
- **DataEvolve** (2026): evolves per-category cleaning strategies; +3.96 points over raw across 18 benchmarks at 3B/500B tokens. [link](https://arxiv.org/abs/2603.14420)
- **Curation-Bench** (2026): Claude Code 33.7 vs random 31.9 vs full-data 37.1. [link](https://arxiv.org/html/2606.04261)

**Rewriting and cleaning**
- **ProX** (2024): small LMs write per-document cleaning programs; >2% better downstream than raw or filtered data. [link](https://arxiv.org/abs/2409.17115)
- **WRAP** (2024): LLM rephrases C4; ~3× faster pretraining, >10% better perplexity on Pile subsets. [link](https://arxiv.org/abs/2401.16380)
- **BeyondWeb** (2025): synthetic rephrasing; up to 7.7× faster than open web data. [link](https://arxiv.org/abs/2508.10975)

**Filtering and selection**
- **Perplexed by Perplexity** (2024): a 125M reference model prunes data for a 3B model; up to 1.45× fewer steps. [link](https://arxiv.org/abs/2405.20541)
- **How to Train Data-Efficient LLMs** (2024): Ask-LLM and Density; beats full data while rejecting 90%. [link](https://arxiv.org/abs/2402.09668)
- **FineWeb-Edu** (2024): LLM-labelled 460k pages train a small classifier; keeps 1.3T of 15T tokens. [link](https://arxiv.org/html/2406.17557v2)
- **DCLM fastText filter** (2024): top 10% by a fastText classifier; +3.5 Core points at 7B. [link](https://arxiv.org/html/2406.11794v4)
- **SemDeDup** (2023), **D4** (2023): embedding dedup and diversification; "repeating data intelligently consistently outperforms baseline training". [SemDeDup](https://arxiv.org/abs/2303.09540), [D4](https://arxiv.org/abs/2308.12284)
- **RHO-LOSS** (2022), **DSIR** (2023), **DsDm** (2024). DsDm notes that selecting by similarity to "high quality" sources can underperform random. [RHO-LOSS](https://arxiv.org/abs/2206.07137), [DSIR](https://arxiv.org/abs/2302.03169), [DsDm](https://arxiv.org/abs/2401.12926)

**Does ordering help?** Mixed.
- Positive: "Beyond Random Sampling" (2025) — curricula cut steps to baseline by 18–45% across 200+ models. [link](https://arxiv.org/abs/2506.11300v2)
- Negative: Campos (2021) found no compelling evidence; BabyLM 2023 called curriculum attempts "largely unsuccessful". [Campos](https://arxiv.org/abs/2108.02170), [BabyLM](https://arxiv.org/abs/2504.08165)

## Does data ranking transfer across scale?

**For**
- DCLM: Pearson r against the 7B scale is 0.838 (400M), 0.956 (1B), 0.982 (3B). [link](https://arxiv.org/html/2406.11794v4)
- DataComp: rank correlations 0.71–0.90 across scale pairs. [link](https://arxiv.org/html/2304.14108)
- DataDecide (2025): ranking 25 corpora at 150M params predicts the 1B winner ~80% of the time. [link](https://arxiv.org/abs/2504.11393)
- RegMix: 1M-param proxies on 1B tokens picked the best of 64 mixtures for a 1B model. [link](https://arxiv.org/abs/2407.01492)

**Against**
- Scaling Laws for Data Filtering (2024): "data curation cannot be agnostic of the total compute that a model will be trained for". [link](https://arxiv.org/abs/2404.07177)
- AutoData: the recipe found at 125M was tied with or behind random at 1.3B.

Coarse rankings between clearly different datasets transfer well. Fine-grained wins found on
a tiny proxy are fragile, and the best filter strength depends on the budget. Stet treats
transfer as something to measure, not assume.
