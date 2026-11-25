# Reference curators

Nine complete curators, entered in the competition by the organisers so that the board has
fixed points: eight references (including the `reader` neural baseline), and [one red-team curator](#red-team).
`fineweb2` and `reader` run in every round; each of the others runs in one round in eight. Fork any of them:
`stet fork stet/tidy`, `stet fork stet/fineweb2`, `stet fork stet/reader`.

Three of the organisers' own:

| | What it does |
|---|---|
| `identity` | Nothing. The unedited corpus; gain 0 by definition. |
| `english` | Keeps documents that look like English prose. |
| `tidy` | Pages that read as writing, in any language: deduplicated, ranked by how much of each is sentences, the best kept whole, shuffled. |

Half of the exam's score is fresh twin-context comprehension in every language, each by its share of
the round's corpus, so `english`, which keeps only English, gives up comprehension in every
other language. `tidy` asks no page what language it is in, and copies the pages it chooses
whole: when it kept only their lines of prose it cut the code out with the menus, and scored
0.03 bits per byte lower on the writing ([measured](../docs/EXPERIMENTS.md#text-a-model-writes)).

Each of these `curator.py` files is the same file as `src/stet/editors/<name>.py` (a test keeps them identical).

`english`, `tidy` and the four below read the corpus on every core a curator is given, with
`task.map`: the page-by-page work in worker processes, and whatever depends on the order of
the pages decided in the loop that reads their results. `fineweb2` also writes a script
before its slowest step, so that there is one if time runs out.

And four that rebuild published recipes, so the board shows where what is already known lands.
On English benchmarks, large models trained on data made the first two ways beat data made
the third way, which beats raw crawl text; the fourth keeps every language
([the reasoning](../docs/research/exam-first-principles.md#42-trust-the-known-answer-test)).

| | What it does |
|---|---|
| `dclm` | DCLM-Baseline's filter: English pages ranked by its published fastText classifier (OpenHermes 2.5 and ELI5 against web), the top kept. |
| `fineweb-edu` | FineWeb-Edu's filter: English pages ranked by its published educational-value classifier, the top kept. |
| `heuristic` | Gopher's quality and repetition rules, C4's line filters and exact deduplication, on English pages. |
| `fineweb2` | FineWeb-2's multilingual recipe without its per-language tuning: every language kept, cleaned and deduplicated. |

The English ones decide what is English as the published recipes do: Meta's fastText language identification, English at 0.65 or more.
The classifiers are the published weights, named by reference in `stet.toml` and fetched by
whoever runs the curator.

And two neural patterns (`reader` runs every round as the neural anchor; `agent` is a pattern not entered):

| | What it does |
|---|---|
| `reader` | Multilingual prefilter and exact dedup across all CPU cores (`task.map`), then `Qwen/Qwen3-0.6B` scores candidate openings on a 1–5 educational/reasoning rubric using `vLLM` prefix caching on NVIDIA (or batched `transformers` on AMD/macOS), applies per-language/script quantile normalization so English-rubric score shifts never starve Korean/Russian/Arabic/Japanese, writes periodic neural-scored checkpoints and fails loudly if the model cannot run, before ordering kept pages as a 4-band curriculum. Shows `[[assets]]`, `task.asset`, `vLLM` (with `gpu_memory_utilization` capped to `task.gpu_memory_gb`), quantile normalization, curriculum ordering, and checkpointing neural progress. |
| `agent` | English prose with its boilerplate lines cut; how strictly to filter it is decided by an agent: the pi harness, on Node, driving a 0.6B model served on loopback inside the sandbox. Shows how to bring a harness and a model of your own. Run `./pack.sh` first. |

## Red team

And one that tries to game surface style instead of curating well, entered by the organisers so
that the board shows whether style mimicry works.

| | What it tries |
|---|---|
| `mimic` | To game the exam by surface form: the pages that look most like the exam's sources and formats (terminals, logs, JSON, code, questions and answers, reference, news, threads; any language) fill the budget first, and `tidy`'s selection tops it up only if they run short. |

In Season 2's raw-perplexity exam, `mimic` beat every other reference curator (+0.10 bpb over raw crawl, ahead of `fineweb2`'s +0.05) simply by matching the exam's surface style. Season 3 replaced raw perplexity with 100%-fresh same-source twin-context comprehension and harvest-derived reasoning (`explain`, `infer`, `synthesis`) after a fixed multilingual SFT probe, so surface style lifts both real and wrong openings equally and `mimic` falls behind real quality curation.

There was a second, `scribe`, which spent the 16,000 tokens of its own text a script could
then hold on worked problems in the exam's forms. That bought 0.10 bits per byte on the
skills it was aimed at, and 0.004 on the whole score as it now is. A script now holds no
text of its own ([why](../docs/EXPERIMENTS.md#which-parts-of-the-exam-count)), and `scribe`
is gone.
