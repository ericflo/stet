# Reference curators

Seven complete curators, entered in the competition by the organisers so that the board has
fixed points. `fineweb2` and `dclm` run in every round; each of the others runs in one round
in eight. Fork any of them: `stet fork stet/tidy`, `stet fork stet/fineweb2`.

Three of the organisers' own:

| | What it does |
|---|---|
| `identity` | Nothing. The unedited corpus; gain 0 by definition. |
| `english` | Keeps documents that look like English prose. |
| `tidy` | Pages that read as writing, in any language: deduplicated, ranked by how much of each is sentences, the best kept whole, shuffled. |

Three quarters of the exam's score is human writing in every language, each by its share of
the round's corpus, so `english`, which keeps only English, gives up the writing in every
other language. `tidy` asks no page what language it is in.

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

And two that are not entered, as patterns to build on:

| | What it does |
|---|---|
| `reader` | English prose with its boilerplate lines cut (the model reads English), then a 135M-parameter language model reads the opening of each and the least and most surprising are dropped. Shows `[[assets]]`, `task.asset`, the GPU through PyTorch, and writing a script early. |
| `agent` | English prose with its boilerplate lines cut; how strictly to filter it is decided by an agent: the pi harness, on Node, driving a 0.6B model served on loopback inside the sandbox. Shows how to bring a harness and a model of your own. Run `./pack.sh` first. |

## Red team

And one that tries to buy the exam instead of curating well, entered by the organisers so
that the board shows what that buys.

| | What it tries |
|---|---|
| `mimic` | To buy the exam by selection: the pages that look most like the exam's writing and the forms its skills come in (terminals, logs, JSON, code, questions and answers, reference, news, threads; any language) fill the budget first, and `tidy`'s selection tops it up only if they run short. |

If it ever beats `tidy` by more than noise, the exam has a hole, and it is the exam that gets fixed.

There was a second, `scribe`, which spent the 16,000 tokens of its own text a script could then hold on worked problems in the exam's forms. It beat `tidy` by 0.01 bits per byte, so a script now holds no text of its own, and `scribe` is gone.
