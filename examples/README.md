# Reference curators

Seven complete curators, entered in the competition by the organisers so that the board has
fixed points. Fork any of them: `stet fork stet/tidy`, `stet fork stet/fineweb2`.

Three of the organisers' own:

| | What it does |
|---|---|
| `identity` | Nothing. The unedited corpus; gain 0 by definition. |
| `english` | Keeps documents that look like English prose. |
| `tidy` | English, deduplicated, boilerplate lines cut, best budget's worth, shuffled. |

Each of these `curator.py` files is the same file as `src/stet/editors/<name>.py` (a test keeps them identical).

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
| `reader` | `tidy`'s candidates, then a 135M-parameter language model reads the opening of each and the least and most surprising are dropped. Shows `[[assets]]`, `task.asset`, the GPU through PyTorch, and writing a script early. |
| `agent` | `tidy`'s candidates; how strictly to filter them is decided by an agent: the pi harness, on Node, driving a 0.6B model served on loopback inside the sandbox. Shows how to bring a harness and a model of your own. Run `./pack.sh` first. |
