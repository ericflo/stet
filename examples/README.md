# Reference editors

Three complete editors, entered in the competition by the organisers so that the board has
fixed points. Fork any of them: `stet fork stet/tidy`.

| | What it does |
|---|---|
| `identity` | Nothing. The unedited corpus; gain 0 by definition. |
| `english` | Keeps documents that look like English prose. |
| `tidy` | English, deduplicated, boilerplate lines cut, best budget's worth, shuffled. |

Each `editor.py` is the same file as `src/stet/editors/<name>.py` (a test keeps them identical).

And two that are not entered, as patterns to build on:

| | What it does |
|---|---|
| `reader` | `tidy`'s candidates, then a 135M-parameter language model reads the opening of each and the least and most surprising are dropped. Shows `[[assets]]`, `task.asset`, the GPU through PyTorch, and writing a script early. |
| `agent` | `tidy`'s candidates; how strictly to filter them is decided by an agent: the pi harness, on Node, driving a 0.6B model served on loopback inside the sandbox. Shows how to bring a harness and a model of your own. Run `./pack.sh` first. |
