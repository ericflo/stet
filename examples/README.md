# Reference editors

Three complete editors, entered in the competition by the organisers so that the board has
fixed points. Fork any of them: `stet fork stet/tidy`.

| | What it does |
|---|---|
| `identity` | Nothing. The unedited corpus; gain 0 by definition. |
| `english` | Keeps documents that look like English prose. |
| `tidy` | English, deduplicated, boilerplate lines cut, best budget's worth, shuffled. |

Each `editor.py` is the same file as `src/stet/editors/<name>.py` (a test keeps them identical).

And one that is not entered, as a pattern to build on:

| | What it does |
|---|---|
| `reader` | `tidy`'s candidates, then a 135M-parameter language model reads the opening of each and the least and most surprising are dropped. Shows `[[assets]]`, `task.asset`, the GPU through PyTorch, and writing a script early. |
