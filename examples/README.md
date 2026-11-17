# Reference editors

Three complete editors, entered in the competition by the organisers so that the board has
fixed points. Fork any of them: `stet fork stet/tidy`.

| | What it does |
|---|---|
| `identity` | Nothing. The unedited corpus; gain 0 by definition. |
| `english` | Keeps documents that look like English prose. |
| `tidy` | English, deduplicated, boilerplate lines cut, best budget's worth, shuffled. |

Each `editor.py` is the same file as `src/stet/editors/<name>.py` (a test keeps them identical).
