"""Reference editors. Each is a complete, runnable editor: `python -m stet.editors.<name>`.

They exist to anchor the leaderboard (what does doing nothing score? what does an
afternoon of heuristics score?) and to be forked. Each file stands alone, and is copied
byte for byte into examples/<name>/editor.py, which is what is entered in the competition.
"""

BUILTIN = ("identity", "english", "tidy")
