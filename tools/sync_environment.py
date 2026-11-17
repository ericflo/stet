"""Write the `run` extra in pyproject.toml from the season's environment.

The season file names the Python and the exact version of every library a curator may
count on. Installing `stet-cli[run]` must give exactly those, so the list in
pyproject.toml is written out from the season, not kept by hand:

    python tools/sync_environment.py          rewrite pyproject.toml
    python tools/sync_environment.py --check  exit 1 if it is out of date (the tests do this)

To change a library: edit "environment" in src/stet/seasons/s0.json (both "packages", what
curators are promised, and "also", what those bring with them), run this, and bump the
harness version: a curator may behave differently under the new one.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN, END = "# --- written by tools/sync_environment.py; do not edit by hand ---\n", "# --- end ---\n"


def lines(env: dict) -> list[str]:
    only = set(env.get("linux_x86_64_only") or [])
    out = []
    for name, version in sorted({**env["also"], **env["packages"]}.items()):
        marker = "; sys_platform == 'linux' and platform_machine == 'x86_64'" if name in only else ""
        out.append(f'    "{name}=={version}{marker}",')
    return out


def render(env: dict) -> str:
    return BEGIN + "run = [\n" + "\n".join(lines(env)) + "\n]\n" + END


def main() -> int:
    env = json.loads((ROOT / "src/stet/seasons/s0.json").read_text())["environment"]
    path = ROOT / "pyproject.toml"
    text = path.read_text()
    block = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not block.search(text):
        sys.stderr.write("pyproject.toml has no block for the run extra\n")
        return 1
    new = block.sub(lambda _: render(env), text)
    if "--check" in sys.argv:
        if new != text:
            sys.stderr.write("pyproject.toml's run extra is out of date: python tools/sync_environment.py\n")
            return 1
        return 0
    path.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
