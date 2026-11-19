"""Short programs and what they print. The output is worked out here, not by running the code,
so every machine agrees; the tests run the code to check."""

from __future__ import annotations

from ..items import Item
from ..rng import Rng
from .lexicon import WORDS

NAME = "prog"
PY = ["arith", "concat", "upper", "length", "reverse", "slice", "list-sum", "sorted", "last", "loop-sum", "loop-rev",
      "if-even", "func", "dict", "while-halve", "squares", "count"]
JS = ["arith", "upper", "length", "slice", "push-length", "reduce", "loop-sum", "ternary", "func", "map-join"]
SMALL = ["arith", "loop-sum", "func", "if-even", "length"]  # in Go, C and Lua
OTHER = ["go", "c", "lua"]


def corners() -> list[str]:
    return ([f"{NAME}/python/{k}" for k in PY] + [f"{NAME}/javascript/{k}" for k in JS]
            + [f"{NAME}/{lang}/{k}" for lang in OTHER for k in SMALL])


def _python(rng: Rng, k: str) -> tuple[str, str]:
    a, b, c = rng.between(2, 15), rng.between(2, 15), rng.between(1, 9)
    w = rng.choice(WORDS)
    xs = [rng.between(1, 30) for _ in range(rng.between(3, 6))]
    if k == "arith":
        return f"a = {a}\nb = {b}\nprint(a * b - {c})", str(a * b - c)
    if k == "concat":
        s, t = rng.sample(WORDS, 2)
        return f's = "{s}"\nt = "{t}"\nprint(s + t * 2)', s + t * 2
    if k == "upper":
        return f'word = "{w}"\nprint(word.upper())', w.upper()
    if k == "length":
        return f'word = "{w}"\nprint(len(word) * {c})', str(len(w) * c)
    if k == "reverse":
        return f'word = "{w}"\nprint(word[::-1])', w[::-1]
    if k == "slice":
        i = rng.between(0, 2)
        return f'word = "{w}"\nprint(word[{i}:{i + 3}])', w[i:i + 3]
    if k == "list-sum":
        return f"xs = {xs}\nxs.append({c})\nprint(sum(xs))", str(sum(xs) + c)
    if k == "sorted":
        return f"xs = {xs}\nprint(sorted(xs))", str(sorted(xs))
    if k == "last":
        return f"xs = {xs}\nxs.append({a})\nprint(xs[-2])", str(xs[-1])
    if k == "loop-sum":
        n = rng.between(3, 9)
        return f"total = 0\nfor i in range({n}):\n    total += i * {c}\nprint(total)", str(sum(i * c for i in range(n)))
    if k == "loop-rev":
        return f'out = ""\nfor ch in "{w}":\n    out = ch + out\nprint(out)', w[::-1]
    if k == "if-even":
        x = rng.between(1, 99)
        return f'x = {x}\nif x % 2 == 0:\n    print("even")\nelse:\n    print("odd")', "even" if x % 2 == 0 else "odd"
    if k == "func":
        x = rng.between(1, 12)
        return f"def f(n):\n    return n * {a} + {b}\n\nprint(f({x}))", str(x * a + b)
    if k == "dict":
        return f'd = {{"x": {a}, "y": {b}}}\nd["z"] = d["x"] + d["y"] * {c}\nprint(d["z"])', str(a + b * c)
    if k == "while-halve":
        n = rng.between(10, 500)
        steps, m = 0, n
        while m > 1:
            m //= 2
            steps += 1
        return f"n = {n}\nsteps = 0\nwhile n > 1:\n    n = n // 2\n    steps += 1\nprint(steps)", str(steps)
    if k == "squares":
        lo = rng.between(1, 6)
        return f"print([x * x for x in range({lo}, {lo + 4})])", str([x * x for x in range(lo, lo + 4)])
    if k == "count":
        ch = rng.choice(sorted(set(w)))
        return f'word = "{w}"\nprint(word.count("{ch}"))', str(w.count(ch))
    raise ValueError(k)


def _javascript(rng: Rng, k: str) -> tuple[str, str]:
    a, b, c = rng.between(2, 15), rng.between(2, 15), rng.between(1, 9)
    w = rng.choice(WORDS)
    xs = [rng.between(1, 30) for _ in range(rng.between(3, 6))]
    arr = "[" + ", ".join(map(str, xs)) + "]"
    if k == "arith":
        return f"let a = {a};\nlet b = {b};\nconsole.log(a * b - {c});", str(a * b - c)
    if k == "upper":
        return f'const word = "{w}";\nconsole.log(word.toUpperCase());', w.upper()
    if k == "length":
        return f'const word = "{w}";\nconsole.log(word.length + {c});', str(len(w) + c)
    if k == "slice":
        i = rng.between(0, 2)
        return f'const word = "{w}";\nconsole.log(word.slice({i}, {i + 3}));', w[i:i + 3]
    if k == "push-length":
        return f"const xs = {arr};\nxs.push({c});\nconsole.log(xs.length);", str(len(xs) + 1)
    if k == "reduce":
        return f"const xs = {arr};\nconsole.log(xs.reduce((s, x) => s + x, 0));", str(sum(xs))
    if k == "loop-sum":
        n = rng.between(3, 9)
        return (f"let total = 0;\nfor (let i = 0; i < {n}; i++) {{\n  total += i * {c};\n}}\nconsole.log(total);",
                str(sum(i * c for i in range(n))))
    if k == "ternary":
        x = rng.between(1, 99)
        return f'const x = {x};\nconsole.log(x > 50 ? "big" : "small");', "big" if x > 50 else "small"
    if k == "func":
        x = rng.between(1, 12)
        return f"function f(n) {{\n  return n * {a} + {b};\n}}\nconsole.log(f({x}));", str(x * a + b)
    if k == "map-join":
        return f'const xs = {arr};\nconsole.log(xs.map(x => x * {c}).join(","));', ",".join(str(x * c) for x in xs)
    raise ValueError(k)


def _other(rng: Rng, lang: str, k: str) -> tuple[str, str]:
    a, b, c = rng.between(2, 15), rng.between(2, 15), rng.between(1, 9)
    w = rng.choice(WORDS)
    n, x, v = rng.between(3, 9), rng.between(1, 12), rng.between(1, 99)
    want = {"arith": str(a * b - c), "loop-sum": str(sum(i * c for i in range(n))), "func": str(x * a + b),
            "if-even": "even" if v % 2 == 0 else "odd", "length": str(len(w) * c)}[k]
    if lang == "go":
        body = {
            "arith": f"\ta, b := {a}, {b}\n\tfmt.Println(a*b - {c})",
            "loop-sum": f"\ttotal := 0\n\tfor i := 0; i < {n}; i++ {{\n\t\ttotal += i * {c}\n\t}}\n\tfmt.Println(total)",
            "func": f"\tfmt.Println(f({x}))",
            "if-even": f'\tx := {v}\n\tif x%2 == 0 {{\n\t\tfmt.Println("even")\n\t}} else {{\n\t\tfmt.Println("odd")\n\t}}',
            "length": f'\tword := "{w}"\n\tfmt.Println(len(word) * {c})',
        }[k]
        extra = f"func f(n int) int {{\n\treturn n*{a} + {b}\n}}\n\n" if k == "func" else ""
        return f'package main\n\nimport "fmt"\n\n{extra}func main() {{\n{body}\n}}', want
    if lang == "c":
        body = {
            "arith": f'    int a = {a}, b = {b};\n    printf("%d\\n", a * b - {c});',
            "loop-sum": f'    int total = 0;\n    for (int i = 0; i < {n}; i++) {{\n        total += i * {c};\n    }}\n    printf("%d\\n", total);',
            "func": f'    printf("%d\\n", f({x}));',
            "if-even": f'    int x = {v};\n    if (x % 2 == 0) {{\n        printf("even\\n");\n    }} else {{\n        printf("odd\\n");\n    }}',
            "length": f'    const char *word = "{w}";\n    printf("%d\\n", (int)strlen(word) * {c});',
        }[k]
        head = "#include <stdio.h>\n" + ("#include <string.h>\n" if k == "length" else "")
        extra = f"\nint f(int n) {{\n    return n * {a} + {b};\n}}\n" if k == "func" else ""
        return f"{head}{extra}\nint main(void) {{\n{body}\n    return 0;\n}}", want
    body = {
        "arith": f"local a, b = {a}, {b}\nprint(a * b - {c})",
        "loop-sum": f"local total = 0\nfor i = 0, {n - 1} do\n  total = total + i * {c}\nend\nprint(total)",
        "func": f"local function f(n)\n  return n * {a} + {b}\nend\nprint(f({x}))",
        "if-even": f'local x = {v}\nif x % 2 == 0 then\n  print("even")\nelse\n  print("odd")\nend',
        "length": f'local word = "{w}"\nprint(#word * {c})',
    }[k]
    return body, want


def program(rng: Rng, corner: str) -> tuple[str, str, str]:
    """(language, code, what it prints)"""
    _, lang, k = corner.split("/")
    if lang == "python":
        code, out = _python(rng, k)
    elif lang == "javascript":
        code, out = _javascript(rng, k)
    else:
        code, out = _other(rng, lang, k)
    return lang, code, out


def make(rng: Rng, corner: str) -> Item:
    lang, code, out = program(rng, corner)
    return Item(kind="answer", corner=corner, fmt="code", lang=f"code:{lang}", source=f"gen:{NAME}",
                ctx=f"```{lang}\n{code}\n```\nOutput:", cont=f" {out}")
