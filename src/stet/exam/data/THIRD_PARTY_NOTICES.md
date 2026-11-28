# Third-Party Data Notices and Licenses

While the `stet` codebase itself is licensed under the MIT License, the `stet-cli`
wheel bundles two small compressed data packs under `stet/exam/data/` that incorporate
or derive from third-party openly licensed datasets. Every bundled dataset uses a
commercial-permissive open license (Apache 2.0, MIT, CC BY 4.0, or CC BY-SA 4.0);
zero non-commercial (NC) datasets are included.

---

## 1. `s3-sft-ml.jsonl.zst` (512 multilingual SFT probe examples)

Used exclusively for Season 3's brief 1-epoch (`32`-step) post-pretraining SFT probe
during grading (`src/stet/exam/sft.py`). No rows from this pack appear in any exam.

| Pack `source` label | Rows | Upstream Dataset & Authors | License & Terms | Transformations |
|---|---:|---|---|---|
| `global_mmlu_dev` | 235 | **CohereLabs/Global-MMLU** (Cohere For AI) — https://huggingface.co/datasets/CohereLabs/Global-MMLU | **Apache License 2.0** (full text in [`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt)) — https://www.apache.org/licenses/LICENSE-2.0 | Selected `dev` split across 11 languages (`en, ru, zh, es, ar, pt, de, fr, ko, ja, hi`); formatted as `prompt` (question + choices) and `comp` (answer text). |
| `xstory_cloze_train` | 79 | **XStoryCloze** (Meta AI / FAIR, Lin et al. 2022) — https://huggingface.co/datasets/juletxara/xstory_cloze · FAIR notice: https://github.com/facebookresearch/fairseq/blob/main/examples/xglm/XStoryCloze.md | **CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0/ | Selected `train` split across multilingual story completions; formatted as 4-sentence story `prompt` and winning ending `comp`. Derivatives of these rows remain under CC BY-SA 4.0. |
| `gsm8k_train` | 58 | **GSM8K** (OpenAI, Cobbe et al. 2021) — https://huggingface.co/datasets/openai/gsm8k | **MIT License** (upstream copyright & license in [`licenses/GSM8K-MIT.txt`](licenses/GSM8K-MIT.txt)) — https://github.com/openai/grade-school-math/blob/master/LICENSE | Selected `train` split (`main` config); formatted as math word problem `prompt` and step-by-step rationale/answer `comp`. |
| `arc_easy_train` | 54 | **AI2 Reasoning Challenge (ARC-Easy)** (Allen Institute for AI, Clark et al. 2018) — https://huggingface.co/datasets/allenai/ai2_arc | **CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0/ | Selected `train` split (`ARC-Easy` config); formatted as science question + choices `prompt` and answer `comp`. Derivatives of these rows remain under CC BY-SA 4.0. |
| `openbookqa_train` | 48 | **OpenBookQA** (Allen Institute for AI, Mihaylov et al. 2018) — https://huggingface.co/datasets/allenai/openbookqa | **Apache License 2.0** (full text in [`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt)) — https://www.apache.org/licenses/LICENSE-2.0 | Selected `train` split (`main` config); formatted as multiple-choice question `prompt` and answer `comp`. |
| `mbpp_train` | 38 | **MBPP** (Google Research, Austin et al. 2021) — https://huggingface.co/datasets/google-research-datasets/mbpp | **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/ | Selected `train` split; formatted as Python task specification `prompt` and reference solution `comp`. |

Full upstream license texts required for redistribution (`Apache License 2.0` for Global-MMLU and OpenBookQA, and OpenAI's upstream `MIT License` copyright and permission notice for GSM8K) are bundled under `stet/exam/data/licenses/` and in the wheel's `.dist-info/licenses/` directory.

---

## 2. `s3-practice.jsonl.zst` (508 offline practice exam items)

Used for local `practice` and `gym` rounds (offline fallback when no fresh harvest hours are
present, and default scored reasoning items when `stet harvest` is used without
`STET_SYNTHESIZE_PRACTICE=1`). Official competition rounds never use this pack.

- **Source material:** Wikipedia articles (`wikipedia`), Wikipedia article additions
  (`wikipedia-added`), and Wikipedia talk pages (`wikipedia-talk`) across 42 languages,
  authored by **Wikipedia contributors** (https://www.wikipedia.org/).
- **Per-passage attribution manifest:** [`WIKIPEDIA_ATTRIBUTION.jsonl`](WIKIPEDIA_ATTRIBUTION.jsonl)
  (bundled both alongside this notice in `stet/exam/data/` and in the wheel's `.dist-info/licenses/`
  directory) lists all **744** unique Wikipedia pages and revisions referenced by the practice pack
  — including all `221` primary comprehension openings, all `376` distinct donor alternative openings
  (`611` occurrences, including all `236` donor-only pages preserved in each item's `wrong_sources`),
  and all `287` reasoning source passages — with each page's language, revision ID, title, canonical
  page URL, direct revision permalink (`?oldid=...`), and direct page-history URL
  (`?oldid=...&action=history`). Published official settled exams (`public_items()`) likewise
  preserve `wrong_sources` and per-passage `attribution` revision/history URLs for every openly
  licensed Wikipedia twin-context and reasoning item.
- **License:** **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/) and
  GFDL, per Wikimedia Foundation Terms of Use (§7,
  https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content).
- **Transformations:**
  - `221` twin-context comprehension items (`kind: "context"`): Wikipedia passages split
    at a sentence/paragraph boundary with length-matched wrong openings (`wrongs`) and their
    donor revision IDs (`wrong_sources`) from the same source and language, normalized to crawl
    whitespace (`src/stet/exam/fresh/select.py`).
  - `287` reasoning items (`explain: 96`, `infer: 95`, `synthesis: 96`): deterministic
    `thinkingmachines/Inkling-2-Mini` (`c9d02830`, `reasoning_effort="low"`, Apache 2.0)
    multi-sentence comprehension/reasoning prompts and responses derived from decontaminated
    Wikipedia excerpts (`src/stet/exam/fresh/reason.py`), filtered to fit within the 1,024-token
    student scoring window and distributed under CC BY-SA 4.0 as adaptations of Wikipedia text.

---

## 3. Referenced Models (Not Bundled in Wheel)

- **`baidu/ERNIE-4.5-0.3B-Base-PT`** (Season 3 runtime student & tokenizer): **Apache License 2.0** — https://huggingface.co/baidu/ERNIE-4.5-0.3B-Base-PT
- **`thinkingmachines/Inkling-2-Mini`** (Season 3 official runtime examiner reasoning teacher & generator of the bundled `s3-practice.jsonl.zst` reasoning items): **Apache License 2.0** — https://huggingface.co/thinkingmachines/Inkling-2-Mini
- **`Qwen/Qwen3-0.6B`** (`examples/reader` runtime reference curator): **Apache License 2.0** — https://huggingface.co/Qwen/Qwen3-0.6B
