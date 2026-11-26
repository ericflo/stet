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
| `global_mmlu_dev` | 235 | **CohereLabs/Global-MMLU** (Cohere For AI) — https://huggingface.co/datasets/CohereLabs/Global-MMLU | **Apache License 2.0** — https://www.apache.org/licenses/LICENSE-2.0 | Selected `dev` split across 11 languages (`en, ru, zh, es, ar, pt, de, fr, ko, ja, hi`); formatted as `prompt` (question + choices) and `comp` (answer text). |
| `xstory_cloze_train` | 79 | **XStoryCloze** (Meta AI / FAIR, Lin et al. 2022) — https://huggingface.co/datasets/juletxara/xstory_cloze · FAIR notice: https://github.com/facebookresearch/fairseq/blob/main/examples/xglm/XStoryCloze.md | **CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0/ | Selected `train` split across multilingual story completions; formatted as 4-sentence story `prompt` and winning ending `comp`. Derivatives of these rows remain under CC BY-SA 4.0. |
| `gsm8k_train` | 58 | **GSM8K** (OpenAI, Cobbe et al. 2021) — https://huggingface.co/datasets/openai/gsm8k | **MIT License** — https://opensource.org/licenses/MIT | Selected `train` split (`main` config); formatted as math word problem `prompt` and step-by-step rationale/answer `comp`. |
| `arc_easy_train` | 54 | **AI2 Reasoning Challenge (ARC-Easy)** (Allen Institute for AI, Clark et al. 2018) — https://huggingface.co/datasets/allenai/ai2_arc | **CC BY-SA 4.0** — https://creativecommons.org/licenses/by-sa/4.0/ | Selected `train` split (`ARC-Easy` config); formatted as science question + choices `prompt` and answer `comp`. Derivatives of these rows remain under CC BY-SA 4.0. |
| `openbookqa_train` | 48 | **OpenBookQA** (Allen Institute for AI, Mihaylov et al. 2018) — https://huggingface.co/datasets/allenai/openbookqa | **Apache License 2.0** — https://www.apache.org/licenses/LICENSE-2.0 | Selected `train` split (`main` config); formatted as multiple-choice question `prompt` and answer `comp`. |
| `mbpp_train` | 38 | **MBPP** (Google Research, Austin et al. 2021) — https://huggingface.co/datasets/google-research-datasets/mbpp | **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/ | Selected `train` split; formatted as Python task specification `prompt` and reference solution `comp`. |

---

## 2. `s3-practice.jsonl.zst` (509 offline practice exam items)

Used solely as an offline fallback for local `practice` and `gym` rounds when no local
fresh harvest hours are present on disk. Official competition rounds never use this pack.

- **Source material:** Wikipedia articles (`wikipedia`), Wikipedia article additions
  (`wikipedia-added`), and Wikipedia talk pages (`wikipedia-talk`) across 15 languages
  (`en, ru, de, ja, fr, zh, es, fa, uk, it, ar, nl, pl, ko, pt`), authored by
  **Wikipedia contributors** (https://www.wikipedia.org/).
- **License:** **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/) and
  GFDL, per Wikimedia Foundation Terms of Use.
- **Transformations:**
  - `221` twin-context comprehension items (`kind: "context"`): Wikipedia passages split
    at a sentence/paragraph boundary with length-matched wrong openings from the same
    source and language, normalized to crawl whitespace (`src/stet/exam/fresh/select.py`).
  - `288` reasoning items (`explain: 101`, `infer: 93`, `synthesis: 94`): deterministic
    `Qwen/Qwen3-8B` (Apache 2.0) multi-sentence comprehension/reasoning prompts and
    responses derived from the Wikipedia excerpts (`src/stet/exam/fresh/reason.py`),
    distributed under CC BY-SA 4.0 as adaptations of Wikipedia text.

---

## 3. Referenced Runtime Models (Downloaded at Runtime, Not Bundled in Wheel)

- **`baidu/ERNIE-4.5-0.3B-Base-PT`** (Season 3 student & tokenizer): **Apache License 2.0** — https://huggingface.co/baidu/ERNIE-4.5-0.3B-Base-PT
- **`Qwen/Qwen3-8B`** (Season 3 official examiner reasoning teacher): **Apache License 2.0** — https://huggingface.co/Qwen/Qwen3-8B
- **`Qwen/Qwen3-0.6B`** (`examples/reader` reference curator): **Apache License 2.0** — https://huggingface.co/Qwen/Qwen3-0.6B
