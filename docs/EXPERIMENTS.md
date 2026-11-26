# Experiments

The measurements the rules rest on, and what is still to be measured. Each says what was
run, what came out, and what it decided. Scores are in bits per byte; higher is better. Each
says which exam it was measured on: Season 1's (seven capabilities, each counting the same)
or Season 2's ([the rules](RULES.md#the-exam)). The reasoning behind the exam, and the pilot
that chose the student's size, are in [the research](research/exam-first-principles.md).

## How much is noise?

GPU arithmetic is not exactly repeatable, and the differences grow over training: the same
script, trained twice from the same initial weights, does not give the same student. Nine
curators on one official round's corpus were trained again on rented NVIDIA GPUs, three times
at micro and twice at standard, and each training sat the exam. On Season 1's exam:

| | micro | standard |
|---|---|---|
| A curator's score, sd between trainings of the same script | 0.025 | 0.013 |
| Spread between curators | 0.095 | 0.077 |
| Signal to noise | 3.8 | 5.8 |
| Rank agreement with the large student (Spearman) | 0.69 | 0.95 |

The unedited corpus trains least steadily of all.

**What it decided.** Rounds train the standard student: it is steadier, and it orders
curators as the large one does, where micro does not
([why](research/exam-first-principles.md#75-scale-rounds-at-the-standard-student-finals-at-the-large-one-decide-the-season)).

**The pilot, scored other ways.** The scale pilot
([research](research/exam-first-principles.md#6-the-scale-pilot-round-12-november-2026):
nine curators, three sizes of student, repeated trainings) kept every item's score, so it
could be scored again under any way of adding the items up, with nothing trained again.
The rule was fixed before the numbers: take the one with the best spread over noise at the
official size, provided it agrees with the large student no worse than Season 1's.

| How the items are added up | Noise: sd between two trainings of one script | Spread between curators, over noise | Rank agreement with the large student |
|---|---|---|---|
| Season 1's | 0.0196 | 3.4 | 0.90 |
| Bits pooled within a stratum | 0.0157 | 4.0 | 0.95 |
| Square-root weights | 0.0172 | 3.8 | 0.98 |
| No item's bits put over fewer than 16 bytes | 0.0097 | 6.0 | 0.90 |
| Fresh writing only | 0.0049 | 28.0 | 0.88 |
| Generated items only | 0.0225 | 2.8 | 0.93 |
| **Season 2's: the 16-byte floor, writing three quarters, the skills that pass one quarter, small languages pooled** | **0.0052** | **21.7** | **0.95** |

Part by part, on the same pilot:

| Part | Items | Spread over noise | Order carries to the large student |
|---|---|---|---|
| Discussion (fresh) | 74 | 23.5 | 0.95 |
| Reference (fresh) | 729 | 16.3 | 0.88 |
| Agents | 32 | 4.5 | 0.90 |
| Transformation | 28 | 3.4 | 0.90 |
| Grammar | 26 | 2.2 | 0.81 |
| Reading | 72 | 1.6 | 0.71 |
| Logic | 44 | 1.6 | 0.57 |
| Facts | 200 | 2.1 | 0.57 |
| Algorithms, algebra, word problems | | 1.6 to 2.6 | 0.17 to 0.43 |
| Arithmetic, multi-step, units, structure, programs | | 0.8 to 1.8 | at chance at every size |

**The pilot's fresh writing was Wikipedia only**: new articles, additions to old ones, and
talk pages. News, technical writing and posts were not in it, and were taken to behave as
the same kind of thing, long passages of human text. The next section measured them. And
with eight curators, most of them English-only filters that the writing nearly ties, rank
agreement cannot tell a writing share of 67% from one of 90%: three quarters is a
declared value, not a measured one.

**What it decided.** The shape of Season 2's score: the 16-byte floor, writing three
quarters and skills one quarter, and the two tests a part must pass to count: curators
spread on it by at least 1.5 times its noise, and the order it puts them in carries to
the large student at 0.55 or better. Fresh writing alone is steadier still, and carries
a little less well than the mix, so the skills stay. Which parts pass was then measured
on purpose, below.

## Which parts of the exam count

Before Season 2 began, every part the exam could have was put to those two tests, with
the rule written down first
([research](research/exam-first-principles.md#97-every-part-is-in-the-score-or-off-the-exam)).
Seven curators that differ in kind (the unedited corpus, `tidy`, `english`, `fineweb2`,
`dclm`, and two entered to attack the exam) were trained on one practice round: four
times each at micro, three times at standard, once at large. Every student sat one
candidate exam of about 4,700 items: every kind of writing a harvest takes, and every generated
family, asked cold and again after a few worked examples.

| Part | Spread over noise, standard | Order carries to large | |
|---|---|---|---|
| News, reference, discussion, social posts, technical writing | 24 to 93 | 0.77 to 1.00 | in |
| Science abstracts | 19 | 0.49 | out |
| Logic, agent sessions, knowledge, reading, grammar | 4.9 to 12 | 0.71 to 1.00 | in |
| Arithmetic, units, word problems, algebra, an algorithm's output: after worked examples | 1.6 to 3.8 | 0.66 to 0.94 | in |
| Transformation; structured data after worked examples | 2.1; 1.6 | 0.83; 0.60 | passed here, then out: [the order of a script](#the-order-of-a-script) |
| The same six asked cold | 0.8 to 3.4 | −0.09 to 0.94 | out (one passes; its other form counts) |
| Running a program; several steps of arithmetic; answers offered as a pair | 0.4 to 2.4 | −0.71 to 0.66 | out |

The score the final parts make has a noise of 0.0036 bits per byte between two trainings
of one script at standard, and the six curators spread by forty-five times that. They
stand in the same order at standard and at large.

**What it decided (Season 2).** The exam is four kinds of writing (reference, news, technical,
conversation) and seven skills (agents, reading, logic, grammar, numbers, programs,
knowledge). A part that failed is not on the exam, and its generator or its
source is deleted: nothing is sat that does not count. Season 2's 39M student took two trainings of
one script to differ by 0.005 at standard and 0.03 at micro (`noise` in `s2.json`), with
provisional official tolerances of 0.025 (grade), 0.04 (baseline), and 0.04 (run). Season 3
uses `baidu/ERNIE-4.5-0.3B-Base-PT` (bottom half frozen, top half reset, 65.5M tokens +
1-epoch multilingual SFT probe) on a 100%-fresh exam (100% harvest-derived multi-sentence
reasoning scored; 4 human-written twin-context kinds sat at 0% weight as diagnostics); its
own noise figures, calibrated NVIDIA CUDA tolerances (`0.012` grade, `0.015` baseline,
`0.015` run), DataDecide 1B agreement, and adversarial red-team measurements live in
`src/stet/seasons/s3.json` and the Season 3 validation sections below. Every round's page
publishes each grade's difference from the others.

**Can the exam be bought with a curator's own text?** One of the two attackers, `scribe`,
was `tidy` plus the 16,000 tokens of its own text a script could then hold, spent on
worked problems in the forms of five skills. Where that worked it bought 0.10 bits per
byte (agent sessions and transformation, 5.6 and 3.2 standard errors); on the score as it
stood it beat `tidy` by 0.010, 2.6 standard errors. On the final score, where
transformation is gone and agent sessions weigh less, the same students differ by 0.004,
1.3 standard errors.

**What it decided.** A script holds no text of its own: it is copies of the crawl and no
more. That is partly a judgement: by the letter of the test, on the final score, the text
bought too little to count. It bought the parts it was aimed at, nothing a curator does
for the data needs it, and [text written for the purpose](#text-a-model-writes) taught no
more than the pages it was written from.

**Is the pool a curator chooses from big enough?** A standard round's pool is 16 files of
the crawl, six times what the student reads. One student of each script was trained on a
pool of 64 files. The rule: take the larger pool if the curators' scores spread a quarter
more on it, or their order changes. The spread was a fifth smaller, and the order agreed
at 0.94.

**What it decided.** The pool stays at 16 files; four times the download would have
bought nothing. Micro's pool, which was twelve times its budget, becomes six times like
the others (4 files, 300 seconds): on it the micro score's noise fell from 0.045 to 0.030.

**Does averaging the student's last weights steady it?** Six ways of averaging were
carried through every one of those trainings. Every one made the score noisier at micro,
by 1.5% to 6.5%, where the rule asked for a quarter less.

**What it decided.** The trainer stays as it is.

**Micro still does not order curators as standard does.** Even on a pool of the same
ratio, the micro and standard scores agree at only 0.43 (seven scripts). Practice at
micro is for finding mistakes; a curator is compared at standard.

## Text a model writes

Would text that a small open model writes from the round's own pages teach the student
more than the pages do? One practice round at standard, 69 students on rented GPUs
([research](research/exam-first-principles.md#99-text-a-model-writes-a-pilot), with the
rule written first). A fixed 4-billion-parameter model rewrote pages of `tidy`'s script in
its own words, for 1%, 5% and 15% of what the student reads; in two more arms it wrote up
pages `tidy` had not chosen, in place of `tidy`'s weakest. Every arm had a twin with the
same pages in the same places, copied and not rewritten, and every pair was trained in up
to three orders.

| On the exam's writing, over thirteen pairs | The model's text less a copy of the same pages | Pairs where the model's text is ahead |
|---|---|---|
| The exam as it was | +0.008 | 12 |
| Empty lines closed | −0.005 | 4 |
| White space as the crawl has it | −0.002 | 5 |

The first row is not the model's writing. The exam's passages held an empty line every 500
bytes, tabs, and runs of spaces: the sources' own layout. Common Crawl's text has none of
them (an empty line once in 140,000 bytes), so a student had hardly seen one, and each cost
about twenty bits. The model writes paragraphs with an empty line between them. An arm
with no model's text at all, only an empty line put after every line of 396 pages, gained
half of what the 1% arm did.

**What it decided.** Two things.

- **No text a model writes.** With the layout taken out, rewritten pages are level with
  their copies or below them, and worse as the dose grows (0.006 to 0.018 below at 15% in
  place). No arm cleared the rule on the corrected exam.
- **The exam's writing has the crawl's white space.** One space between words, one line end
  between lines, no tabs. `tidy`, which had cut pages to their prose lines and so never
  showed a student an empty line, was being marked 0.02 down for it.

**`tidy` copies its pages whole.** The twins showed it, and four orders confirmed it: the
pages `tidy` chooses, copied whole, beat `tidy`'s lines of them by 0.018 to 0.034 on the
writing, every time. Nearly all of it is technical writing and code, whose lines are not
sentences. And the starter (`stet init`) keeps 93% of pages, so it scores as the raw pages
do; the site said it beat them, and no longer does.

## The order of a script

A script's order is the curator's to choose, and most shuffle. The pilot above trained
one curator's pages in many orders: `tidy`'s lines ten ways, the same pages whole four
ways, and the raw pages four ways. One student each.

| The standard score, sd | bits per byte |
|---|---|
| Between two trainings of one script | 0.0037 |
| Between two orders of `tidy`'s lines (ten) | 0.0084 |
| The same without the one order that stands apart (nine) | 0.0052 |
| Between two orders of `tidy`'s pages whole (four) | 0.0056 |
| Between four draws of the raw pages | 0.019 |

Nine of the ten orders of `tidy` agree about as two trainings do. One, the order a
harness of a day earlier happened to give, is 0.02 better, and loses it when its lines
are shuffled again. Why is not known. The raw pages vary far more: a student reads a
sixth of them, and which sixth is worth up to 0.04.

**What it decided.**

- **The noise a part must clear is this one.** A round gives a curator one script. Set
  against the spread between arrangements of one curator's pages, and not between
  trainings of one script, two parts no longer told curators apart by 1.5 times the
  noise: transformation (1.3) and structured data (0.8). Both are off the exam. Every
  kind of writing still passes by 20 or more, but for technical writing (3.5, or 8.9
  without one draw of raw pages); the other skills by 1.6 to 6.2.
- **A practice run prints both numbers**, and the curator's brief says that a change which
  also changes the order has to beat 0.008, not 0.005.
- **The baseline stays four trainings of the crawl's own order.** A lucky baseline moves
  every curator in the round alike, and the board fits each round's difficulty and takes
  that out. It does move the gain a single round shows.

## Where a round's training went

In Season 1 every grading machine trained the unedited corpus for itself each round, and
all seven reference curators ran in every round. A third of the season's training was
baselines, and half was the references.

**What it decided.** Four established graders measure a round's baseline, and its score on
each item is the median of theirs: every grader's would be no better a baseline than
a few, and it is still several, so that one unsteady training of the unedited
corpus does not move every gain in the round, whichever machine reports first. Two references, `fineweb2` and `dclm`, run
in every round and tie the rounds together; each of the others runs in one round in eight.

## What the record rule can tell apart

The record changes hands on a one-sided paired test against a margin of 0.005, at looks after
5, 10, 20 and 40 shared rounds ([the rule](RULES.md#the-record)). How often does a
challenger with a given true lead over the holder end up with the record? It depends on how
much two curators' gains differ from round to round, which is not measured yet, so here it is
at four guesses. The season takes one curator's gain to move by 0.015 from round to round
(`noise.standard.round_sd`: Season 1's board, scaled by the pilot), which puts the
difference of two near 0.02 if they move independently.

| Spread between rounds \ true lead | 0 (a copy) | 0.005 | 0.01 | 0.015 | 0.02 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.1% | 4% | 82% | 100% | 100% | 100% | 100% |
| 0.02 | 0.6% | 4% | 30% | 82% | 99% | 100% | 100% |
| 0.03 | 1.0% | 4% | 17% | 48% | 82% | 100% | 100% |
| 0.05 | 1.6% | 4% | 10% | 22% | 40% | 82% | 98% |

```
uv run --no-project --with numpy --with scipy python tools/record_power.py --sd 0.02 --margin 0.005 --leads 0 0.005 0.01 0.015 0.02 0.03 0.04
```

**What it decided.** A lead equal to the margin takes the record about one time in
twenty-five, inside the 5% the rule allows. A copy of the holder takes it once in a hundred
tries or less while rounds differ by 0.03 or less, and 1.6 times in a hundred if they
differ by 0.05. A lead of 0.02, four times the margin, is confirmed nearly always if rounds
differ by 0.02 or less, and four times in ten if they differ by 0.05. The margin is
*provisional* until the first rounds measure the spread between them.

## How long a round takes

| Step | When |
|---|---|
| The draw. Its seed picks three of the hours already harvested | 0 |
| The examiner builds the round's corpus, makes the exam from those hours' harvests and posts it. It reads no source | From the draw |
| Curators run, two runs each of up to 1,200 seconds | From the draw, on the machines around |
| Grades: one standard training each, about 18 minutes on a rented RTX 3090 or 4070 Ti, under $0.10 | From when a curator has written a script |
| Each grader sits the exam on its trained students | Once the exam is posted |
| The round settles | When its runs and its grades are in |

A large student, for the finals, trains in about an hour on an RTX 4090.

Season 1 harvested the hour after each draw and then waited an hour more for the sources
that publish late. Its exam was posted about 2 h 15 m after the draw and its rounds settled
at about 2 h 20 m, while the machines that had trained sat idle.

**What it decided.** The harvest moved before the draw: every hour of the clock is
harvested an hour after it ends, a draw picks three of the 26 before it, and no round waits
for its exam.

## What a harvest gives

On Season 1's sources, a live harvest of a two-hour window took 26 minutes and gave 1,683
documents. Chosen for an exam, they made 901 items:

| Kind of writing | Items | Sources (items) | Languages |
|---|---|---|---|
| Reference | 255 | Wikipedia additions 151, new articles 104 | 36, 29% English |
| Discussion | 255 | Wikipedia talk 70, Bluesky replies 65, Lemmy 47, Mastodon replies 39, Hacker News 34 | 20, 65% English |
| News | 140 | Common Crawl's news crawl 140, from 110 sites | 39, 45% English |
| Social | 129 | Bluesky 74, Mastodon 55 | 13, 53% English |
| Technical | 122 | GitHub 122 | 12, 72% English |

News, social and technical had fewer candidates than their part, and gave the rest to the
other two. GitHub's comments are left out: about half of those written now come from coding
agents under people's own accounts.

**One hour is enough.** Round s1-o00003's window (20 November, from 11:40 UTC), harvested
again with every source read at once: the two hours took 645 seconds and gave 1,592
documents; its first hour took 629 seconds and gave 1,156. Chosen for an exam, both made 900
items, in nearly the same parts (discussion 257 and 267, reference 257 and 267, news 143 and
141, social 126 and 111, technical 118 and 114), the hour in 70 languages to the two hours' 51.
Each source still gave most of its share in the hour (Hacker News 97 of 150, Lemmy 107,
Mastodon 83), and the parts are kept equal by the choosing, not by the harvest.

**What it decided.** A harvest is of one hour of the clock. A Season 2 exam is made from
three of them.

**Two more sources, and two left out.** Forums that run Discourse, 31 of them, give about
100 replies an hour. The kernel's mailing lists give first versions of patches: fresh code
written by people. Stack Exchange was tried and left out: four questions an hour. Crossref
gives about 40 new works' abstracts an hour; science abstracts did not pass the season's
test ([above](#which-parts-of-the-exam-count)), so it is not harvested.

## What a small model can read in twenty minutes

A curator that uses a language model is limited by the clock before it is limited by
memory. Measured on the reference machine (Radeon 8060S, ROCm, PyTorch 2.15) with nothing
else running, half precision, batches of 128 passages of 128 tokens, one forward pass:

| Model | Parameters | Tokens read per second | In 19 minutes |
|---|---|---|---|
| MiniLM-L6 (encoder) | 22M | 669,000 | 763M |
| ModernBERT-base (encoder) | 149M | 76,000 | 87M |
| SmolLM2-135M | 135M | 65,000 | 74M |
| SmolLM2-360M | 362M | 29,000 | 33M |
| Qwen3-0.6B | 596M | 15,800 | 18M |
| Qwen2.5-1.5B | 1,544M | 7,900 | 9.0M |
| SmolLM2-1.7B | 1,711M | 6,300 | 7.2M |

A standard round's corpus is about 2.5 GB in 335,000 documents. So a small encoder can
read most of it, a 0.6-billion-parameter model the opening of the tens of thousands of
documents cheaper code liked best, and nothing larger much at all. Writing is far slower
than reading, 350 to 1,250 tokens a second in batches of 16, and a script holds no text of
its own, so there is nothing for written text to do.

That machine is about 1.2 times the floor for running curators
(`runner.gpu_tokens_per_second` in the season file is this measurement's cousin: a fixed
110M-parameter model, 80,000 tokens a second).

**What it decided.** A curator is given 8 GB of GPU memory, and model files may total
4 GiB: enough for any model in this table.

## Season 3: ERNIE-4.5-0.3B and the 100%-fresh exam

Before Season 3 opened, a 32-run 8× RTX 5090 gate on an earlier `Qwen3-0.6B-Base` draft
(32.8M tokens, 7 synthetic procedural skills + 4 fresh twin-context pillars) failed to clear
chance on Allen AI's DataDecide 1B known-answer ordering (`7/15` pairs, `p = 0.640` across
4 seeds and disjoint slices), because synthetic templates rewarded short-window textbook/code
enrichment and let a 3% overfiltering trap beat 1B-optimal recipes.

Four follow-up validation stages (~125 GPU runs total across RTX 5090 and RTX 3090 hardware,
archived in `~/stet-measurements/s3-validation/{sweep10,adv_v2,prod_gate,live_gate}`) redesigned
both the student and the exam from first principles and verified the exact live official pipeline:

1. **Frozen bottom-half `baidu/ERNIE-4.5-0.3B-Base-PT` at `65.5M` tokens (12-recipe DataDecide 1B sweep).**
   Freezing the tied embedding/head (`vocab = 103,424`) and lower 9 layers while resetting the
   upper 9 layers (`127.4M` trainable parameters) lets the student train on **2× more tokens
   (`65.5M`)** in **nearly half the time** (`19.7–20.1 min` per pass on RTX 5090, `~30 min` on 4090,
   `~55 min` on 3090) and **`8.90 GiB` allocated / `9.33 GiB` reserved VRAM** across training,
   SFT, and exam sitting (lowering the volunteer grading floor to **12 GB**; independently
   reproduced on AMD Radeon 8060S ROCm 7.14 at `9.43 GiB` reserved matching NVIDIA's `init_sha256`).
2. **Live official 100%-fresh 7-pillar exam (evaluated initially at `50%` twin-context comprehension + `50%` harvest-derived reasoning before stage 4 moved comprehension to `0%` diagnostic).**
   Running unmodified `stet.exam.examine.build()` on a brand-new official draw (`s3-o00001`,
   `507,785` CC-MAIN-2026-43 docs) and 3 fresh Nov 25 harvest hours (`1795622400`, `1795626000`,
   `1795629600`, `0/0` overlap with `s3-practice.jsonl.zst`) produced **`1,370` fresh items**
   (`items_digest = 5c0f0579f368666a...`, `init_sha256 = 3a3a76cbc989eaf8...`) across **`74` languages**
   (`73` in reasoning; `reference: 208, news: 130, technical: 209, conversation: 351, explain: 211,
   infer: 204, synthesis: 57` across all 3 decoupled wrappers `26 / 16 / 15`) in **`555.7s` (`9.3 min`)**
   draw-to-publication (`93.3s` corpus fetch + `462.4s` full 4-GPU 640-doc `Qwen/Qwen3-8B` synthesis,
   decontamination, `min_pillar_items=20` enforcement, and authoritative CPU `init_sha256`),
   passing public round-trip and replay determinism checks and matching every CUDA student's
   `init_sha256` bit-for-bit.
   - **DataDecide 1B (12 recipes vs `dolma1_7`, ONE pinned `macro.parquet` 1B truth table):**
     - **Draw 1 (`prod_gate`, `s0`, English subset):** `46/66` (`69.7%`, perm one-sided `p = 0.0426`),
       `2pt+ = 25/31` (`80.6%`), Spearman `ρ = +0.566` (`p = 0.0548`), `11/11` filtered-vs-unfiltered
       pairs right, 3% trap `#11/12`.
     - **Draw 2 (`live_gate`, `s1`, brand-new live official exam, English subset):** **`48/66` (`72.7%`,
       perm one-sided `p = 0.0225`)**, **`2pt+ = 26/31` (`83.9%`)**, **`ρ = +0.594` (`p = 0.0415`)**,
       `11/11` filtered-vs-unfiltered pairs right, 3% trap `#11/12`.
     - **2-Draw Independent Mean (`Draw 1 + Draw 2`, English subset):** **`47/66` (`71.2%`, perm
       one-sided `p = 0.0314`)**, **`2pt+ = 26/31` (`83.9%`)**, **`ρ = +0.580` (`p = 0.0479`)**,
       3% trap `#11/12`. *(On the full multilingual exam, English-only DataDecide recipes score
       `28/66`, `28/66`, `30/66` [Draw 1 / Draw 2 / 2-draw mean] under the pre-Review #6 unweighted
       `"other"`-pool scorer, and **`27/66` (`ρ = −0.238`), `37/66` (`ρ = +0.084`), `34/66`
       (`ρ = −0.042`)** under the shipping `weight_other_by_share=true` Season 3 scorer, because
       strict English-only DCLM-QC filters regress against `dolma1_7` on the non-English crawl;
       the English-subset totals `46/66`, `48/66`, `47/66` are identical under both scorers.)*
     - **Honest bias diagnosis (not draw noise):** Draw 1 and Draw 2 correlate at **`ρ = +0.94`**
       (`60/66` pairs) with each other, which means the gap from DataDecide's 1B truth table is a
       **systematic lean**, not sampling noise that averages out across rounds. Both draws reliably
       separate filtered from unfiltered recipes (`11/11`) and catch severe overfiltering (`#11/12`),
       but both place `FineWeb-Edu` (`#7/12` in 1B macro, `1.95 pp` below `#1`) at `#1` by ~2× the
       runner-up (`+0.0205` vs `+0.0097` on Draw 1, `+0.0248` vs `+0.0153` on Draw 2) because the
       teacher's explanatory prose over-rewards educational/textbook style; among the top-8 recipes
       within 2 pp of each other, shipped-exam ordering is near chance (`16/28` and `17/28`,
       `ρ ≈ +0.2`). *(An earlier historical prototype sweep `sweep10` reached `55/66 = 83.3%`,
       `96.8%` `2pt+`, `ρ = +0.818`, `p = 0.0010` on a pre-shipped prototype exam; future sweeps
       should keep trained student checkpoints so scoring changes can be re-evaluated without
       retraining.)*
   - **Cross-GPU determinism on real materialized stream:** identical `127.4M`-parameter
     `trainable_sha256` (`4c949103d67fc83d`) across GPU 4 and GPU 5 (`max_item_diff = 2.0e-6`).
   - **Live official exam gauntlet (100% unmodified `stet.pipeline.edit_round` + `stet.script.materialize`
     + `round_score`; raw 2-seed unedited mean diff = `0.0004 bpb`, paired item-bootstrap SE ≈
     `±0.0010 bpb`).** Both the historical pre-Review #6 scorer (`weight_other_by_share=false`)
     and the **shipping Season 3 scorer (`weight_other_by_share=true`)** on the exact same saved
     Phase 9 item score arrays and 2-seed unedited baseline are shown side by side:

| Curator (all via `stet.script.materialize`) | Pre-Review #6 scorer (bpb) | **Shipping S3 scorer (bpb)** | Shipping Comp | Shipping Reas | Notes |
|---|---:|---:|---:|---:|---|
| **`fineweb2` (2-seed)** | `+0.01911` | **`+0.02056`** | **`+0.00379`** | `+0.03733` | Positive on all 7 pillars (`s0=+0.02010`, `s1=+0.02102`, diff `0.00092`) |
| **`reader` v2 (shipped, 2-seed)** | `+0.01922` | **`+0.01993`** | `+0.00178` | **`+0.03808`** | `vllm` `259.1s/1200s` *(unsandboxed host cores)*, `py3langid` per-lang quantile, `80` langs (`s0=+0.01916`, `s1=+0.02070`, diff `0.00154`) |
| `mimic` | `+0.00740` | **`+0.01527`** | `+0.00141` | `+0.02912` | Rises under crawl-weighted `"other"` pool; still `~0.0053 bpb` below `fineweb2` |
| `reader` v1 *(pre-fix diagnosis, 2-seed)* | `−0.00340` | **`+0.00348`** | `+0.00120` | `+0.00575` | Exposed `BYTES_PER_TOKEN=4.0` top-band truncation + coarse script buckets |
| `adv_qa_spam` (Q/A-heavy pages) | `−0.00630` | **`+0.00153`** | `+0.00339` | `−0.00034` | **Crosses slightly above unedited** under shipping scorer (`+0.00339` comp, `−0.00034` reas; `13×` below `fineweb2` / `reader` v2) |
| `unedited` (2-seed baseline) | `0.00000` | **`0.00000`** | `0.00000` | `0.00000` | Baseline (2-seed raw mean diff `0.0004 bpb`) |
| `adv_splice_qa` (synthetic Q/A splices) | `−0.03480` | **`−0.03696`** | `−0.00538` | `−0.06854` | Crushed below unedited |
| `english` (English-only filter) | `−0.08510` | **`−0.08671`** | `+0.00884` | `−0.18226` | Monolingual collapse penalized |
| `adv_overfilter` (top-3% narrow keep) | `−0.54270` | **`−0.51935`** | `−0.02921` | `−1.00948` | Obliterated |

**What it decided.** Season 3 ships `baidu/ERNIE-4.5-0.3B-Base-PT` (`freeze_bottom: true`,
`65.5M` tokens, 1-epoch commercial-permissive multilingual SFT probe) with the 100%-fresh exam
(100% fresh reasoning scored after stage 4 below, 4 human-written comprehension kinds at `0%`
diagnostic; `rules_version: 7`, `weight_other_by_share: true`, `code_share: 0.15`,
`min_pillar_items: 20` with sparse-pillar omission and `STET_EXAM_TIMEOUT = 6h` +
`POST /api/v1/admin/rounds/{id}/cancel`, `baselines.per_round = 2`),
a **12 GB** volunteer grading floor, **24 GB+** neural curator sandboxes, and on-demand rounds (`$0/day` idle).
Running unmodified `stet.pipeline.edit_round` + `stet.script.materialize` caught and fixed two real
bugs in `examples/reader/curator.py` (`BYTES_PER_TOKEN=4.0` truncating the top 3,634 Band-3 docs
and coarse 10-script buckets starving `fa`, `pl`, `uk`), jumping `reader` from `+0.00348` to
**`+0.01993 bpb`** under the shipping scorer (`−0.00340 → +0.01922` under the historical scorer;
`fa −0.150 → +0.0168`, `pl −0.081 → +0.0410`, `uk −0.030 → +0.0398`, `zh −0.018 → +0.0002`,
`ja −0.011 → −0.0014`, `ko −0.029 → −0.0184`) in `259.1s / 1,200s` on `vllm==0.34.0`
*(unsandboxed host cores; standalone 6-core `taskset -c 0-5` benchmark on `s3-p00000`'s `507,387` docs
completed prefiltering + `py3langid` in **`37.49s`** -> `75,678` candidates, leaving `>1,160s` for
GPU scoring)*.

3. **Production AMD Radeon 8060S examiner verification & language-mix adversary gate (`amd_examiner`, `lang_adv_gate`).**
   After fixing the `synthesis` `PROMPT: ...` placeholder, adding crawl-share weighting inside the
   pooled `"other"` small-language cell (`weight_other_by_share: true` on `s3`, preserving `false`
   for `s2`), collapsing Wikipedia variants into one source bucket, lowering CJK `min_len` to 180
   chars, adding wrong-language / mid-sentence / `\ufffd` filters, enforcing `min_keep_rate = 0.5`,
   and publishing teacher `prompt` + `cont` + `wrapper_id` + `excerpt_sha256` for every withheld
   reasoning item:
   - **Two cold official draws on the production AMD Radeon 8060S (`124 GiB`, `PyTorch 2.15.0+rocm7.14`, `transformers 5.23.0`, `$0/round`):**
     - **Draw A (`s3-o00002`, real `news=0` draw, hours `1795636800/44000/51200`):** omitted `news=0`
       cleanly (redistributing `16.7%` each to `reference/technical/conversation`), built **`1,517` fresh
       items** (`explain=206, infer=210, synthesis=203`; `620/640 = 96.9%` kept, `synthesis` `95.3%`,
       automated parser drops `wrong_lang=13, truncated_sentence=6, broken_utf8=1`) in **`1,222.5s`
       synthesis / `1,400.6s` build (`~24.5 min` incl. `68s` corpus fetch)**; `N_eff = 782.7`; tiny
       `<0.1%` langs **`0.39%`** of score; `294/294` withheld reasoning items published and
       bit-verified to exact `Item.id`.
     - **Draw B (`s3-o00003`, full 7-pillar draw, hours `1795633200/40400/47600`):** built **`1,491` fresh
       items** (`ref=192, news=180, tech=193, conv=315, explain=204, infer=203, synthesis=204`;
       `617/640 = 96.4%` kept, `synthesis` `95.8%`; automated parser drops `wrong_lang=11,
       broken_utf8=7, truncated_sentence=5`) in **`1,212.6s` synthesis / `1,357.9s` build (`~23.8 min`
       incl. `~68s` corpus fetch)**; Wikipedia share of reasoning dropped **`56.9% → 36.0%`**; `N_eff`
       rose **`566 → 799.4`**; tiny `<0.1%` langs dropped **`10.3% → 0.01%`** of score; 9 major
       languages cleared `SMALL=8` across all 3 reasoning kinds (`de, en, es, fr, it, ja, pl, pt, ru`);
       major language weights tracked crawl shares closely (`en 36.1% vs 34.8%`, `ru 14.2% vs 14.0%`,
       `ja 6.7% vs 6.8%`, `de 5.6% vs 5.3%`, `fr 4.9% vs 4.2%`, `zh 4.7% vs 3.7%`, `es 4.1% vs 4.0%`);
       `391/391` withheld reasoning items published and bit-verified.
     - **Independent inspection & prompt-language hardening (Review #7):** those two draws were built
       before lead-in language checking and foreign-script leakage filtering were added; automated
       response-only checks showed `0` drops on re-check, whereas independent inspection caught
       mixed-language Thai-in-Arabic prompts (`3fe4e6696726367f` on Draw A, `a371cfa3983dca24` on
       Draw B) plus 4 short 2–3 character foreign-script leaks (Han-in-Hebrew prompt `959cf9ee516c7757`,
       Han-in-Russian `9ad6cf24e01e5a64`, Arabic-in-Hebrew `dcf36ba39d52a0eb`, Han-in-Bulgarian
       `de426502d930db4a`). `parse_generated_items` now checks **both** teacher prompts and responses
       for language/script fidelity and foreign non-Latin script leakage (`>= 2` chars absent from the
       excerpt), catching all 6 defects when re-run over the archived Draw A and Draw B outputs.
   - **Cross-vendor `init_sha256` parity & 4× RTX 5090 language-mix adversary gate (`lang_adv_gate`):**
     - Local AMD Radeon 8060S (`ROCm 7.14`, `PyTorch 2.15.0`) `init_sha256` (`5bd1d5ce4547697e...`)
       matched **4/4 NVIDIA RTX 5090 (`CUDA 13`, `PyTorch 2.14.0`) graders bit-for-bit**.
     - **`adv_tiny_lang_boost` (`32.5%` `<0.5%`-crawl small languages):** on Phase 9's exam (`s3-o00001`),
       crawl-share weighting inside `"other"` cut its score from **`+0.01256` (old unweighted) to
       `+0.00342` (`−0.00914 bpb`)**, `6×` below `fineweb2` (`+0.02060`); on Draw B (`s3-o00003`),
       it scored **`+0.00876`** vs `fineweb2` **`+0.02795`** (`3.2×` below), confirming the small-language
       `"other"`-pool exploit is closed.
     - **`adv_exam_lang_mix` (clean deduplicated pool resampled to Draw B's exam item counts):** scored
       **`+0.03193`** vs `fineweb2` `+0.02795` (`+0.0040`) on matched Draw B and **`+0.02296`** vs
       `+0.02060` (`+0.0024`) on unmatched Phase 9 (`s3-o00001`), an **observed single-seed point
       difference of `+0.0016 bpb`** between the matched and unmatched draws. Because
       `adv_exam_lang_mix` and `fineweb2` also differed in prefiltering and deduplication (using
       `reader`'s CJK-aware prefilter restored `ja` from `1.99% → 3.59%`, lifting `ja` by
       `+0.0259 bpb`, alongside mild temperature upsampling of `1%–5%` major languages over `<0.5%`
       micro-languages), this single-seed comparison does **not** establish a formal upper bound on
       exam-specific language-mix exploitation; estimating a bound requires toggling reweighting on/off
       over the same filtered pool across multiple independently held-out draws and training seeds.

4. **Pre-launch exam decomposition, perplexity-pruning adversary gate, cross-card hardware parity, and round cost (`r8_gate`).**
   Before opening Season 3, four empirical gates audited the shipped exam and hardware fleet:
   - **Half-by-half decomposition (why Season 3 scores `reasoning = 1.0`, `comprehension = 0.0` diagnostic):**
     Re-scoring Draw 1 and Draw 2 with each half alone proved that the human-written twin-context
     half carried 50% of the declared weight but supplied only `3%–12%` of `fineweb2`'s actual
     gain, scored at or below chance on DataDecide (`24/66` on Draw 1, `34/66` on Draw 2, putting
     the 11th-of-12 over-filtered recipe first/second), could not separate `adv_qa_spam`
     (`+0.0028` / `+0.0034`) from `fineweb2`/`reader`, had cross-card noise (`5090 vs 4090 =
     0.0027–0.0037 bpb`) larger than `fineweb2`'s entire gain (`+0.0007–+0.0017 bpb`), and
     scored `english` (zero non-English training data) higher than `fineweb2` even on non-English
     items (`+0.0088 / +0.0046` vs `+0.0013–+0.0035`) because subtracting wrong openings cancels
     fluency while the frozen lower half of `ERNIE-4.5-0.3B` already reads every language.
     By contrast, the **model-written reasoning half alone** (`explain`, `infer`, `synthesis`):
     - scores **`49/66` (Draw 1) and `46/66` (Draw 2), mean `47.5/66`** on DataDecide (matching or
       beating the 50/50 mix's `47.0/66`);
     - increases `fineweb2` signal-to-noise from `14–17×` to **`24–55×`** (`+0.0373 / +0.0542`
       vs 2-seed diff `0.0006–0.0012`);
     - places `adv_qa_spam` strictly below unedited (`−0.00034` on Phase 9);
     - strongly penalizes missing languages (`english = −0.162 / −0.182`).
     Season 3 therefore sets `groups = {"reasoning": 1.0, "comprehension": 0.0}` (with `code_share = 0.15`
     and per-source caps inside `select_context` so LKML never dominates even diagnostic `technical`),
     showing the four human-written kinds on the round and curator pages beside the score as
     0%-weight diagnostic telemetry until a human-written measure clears DataDecide.
   - **Perplexity-pruning / model-predictable prose adversaries (`adv_ppl_qwen06b` & `adv_ppl_ernie03b`,
     2× RTX 5090, Draw B `s3-o00003`, 65.5M tokens):**
     We built two pure lowest-perplexity curators on `reader._prefilter`'s `75,565` deduplicated
     multilingual docs with **zero quality rubric**, matching `fineweb2`'s exact per-language byte
     mix (`~21.9k` unique docs = bottom `~29%` NLL/byte, `passes = 1.0`): one scored by
     `Qwen/Qwen3-0.6B` (same family as teacher `Qwen3-8B`) and a cross-family control scored by
     `baidu/ERNIE-4.5-0.3B-Base-PT` (independent Baidu weights).

| Curator (all Draw B crawl, 65.5M tok, vs Draw B `unedited`) | Draw B Reasoning (official S3) | Draw B Comp (0% diag) | Draw B 50/50 | Phase 9 Reasoning (official S3) | Phase 9 Comp (0% diag) |
|---|---:|---:|---:|---:|---:|
| **`adv_ppl_qwen06b`** (lowest `Qwen3-0.6B` NLL/byte) | **`+0.07418`** (+37%) | **`+0.00726`** (4.2×) | `+0.04072` | **`+0.05664`** (+40%) | **`+0.00497`** (7.3×) |
| **`adv_ppl_ernie03b`** (lowest `ERNIE-4.5-0.3B` NLL/byte) | **`+0.07247`** (+34%) | **`+0.00867`** (5.0×) | `+0.04057` | **`+0.05363`** (+32%) | **`+0.00521`** (7.7×) |
| `adv_exam_lang_mix` (`reader._prefilter` random) | `+0.06027` | `+0.00358` | `+0.03193` | `+0.04433` | `+0.00158` |
| `fineweb2` (heuristic-only, no neural filter) | `+0.05417` | `+0.00173` | `+0.02795` | `+0.04051` | `+0.00068` |
| `adv_tiny_lang_boost` | `+0.01763` | `−0.00012` | `+0.00876` | `+0.01067` | `−0.00382` |

     **Honest diagnosis (both sides of this result):**
     1. *Not Qwen3 teacher-family overfitting:* `ERNIE-4.5-0.3B` overlapped `Qwen3-0.6B` on `79.3%`
        of selected documents, beat `Qwen3-0.6B` on `explain` (`+0.02077` vs `+0.02020`),
        human-written `technical` (`+0.0154` vs `+0.0064`), and Phase 9 English reasoning
        (`+0.06534` vs `+0.06455`), and **both** crushed `fineweb2` by `4.2×–7.7×` on 100%
        human-written twin-context comprehension (where unconditional style cancels identically).
        Keeping the lowest-PPL `~29%` of deduplicated Gopher-filtered Common Crawl is literally
        **CCNet-head / perplexity pruning** (Wenzek et al. 2019; Marion et al. 2023; Ankner et al.
        2024), which drops garbled web scrapes and increases `.edu/.gov/.ac/.org` share
        (`8.2% → 9.0–9.4%`, `pmc.ncbi.nlm.nih.gov` `8 → 18`, `med.stanford.edu` `5 → 14`).
     2. *Real Season 3 caveat:* uncapped lowest-PPL with zero quality judgment **also** increases
        privacy/TOS/copyright boilerplate (`3.0% → 4.6%`) and AI-SEO phrase density (`1.3% → 2.5%`),
        lowers lexical diversity (`TTR 0.713 → 0.676`), and puts PHP deprecation logs and AI
        listicles in its extreme bottom tail — yet beats both `fineweb2` (`+32–40%`) and `reader`
        v2 (`+0.0381`, whose zero-shot 0.6B rubric ratings collapsed to `[2.57, 2.81]`). On Phase 9
        English reasoning, `FineWeb-Edu` (`+0.10562`) and `DCLM-qc-10p` (`+0.08705`) still lead
        generic English Common Crawl (`ml_english` `+0.08008`), showing strong quality classifiers
        retain headroom above generic low-PPL filtering. Entrants should treat CCNet-style
        perplexity pruning as an established, strong neural baseline on NLL exams.
   - **Cross-card hardware parity (`5090` vs `4090` vs `3090` at `standard` 65.5M tokens; `5090` vs `AMD 8060S` at `nano`/`micro`):**
     Training the exact archived `fineweb2.jsonl` (`tokens_sha256 = 232c6bee572bc914...`) from the
     exact Draw B seed produced **bit-identical `init_sha256` (`5bd1d5ce4547697e...`) and
     `tokens_sha256`** across all four GPUs:

| Hardware pair & scale | Runtime (`standard`) | Draw B Reasoning `|da−db|` / `|direct|` | Phase 9 Reasoning `|da−db|` / `|direct|` | Draw B 50/50 `|direct|` | Phase 9 50/50 `|direct|` |
|---|---:|---:|---:|---:|---:|
| **RTX 5090 (Blackwell) vs RTX 4090 (Ada)**, `standard` | `1,172s` / `1,803s` | `0.00205` / `0.00138` | **`0.00374`** / `0.00329` | `0.00255` | `0.00273` |
| **RTX 5090 (Blackwell) vs RTX 3090 (Ampere)**, `standard` | `1,172s` / `3,313s` | `0.00179` / `0.00162` | `0.00367` / **`0.00340`** | `0.00162` | `0.00205` |
| **RTX 4090 (Ada) vs RTX 3090 (Ampere)**, `standard` | `1,803s` / `3,313s` | `0.00027` / `0.00005` | `0.00008` / `0.00022` | `0.00072` | `0.00070` |
| RTX 5090 vs AMD Radeon 8060S (ROCm 7.14), `nano` / `micro` | `~3.4h` est. `standard` | `0.00155` / `0.00099` | `0.00156` / `0.00122` | `0.00075` / `0.00062` | `0.00080` / `0.00100` |

     **What it decided:** Official Season 3 tolerances are calibrated to `~3.2–4.4×` the maximum
     measured cross-architecture NVIDIA diff (`0.00374` grade / `0.00340` baseline):
     **`grade = 0.012`, `baseline = 0.015`, `run = 0.015`** (down from provisional `0.03 / 0.05 /
     0.05`, catching real drift at `22%–30%` of `fineweb2`'s `+0.041–+0.054` reasoning gain).
     Because AMD ROCm Flash Attention backward is non-deterministic, takes `~3.4h` per standard
     pass, and experienced a 1-in-6 transient `FloatingPointError` divergence under card contention,
     Season 3 restricts official verifying (grading and, since runners must earn 5 agreed official
     grades first, running curators) to NVIDIA CUDA (`grade_backends: ["cuda"]`), while
     retaining AMD ROCm for local practice (`stet run`) and the `$0/round` production examiner.
   - **Per-round compute & cost vs Season 2:**
     At `standard` scale (`65.5M` tokens), one grading pass takes `~19.5 min` on an RTX 5090
     (`~$0.15` at `$0.45/h`), `~30 min` on an RTX 4090 (`~$0.22` at `$0.43/h`), or `~55 min` on an
     RTX 3090 (`~$0.20` at `$0.22/h`), versus `$0.041–$0.046` per training in Season 2. A typical
     10-script Season 3 round runs 12 trainings (`10` scripts + `2` baselines) = **`~$1.80` of
     grading compute** (vs `~$0.55` in Season 2) plus curator runs and boot overhead (**`~$3–$5`
     all-in per round**, ~`2.5–3×` Season 2's `$1.57–$2.16`, while on-demand draws keep idle days
     at `$0/day` and the local AMD examiner costs `$0/round`).

## Still to measure

- Exam-specific language-mix sensitivity with controlled prefiltering: compare reweighting on vs off
  on the **same** filtered pool across multiple independently held-out draws and training seeds.
- A human-written comprehension signal that passes DataDecide and responds to non-English training
  data (the four twin-context kinds stay at `0%` weight beside the score until one does).
- How far two curators' gains differ from round to round on the 100%-reasoning score: what the
  record's margin should be set from.
- Grades of the same scripts across additional NVIDIA GPU generations (e.g. L40S, RTX 3080/4070)
  and whether the calibrated `0.012 / 0.015` tolerances hold in live rounds.
- The unedited corpus's score against training tokens, on this exam: the curve a multiplier
  would be read from.
- Why one order of the same pages scored 0.02 above nine others, and whether an order
  like it can be found on purpose.
- How long a round takes from draw to settlement with `--exam-after 25m`.
