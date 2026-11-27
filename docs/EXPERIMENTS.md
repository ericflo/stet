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
     - **Official Season 3 reasoning (`explain`, `infer`, `synthesis`, English subset):**
       - **Draw 1 (`prod_gate`, `s0`, `80 / 80 / 80` EN items):** **`49/66` (`74.2%`)**, `2pt+ = 26/31` (`83.9%`), Spearman `ρ = +0.636`, `11/11` filtered-vs-unfiltered pairs right, 3% trap `#12/12`. Per-pillar: `explain = 49/66` (`ρ = +0.678`), `infer = 47/66` (`ρ = +0.629`), **`synthesis = 45/66` (`68.2%`, `ρ = +0.517`)**.
       - **Draw 2 (`live_gate`, `s1`, brand-new live official exam, `54 / 54 / 8` EN items):** **`46/66` (`69.7%`)**, `2pt+ = 24/31` (`77.4%`), Spearman `ρ = +0.531`, `11/11` filtered-vs-unfiltered pairs right, 3% trap `#11/12`. Per-pillar: `explain = 52/66` (`ρ = +0.762`), `infer = 47/66` (`ρ = +0.524`), **`synthesis = 34/66` (`51.5%`, `ρ = +0.042`, trap `#10/12`)** — Draw 2 was built before the `synthesis` `PROMPT: ...` placeholder fix and had only **`8` English `synthesis` items** (post-fix Draw B has `47` English `synthesis` items, matching `explain = 48` and `infer = 48`).
       - **2-Draw Independent Mean (`Draw 1 + Draw 2`, English subset):** **`47/66` (`71.2%`)**, `ρ = +0.587` (per-pillar 2-draw means: `explain = 51/66` `ρ = +0.706`, `infer = 48/66` `ρ = +0.580`, `synthesis = 37/66` `ρ = +0.224`).
     - **Historical 50/50 & 0%-weight comprehension reference (English subset):** the pre-stage-4 50/50 mix scored `46/66` (`ρ = +0.566`, perm `p = 0.0426`), `48/66` (`ρ = +0.594`, `p = 0.0225`), and `47/66` (`ρ = +0.580`, `p = 0.0314`), while 0%-weight twin-context comprehension alone scored **`24/66` (`ρ = −0.301`, trap `#1/12`)** on Draw 1, **`34/66` (`ρ = +0.035`, trap `#2/12`)** on Draw 2, and `30/66` (`ρ = −0.112`) 2-draw mean. *(On the full 74-language exam, English-only DataDecide recipes score **`27/66` (`ρ = −0.238`), `37/66` (`ρ = +0.049`), `33/66` (`ρ = −0.049`)** under the final Season 3 scorer, because strict English-only DCLM-QC filters regress against `dolma1_7` on the non-English crawl; English-subset figures are invariant to `"other"`-pool changes.)*
     - **Honest bias diagnosis (not draw noise):** Draw 1 and Draw 2 correlate at **`ρ = +0.94`**
       (`60/66` pairs) with each other, which means the gap from DataDecide's 1B truth table is a
       **systematic lean**, not sampling noise that averages out across rounds. Both draws reliably
       separate filtered from unfiltered recipes (`11/11`) and catch severe overfiltering (`#12/12`
       on Draw 1, `#11/12` on Draw 2), but both place `FineWeb-Edu` (`#7/12` in 1B macro, `1.95 pp`
       below `#1`) at `#1` by ~2× the runner-up (`+0.0356` vs `+0.0163` on Draw 1 and `+0.0414` vs
       `+0.0244` on Draw 2 under official 100% reasoning; `+0.0205` vs `+0.0097` and `+0.0248` vs
       `+0.0153` on historical 50/50) because the
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
     `±0.0010 bpb`).** Every row below is scored under the **final frozen Season 3 scorer**
     (`groups = {"reasoning": 1.0, "comprehension": 0.0}`, `weight_other_by_share = true`,
     `trim = 0.10` applied consistently inside pooled `"other"` via `weighted_trimmed_mean`,
     `code_share = 0.15`) on the archived Phase 9 (`s3-o00001`) item score arrays against both
     the 2-seed `unedited` average and same-seed `unedited_s0`, alongside the historical
     pre-Review #6 50/50 score for reference:

| Curator (all via `stet.script.materialize`) | **Final S3 Official (Reas, vs 2-seed avg)** | **Final S3 (vs same-seed `s0`)** | Final Comp (0% diag, `code=0.15`) | Final 50/50 (ref) | Pre-R6 50/50 (hist) | Notes |
|---|---:|---:|---:|---:|---:|---|
| **`reader` v2 (shipped, 2-seed)** | **`+0.03833`** | **`+0.03834`** | `+0.00238` | `+0.02035` | `+0.01922` | `vllm` `259.1s/1200s` *(unsandboxed host cores)*, `py3langid` per-lang quantile, `80` langs (`s0=+0.03786`, `s1=+0.03904`, diff `0.00118`) |
| **`fineweb2` (2-seed)** | **`+0.03689`** | **`+0.03682`** | **`+0.00383`** | `+0.02036` | `+0.01911` | Positive on all 7 pillars (`s0=+0.03652`, `s1=+0.03728`, diff `0.00075`) |
| `mimic` (`s0`) | **`+0.03110`** | **`+0.03052`** | `+0.00165` | `+0.01638` | `+0.00740` | `~0.0058–0.0072 bpb` below `fineweb2` / `reader` v2 |
| `reader` v1 *(pre-fix diagnosis, 2-seed)* | **`+0.01056`** | **`+0.01055`** | `+0.00154` | `+0.00605` | `−0.00340` | Exposed `BYTES_PER_TOKEN=4.0` top-band truncation + coarse script buckets |
| `adv_qa_spam` (`s0`, Q/A-heavy pages) | **`+0.00197`** | **`+0.00215`** | `+0.00414` | `+0.00306` | `−0.00630` | Small gain near noise (`~18–19×` below `fineweb2`/`reader` v2; `explain −0.0004`, `infer −0.0030`, `synthesis +0.0093` on only 8 EN items; Phase 8 Draw 1: `+0.00195` vs avg / `+0.00338` vs `s0`; pre-pooled-trim Phase 9 was `−0.00034` vs avg / `−0.00015` vs `s0`, Phase 8 `+0.00348` vs `s0`) |
| `unedited` (2-seed baseline) | **`0.00000`** | **`0.00000`** | `0.00000` | `0.00000` | `0.00000` | Baseline (2-seed raw mean diff `0.0004 bpb`) |
| `adv_splice_qa` (`s0`, synthetic splices) | **`−0.06778`** | **`−0.06794`** | `−0.00404` | `−0.03591` | `−0.03480` | Crushed below unedited (Phase 8 Draw 1: `−0.08293`) |
| `english` (`s0`, English-only filter) | **`−0.18500`** | **`−0.18483`** | `+0.00873` | `−0.08813` | `−0.08510` | Monolingual collapse penalized (Phase 8 Draw 1: `−0.16282`) |
| `adv_overfilter` (`s0`, top-3% keep) | **`−0.98669`** | **`−0.98669`** | `−0.02594` | `−0.50631` | `−0.54270` | Obliterated (Phase 8 Draw 1: `−1.01696`) |

**What it decided.** Season 3 ships `baidu/ERNIE-4.5-0.3B-Base-PT` (`freeze_bottom: true`,
`65.5M` tokens, 1-epoch commercial-permissive multilingual SFT probe) with the 100%-fresh exam
(100% fresh reasoning scored after stage 4 below, 4 human-written comprehension kinds at `0%`
diagnostic; `rules_version: 7`, `weight_other_by_share: true` with symmetric `10%` weight-tail
trimming inside pooled `"other"`, `code_share: 0.15`, `min_pillar_items: 20` with sparse-pillar
omission and `STET_EXAM_TIMEOUT = 6h` + `POST /api/v1/admin/rounds/{id}/cancel`,
`baselines.per_round = 2`), a **12 GB** volunteer grading floor, **24 GB+** neural curator
sandboxes, and on-demand rounds (`$0/day` idle).
Running unmodified `stet.pipeline.edit_round` + `stet.script.materialize` caught and fixed two real
bugs in `examples/reader/curator.py` (`BYTES_PER_TOKEN=4.0` truncating the top 3,634 Band-3 docs
and coarse 10-script buckets starving `fa`, `pl`, `uk`), jumping `reader` from `+0.01056` to
**`+0.03833 bpb`** on the final Season 3 official reasoning score (`+0.00605 → +0.02035` on 50/50,
`−0.00340 → +0.01922` under the historical scorer;
`fa −0.150 → +0.0168`, `pl −0.081 → +0.0410`, `uk −0.030 → +0.0398`, `zh −0.018 → +0.0002`,
`ja −0.011 → −0.0014`, `ko −0.029 → −0.0184`) in `259.1s / 1,200s` on `vllm==0.34.0`
*(unsandboxed host cores; standalone 6-core `taskset -c 0-5` benchmark on `s3-p00000`'s `507,387` docs
completed prefiltering + `py3langid` in **`37.49s`** -> `75,678` candidates, leaving `>1,160s` for
GPU scoring)*.

3. **AMD Radeon 8060S examiner verification (prior `Qwen3-8B` baseline) & language-mix adversary gate (`amd_examiner`, `lang_adv_gate`).**
   After fixing the `synthesis` `PROMPT: ...` placeholder, adding crawl-share weighting inside the
   pooled `"other"` small-language cell (`weight_other_by_share: true` on `s3`, preserving `false`
   for `s2`), collapsing Wikipedia variants into one source bucket, lowering CJK `min_len` to 180
   chars, adding wrong-language / mid-sentence / `\ufffd` filters, enforcing `min_keep_rate = 0.5`,
   and publishing teacher `prompt` + `cont` + `wrapper_id` + `excerpt_sha256` for every withheld
   reasoning item:
   - **Two cold official draws on AMD Radeon 8060S (`124 GiB`, `PyTorch 2.15.0+rocm7.14`, `transformers 5.23.0`, prior `Qwen3-8B` `$0/round` baseline):**
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
       under the final Season 3 reasoning scorer (`weight_other_by_share=true` + pooled 10% trim),
       it scores **`+0.00854`** vs `fineweb2` **`+0.03989`** (`4.7×` below; on 50/50 it fell from
       `+0.01256` unweighted to `+0.00361`, `6×` below `fineweb2`'s `+0.02172`); on Draw B
       (`s3-o00003`), it scores **`+0.01644`** vs `fineweb2` **`+0.05423`** (`3.3×` below; 50/50
       `+0.00793` vs `+0.02838`), confirming the small-language `"other"`-pool exploit is closed.
     - **`adv_exam_lang_mix` (clean deduplicated pool resampled to Draw B's exam item counts):** scored
       **`+0.06046`** vs `fineweb2` `+0.05423` (`+0.00623`) on matched Draw B and **`+0.04632`** vs
       `+0.03989` (`+0.00643`) on unmatched Phase 9 (`s3-o00001`) on the final official reasoning
       score (50/50: `+0.03218` vs `+0.02838` on Draw B, `+0.02476` vs `+0.02172` on Phase 9). Because
       `adv_exam_lang_mix` and `fineweb2` also differed in prefiltering and deduplication (using
       `reader`'s CJK-aware prefilter restored `ja` from `1.99% → 3.59%`, lifting `ja` by
       `+0.0259 bpb`, alongside mild temperature upsampling of `1%–5%` major languages over `<0.5%`
       micro-languages), this single-seed comparison does **not** establish a formal upper bound on
       exam-specific language-mix exploitation; estimating a bound requires toggling reweighting on/off
       over the same filtered pool across multiple independently held-out draws and training seeds.

4. **Pre-launch exam decomposition, perplexity-pruning adversary gate, cross-card hardware parity, and round cost (`r8_gate`).**
   Before opening Season 3, four empirical gates audited the shipped exam and hardware fleet:
   - **Half-by-half decomposition (why Season 3 scores `reasoning = 1.0`, `comprehension = 0.0` diagnostic):**
     Re-scoring Draw 1 and Draw 2 under the final frozen Season 3 scorer (`code_share = 0.15`,
     pooled 10% tail trim) with each half alone proved that the human-written twin-context half
     carried 50% of the declared weight but supplied only `~8%–10%` of `fineweb2`'s actual gain
     (`+0.00368` / `+0.00383` vs `+0.04147` / `+0.03689`), scored at or below chance on DataDecide
     (`24/66` on Draw 1, `34/66` on Draw 2, putting the 11th-of-12 over-filtered recipe first/second),
     could not separate `adv_qa_spam` (`+0.00292` / `+0.00414`) from `fineweb2`/`reader`, had
     cross-card noise (`5090 vs 4090 = 0.00201 bpb` on Draw B) comparable to `fineweb2`'s entire
     gain (`+0.00252` on Draw B), and scored `english` (zero non-English training data) higher
     than `fineweb2` (`+0.01179` / `+0.00873` vs `+0.00368` / `+0.00383`) because subtracting
     wrong openings cancels fluency while the frozen lower half of `ERNIE-4.5-0.3B` already reads
     every language.
     By contrast, the **model-written reasoning half alone** (`explain`, `infer`, `synthesis`):
     - scores **`49/66` (Draw 1) and `46/66` (Draw 2), `47/66` 2-draw combined** on DataDecide
       (per-pillar: `explain 49/66 & 52/66`, `infer 47/66 & 47/66`, `synthesis 45/66` on Draw 1
       with 80 EN items vs `34/66` on Draw 2 with only 8 pre-fix EN items);
     - increases `fineweb2` signal-to-noise from `14–17×` to **`24–55×`** (`+0.0369` Phase 9 /
       `+0.0542` Draw B vs 2-seed diff `0.0008`);
     - keeps `adv_qa_spam` near unedited (`+0.00195` Draw 1 / `+0.00197` Draw 2 vs 2-seed avg;
       `+0.00338` / `+0.00215` vs same-seed `s0` — `~18–21×` below `fineweb2`/`reader` v2;
       pre-pooled-trim Phase 9 was `−0.00034` vs avg / `−0.00015` vs `s0`);
     - strongly penalizes missing languages (`english = −0.1628 / −0.1850`).
     Season 3 therefore sets `groups = {"reasoning": 1.0, "comprehension": 0.0}` (with `code_share = 0.15`
     and per-source caps inside `select_context` so LKML never dominates even diagnostic `technical`),
     showing the four human-written kinds on the round and curator pages beside the score as
     0%-weight diagnostic telemetry until a human-written measure clears DataDecide.
   - **Perplexity-pruning / model-predictable prose adversaries (`adv_ppl_qwen06b` & `adv_ppl_ernie03b`,
     2× RTX 5090, Draw B `s3-o00003`, 65.5M tokens; scored under final frozen `code_share = 0.15` + pooled-trim scorer):**
     We built two pure lowest-perplexity curators on `reader._prefilter`'s `75,565` deduplicated
     multilingual docs with **zero quality rubric**, matching `fineweb2`'s exact per-language byte
     mix (`~21.9k` unique docs = bottom `~29%` NLL/byte, `passes = 1.0`): one scored by
     `Qwen/Qwen3-0.6B` (same family as teacher `Qwen3-8B`) and a cross-family control scored by
     `baidu/ERNIE-4.5-0.3B-Base-PT` (independent Baidu weights).

| Curator (all Draw B crawl, 65.5M tok, vs Draw B `unedited`) | Draw B Reasoning (official S3) | Draw B Comp (0% diag, `code=0.15`) | Draw B 50/50 | Phase 9 Reasoning (official S3) | Phase 9 Comp (0% diag, `code=0.15`) |
|---|---:|---:|---:|---:|---:|
| **`adv_ppl_qwen06b`** (lowest `Qwen3-0.6B` NLL/byte) | **`+0.07457`** (+37.5%) | **`+0.00761`** (3.02×) | `+0.04109` | **`+0.05619`** (+40.8%) | **`+0.00668`** (1.89×) |
| **`adv_ppl_ernie03b`** (lowest `ERNIE-4.5-0.3B` NLL/byte) | **`+0.07388`** (+36.2%) | **`+0.00727`** (2.89×) | `+0.04057` | **`+0.05507`** (+38.1%) | **`+0.00606`** (1.71×) |
| `adv_exam_lang_mix` (`reader._prefilter` random) | `+0.06046` | `+0.00390` | `+0.03218` | `+0.04632` | `+0.00319` |
| `fineweb2` (heuristic-only, no neural filter) | `+0.05423` | `+0.00252` | `+0.02838` | `+0.03989` | `+0.00354` |
| `adv_tiny_lang_boost` | `+0.01644` | `−0.00057` | `+0.00793` | `+0.00854` | `−0.00131` |

     **Honest diagnosis (both sides of this result):**
     1. *Not Qwen3 teacher-family overfitting:* `ERNIE-4.5-0.3B` overlapped `Qwen3-0.6B` on `79.3%`
        of selected documents, beat `Qwen3-0.6B` on Draw B `explain` (`+0.07042` vs `+0.06839`),
        human-written `technical` (`+0.01017` vs `+0.00746`), and Phase 9 English reasoning
        (`+0.06534` vs `+0.06455`), and **both** beat `fineweb2` by `1.71×–3.02×` on 100%
        human-written twin-context comprehension under the final `code_share = 0.15` pooled-trim
        scorer (where unconditional style cancels identically).
        Keeping the lowest-PPL `~29%` of deduplicated Gopher-filtered Common Crawl is literally
        **CCNet-head / perplexity pruning** (Wenzek et al. 2019; Marion et al. 2023; Ankner et al.
        2024), which drops garbled web scrapes and increases `.edu/.gov/.ac/.org` share
        (`8.2% → 9.0–9.4%`, `pmc.ncbi.nlm.nih.gov` `8 → 18`, `med.stanford.edu` `5 → 14`).
     2. *Real Season 3 caveat:* uncapped lowest-PPL with zero quality judgment **also** increases
        privacy/TOS/copyright boilerplate (`3.0% → 4.6%`) and AI-SEO phrase density (`1.3% → 2.5%`),
        lowers lexical diversity (`TTR 0.713 → 0.676`), and puts PHP deprecation logs and AI
        listicles in its extreme bottom tail — yet beats both `fineweb2` (`+36–41%`) and `reader`
        v2 (`+0.0383`, whose zero-shot 0.6B rubric ratings collapsed to `[2.57, 2.81]`). On Phase 9
        English reasoning, `FineWeb-Edu` (`+0.10562`) and `DCLM-qc-10p` (`+0.08705`) still lead
        generic English Common Crawl (`ml_english` `+0.08008`), showing strong quality classifiers
        retain headroom above generic low-PPL filtering. Entrants should treat CCNet-style
        perplexity pruning as an established, strong neural baseline on NLL exams.
   - **Rotated-excerpt & prompt-only control (`r9_excerpt`, 4× RTX 5090, official CUDA `standard` 65.5M tokens):**
     To test whether between-curator gains on the 100%-reasoning score reflect excerpt-grounded
     reading comprehension or primarily prompt-conditioned modeling of the teacher's answer prose,
     we trained four Draw B `standard` students in parallel (`unedited`, `fineweb2`,
     `adv_ppl_qwen06b`, and Draw B `mimic`; bit-identical `init_sha256 = 5bd1d5ce4547697e...`,
     reproducing archived `fineweb2` `+0.05423 / +0.03989` and `adv_ppl_qwen06b`
     `+0.07457 / +0.05619` bit-for-bit) and evaluated every student on both Draw B (`s3-o00003`,
     `573/611` reasoning items rotated across `48` `(lang, task)` groups and `17` languages) and
     Phase 9 (`s3-o00001`, `399/472` rotated across `68` groups and `32` languages) under three
     conditions: **original** (`orig`), **rotated excerpt** within `(lang, task)` sorted by
     `Item.id` (`rot`, keeping prompt, wrapper, and target answer identical), and **prompt-only**
     with empty excerpt (`empty`):

| Curator pair (65.5M CUDA `standard`, final S3 scorer, all reasoning items) | Draw B `orig` | Draw B `rot` (`%` of `orig`) | Draw B `empty` (`%`) | Draw B rot-only `orig − rot` | Phase 9 `orig` | Phase 9 `rot` (`%` of `orig`) | Phase 9 `empty` (`%`) | Phase 9 rot-only `orig − rot` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`fineweb2 − unedited`** | `+0.05423` | `+0.05593` (`103.1%`) | `+0.05685` (`104.8%`) | `−0.00113` (`−2.1%`) | `+0.03989` | `+0.04349` (`109.0%`) | `+0.04462` (`111.8%`) | `−0.00316` (`−7.7%`) |
| **`adv_ppl_qwen06b − unedited`** | `+0.07457` | `+0.07523` (`100.9%`) | `+0.07907` (`106.0%`) | `+0.00057` (`+0.8%`) | `+0.05619` | `+0.05756` (`102.4%`) | `+0.05964` (`106.1%`) | `−0.00085` (`−1.5%`) |
| **`mimic − unedited`** | `+0.02691` | `+0.02893` (`107.5%`) | `+0.02904` (`107.9%`) | `−0.00203` (`−6.7%`) | `+0.02906` | `+0.02930` (`100.8%`) | `+0.03025` (`104.1%`) | `−0.00096` (`−3.0%`) |
| **`adv_ppl_qwen06b − fineweb2`** | `+0.02086` | `+0.01945` (`93.3%`) | `+0.02229` (`106.9%`) | `+0.00176` (`+8.4%`) | `+0.01575` | `+0.01263` (`80.2%`) | `+0.01368` (`86.9%`) | `+0.00302` (`+17.9%`) |
| **`fineweb2 − mimic`** | `+0.02862` | `+0.02865` (`100.1%`) | `+0.02916` (`101.9%`) | `+0.00072` (`+3.0%`) | `+0.01335` | `+0.01635` (`122.4%`) | `+0.01569` (`117.5%`) | `−0.00248` (`−22.9%`) |

     **Candid scientific readout:** Every individual student strongly uses the source excerpt in
     absolute terms — rotating the excerpt worsens mean reasoning score by `0.136–0.139 bpb` on
     Draw B (`unedited: −1.1276 → −1.2671`, `fineweb2: −1.0682 → −1.2047`, `adv_ppl_qwen06b:
     −1.0431 → −1.1812`, `mimic: −1.1077 → −1.2451`) and `0.140–0.144 bpb` on Phase 9, while
     deleting the excerpt (`empty`) worsens mean score by `0.128–0.140 bpb`. **However, virtually
     all (`100.8%–109.0%` vs `unedited`, `80.2%–93.3%` for `adv_ppl_qwen06b − fineweb2`, and
     `100.1%–122.4%` for `fineweb2 − mimic`) of the between-curator gain survives both excerpt
     rotation and complete excerpt removal** (on the rotated-only subset, excerpt-conditioned
     `orig − rot` differences are near zero: `−0.003` to `+0.003 bpb`). Because the frozen lower
     9 layers of `ERNIE-4.5-0.3B-Base-PT` already supply roughly equal in-context excerpt copying
     across curators, Season 3's official score at `65.5M` tokens empirically functions primarily
     as a **fresh multilingual prompt-conditioned answer-prose likelihood metric** rather than
     excerpt-grounded deduction. That mechanism directly explains both its strengths (strong
     DataDecide 1B agreement `49/66 & 46/66`, `11/11` filtered-vs-unfiltered separation, `fineweb2`
     beating `mimic` by `2×` on Draw B, strong penalty on monolingual `english` and synthetic
     `adv_splice_qa`) and its known caveats (educational-prose lean toward `FineWeb-Edu` and strong
     CCNet-head low-PPL scores) — motivating the head-to-head teacher benchmark below.
   - **Teacher model head-to-head (`Qwen/Qwen3-8B` vs `Qwen/Qwen3.5-9B` vs `thinkingmachines/Inkling-2-Mini`,
     4× RTX 5090, official CUDA `standard` 65.5M tokens, exact Draw B 640 docs & `decontam.seen()` parity):**
     To test whether a newer Apache-2.0 teacher fixes `Qwen3-8B`'s `FineWeb-Edu` textbook lean and
     `adv_ppl_qwen06b` perplexity-pruning edge, we synthesized Draw B (`s3-o00003`) reasoning exams
     from the **exact same 640 Draw B source documents across 43 languages** with `Qwen/Qwen3.5-9B`
     (`c2022362`, `enable_thinking=False`) and `thinkingmachines/Inkling-2-Mini` (`c9d02830`, both
     `reasoning_effort="none"` and `reasoning_effort="low"`), applied exact `stet.exam.decontam.seen()`
     against the Draw B corpus, and evaluated **16 official CUDA `standard` (`65.5M`-token)
     `ERNIE-4.5-0.3B` students** (4 multilingual/adversarial streams + all 12 DataDecide 1B recipes,
     all bit-identical `init_sha256 = 5bd1d5ce4547697e...`) on every exam under both original (`orig`)
     and within-`(lang, task)` rotated excerpts (`rot`). Two experimental notes for transparency:
     (a) `Qwen/Qwen3.5-9B` with `enable_thinking=True` has no `reasoning_effort` knob and burned
     `>2,048` CoT tokens per 2-sentence item even under anti-loop sampling (`temp=0.6, pp=1.5`),
     producing an empty exam (`0/640` kept, not compared in the table); (b) `Inkling-2-Mini` with
     `reasoning_effort="none"` kept only `3/640` under greedy `t=0, max_tokens=512` and was rerun
     non-greedily (`t=0.6, top_p=0.95, pp=1.0, max_tokens=1536`) to yield `621/640`, whereas shipped
     `Inkling-2-Mini` (`effort="low"`, greedy `t=0.0, max_tokens=1536`) kept `626/640` cleanly.

| Teacher (exact same 640 Draw B docs) | Synth Wall Time | Kept / 640 (Post-Decontam) | Langs | DD 1B Pairs (`orig` / `rot`, paired boot 95% CI) | DD 1B `ρ` (`orig`) | `FineWeb-Edu` Rank *(true `#7`)* / `P(#1)` | Predicted Top-1 *(1B Shortfall)* |
|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3-8B` (prior S3) | `1213s` *(AMD 8060S)* | `617` (`96.4%` → `611`) | `39` | `54/66` / `51/66` (`[47, 57]`) | `+0.7762` | `#1/12` / `100%` | `fineweb-edu` (`+1.95pp`) |
| `Qwen/Qwen3.5-9B` (`nothink`) | `58.6s` *(1× 5090)* | `612` (`95.6%` → `606`) | `42` | `58/66` / `51/66` (`[49, 59]`) | `+0.8392` | `#1/12` / `99%` | `fineweb-edu` (`+1.95pp`) |
| `Inkling-2-Mini` (`effort="none"`, `t=0.6` rerun) | `246.6s` *(2× 5090)* | `621` (`97.0%` → `614`) | `40` | `55/66` / `49/66` (`[49, 58]`) | `+0.8042` | `#7/12` / `0%` | `dclm-baseline-qc-7p-fw2` (`+0.14pp`) |
| **`Inkling-2-Mini` (`effort="low"`, `t=0.0`)** | `562.2s` *(1× 5090 warm)* | **`626` (`97.8%` → `617`)** | `41` | **`58/66`** / `52/66` (`[50, 61]`) | **`+0.8601`** | **`#8/12` / `0%`** | `dclm-baseline-qc-20p` (`0.00pp`) |

| Teacher (43-lang Draw B ML + adversarial + rotated control) | `fineweb2` | `adv_ppl_qwen06b` (`%` of `fw2`) | `mimic` (`%` of `fw2`) | `adv − fw2` (`orig` → `rot`) | `fw2 − mimic` | CJK `fw2` raw mean (`n=50`) | Abs Drop on Rot (`u / fw2`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-8B` | `+0.05424` | `+0.07458` (`137.5%`) | `+0.02692` (`49.6%`) | `+0.02034 → +0.01930` | `+0.02732` | `+0.018` | `+0.139 / +0.136` |
| `Qwen/Qwen3.5-9B` (`nothink`) | `+0.05332` | `+0.06525` (`122.4%`) | `+0.03155` (`59.2%`) | `+0.01193 → +0.00695` | `+0.02177` | — | `+0.148 / +0.141` |
| `Inkling-2-Mini` (`effort="none"`) | `+0.04955` | `+0.05248` (`105.9%`) | `+0.02743` (`55.4%`) | `+0.00293 → −0.00037` | `+0.02212` | — | `+0.139 / +0.134` |
| **`Inkling-2-Mini` (`effort="low"`)** | `+0.04710` | **`+0.04820` (`102.3%`)** | `+0.02922` (`62.0%`) | **`+0.00110 → −0.00143`** | `+0.01788` | `+0.005` | `+0.211 / +0.205` |

     **Honest statistical readout (paired stratified bootstrap over the 143 common EN docs and
     587 common multilingual docs):**
     1. *What IS statistically established:* (a) **Elimination of `FineWeb-Edu`'s `#1` textbook
        lean** — `Qwen3-8B` ranks `FineWeb-Edu` `#1/12` in `100%` of bootstrap resamples and
        `Qwen3.5-9B` in `99%`, while **`Inkling-2-Mini` ranks it `#1/12` in `0%`** (median `#7`,
        matching its true `#7` 1B OLMES rank); (b) **`adv_ppl_qwen06b − fineweb2` shrinks by
        `0.019 bpb`** (95% CI `[−0.022, −0.015]`), moving `adv_ppl_qwen06b` from a `+37.5%`
        runaway lead to statistically tied with `fineweb2` (`+0.0019`, 95% CI `[−0.0011, +0.0047]`).
     2. *What is NOT statistically proven by a single draw:* single-draw pairwise DataDecide
        accuracy differences are within item-sampling noise (`Inkling-low` vs `Qwen3-8B`: `+3.7`
        pairs, 95% CI `[−3, +10]`, `18%` not ahead; vs `Qwen3.5-9B nothink`: `+1.2`, `[−5, +7]`,
        `41%` not ahead; vs `Inkling-none`: `+2.2`, `[−4, +8]`, `29%` not ahead). Likewise,
        picking exact `#1` (`dclm-baseline-qc-20p`) on one Draw B slice is not stable across
        resamples or draws (see the 3-draw validation below).
     3. *Real trade-offs stated candidly:* on Draw B common docs, `Inkling-2-Mini` lowers raw
        `fineweb2` gain by `0.0069 bpb` (`[−0.0105, −0.0032]`), narrows `fineweb2 − mimic` by
        `0.0090` (`[−0.0133, −0.0051]`, raising `mimic` from `50%` to `62%` of `fineweb2` on
        Draw B), and compresses `fineweb2` CJK raw mean gain (`+0.018 → +0.005`, 95% CI diff
        `[−0.026, −0.001]` across `n=50` CJK items). And while rotating excerpts raises absolute
        mean NLL by `~0.19–0.21 bpb` (worse likelihood; vs `~0.14 bpb` on `Qwen3-8B`),
        **`fineweb2 − unedited` still retains `110.1%` under rotation**, so between-curator
        rankings on multilingual crawls still do not rely on the source excerpt.

   - **Full Safety-Case Re-Validation on Shipped `Inkling-2-Mini` Exams (`r10_gate`, 3 Disjoint
     Harvest Draws, 34 Official CUDA `standard` Students, Saved `bf16` Weights, Cold-Start 5090
     Examiner & Cross-Card RTX 4090 Parity):**
     Because the initial head-to-head only scored 4 multilingual students on 1 draw, we ran the
     **complete Season 3 safety case directly on the shipped `Inkling-2-Mini` code** across
     three completely disjoint harvest windows (`draw_b`: Nov 25 late, `618` clean / `42` langs /
     `143` EN; `phase9`: Nov 25 mid, `615` clean / `39` langs / `128` EN; `draw1`: Nov 24, `621`
     clean / `38` langs / `149` EN, after applying the shipped `prompt_copies_excerpt` filter
     which removes `0` / `6` / `3` = `9` prompt-copying rows from the archived draws), synthesized
     live by the shipped `stet.exam.fresh.reason.synthesize_reasoning()` client against the
     immutable pinned vLLM container (`vllm/vllm-openai@sha256:d24e5fc73f0b859bd76c9d2f6f6fa4da3001afd7637f2610876420f3027ea6c7`,
     `1,874/1,920 = 97.6%` kept raw under the shipped filter, zero CoT/control-token leaks,
     cold-start Draw B rent-to-synth **`856.3s` (`14.27 min`), `$0.147` GPU cost** at `$0.619/h`:
     `125.6s` container start + `141.9s` model load & `/v1/models` check + `588.7s` cold synthesis;
     warm synthesis `367–379s`). All 34 trained students (all 12 Phase 9 gauntlet students, all 8
     Draw B controls/adversaries, all 12 DataDecide 1B recipes on brand-new shortest-row 32-lane
     dealt slices, and 2 RTX 4090 cross-card parity students) sat all 3 Inkling draws (`orig` and
     `rot`), all 34 trainable upper-half `bf16` checkpoints (`244 MB` each, `8.1 GB` total) were
     archived, and the tables below rescore the archived student outputs on the exact `1,854`-item
     post-filter population.

     - **Multi-draw DataDecide 1B (3 disjoint Inkling draws under shipped filter; English subset vs. full multilingual-scored diagnostic, 2,000-resample stratified bootstrap):**

| Shipped Inkling Draw / Scope | Items | DD 1B Pairs (`orig` / `rot`, 95% CI) | `2pt+` Pairs | Spearman `ρ` (95% CI) | `FineWeb-Edu` Rank *(true `#7`)* / `P(#1)` | Predicted Top-1 *(1B Shortfall)* |
|---|---:|---:|---:|---:|---:|---|
| Draw B (`s3-o00003`, EN subset) | `143` | `52/66` (`78.8%`) / `48/66` (`[48, 56]`) | `26/31` | `+0.7902` (`[+0.622, +0.867]`) | `#7/12` / `0.0%` | `dclm-baseline-qc-10p` (`+0.49pp`) |
| Phase 9 (`s3-o00001`, EN subset) | `128` | `53/66` (`80.3%`) / `52/66` (`[49, 57]`) | `27/31` | `+0.8042` (`[+0.636, +0.895]`) | `#5/12` / `0.4%` | `dclm-baseline-qc-7p-fw2` (`+0.14pp`) |
| Draw 1 (`s3-p00000`, EN subset) | `149` | `52/66` (`78.8%`) / `50/66` (`[49, 55]`) | `27/31` | `+0.7972` (`[+0.636, +0.825]`) | `#6/12` / `0.0%` | `dclm-baseline-qc-10p` (`+0.49pp`) |
| **Pooled 3-Draw Mean (EN subset)** | **`420`** | **`53/66` (`80.3%`) / `49/66` (`[49, 55]`)** | **`27/31`** | **`+0.8042` (`[+0.699, +0.832]`)** | **`#6/12` / `0.0%`** | **`dclm-baseline-qc-7p-fw2` (`+0.14pp`)** |
| **All scored reasoning (official lang weights, EN target diagnostic)** | **`1,854`** | **`25/66` (`25/66`, `26/66`, `24/66`)** | **`9/31`** | **`−0.294`** | — | `dolma1_7` `#1` *(true `#12` on EN)* |

       *Scope qualification:* the **`53/66` (`80.3%`)** DataDecide 1B agreement is measured specifically on the **English reasoning subset** (`420` items across the three draws) because DataDecide's 1B OLMES macro target is English-only. Scoring the same 12 recipes with the full official multilingual leaderboard scorer (`1,854` items under each draw's recorded crawl language weights) ranks `dolma1_7` first on every draw (`25/66`, `26/66`, `24/66`; pooled `25/66`, `9/31` `2pt+`, `ρ = −0.294` against English OLMES) because `dolma1_7` contains non-English text while most higher-ranked DataDecide recipes filter strictly to English. Validating downstream ordering of the full multilingual leaderboard requires a multilingual downstream benchmark target.

     - **Complete Phase 9 (`s3-o00001`) & Draw B (`s3-o00003`) Gauntlets on Shipped Inkling-2-Mini (3-draw mean + 1,000-resample `(part, lang)` stratified bootstrap CIs):**

| Curator (65.5M CUDA `standard`, shipped Inkling-2-Mini) | Draw B | Phase 9 | Draw 1 | **3-Draw Mean (`orig` / `rot`)** | EN / CJK / Other raw (`n=420/160/1274`) |
|---|---:|---:|---:|---:|---|
| **Phase 9 (`s3-o00001`) corpus:** `p9_reader_v2` (`s0` / `s1`) | `+0.04315` / `+0.04234` | `+0.04760` / `+0.04660` | `+0.04372` / `+0.04329` | **`+0.04482`** (`s1 +0.04408`) / `+0.05100` | `+0.0506 / −0.0103 / +0.0567` |
| `p9_fineweb2` (`s0` / `s1`) | `+0.04063` / `+0.04034` | `+0.04366` / `+0.04472` | `+0.03871` / `+0.04120` | **`+0.04100`** (`s1 +0.04209`) / `+0.04708` | `+0.0455 / −0.0069 / +0.0534` |
| `p9_mimic` | `+0.02359` | `+0.02249` | `+0.02383` | **`+0.02330`** (`56.8%` of `fw2`) / `+0.02652` | `+0.0471 / +0.0108 / +0.0043` |
| `p9_reader_v1` (un-normalized) | `+0.00381` | `+0.00256` | `+0.00339` | `+0.00325` / `+0.00546` | `+0.0280 / −0.0434 / −0.0212` |
| `p9_adv_qa_spam` | `+0.00294` | `−0.00471` | `+0.00046` | **`−0.00044`** / `−0.00571` | `+0.0154 / +0.0365 / −0.0333` |
| `p9_adv_splice_qa` | `−0.09313` | `−0.09280` | `−0.09565` | **`−0.09386`** / `−0.05750` | — |
| `p9_english` (monolingual) | `−0.18501` | `−0.17615` | `−0.18755` | **`−0.18290`** / `−0.20322` | — |
| `p9_adv_overfilter` | `−1.07806` | `−1.11557` | `−1.08021` | **`−1.09128`** / `−1.06998` | — |
| **Draw B (`s3-o00003`) corpus:** `db_adv_exam_lang_mix` | `+0.05140` | `+0.05649` | `+0.05250` | `+0.05346` / `+0.06065` | `+0.0543 / +0.0314 / +0.0567` |
| `db_adv_ppl_qwen06b` | `+0.04797` | `+0.05248` | `+0.04915` | **`+0.04987`** (`103.8%` of `fw2`) / `+0.05150` | `+0.0584 / −0.0042 / +0.0621` |
| `db_fineweb2` | `+0.04644` | `+0.05077` | `+0.04687` | **`+0.04803`** / `+0.05273` | `+0.0562 / +0.0058 / +0.0574` |
| `db_adv_ppl_ernie03b` (cross-family) | `+0.04531` | `+0.04740` | `+0.04424` | **`+0.04565`** (`95.0%` of `fw2`) / `+0.04774` | `+0.0526 / −0.0206 / +0.0595` |
| `db_mimic` | `+0.02718` | `+0.02331` | `+0.02680` | **`+0.02576`** (`53.6%` of `fw2`) / `+0.03006` | `+0.0502 / +0.0221 / +0.0045` |
| `db_adv_tiny_lang_boost` | `+0.01089` | `+0.01330` | `+0.01046` | **`+0.01155`** (`4.2×` below `fw2`) / `+0.01741` | `+0.0117 / −0.0021 / +0.0232` |

     - **Key 3-draw stratified bootstrap margins (pooled across all 3 Inkling draws, 1,000 `(part, lang)` resamples; 95% CIs reflect item-resampling uncertainty conditional on the seed-0 students and training corpus):**
       `p9_reader_v2 − p9_fineweb2 = +0.00382` (`s0`, 95% item-resampling CI `[+0.00251, +0.00499]`,
       `P(<=0)=0.000`, conditional on the seed-0 students and Phase 9 corpus; `s1` paired 3-draw
       mean margin **`+0.00199`**, `+0.04408 − +0.04209`, `[+0.00095, +0.00312]`, winning all 3
       draws and sitting below the seed-0 item-resampling lower endpoint of `+0.00251`);
       `p9_fineweb2 − p9_mimic = +0.01770` (`[+0.01618, +0.01915]`, `P(<=0)=0.000`);
       `p9_fineweb2 − p9_adv_qa_spam = +0.04144` (`[+0.03974, +0.04293]`, `P(<=0)=0.000`);
       `db_fineweb2 − db_mimic = +0.02226` (`[+0.02080, +0.02357]`, `P(<=0)=0.000`);
       `db_adv_ppl_qwen06b − db_fineweb2 = +0.00184` (`[+0.00036, +0.00330]`, `P(<=0)=0.008`);
       `db_adv_ppl_ernie03b − db_fineweb2 = −0.00238` (`[−0.00428, −0.00076]`, `P(<=0)=0.995`,
       meaning cross-family ERNIE-0.3B lowest-PPL loses to `fineweb2` under `Inkling-2-Mini`!);
       `db_fineweb2 − db_adv_tiny_lang_boost = +0.03647` (`[+0.03515, +0.03753]`, `P(<=0)=0.000`).
       On `db_adv_exam_lang_mix` (`+0.05346` vs `db_fineweb2` `+0.04803`, margin `+0.00544`, 95%
       item-resampling CI `[+0.00432, +0.00667]` spanning the provisional `0.005` record margin):
       substituting `fineweb2`'s CJK item gains drops the lead to `+0.00125`, locating most of the
       difference in CJK. Because the two training streams differed in both prefiltering and
       language weighting, this experiment cannot isolate which change caused the gap; an
       apples-to-apples comparison of crawl-share vs exam-share weighting on the identical filtered
       pool and training budget across held-out draws and seeds is the right follow-up test.
       Multi-seed paired gain deltas (`s1 − s0`): `p9_reader_v2` `−0.00081 / −0.00100 / −0.00043`,
       `p9_fineweb2` `−0.00029 / +0.00106 / +0.00250`, `db_unedited` `+0.00009 / −0.00011 / −0.00012`.
       Between-curator rotation survival across all 3 Inkling draws remains `107.8%–118.0%` on
       `fineweb2` even though rotating excerpts raises absolute mean NLL by `+0.180–+0.190 bpb`
       (e.g. Draw B unedited seed-0 mean NLL rose from `1.337` to `1.522 bpb`), confirming that
       between-curator rankings still rely on prompt-conditioned answer-prose modeling rather than
       the excerpt.
     - **Direct Cross-Card NVIDIA RTX 5090 vs RTX 4090 Parity on Shipped `Inkling-2-Mini` (~525-byte answers):**
       Training `db_unedited_s0` and `db_fineweb2_s0` on RTX 5090 vs RTX 4090 produced bit-identical
       `tokens_sha256` and `init_sha256` (`5bd1d5ce4547697e...`). Across all 3 shipped Inkling draws,
       aggregate cross-card baseline diffs were **`+0.000463` (`draw_b`), `+0.001068` (`phase9`), and
       `+0.001732` (`draw1`)** (`8.7×–32×` inside the `0.015` baseline tolerance), and `fineweb2` gain
       diffs were **`+0.001127 / +0.000797 / −0.000364` own-base** and **`+0.002008 / +0.001724 /
       +0.001173` shared-base** (`6.0×–10.2×` inside the `0.012` grade tolerance), directly confirming
       that calibrated `0.012 / 0.015` tolerances hold on `Inkling-2-Mini` exams.

   - **Prior 3-Architecture NVIDIA Cross-Card Parity (`Qwen3-8B` exams, `r8_gate`: `5090` vs `4090` vs `3090` at `standard` 65.5M tokens; `5090` vs `AMD 8060S` at `nano`/`micro`; final pooled-trim `code_share = 0.15` scorer):**
     Training the exact archived `fineweb2.jsonl` from the Draw B seed produced **bit-identical
     `init_sha256` (`5bd1d5ce4547697e...`) across all four GPUs** and **bit-identical
     `tokens_sha256` at each matched scale** (`232c6bee572bc914...` across RTX 5090 / 4090 / 3090
     at `standard`, `96cb181c1abcd108...` across RTX 5090 vs AMD 8060S at `nano`, and
     `0e548cd8bb6780a6...` across RTX 5090 vs AMD 8060S at `micro`):

| Hardware pair & scale | Runtime (`standard`) | Draw B Reasoning `|da−db|` / `|direct|` | Phase 9 Reasoning `|da−db|` / `|direct|` | Draw B 50/50 `|direct|` | Phase 9 50/50 `|direct|` |
|---|---:|---:|---:|---:|---:|
| **RTX 5090 (Blackwell) vs RTX 4090 (Ada)**, `standard` | `1,172s` / `1,803s` | `0.00169` / `0.00131` | `0.00397` / `0.00338` | `0.00166` | `0.00192` |
| **RTX 5090 (Blackwell) vs RTX 3090 (Ampere)**, `standard` | `1,172s` / `3,313s` | `0.00133` / `0.00161` | **`0.00449`** / **`0.00362`** | `0.00163` | `0.00152` |
| **RTX 4090 (Ada) vs RTX 3090 (Ampere)**, `standard` | `1,803s` / `3,313s` | `0.00036` / `0.00002` | `0.00052` / `0.00019` | `0.00000` | `0.00069` |
| RTX 5090 vs AMD Radeon 8060S (ROCm 7.14), `nano` / `micro` | `~3.4h` est. `standard` | `0.00154` / `0.00095` | `0.00156` / `0.00125` | `0.00075` / `0.00072` | `0.00080` / `0.00110` |

     **What it decided:** Official Season 3 `standard` tolerances are calibrated to `~2.7–4.1×`
     the maximum measured cross-architecture NVIDIA diff (`0.00449` grade / `0.00362` baseline):
     **`grade = 0.012`, `baseline = 0.015`, `run = 0.015`** (down from provisional `0.03 / 0.05 /
     0.05`, catching real drift at `22%–30%` of `fineweb2`'s `+0.040–+0.054` reasoning gain);
     `large` tolerances (`0.012 / 0.015 / 0.015`) are extrapolated from `standard`.
     Because AMD ROCm Flash Attention backward is non-deterministic, takes `~3.4h` per standard
     pass, and experienced a 1-in-6 transient `FloatingPointError` divergence under card contention,
     Season 3 restricts official verifying (grading and, since runners must earn 5 agreed official
     grades first, running curators) to NVIDIA CUDA (`grade_backends: ["cuda"]`), while
     retaining AMD ROCm for local practice (`stet run`).
   - **Per-round compute & cost vs Season 2:**
     At `standard` scale (`65.5M` tokens), one grading pass takes `~19.5 min` on an RTX 5090
     (`~$0.15` at `$0.45/h`), `~30 min` on an RTX 4090 (`~$0.22` at `$0.43/h`), or `~55 min` on an
     RTX 3090 (`~$0.20` at `$0.22/h`), versus `$0.041–$0.046` per training in Season 2. A typical
     10-script Season 3 round runs 12 trainings (`10` scripts + `2` baselines) = **`~$1.80` of
     grading compute** (vs `~$0.55` in Season 2) plus **`~$0.15–$0.20/round`** for the automated
     ephemeral 1× RTX 5090 `Inkling-2-Mini` examiner (measured cold-start `14.27 min` / `$0.147`
     rent-to-synth-done at `$0.619/h` via `stet-cloud --examiner-addr 127.0.0.1:8000`) and curator
     runs/boot overhead (**`~$3–$5` all-in per round**, ~`2.5–3×` Season 2's `$1.57–$2.16`, while
     on-demand draws keep idle days at `$0/day`).

## Still to measure

- An exam and student setup whose **between-curator** gain is driven primarily by excerpt-grounded
  deduction rather than prompt-conditioned answer-prose modeling (see the `r9_excerpt`
  rotated-excerpt control above, where `80%–109%` of between-curator separation survived excerpt
  rotation/removal on frozen-bottom `ERNIE-4.5-0.3B` at `65.5M` tokens).
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
