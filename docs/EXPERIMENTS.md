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
1-epoch multilingual SFT probe) on a 100%-fresh exam (twin-context comprehension +
harvest-derived multi-sentence reasoning); its own noise figures, tolerances, DataDecide 1B
agreement, and adversarial red-team measurements live in `src/stet/seasons/s3.json` and the
Season 3 validation report. All tolerances remain *provisional* until grades from many
machines, of both makes, can be compared: every round's page publishes each grade's
difference from the others.

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

Three follow-up validation stages (~100 GPU runs total across RTX 5090 and RTX 3090 hardware,
archived in `~/stet-measurements/s3-validation/{sweep10,adv_v2,prod_gate}`) redesigned both
the student and the exam from first principles:

1. **Frozen bottom-half `baidu/ERNIE-4.5-0.3B-Base-PT` at `65.5M` tokens (12-recipe DataDecide 1B sweep).**
   Freezing the tied embedding/head (`vocab = 103,424`) and lower 9 layers while resetting the
   upper 9 layers (`127.4M` trainable parameters) lets the student train on **2× more tokens
   (`65.5M`)** in **nearly half the time** (`19.7 min` per pass on RTX 5090, `~29 min` on 4090,
   `~52 min` on 3090) and **`8.90 GiB` allocated / `9.33 GiB` reserved VRAM** across training,
   SFT, and exam sitting (lowering the volunteer grading floor to **12 GB**).
2. **100%-fresh 7-pillar exam (`50%` twin-context comprehension + `50%` fresh harvest-derived reasoning).**
   Replacing all synthetic templates and static benchmarks with deterministic (`temp = 0`)
   multi-sentence reasoning items (`explain`, `infer`, `synthesis` — 6 natural wrappers, zero
   `?`/`Q:/A:` syntax in `synthesis`, `<50%` 5-gram verbatim copy filter) generated by
   `Qwen/Qwen3-8B` from the same 3 fresh harvest hours, scored after a 1-epoch (`32`-step,
   `512`-example) commercial-permissive multilingual SFT probe (`s3-sft-ml.jsonl.zst`):
   - **DataDecide 1B (12 recipes, English subset):**
     - **2-seed mean (`sweep10`):** `50/50` cleared **`83.3%` pairwise (`96.8%` on `2pt+` pairs,
       Spearman `ρ = +0.818`, `p = 0.0010`)**, putting the strict 3% overfiltering trap at
       **`#11/12`** and picking a top-tier DCLM recipe (`dclm-baseline-qc-10p`, shortfall `0.49 pt`).
     - **Final production-harness gate (24 runs on exact shipped `stet` code, 1-seed `s0`):**
       `50/50` hit **`69.7%` pairwise (`80.6%` `2pt+`, `ρ = +0.566`, one-sided `p = 0.039`,
       3% trap `#11/12`)** and the 3-pillar reasoning half hit **`74.2%` (`83.9%` `2pt+`,
       `ρ = +0.636`, `p = 0.014`, 3% trap `#12/12`)** — matching and exceeding the single-seed
       `s0` slice of the 2-seed sweep (`69.7%`, `ρ = +0.531`, trap `#10/12`), confirming that
       multi-round averaging (5–40 rounds in Stet's record rule) removes single-slice variance.
   - **Cross-GPU determinism:** identical `127.4M`-parameter weight SHA-256 across separate
     RTX 5090 GPUs (`max_item_diff = 2.0e-6`, `mean = 8.5e-8`).
   - **Multilingual & adversarial gauntlet (exact production `round_score`, `unedited` 2-seed SD = `0.00068 bpb`):**

| Curator | Score (bpb) | Comp | Reas | SNR vs unedited SD |
|---|---:|---:|---:|---:|
| `fineweb2` (2-seed) | **`+0.0226`** | `+0.0046` | `+0.0406` | **`+33.1×`** (`SD = 0.00074`) |
| `mimic` | `+0.0179` | `+0.0029` | `+0.0329` | `+26.3×` (`−6.9 SD` below `fineweb2`) |
| `reader` (2-seed, per-script quantile norm) | **`+0.0169`** | `+0.0031` | `+0.0306` | **`+24.8×`** (`SD = 0.00120`) |
| `adv_qa_spam` (Q/A-heavy pages) | `+0.0025` | `+0.0029` | `+0.0022` | `+3.7×` (`−29 SD` below `fineweb2`) |
| `unedited` (2-seed baseline) | `0.0000` | `0.0000` | `0.0000` | `0.0` (`SD = 0.00068`) |
| `adv_splice_qa` (synthetic Q/A splices) | `−0.0451` | `−0.0060` | `−0.0842` | `−66×` |
| `english` (English-only filter) | `−0.0781` | `+0.0131` | `−0.1692` | `−115×` |
| `adv_overfilter` (top-3% narrow keep) | `−0.5175` | `−0.0410` | `−0.9939` | `−760×` |

**What it decided.** Season 3 ships `baidu/ERNIE-4.5-0.3B-Base-PT` (`freeze_bottom: true`,
`65.5M` tokens, 1-epoch multilingual SFT probe) with the 100%-fresh 7-pillar `comprehension`
+ `reasoning` exam (`rules_version: 7`), a **12 GB** volunteer grading floor, **24 GB+**
neural curator sandboxes, and on-demand rounds (`$0/day` idle). Per-script quantile
normalization in `examples/reader` eliminated severe non-Latin starvation (`ar` `−0.041 → +0.0655`,
`ru` `−0.045 → +0.0077`, `ja` `−0.021 → −0.0077`, `ko` `−0.128 → −0.0221`), leaving
sub-script separation (`fa`, `pl`, `uk`) as clear headroom for community curators.

## Still to measure

- How far two curators' gains differ from round to round: what the record's margin should
  be set from.
- Grades of the same scripts on NVIDIA and AMD machines: whether the tolerances hold across
  makes. The data arrives by itself, on every round's page.
- The unedited corpus's score against training tokens, on this exam: the curve a multiplier
  would be read from.
- What share the writing should have. The pilot cannot tell two thirds from nine tenths.
- Why one order of the same pages scored 0.02 above nine others, and whether an order
  like it can be found on purpose.
- How far a curator's gain moves between two draws of the raw pages for the baseline: four
  draws is all there is.
- How long a round takes from draw to settlement, now that it does not wait for its exam.
