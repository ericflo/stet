# Season 1: how the next exam becomes the score

Status: being built. The decisions are in [first principles, section 7](exam-first-principles.md#7-decisions-november-2026).
This note says how they are carried out.

## What changes for a round

1. **The draw and the curators' runs are as now**, at the standard scale. A run cannot see the
   exam: it has not been made yet.
2. **The exam is made after the draw.** An organisers' machine (the examiner):
   - waits until the harvest window (two hours from the draw) has closed and three more hours
     have passed;
   - harvests the window's fresh text;
   - builds the round's corpus and drops the fresh items it already holds;
   - generates the round's other items from its seed;
   - measures the corpus's languages.

   It posts the exam to the server: the items (for graders), a table of what each item counts
   toward (pillar, part, language, format and kind), the item digest, and the language shares.
   The round now has its exam (`ExamSHA` is the item digest).
3. **Grading starts when the exam is posted.**
   - A grader builds the corpus, applies the script, trains the student and sits every item. It
     reports one score per item, and its own baseline's (the unedited corpus, trained once per
     grader per round) the same way.
   - The round's baseline is the per-item mean of the established graders' baselines. The
     organisers' machines make sure there are at least four. The unedited student is the least
     steady one, and this averages it.
   - A grade's score is the next exam's aggregate (seven pillars, each counting the same) of its
     item scores less the round's baseline. The server computes it with a Go port of
     `newexam/aggregate.py`, checked against the Python on shared test vectors.
4. **Agreement, settlement, the board and the record are as now**, in bits per byte.
   - Two grades of a script agree within `tolerance.grade`.
   - A grader's baseline agrees with the round's when its own score against the round's
     baseline is within `tolerance.baseline`.
   - The timers that give up on missing grades count from when the exam was posted.

## Timing

Draw, then runs (within an hour), then the exam (draw + 5 h), then grades (about an hour at
standard), then settlement: about 6 to 7 hours a round. Rounds never overlap, so a round every
6 hours becomes one every 6 to 7.

## What is kept private

- The items are served to graders only: established verifiers and the organisers' machines.
- A settled round publishes:
  - the generated items;
  - Wikipedia's fresh text;
  - for every other source, the item's id, URL and the sha256 of its text.
- Practice rounds use generated items, and the fresh text of a published practice snapshot made
  from Wikipedia only.

## Seasons

The server keeps one season. Season 0 is closed under the season-0 server (the held-out exam is
published, the final table made). Season 1 then starts on a new database, and season 0's site
stays readable.

## Stages

1. **Done.** `s1.json`, with the next exam's parameters in the season digest and `newexam` in
   the harness hash. Season 1's tokenizer is trained on the crawl alone.
2. **Done.** The grader's side: it fetches the round's items, checks them against the round's
   digest, sits them, and reports item scores, keeping a baseline for each exam.
3. **Done.** The server:
   - the exam endpoints;
   - grades held until the exam is posted, with nothing given up meanwhile;
   - item scores reported and checked;
   - the round's per-item baseline;
   - `NextScore`, a bit-for-bit port of the aggregate with shared test vectors;
   - archiving at settlement.
4. **Done.** The examiner: `stet examine --follow`.
5. **In progress.** The site and API: pillars and bits per byte, no held-out exam, no
   multiplier until its curve is measured.
6. **Done.** Season 1's rules (`docs/RULES-s1.md`) and the switch in `docs/OPERATING.md`.
7. **Calibration**, before the first round:
   - The tolerances are first values from the NVIDIA repeat trainings: 0.013 between
     trainings at standard, so grade 0.06, baseline 0.1, run 0.09.
   - An AMD grader (the organisers' own verifier) must be measured against NVIDIA ones on
     the same scripts before its grades count.
   - The record's margin (0.01) stays provisional until the noise between rounds is known.
8. **The switch.** It needs the organisers' go-ahead: season 0's rules promise two weeks'
   notice of its close.

Practice rounds, `stet run`, `stet score` and the gym all work on season 1 now
(`--season s1`). They use generated items only.
