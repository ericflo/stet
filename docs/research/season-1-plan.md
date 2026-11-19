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

1. `s1.json`, and the client reading the next exam from the season. Its parameters are in the
   season digest, and `newexam` is in the harness hash.
2. The grader's side: fetch the round's items, sit them, report item scores, and keep a
   baseline per round.
3. The server:
   - the exam endpoint;
   - holding grade jobs until the exam is posted;
   - serving items to graders;
   - storing item scores;
   - the round's per-item baseline;
   - the Go aggregate;
   - gains and agreement from it.
4. The examiner, `stet examine --follow`, grown from `stet shadow --follow`.
5. The site and API: pillars in place of the six sources, bits per byte, no held-out exam.
6. The rules and documents for season 1.
7. Calibration at standard: the tolerances, the record's margin, the noise.
8. The switch: close season 0, start season 1. This needs the organisers' go-ahead.
