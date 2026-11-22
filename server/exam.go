package main

// The season's exam. A round's exam is made at its draw: the organisers' examiner takes the
// hours of writing the draw's seed picked from the day before (harvest.go), builds the round's
// items and posts them here (apiExamPost). Grades start then. A grader sits every item
// and reports each one's score, and, if its job asked, the unedited baseline's (a few machines
// measure that for the round: see measuring); the round's baseline is the median, item
// by item, of the established graders' baselines it keeps (baselinesKept); and a grade's score is the exam's
// aggregate (ExamPlan.ScoreItems, a port of src/stet/exam/aggregate.py) of its item scores less the
// round's baseline. From there on the judge works as it always has, on that score.
//
// What a round holds of this (the exam's table, the baseline item by item, every grade's item
// scores) is kept while the round is open, and moved to the file store when it settles: one
// round is open at a time, so the database holds at most one round's worth.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
)

const (
	examFileMax  = 64 << 20 // bytes of a round's items, compressed
	examItemMax  = 100_000
	maxItemScore = 1e3 // bits per byte, either way: only nonsense is further out
)

// The files a round's exam keeps in the file store.
const (
	examItems  = "items.jsonl.zst"  // what graders sit, until the round settles
	examPublic = "public.jsonl.zst" // what is published when it does
	examTable  = "table.json"       // what each item counts toward
	examScores = "scores.json"      // every item score of the settled round
)

// examBlob is where a round's exam keeps one of its files: under the season, so that no
// season's rounds are ever taken for another's.
func (a *App) examBlob(idx int, name string) string {
	return fmt.Sprintf("exams/%s/%08d/%s", a.season.ID, idx, name)
}

// examPosted reports whether a round's exam is there to be sat: it is made after the draw.
func examPosted(rd *Round) bool { return rd.ExamSHA != "" }

// ExamPost is what the examiner sends: the round's items for graders, and what each item counts
// toward for the score.
type ExamPost struct {
	Round    string             `json:"round"`
	Digest   string             `json:"items_digest"`
	File     string             `json:"items_file"` // base64 of the zstd JSON lines graders sit
	Table    []ItemInfo         `json:"table"`
	Shares   map[string]float64 `json:"languages"` // percent of the round's corpus in each language
	Snapshot string             `json:"snapshot_sha256"`
	Fresh    int                `json:"fresh"`
	Public   string             `json:"public_file"` // base64 of the zstd JSON lines published when the round settles
	Hours    []RoundHour        `json:"hours"`       // the harvests its writing was made from: see harvest.go
}

// apiExamPost: POST /api/v1/admin/rounds/{id}/exam, from the organisers. Posting the same exam
// again is harmless; posting another is refused.
func (a *App) apiExamPost(w http.ResponseWriter, r *http.Request) {
	_, _, body, err := a.signedBy(w, r, 4*examFileMax/3+(8<<20), true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	idx, ok := a.parseRoundID(r.PathValue("id"))
	if !ok {
		writeErr(w, fail(404, "no such round"))
		return
	}
	var p ExamPost
	if err := json.Unmarshal(body, &p); err != nil {
		writeErr(w, fail(400, "not an exam: %v", err))
		return
	}
	posted, err := a.postExam(r.Context(), idx, &p)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"round": roundID(a.season.ID, idx), "items": len(p.Table), "posted": posted})
}

// postExam gives a round the exam the examiner made for it, and reports whether it is new.
func (a *App) postExam(ctx context.Context, idx int, p *ExamPost) (bool, error) {
	file, err := base64.StdEncoding.DecodeString(p.File)
	public, perr := base64.StdEncoding.DecodeString(p.Public)
	if err != nil || perr != nil || len(public) == 0 || len(public) > examFileMax || len(file) == 0 || len(file) > examFileMax || !sha256Re.MatchString(p.Digest) || len(p.Table) == 0 ||
		len(p.Table) > examItemMax || (p.Snapshot != "" && !sha256Re.MatchString(p.Snapshot)) || p.Fresh < 0 || p.Fresh > len(p.Table) {
		return false, fail(400, "the exam does not add up")
	}
	if err := a.checkTable(p.Table, p.Shares); err != nil {
		return false, err
	}
	var already bool
	a.st.view(func(s *State) {
		rd := s.Rounds[idx]
		already = rd != nil && rd.ExamSHA == p.Digest
		if rd == nil || rd.Status != "open" {
			err = fail(409, "%s is not open", roundID(a.season.ID, idx))
		} else if rd.ExamSHA != "" && rd.ExamSHA != p.Digest {
			err = fail(409, "%s already has its exam", roundID(a.season.ID, idx))
		} else if !already {
			_, err = a.checkHours(s, rd, p)
		}
	})
	if err != nil || already {
		return false, err
	}
	table, _ := json.Marshal(p.Table)
	for _, put := range []struct {
		name, kind string
		b          []byte
	}{{examItems, "application/zstd", file}, {examPublic, "application/zstd", public}, {examTable, "application/json", table}} {
		if err := a.st.blobs.Put(ctx, a.examBlob(idx, put.name), put.b, put.kind); err != nil {
			slog.Error("keeping a round's exam", "round", idx, "err", err)
			return false, fail(503, "the file store is not answering; try again in a moment")
		}
	}
	err = a.st.update(ctx, func(tx *Tx) error {
		rd := tx.s.Rounds[idx]
		if rd == nil || rd.Status != "open" || (rd.ExamSHA != "" && rd.ExamSHA != p.Digest) {
			return fail(409, "%s changed while its exam was being kept; send it again", roundID(a.season.ID, idx))
		}
		late, err := a.checkHours(tx.s, rd, p)
		if err != nil {
			return err
		}
		if late {
			rd.Hours, rd.HoursLate = p.Hours, true
		}
		rd.ExamSHA, rd.ExamAt, rd.ExamTable, rd.Shares, rd.Snapshot, rd.Fresh = p.Digest, now(), p.Table, p.Shares, p.Snapshot, p.Fresh
		rd.ExamMix = a.examMix(p.Table)
		tx.event("exam", map[string]any{"round": roundID(a.season.ID, idx), "items": len(p.Table), "fresh": p.Fresh})
		// A grade taken before the exam has waited for it: its time to sit it starts now.
		for _, c := range tx.s.Claims {
			if c.RoundIdx == idx && c.Kind == "grade" && c.Status == "claimed" && c.ExpiresAt < rd.ExamAt+int64(a.cfg.Lease.Seconds()) {
				c.ExpiresAt = rd.ExamAt + int64(a.cfg.Lease.Seconds())
				tx.saveClaim(c)
			}
		}
		a.moved(tx, rd) // the round's clock starts again: its grades start now
		return nil
	})
	return err == nil, err
}

// checkTable refuses a table with a part the season does not name, or one under a pillar the
// season does not put it under, or a language share that is not a number. The weights are the
// season's; the examiner only says what each item is.
func (a *App) checkTable(table []ItemInfo, shares map[string]float64) error {
	for i, it := range table {
		if it.Pillar == "" || it.Pillar != a.season.Exam.pillarOf(it.Part) || it.Part == "" || it.Lang == "" || it.Stratum == "" ||
			len(it.Part)+len(it.Lang)+len(it.Stratum) > 200 {
			return fail(400, "item %d of the table is not one the season scores", i)
		}
	}
	if len(shares) == 0 || len(shares) > 1000 {
		return fail(400, "the exam names no languages, or too many: a round's share of each is part of its score")
	}
	for k, v := range shares {
		if len(k) > 32 || !finite(v) || v < 0 || v > 100 {
			return fail(400, "the share of %q is not a percentage", clip(k, 32))
		}
	}
	return nil
}

// apiExamGet: GET /api/v1/rounds/{id}/exam. Once a round settles, its published exam, to anyone:
// the generated items and the openly licensed text whole, every other item as its source, id
// and hash. Until then the whole exam, signed, to an account that holds a grade job in the
// round, or to the organisers.
func (a *App) apiExamGet(w http.ResponseWriter, r *http.Request) {
	idx, ok := a.parseRoundID(r.PathValue("id"))
	if !ok {
		writeErr(w, fail(404, "no such round"))
		return
	}
	var posted, settled bool
	a.st.view(func(s *State) {
		rd := s.Rounds[idx]
		posted = rd != nil && examPosted(rd)
		settled = posted && rd.Status == "settled"
	})
	if !posted {
		writeErr(w, fail(404, "%s has no exam yet", roundID(a.season.ID, idx)))
		return
	}
	blob := a.examBlob(idx, examPublic)
	if !settled {
		acct, _, _, err := a.signedBy(w, r, 1<<10, true, anyone)
		if err != nil {
			writeErr(w, err)
			return
		}
		ok := acct.administers()
		a.st.view(func(s *State) {
			for _, c := range s.Claims {
				ok = ok || (c.RoundIdx == idx && c.AccountID == acct.ID && c.Kind == "grade" && c.live(now()))
			}
		})
		if !ok {
			writeErr(w, fail(403, "until a round settles, its exam is given to the machines grading it"))
			return
		}
		blob = a.examBlob(idx, examItems)
	}
	f, size, err := a.st.blobs.Open(r.Context(), blob)
	if err != nil {
		slog.Error("reading a round's exam", "round", idx, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	io.Copy(w, f)
}

// checkItems refuses a report that is not one score for every item of the round's exam.
func checkItems(sc *ExamScore, n int, what string) error {
	if sc == nil || len(sc.ItemScores) != n || n == 0 {
		return fail(400, "the %s score is not one score for each of the exam's %d items", what, n)
	}
	sum := 0.0
	for _, v := range sc.ItemScores {
		if !finite(v) || math.Abs(v) > maxItemScore {
			return fail(400, "the %s score has an item score that is not a usable number", what)
		}
		sum += v
	}
	if !finite(sc.Mean) || math.Abs(sum/float64(n)-sc.Mean) > 1e-3 {
		return fail(400, "the %s score's mean is not the mean of its items", what)
	}
	return nil
}

// medianItems is the item-by-item median of some item scores: with an even number of them, the
// mean of the two in the middle. One machine that is far out moves it by nothing once three
// have measured.
func medianItems(rows [][]float64) []float64 {
	if len(rows) == 0 {
		return nil
	}
	out, col := make([]float64, len(rows[0])), make([]float64, len(rows))
	for i := range out {
		for k, row := range rows {
			col[k] = row[i]
		}
		sort.Float64s(col)
		if m := len(col) / 2; len(col)%2 == 1 {
			out[i] = col[m]
		} else {
			out[i] = (col[m-1] + col[m]) / 2
		}
	}
	return out
}

func diffItems(x, base []float64) []float64 {
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] - base[i]
	}
	return out
}

// roundBaselines are the round's baselines that may make its own, in the order of their
// accounts: measured on its corpus and exam, by an account established when it measured (or the
// organisers'). Whether a measurement counts was settled when it arrived: standing earned as the
// round settles must not move the round's own baseline.
func (a *App) roundBaselines(s *State, rd *Round) []*Baseline {
	var out []*Baseline
	for _, b := range s.Baselines {
		acct := s.Accounts[b.AccountID]
		if b.RoundIdx == rd.Idx && (b.Established || (acct != nil && acct.Anchor)) && b.CorpusSHA == rd.CorpusSHA && b.ExamSHA == rd.ExamSHA &&
			len(b.Items) == len(rd.ExamTable) {
			out = append(out, b)
		}
	}
	sort.Slice(out, func(i, k int) bool { return out[i].AccountID < out[k].AccountID })
	return out
}

// baselineQuorum is how many baselines it takes to say that one of them is the odd one out:
// with two that differ, nothing says which is wrong.
const baselineQuorum = 3

// itemsOf are the item scores of some baselines.
func itemsOf(bs []*Baseline) [][]float64 {
	rows := make([][]float64, len(bs))
	for i, b := range bs {
		rows[i] = b.Items
	}
	return rows
}

// baselinesKept are the round's baselines that make its own. While fewer than three are in,
// all of them: whichever came first is no more the standard than the next. From three on,
// those within the tolerance of the median of them all, whatever order they came in. (If none
// is, they are all kept: there is no telling.)
func (a *App) baselinesKept(s *State, rd *Round) []*Baseline {
	all := a.roundBaselines(s, rd)
	if len(all) < baselineQuorum {
		return all
	}
	mid, tol := medianItems(itemsOf(all)), a.tolerance().Baseline
	var kept []*Baseline
	for _, b := range all {
		if dev, _, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(b.Items, mid), rd.Shares); err == nil && within(dev, tol) {
			kept = append(kept, b)
		}
	}
	if len(kept) == 0 {
		return all
	}
	return kept
}

// baselineDev is how far a grader's baseline is from the median of the round's baselines, its
// own among them (the others' and this one, whether or not its account is established), on the
// exam's score. ok is false while fewer than two others are in: nobody is turned away for
// differing from one machine, which may be the one that is wrong.
func (a *App) baselineDev(s *State, rd *Round, acctID int64, items []float64) (dev float64, ok bool) {
	var rows [][]float64
	for _, b := range a.roundBaselines(s, rd) {
		if b.AccountID != acctID {
			rows = append(rows, b.Items)
		}
	}
	if len(rows) < baselineQuorum-1 || len(items) != len(rd.ExamTable) {
		return 0, false
	}
	dev, _, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(items, medianItems(append(rows, items))), rd.Shares)
	return dev, err == nil
}

// baselineAgrees says whether a grader's baseline is within the tolerance of the round's: its
// score against the round's baseline, on the exam's aggregate.
func (a *App) baselineAgrees(rd *Round, b *Baseline) bool {
	if b == nil || len(rd.BaseItems) == 0 || len(b.Items) != len(rd.BaseItems) || b.ExamSHA != rd.ExamSHA {
		return false
	}
	dev, _, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(b.Items, rd.BaseItems), rd.Shares)
	return err == nil && within(dev, a.tolerance().Baseline)
}

// --- who measures a round's baseline -----------------------------------------------------------
//
// Measuring the unedited corpus costs a grader a whole training pass, as much as the grade
// itself, and every grader's measurement is no better a baseline for the round
// than a few. So the season says how many machines measure it
// (baselines.per_round), and each grade job says whether its machine is one of them
// (Job.Baseline). A machine that is not trains the script's student and nothing else, and
// its grade is scored against the round's baseline like any other.
//
//   - A job asks for the baseline while fewer than that many established accounts have
//     measured the round's and agree with it, or were asked to by a grade job they still
//     hold (measuring). A job that is handed back, runs out or is turned away takes its
//     promise with it, and the next job handed out is asked instead.
//   - An account that has measured the round's, or was ever asked to, is asked again with
//     every further grade: the measurement is on its disk, and its results keep carrying it.
//   - An account not yet established is always asked. Its baseline is no part of the
//     round's, and is how its machine is checked against the others' (baselineDev).
//   - A round is never scored without one. While it has none and nobody at work on it was
//     asked for one, an established grade that came without is as good as no grade: the
//     script waits for another (weigh), and whoever is handed it is asked. If nobody is
//     there to be, the script is given up like any that nobody was there to grade (giveUp).

// measuring is whose measurement of a round's unedited corpus the round has or is promised:
// the established accounts whose baseline it holds and agrees with, and those a grade job
// they still hold asked for one, who have not sent it yet. Before the exam is posted nothing
// has been sent, and it is the promises alone.
func (j *judge) measuring(rd *Round) map[int64]bool {
	if who, ok := j.measures[rd.Idx]; ok {
		return who
	}
	who := map[int64]bool{}
	for _, b := range j.a.baselinesKept(j.s, rd) {
		if j.a.baselineAgrees(rd, b) {
			who[b.AccountID] = true
		}
	}
	for _, c := range j.s.Claims {
		if c.RoundIdx == rd.Idx && c.Kind == "grade" && c.Baseline && c.live(j.t) && j.est[c.AccountID] &&
			j.s.Baselines[baselineKey(rd.Idx, c.AccountID)] == nil {
			who[c.AccountID] = true
		}
	}
	j.measures[rd.Idx] = who
	return who
}

// baselineComing reports whether a round has a baseline, or somebody at work on it was asked
// for one.
func (j *judge) baselineComing(round int) bool {
	rd := j.s.Rounds[round]
	return rd != nil && len(j.measuring(rd)) > 0
}

// baselinesWanted is how many more machines a round would ask for its baseline, as it stands.
func (j *judge) baselinesWanted(rd *Round) int {
	return max(0, j.a.season.Baselines.PerRound-len(j.measuring(rd)))
}

// asksBaseline says whether a grade job handed to this account now asks its machine for the
// round's unedited baseline: see above. With no number in the season, every job does.
func (j *judge) asksBaseline(rd *Round, acct *Account) bool {
	if j.a.season.Baselines.PerRound <= 0 || !j.est[acct.ID] || j.s.Baselines[baselineKey(rd.Idx, acct.ID)] != nil {
		return true
	}
	for _, c := range j.s.Claims {
		if c.RoundIdx == rd.Idx && c.AccountID == acct.ID && c.Kind == "grade" && c.Baseline {
			return true // asked before, whatever came of that job: it may have it trained
		}
	}
	return j.baselinesWanted(rd) > 0
}

// refreshScores sets the round's baseline to the median of the baselines it keeps
// (baselinesKept), item by item, and every grade's score to the exam's aggregate of its item
// scores less that.
func (a *App) refreshScores(tx *Tx, rd *Round) {
	s := tx.s
	kept := a.baselinesKept(s, rd)
	if len(kept) == 0 {
		return
	}
	base := medianItems(itemsOf(kept))
	rd.BaseItems = base
	mean := 0.0
	for _, v := range base {
		mean += v
	}
	loss := places5(-mean / float64(len(base))) // the unedited student's mean bits per byte: for the record
	rd.BaselineLoss = &loss
	// How many machines measured it, whoever they were: but not one whose measurement the
	// round left out of its own.
	out := map[int64]bool{}
	for _, b := range a.roundBaselines(s, rd) {
		out[b.AccountID] = true
	}
	for _, b := range kept {
		delete(out, b.AccountID)
	}
	rd.BaselineN = 0
	for _, b := range s.Baselines {
		if b.RoundIdx == rd.Idx && !out[b.AccountID] && a.baselineAgrees(rd, b) {
			rd.BaselineN++
		}
	}
	tx.saveRound(rd)
	for _, e := range s.inRound[rd.Idx] {
		for _, r := range s.results(rd.Idx, e.EditorID) {
			if !r.Valid || len(r.Items) != len(base) {
				continue
			}
			score, pillars, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(r.Items, base), rd.Shares)
			if err != nil {
				continue
			}
			score = places5(score)
			for k, v := range pillars {
				pillars[k] = places5(v)
			}
			r.Score, r.BySource = &score, pillars
			r.Loss = places5(loss - score)
			tx.saveResult(r)
		}
	}
}

// ExamScores is what a settled round's exam came to, item by item: kept in the file store,
// and served to anyone (apiScoresGet), so that anyone can work every score out again. The
// items are in the order of the round's published exam.
//
// It says of itself everything that working a score out needs. A grade's score is the
// season's aggregate (ExamPlan.ScoreItems) of its item scores less the round's baseline,
// over Table and Shares; GradesOf says whose grade each row is, of which script of which
// entry, and whether it is one of those its script's gain is the median of.
type ExamScores struct {
	Round      string               `json:"round"`
	Digest     string               `json:"items_digest"`
	Shares     map[string]float64   `json:"languages"`
	Table      []ItemInfo           `json:"table,omitempty"`       // what each item counts toward
	Baseline   []float64            `json:"baseline"`              // the round's: the mean, item by item, of BaselineOf's
	BaselineOf []string             `json:"baseline_of,omitempty"` // the handles whose baselines the round's is the median of
	Baselines  map[string][]float64 `json:"baselines"`             // by account handle
	Grades     map[string][]float64 `json:"grades"`                // by result id
	GradesOf   map[string]GradeOf   `json:"grades_of,omitempty"`   // by result id
}

// GradeOf says what one grade in a round's item scores is a grade of. A result's id is in no
// other public record, so without this nobody could tell the rows of Grades apart. A grade
// that found its script invalid sat no exam, and has no row.
type GradeOf struct {
	Editor  string `json:"editor"`            // the entry: its curator's id
	Script  string `json:"script"`            // sha256 of the script it graded
	Runs    []int  `json:"runs"`              // the entry's runs that wrote that script, by their n
	By      string `json:"by"`                // the grader's handle
	Counted bool   `json:"counted,omitempty"` // the script's gain is the median of these
}

// archiveExam moves what a round held of its exam to the file store, once it has settled, and
// then out of the database. Called after the transaction that settled it; a round that could
// not be archived keeps it all, and is tried again on the next tick.
func (a *App) archiveExam(ctx context.Context) {
	var due []int
	a.st.view(func(s *State) {
		for _, rd := range s.Rounds {
			if rd.Status == "settled" && (len(rd.ExamTable) > 0 || len(rd.BaseItems) > 0) {
				due = append(due, rd.Idx)
			}
		}
	})
	for _, idx := range due {
		var out ExamScores
		a.st.view(func(s *State) {
			rd := s.Rounds[idx]
			out = ExamScores{Round: roundID(a.season.ID, idx), Digest: rd.ExamSHA, Shares: rd.Shares, Table: rd.ExamTable, Baseline: rd.BaseItems,
				Baselines: map[string][]float64{}, Grades: map[string][]float64{}, GradesOf: map[string]GradeOf{}}
			handle := func(id int64) string {
				if acct := s.Accounts[id]; acct != nil {
					return acct.Handle
				}
				return strconv.FormatInt(id, 10)
			}
			for _, b := range a.baselinesKept(s, rd) {
				out.BaselineOf = append(out.BaselineOf, handle(b.AccountID))
			}
			sort.Strings(out.BaselineOf)
			for _, b := range s.Baselines {
				if b.RoundIdx == idx && len(b.Items) > 0 {
					out.Baselines[handle(b.AccountID)] = b.Items
				}
			}
			for _, e := range s.inRound[idx] {
				// Which of the entry's runs wrote each script, and which grades of it decided its
				// gain, as the entry published them when it settled.
				wrote, counted := map[string][]int{}, map[string]bool{}
				for _, run := range e.Runs {
					if run.Script == "" {
						continue
					}
					wrote[run.Script] = append(wrote[run.Script], run.N)
					for _, g := range run.Grades {
						counted[run.Script+"/"+g.Handle] = g.Counted
					}
				}
				for _, r := range s.results(idx, e.EditorID) {
					if len(r.Items) > 0 {
						id, by := strconv.FormatInt(r.ID, 10), handle(r.AccountID)
						out.Grades[id] = r.Items
						out.GradesOf[id] = GradeOf{Editor: e.EditorID, Script: r.Script, Runs: wrote[r.Script], By: by, Counted: counted[r.Script+"/"+by]}
					}
				}
			}
		})
		raw, _ := json.Marshal(out)
		if err := a.st.blobs.Put(ctx, a.examBlob(idx, examScores), raw, "application/json"); err != nil {
			slog.Error("archiving a round's exam scores", "round", idx, "err", err)
			continue
		}
		err := a.st.update(ctx, func(tx *Tx) error {
			s := tx.s
			rd := s.Rounds[idx]
			rd.ExamTable, rd.BaseItems = nil, nil
			tx.saveRound(rd)
			for _, b := range s.Baselines {
				if b.RoundIdx == idx && len(b.Items) > 0 {
					b.Items = nil
					tx.saveBaseline(b)
				}
			}
			for _, e := range s.inRound[idx] {
				for _, r := range s.results(idx, e.EditorID) {
					if len(r.Items) > 0 {
						r.Items = nil
						tx.saveResult(r)
					}
				}
			}
			return nil
		})
		if err != nil {
			slog.Error("archiving a round's exam scores", "round", idx, "err", err)
		}
	}
}

// scoresArchived reports whether a round's item scores are in the file store, to be served:
// it has settled, and what it held of its exam has been moved there (archiveExam), which is
// when the round stops holding it.
func scoresArchived(rd *Round) bool {
	return rd != nil && rd.Status == "settled" && examPosted(rd) && len(rd.ExamTable) == 0 && len(rd.BaseItems) == 0
}

// apiScoresGet: GET /api/v1/rounds/{id}/scores. Every item score of a settled round, to
// anyone (ExamScores): the round's baseline, each grader's baseline, and each grade, with
// what each item counts toward and whose grade each is. With the round's published exam, it
// is all that working any of its scores out again takes.
func (a *App) apiScoresGet(w http.ResponseWriter, r *http.Request) {
	idx, ok := a.parseRoundID(r.PathValue("id"))
	if !ok {
		writeErr(w, fail(404, "no such round"))
		return
	}
	var known, settled, sat, archived bool
	a.st.view(func(s *State) {
		rd := s.Rounds[idx]
		known, settled = rd != nil, rd != nil && rd.Status == "settled"
		sat, archived = known && examPosted(rd), scoresArchived(rd)
	})
	id := roundID(a.season.ID, idx)
	switch {
	case !known:
		writeErr(w, fail(404, "no such round"))
		return
	case !settled:
		writeErr(w, fail(404, "%s has not settled: its item scores are sealed until it does", id))
		return
	case !sat:
		writeErr(w, fail(404, "%s settled before it had an exam, so it has no item scores", id))
		return
	case !archived:
		// Settled a moment ago: its scores are on their way to the file store.
		writeErr(w, fail(404, "%s has only just settled, and its item scores are not published yet; try again in a minute", id))
		return
	}
	f, size, err := a.st.blobs.Open(r.Context(), a.examBlob(idx, examScores))
	if errors.Is(err, errNoBlob) {
		writeErr(w, fail(404, "%s has no item scores published", id))
		return
	}
	if err != nil {
		slog.Error("reading a round's exam scores", "round", idx, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	defer f.Close()
	// It is written once, when the round is archived, and never again. A day, not for ever:
	// the address names a round, not the file's contents.
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	w.Header().Set("Cache-Control", "public, max-age=86400")
	io.Copy(w, f)
}

// ExamMix is what a round's exam was made of, for its page: how many items each of the
// season's pillars has, and how many of those are text people published in the day before the
// draw rather than problems generated from the round's seed; in how many languages people
// wrote; and how many items each kind of writing gave. Kept after the round settles, when the
// table it is counted from is not.
type ExamMix struct {
	Items int          `json:"items"`
	Fresh int          `json:"fresh"`
	Langs int          `json:"languages,omitempty"` // of the text people wrote, programming languages apart
	Parts []ExamPillar `json:"pillars"`
	Kinds []ExamKind   `json:"kinds,omitempty"` // of the text people wrote, the most items first
}

type ExamPillar struct {
	Pillar string  `json:"pillar"`
	Group  string  `json:"group,omitempty"`
	Share  float64 `json:"share,omitempty"` // of the score, among the pillars this exam has
	Items  int     `json:"items"`
	Fresh  int     `json:"fresh,omitempty"`
}

type ExamKind struct {
	Kind  string `json:"kind"`
	Items int    `json:"items"`
}

// examMix counts a table the season's exam can have: checkTable has refused any other. A part
// is a family of generated problems when the season says so (ExamPlan.Generated), and a kind of
// writing people published otherwise.
func (a *App) examMix(table []ItemInfo) *ExamMix {
	if len(table) == 0 {
		return nil
	}
	plan := &a.season.Exam
	m := &ExamMix{Items: len(table)}
	at := map[string]int{}
	for _, p := range plan.Pillars {
		at[p.Name] = len(m.Parts)
		m.Parts = append(m.Parts, ExamPillar{Pillar: p.Name, Group: p.Group})
	}
	langs, kinds, present := map[string]bool{}, map[string]int{}, map[string]bool{}
	for _, it := range table {
		i, ok := at[it.Pillar]
		if !ok {
			continue // not a row of the season's exam
		}
		fresh := !plan.generated(it.Part)
		if fresh {
			m.Fresh++
			kinds[it.Part]++
			if !strings.HasPrefix(it.Lang, "code") {
				langs[it.Lang] = true
			}
		}
		present[it.Pillar] = true
		m.Parts[i].Items++
		if fresh {
			m.Parts[i].Fresh++
		}
	}
	for name, share := range plan.weights(present) {
		m.Parts[at[name]].Share = places5(share)
	}
	m.Langs = len(langs)
	for k, n := range kinds {
		m.Kinds = append(m.Kinds, ExamKind{k, n})
	}
	sort.Slice(m.Kinds, func(x, y int) bool {
		if m.Kinds[x].Items != m.Kinds[y].Items {
			return m.Kinds[x].Items > m.Kinds[y].Items
		}
		return m.Kinds[x].Kind < m.Kinds[y].Kind
	})
	return m
}
