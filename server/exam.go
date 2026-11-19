package main

// The season's exam. A round's exam is made after its draw: the organisers'
// examiner harvests the fresh text published in the window after it, builds the round's items
// and posts them here (apiExamPost). Grades start then. A grader sits every item
// and reports each one's score, and its own baseline's; the round's baseline is the mean, item
// by item, of the established graders' baselines; and a grade's score is the exam's
// aggregate (NextScore, a port of src/stet/exam/aggregate.py) of its item scores less the
// round's baseline. From there on the judge works as it always has, on that score.
//
// What a round holds of this (the exam's table, the baseline item by item, every grade's item
// scores) is kept while the round is open, and moved to the file store when it settles: one
// round is open at a time, so the database holds at most one round's worth.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
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
		rd.ExamSHA, rd.ExamAt, rd.ExamTable, rd.Shares, rd.Snapshot, rd.Fresh = p.Digest, now(), p.Table, p.Shares, p.Snapshot, p.Fresh
		tx.event("exam", map[string]any{"round": roundID(a.season.ID, idx), "items": len(p.Table), "fresh": p.Fresh})
		a.moved(tx, rd) // the round's clock starts again: its grades start now
		return nil
	})
	return err == nil, err
}

// checkTable refuses a table that names a pillar the season does not have, or a language share
// that is not a number.
func (a *App) checkTable(table []ItemInfo, shares map[string]float64) error {
	pillars := map[string]bool{}
	for _, p := range a.season.ExamSources {
		pillars[p] = true
	}
	for i, it := range table {
		if !pillars[it.Pillar] || it.Part == "" || it.Lang == "" || it.Stratum == "" || len(it.Part)+len(it.Lang)+len(it.Stratum) > 200 {
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
	if sc == nil || sc.Kind != "next" || len(sc.ItemScores) != n || n == 0 {
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

// meanItems is the item-by-item mean of some item scores.
func meanItems(rows [][]float64) []float64 {
	if len(rows) == 0 {
		return nil
	}
	out := make([]float64, len(rows[0]))
	for _, row := range rows {
		for i, v := range row {
			out[i] += v
		}
	}
	for i := range out {
		out[i] /= float64(len(rows))
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

// roundBaselines are the round's baselines that make its own: measured on its corpus and exam,
// by an account established when it measured (or the organisers'). Whether a measurement
// counts was settled when it arrived: standing earned as the round settles must not move the
// round's own baseline.
func (a *App) roundBaselines(s *State, rd *Round) []*Baseline {
	var out []*Baseline
	for _, b := range s.Baselines {
		acct := s.Accounts[b.AccountID]
		if b.RoundIdx == rd.Idx && (b.Established || (acct != nil && acct.Anchor)) && b.CorpusSHA == rd.CorpusSHA && b.ExamSHA == rd.ExamSHA &&
			len(b.Items) == len(rd.ExamTable) {
			out = append(out, b)
		}
	}
	return out
}

// baselineDev is how far a grader's baseline is from the round's others (all of them but its
// own), on the exam's score; ok is false while there are no others.
func (a *App) baselineDev(s *State, rd *Round, acctID int64, items []float64) (dev float64, ok bool) {
	var rows [][]float64
	for _, b := range a.roundBaselines(s, rd) {
		if b.AccountID != acctID {
			rows = append(rows, b.Items)
		}
	}
	if len(rows) == 0 || len(items) != len(rd.ExamTable) {
		return 0, false
	}
	dev, _, err := NextScore(rd.ExamTable, diffItems(items, meanItems(rows)), rd.Shares)
	return dev, err == nil
}

// baselineAgrees says whether a grader's baseline is within the tolerance of the round's: its
// score against the round's baseline, on the exam's aggregate.
func (a *App) baselineAgrees(rd *Round, b *Baseline) bool {
	if b == nil || len(rd.BaseItems) == 0 || len(b.Items) != len(rd.BaseItems) || b.ExamSHA != rd.ExamSHA {
		return false
	}
	dev, _, err := NextScore(rd.ExamTable, diffItems(b.Items, rd.BaseItems), rd.Shares)
	return err == nil && within(dev, a.tolerance().Baseline)
}

// refreshScores sets the round's baseline to the mean of its baselines, item by item, and every
// grade's score to the exam's aggregate of its item scores less that.
func (a *App) refreshScores(tx *Tx, rd *Round) {
	s := tx.s
	var rows [][]float64
	for _, b := range a.roundBaselines(s, rd) {
		rows = append(rows, b.Items)
	}
	if len(rows) == 0 {
		return
	}
	base := meanItems(rows)
	rd.BaseItems = base
	mean := 0.0
	for _, v := range base {
		mean += v
	}
	loss := places5(-mean / float64(len(base))) // the unedited student's mean bits per byte: for the record
	rd.BaselineLoss = &loss
	// How many machines measured it, whoever they were.
	rd.BaselineN = 0
	for _, b := range s.Baselines {
		if b.RoundIdx == rd.Idx && a.baselineAgrees(rd, b) {
			rd.BaselineN++
		}
	}
	tx.saveRound(rd)
	for _, e := range s.inRound[rd.Idx] {
		for _, r := range s.results(rd.Idx, e.EditorID) {
			if !r.Valid || len(r.Items) != len(base) {
				continue
			}
			score, pillars, err := NextScore(rd.ExamTable, diffItems(r.Items, base), rd.Shares)
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

// NextScores is what a settled round's exam came to, item by item: kept in the file store, so
// that anyone can work every score out again.
type NextScores struct {
	Round     string               `json:"round"`
	Digest    string               `json:"items_digest"`
	Shares    map[string]float64   `json:"languages"`
	Baseline  []float64            `json:"baseline"`
	Baselines map[string][]float64 `json:"baselines"` // by account handle
	Grades    map[string][]float64 `json:"grades"`    // by result id
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
		var out NextScores
		a.st.view(func(s *State) {
			rd := s.Rounds[idx]
			out = NextScores{Round: roundID(a.season.ID, idx), Digest: rd.ExamSHA, Shares: rd.Shares, Baseline: rd.BaseItems,
				Baselines: map[string][]float64{}, Grades: map[string][]float64{}}
			for _, b := range s.Baselines {
				if b.RoundIdx == idx && len(b.Items) > 0 {
					handle := strconv.FormatInt(b.AccountID, 10)
					if acct := s.Accounts[b.AccountID]; acct != nil {
						handle = acct.Handle
					}
					out.Baselines[handle] = b.Items
				}
			}
			for _, e := range s.inRound[idx] {
				for _, r := range s.results(idx, e.EditorID) {
					if len(r.Items) > 0 {
						out.Grades[strconv.FormatInt(r.ID, 10)] = r.Items
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
