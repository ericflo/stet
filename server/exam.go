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
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
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

// examDigestBlob is an immutable, digest-addressed blob key for a round's exam files so two
// overlapping publications cannot overwrite each other before commit.
func (a *App) examDigestBlob(idx int, digest, name string) string {
	return fmt.Sprintf("exams/%s/%08d/%s/%s", a.season.ID, idx, digest, name)
}

// examBlobSHA hashes the full publication payload (ordered items digest + compressed items,
// public, and table bytes) so overlapping posts with the same items_digest but different derived
// public_file or table stage into distinct immutable prefixes and cannot overwrite the winner.
func examBlobSHA(digest string, file, public, table []byte) string {
	fh := sha256.Sum256(file)
	ph := sha256.Sum256(public)
	th := sha256.Sum256(table)
	h := sha256.New()
	fmt.Fprintf(h, "%s\n%x\n%x\n%x", digest, fh, ph, th)
	return hex.EncodeToString(h.Sum(nil))
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
	Public   string             `json:"public_file"`           // base64 of the zstd JSON lines published when the round settles
	Hours    []RoundHour        `json:"hours"`                 // the harvests its writing was made from: see harvest.go
	InitSHA  string             `json:"init_sha256,omitempty"` // optional authoritative initial weights sha256 from the examiner
}

func sharesEqual(a, b map[string]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, va := range a {
		vb, ok := b[k]
		if !ok || va != vb {
			return false
		}
	}
	return true
}

// apiExamPost: POST /api/v1/admin/rounds/{id}/exam, from the organisers. Posting the exact same
// exam (items digest and language shares) again is harmless; posting another is refused.
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
		len(p.Table) > examItemMax || (p.Snapshot != "" && !sha256Re.MatchString(p.Snapshot)) || (p.InitSHA != "" && !sha256Re.MatchString(p.InitSHA)) || p.Fresh < 0 || p.Fresh > len(p.Table) {
		return false, fail(400, "the exam does not add up")
	}
	if err := a.checkTable(p.Table, p.Shares); err != nil {
		return false, err
	}
	var already bool
	a.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if rd == nil || rd.Status != "open" {
			err = fail(409, "%s is not open", roundID(a.season.ID, idx))
		} else if rd.ExamSHA != "" && rd.ExamSHA != p.Digest {
			err = fail(409, "%s already has its exam", roundID(a.season.ID, idx))
		} else if rd.ExamSHA == p.Digest {
			if !sharesEqual(rd.Shares, p.Shares) {
				err = fail(409, "%s already has its exam with different language shares", roundID(a.season.ID, idx))
			} else {
				already = true
			}
		} else {
			_, err = a.checkHours(s, rd, p)
		}
	})
	if err != nil || already {
		return false, err
	}
	table, _ := json.Marshal(p.Table)
	blobSHA := examBlobSHA(p.Digest, file, public, table)
	for _, put := range []struct {
		name, kind string
		b          []byte
	}{{examItems, "application/zstd", file}, {examPublic, "application/zstd", public}, {examTable, "application/json", table}} {
		if err := a.st.blobs.Put(ctx, a.examDigestBlob(idx, blobSHA, put.name), put.b, put.kind); err != nil {
			slog.Error("keeping a round's exam", "round", idx, "err", err)
			return false, fail(503, "the file store is not answering; try again in a moment")
		}
	}
	err = a.st.update(ctx, func(tx *Tx) error {
		rd := tx.s.Rounds[idx]
		if rd == nil || rd.Status != "open" || (rd.ExamSHA != "" && rd.ExamSHA != p.Digest) {
			return fail(409, "%s changed while its exam was being kept; send it again", roundID(a.season.ID, idx))
		}
		if rd.ExamSHA == p.Digest {
			if !sharesEqual(rd.Shares, p.Shares) {
				return fail(409, "%s already has its exam with different language shares", roundID(a.season.ID, idx))
			}
			// The same exam and language shares, posted twice at once: the other post was first,
			// and the round's clock, its event and its grades' time are that one's.
			already = true
			return nil
		}
		late, err := a.checkHours(tx.s, rd, p)
		if err != nil {
			return err
		}
		if late {
			rd.Hours, rd.HoursLate = p.Hours, true
		}
		rd.ExamSHA, rd.ExamBlobSHA, rd.ExamAt, rd.ExamTable, rd.Shares, rd.Snapshot, rd.Fresh = p.Digest, blobSHA, now(), p.Table, p.Shares, p.Snapshot, p.Fresh
		if p.InitSHA != "" && rd.InitSHA == "" {
			rd.InitSHA = p.InitSHA
		}
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
	return err == nil && !already, err
}

// checkTable refuses a table with a part the season does not name, or one under a pillar the
// season does not put it under, or a language share that is not a number. A pillar with zero
// items is allowed (its share goes to the remaining pillars in its group, per RULES.md), while
// a pillar that is present with fewer than min_pillar_items or a declared group with zero items
// is refused. The weights are the season's; the examiner only says what each item is.
func (a *App) checkTable(table []ItemInfo, shares map[string]float64) error {
	counts := map[string]int{}
	groupOf := map[string]string{}
	for _, pl := range a.season.Exam.Pillars {
		groupOf[pl.Name] = pl.Group
	}
	groupCounts := map[string]int{}
	for i, it := range table {
		if it.Pillar == "" || it.Pillar != a.season.Exam.pillarOf(it.Part) || it.Part == "" || it.Lang == "" || it.Stratum == "" ||
			len(it.Part)+len(it.Lang)+len(it.Stratum) > 200 {
			return fail(400, "item %d of the table is not one the season scores", i)
		}
		counts[it.Pillar]++
		if g := groupOf[it.Pillar]; g != "" {
			groupCounts[g]++
		}
	}
	if minP := a.season.Exam.MinPillarItems; minP > 0 {
		for _, pl := range a.season.Exam.Pillars {
			if counts[pl.Name] > 0 && counts[pl.Name] < minP {
				return fail(400, "exam pillar %q has %d items, fewer than the season's minimum of %d", pl.Name, counts[pl.Name], minP)
			}
		}
	}
	if len(a.season.Exam.Groups) > 0 {
		groups := make([]string, 0, len(a.season.Exam.Groups))
		for g := range a.season.Exam.Groups {
			groups = append(groups, g)
		}
		sort.Strings(groups)
		for _, g := range groups {
			if a.season.Exam.Groups[g] > 0 && groupCounts[g] == 0 {
				return fail(400, "exam group %q has 0 items; every scored group must be represented", g)
			}
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

// cancelRound voids every open entry of a round whose exam could not be built, with nothing
// held against any entry (unattended=true), expires all outstanding claims on the round, and
// emits a public cancel event. Callers run settleRounds(tx) afterwards so the round settles
// cleanly with BaselineLoss == nil and unblocks the next draw.
func (a *App) cancelRound(tx *Tx, rd *Round, why string) {
	if rd == nil || rd.Status != "open" {
		return
	}
	why = clip(strings.TrimSpace(why), 300)
	if why == "" {
		why = "the round's exam could not be built; nothing is held against any entry"
	}
	for _, e := range tx.s.inRound[rd.Idx] {
		for _, run := range e.Runs {
			if !run.Dropped {
				run.Dropped = true
				if run.Reason == "" {
					run.Reason = why
				}
			}
			run.Owed = false
		}
		e.Loss, e.Gain, e.BySource, e.Detail = nil, 0, nil, nil
		a.voidEntry(tx, e, why, true)
	}
	for _, c := range tx.s.Claims {
		if c.RoundIdx == rd.Idx && c.Status == "claimed" {
			c.Status, c.EndedAt = "expired", now()
			tx.saveClaim(c)
		}
	}
	var dropBases []*Baseline
	for _, b := range tx.s.Baselines {
		if b.RoundIdx == rd.Idx {
			dropBases = append(dropBases, b)
		}
	}
	for _, b := range dropBases {
		tx.dropBaseline(b)
	}
	if len(rd.BaseItems) > 0 || rd.BaselineLoss != nil || rd.BaselineN != 0 {
		rd.BaseItems, rd.BaselineLoss, rd.BaselineN = nil, nil, 0
		tx.saveRound(rd)
	}
	tx.event("cancel", map[string]any{"round": roundID(a.season.ID, rd.Idx), "reason": why, "at": now()})
	if len(tx.s.inRound[rd.Idx]) == 0 {
		rd.Status, rd.SettledAt = "settled", now()
		tx.saveRound(rd)
	}
}

// apiRoundCancel: POST /api/v1/admin/rounds/{id}/cancel, from the organisers. Cancels an open
// round, voiding all entries as unattended so nothing counts against any curator and the next
// round can be drawn. Once an exam has been posted, requires {"force": true}.
func (a *App) apiRoundCancel(w http.ResponseWriter, r *http.Request) {
	_, _, body, err := a.signedBy(w, r, 1<<16, true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	idx, ok := a.parseRoundID(r.PathValue("id"))
	if !ok {
		writeErr(w, fail(404, "no such round"))
		return
	}
	var req struct {
		Reason string `json:"reason"`
		Force  bool   `json:"force"`
	}
	if trimmed := bytes.TrimSpace(body); len(trimmed) > 0 {
		if trimmed[0] != '{' {
			writeErr(w, fail(400, "bad cancel request: expected JSON object"))
			return
		}
		dec := json.NewDecoder(bytes.NewReader(trimmed))
		if err := dec.Decode(&req); err != nil {
			writeErr(w, fail(400, "bad cancel request: %v", err))
			return
		}
		var extra any
		if err := dec.Decode(&extra); err != io.EOF {
			writeErr(w, fail(400, "bad cancel request: unexpected trailing data"))
			return
		}
	}
	err = a.st.update(r.Context(), func(tx *Tx) error {
		rd := tx.s.Rounds[idx]
		if rd == nil || rd.Status != "open" {
			return fail(409, "%s is not open", roundID(a.season.ID, idx))
		}
		if examPosted(rd) && !req.Force {
			return fail(409, "%s already has its exam; pass {\"force\": true} to cancel an examined round", roundID(a.season.ID, idx))
		}
		a.cancelRound(tx, rd, req.Reason)
		a.settleRounds(tx)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"round": roundID(a.season.ID, idx), "status": "settled", "cancelled": true})
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
	var examSHA, blobSHA string
	a.st.view(func(s *State) {
		rd := s.Rounds[idx]
		posted = rd != nil && examPosted(rd)
		settled = posted && rd.Status == "settled"
		if posted {
			examSHA = rd.ExamSHA
			blobSHA = rd.ExamBlobSHA
		}
	})
	if !posted {
		writeErr(w, fail(404, "%s has no exam yet", roundID(a.season.ID, idx)))
		return
	}
	name := examPublic
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
		name = examItems
	}
	var (
		f    io.ReadCloser
		size int64
		err  error = errNoBlob
	)
	if blobSHA != "" {
		f, size, err = a.st.blobs.Open(r.Context(), a.examDigestBlob(idx, blobSHA, name))
	}
	if errors.Is(err, errNoBlob) {
		// Backwards compatibility for rounds stored under items_digest-only or unkeyed exam blobs.
		f, size, err = a.st.blobs.Open(r.Context(), a.examDigestBlob(idx, examSHA, name))
	}
	if errors.Is(err, errNoBlob) {
		f, size, err = a.st.blobs.Open(r.Context(), a.examBlob(idx, name))
	}
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

// baselineMatchesInit reports whether a stored baseline matches the round's agreed initial
// student weights digest (always true before the round locks an InitSHA, or on seasons without
// a pinned student model).
func (a *App) baselineMatchesInit(rd *Round, b *Baseline) bool {
	if rd == nil || b == nil || rd.InitSHA == "" {
		return true
	}
	if b.InitSHA == "" {
		return len(a.season.Student) == 0
	}
	return b.InitSHA == rd.InitSHA
}

// roundBaselines are the round's baselines that may make its own, in the order of their
// accounts: measured on its corpus, exam and agreed initial weights, by an account established
// when it measured (or the organisers'). Whether a measurement counts was settled when it
// arrived: standing earned as the round settles must not move the round's own baseline.
func (a *App) roundBaselines(s *State, rd *Round) []*Baseline {
	var out []*Baseline
	for _, b := range s.Baselines {
		acct := s.Accounts[b.AccountID]
		if b.RoundIdx == rd.Idx && (b.Established || (acct != nil && acct.Anchor)) && b.CorpusSHA == rd.CorpusSHA && b.ExamSHA == rd.ExamSHA &&
			a.baselineMatchesInit(rd, b) && len(b.Items) == len(rd.ExamTable) {
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

// baselinesAgree reports whether two baselines for the same round are mutually within baseline
// tolerance on the exam's aggregate (and share InitSHA when the season pins initial weights).
func (a *App) baselinesAgree(rd *Round, x, y *Baseline) bool {
	if x == nil || y == nil || len(x.Items) != len(rd.ExamTable) || len(y.Items) != len(rd.ExamTable) {
		return false
	}
	if len(a.season.Student) > 0 && x.InitSHA != "" && y.InitSHA != "" && x.InitSHA != y.InitSHA {
		return false
	}
	dev, _, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(x.Items, y.Items), rd.Shares)
	return err == nil && within(dev, a.tolerance().Baseline)
}

// baselinesPairAgree reports whether two baselines mutually agree symmetrically within baseline
// tolerance. Checking both directions guarantees an undirected agreement graph even in the
// presence of floating-point rounding.
func (a *App) baselinesPairAgree(rd *Round, x, y *Baseline) bool {
	return a.baselinesAgree(rd, x, y) && a.baselinesAgree(rd, y, x)
}

// allBaselinesAgree reports whether every pair of baselines in bs mutually agrees within baseline
// tolerance.
func (a *App) allBaselinesAgree(rd *Round, bs []*Baseline) bool {
	for i := 0; i < len(bs); i++ {
		for k := i + 1; k < len(bs); k++ {
			if !a.baselinesPairAgree(rd, bs[i], bs[k]) {
				return false
			}
		}
	}
	return true
}

// baselineAdj builds the symmetric pairwise-agreement adjacency matrix for bs.
func (a *App) baselineAdj(rd *Round, bs []*Baseline) [][]bool {
	n := len(bs)
	adj := make([][]bool, n)
	for i := range n {
		adj[i] = make([]bool, n)
		adj[i][i] = true
	}
	for i := 0; i < n; i++ {
		for k := i + 1; k < n; k++ {
			if a.baselinesPairAgree(rd, bs[i], bs[k]) {
				adj[i][k] = true
				adj[k][i] = true
			}
		}
	}
	return adj
}

// largestBaselineClique finds the largest pairwise-mutually-agreeing clique (size >= 2) in bs
// using exact Bron-Kerbosch enumeration with Tomita pivoting, and sets tied=true whenever two or
// more distinct maximal cliques tie for that maximum size.
func (a *App) largestBaselineClique(rd *Round, bs []*Baseline) (best []*Baseline, tied bool) {
	n := len(bs)
	if n < 2 {
		return nil, false
	}
	adj := a.baselineAdj(rd, bs)
	allAgree := true
	for i := 0; i < n && allAgree; i++ {
		for k := i + 1; k < n; k++ {
			if !adj[i][k] {
				allAgree = false
				break
			}
		}
	}
	if allAgree {
		out := make([]*Baseline, n)
		copy(out, bs)
		return out, false
	}
	var bestIdx []int
	var bk func(r, p, x []int)
	bk = func(r, p, x []int) {
		if len(p) == 0 && len(x) == 0 {
			if len(r) < 2 {
				return
			}
			switch {
			case len(r) > len(bestIdx):
				bestIdx = make([]int, len(r))
				copy(bestIdx, r)
				tied = false
			case len(r) == len(bestIdx):
				tied = true
			}
			return
		}
		// Upper-bound prune strictly below max(2, len(bestIdx)) so equal-size maximal cliques are
		// still visited and mark tied=true.
		if len(r)+len(p) < max(2, len(bestIdx)) {
			return
		}
		pivot, bestDeg := -1, -1
		for _, u := range p {
			deg := 0
			for _, v := range p {
				if u != v && adj[u][v] {
					deg++
				}
			}
			if deg > bestDeg {
				bestDeg, pivot = deg, u
			}
		}
		for _, u := range x {
			deg := 0
			for _, v := range p {
				if adj[u][v] {
					deg++
				}
			}
			if deg > bestDeg {
				bestDeg, pivot = deg, u
			}
		}
		pCur := append([]int(nil), p...)
		xCur := append([]int(nil), x...)
		for _, v := range p {
			if pivot >= 0 && v != pivot && adj[pivot][v] {
				continue
			}
			var nextP, nextX []int
			for _, w := range pCur {
				if w != v && adj[v][w] {
					nextP = append(nextP, w)
				}
			}
			for _, w := range xCur {
				if adj[v][w] {
					nextX = append(nextX, w)
				}
			}
			nextR := make([]int, len(r)+1)
			copy(nextR, r)
			nextR[len(r)] = v
			bk(nextR, nextP, nextX)
			for i, w := range pCur {
				if w == v {
					pCur = append(pCur[:i], pCur[i+1:]...)
					break
				}
			}
			xCur = append(xCur, v)
		}
	}
	initP := make([]int, n)
	for i := range n {
		initP[i] = i
	}
	bk(nil, initP, nil)
	if len(bestIdx) < 2 {
		return nil, false
	}
	sort.Ints(bestIdx)
	best = make([]*Baseline, len(bestIdx))
	for i, idx := range bestIdx {
		best[i] = bs[idx]
	}
	return best, tied
}

// hasCompetingBaselineClique reports whether bs contains any maximal clique not contained in
// winning whose size is at least minSize. Used only when median fixed-point pruning shrinks the
// winning clique (possible under multi-item MAD clipping) so a shrunken group cannot beat or tie
// another pairwise-agreeing clique of at least the same size.
func (a *App) hasCompetingBaselineClique(rd *Round, bs, winning []*Baseline, minSize int) bool {
	inWin := make(map[int64]bool, len(winning))
	for _, b := range winning {
		inWin[b.AccountID] = true
	}
	n := len(bs)
	adj := a.baselineAdj(rd, bs)
	found := false
	var bk func(r, p, x []int)
	bk = func(r, p, x []int) {
		if found {
			return
		}
		if len(p) == 0 && len(x) == 0 {
			if len(r) >= minSize {
				for _, idx := range r {
					if !inWin[bs[idx].AccountID] {
						found = true
						return
					}
				}
			}
			return
		}
		if len(r)+len(p) < minSize {
			return
		}
		pivot, bestDeg := -1, -1
		for _, u := range p {
			deg := 0
			for _, v := range p {
				if u != v && adj[u][v] {
					deg++
				}
			}
			if deg > bestDeg {
				bestDeg, pivot = deg, u
			}
		}
		for _, u := range x {
			deg := 0
			for _, v := range p {
				if adj[u][v] {
					deg++
				}
			}
			if deg > bestDeg {
				bestDeg, pivot = deg, u
			}
		}
		pCur := append([]int(nil), p...)
		xCur := append([]int(nil), x...)
		for _, v := range p {
			if found {
				return
			}
			if pivot >= 0 && v != pivot && adj[pivot][v] {
				continue
			}
			var nextP, nextX []int
			for _, w := range pCur {
				if w != v && adj[v][w] {
					nextP = append(nextP, w)
				}
			}
			for _, w := range xCur {
				if adj[v][w] {
					nextX = append(nextX, w)
				}
			}
			nextR := make([]int, len(r)+1)
			copy(nextR, r)
			nextR[len(r)] = v
			bk(nextR, nextP, nextX)
			for i, w := range pCur {
				if w == v {
					pCur = append(pCur[:i], pCur[i+1:]...)
					break
				}
			}
			xCur = append(xCur, v)
		}
	}
	initP := make([]int, n)
	for i := range n {
		initP[i] = i
	}
	bk(nil, initP, nil)
	return found
}

// pruneBaselinesToMedian iteratively drops baselines outside tolerance of the current set's
// item-wise median until fixed point or fewer than two baselines remain.
func (a *App) pruneBaselinesToMedian(rd *Round, bs []*Baseline) []*Baseline {
	tol := a.tolerance().Baseline
	kept := bs
	for len(kept) >= 2 {
		mid := medianItems(itemsOf(kept))
		var next []*Baseline
		for _, b := range kept {
			if dev, _, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(b.Items, mid), rd.Shares); err == nil && within(dev, tol) {
				next = append(next, b)
			}
		}
		if len(next) == len(kept) {
			break
		}
		kept = next
	}
	return kept
}

// baselinesKept are the round's baselines that make its own. While fewer than three are in,
// all of them: whichever came first is no more the standard than the next. From three on, there
// must be a unique strictly largest pairwise-agreeing clique of size >= 2 across all established
// baselines (so neither a 2-2 split nor a bridge baseline spanning two disagreeing pairs can settle
// a tie, and a genuine agreeing majority is never voided by wider disagreeing neighborhoods). The
// accepted baseline is computed from that winning clique (pruned to its own median fixed point so
// every kept baseline also agrees with the round's item-wise median). If no unique agreeing
// consensus exists, all baselines are kept so baselineDisputed reports the dispute.
func (a *App) baselinesKept(s *State, rd *Round) []*Baseline {
	all := a.roundBaselines(s, rd)
	if len(all) < baselineQuorum {
		return all
	}
	best, tied := a.largestBaselineClique(rd, all)
	if len(best) < 2 || tied {
		return all
	}
	kept := a.pruneBaselinesToMedian(rd, best)
	if len(kept) >= 2 && a.allBaselinesAgree(rd, kept) {
		if len(kept) == len(best) || !a.hasCompetingBaselineClique(rd, all, best, len(kept)) {
			return kept
		}
	}
	return all
}

// baselinesLeftOut is whose baseline of a round is the odd one out: one that may make the
// round's (roundBaselines) and that the round does not keep (baselinesKept).
func (a *App) baselinesLeftOut(s *State, rd *Round) map[int64]bool {
	out := map[int64]bool{}
	for _, b := range a.roundBaselines(s, rd) {
		out[b.AccountID] = true
	}
	for _, b := range a.baselinesKept(s, rd) {
		delete(out, b.AccountID)
	}
	return out
}

// oddOut reports whether a grade is by a machine whose baseline of its round is the odd one
// out. What such a machine made of a script is no more to be relied on than what it made of
// the unedited corpus: its grade decides nothing, as if it had been turned away when it came.
func (j *judge) oddOut(r *Result) bool {
	out, ok := j.left[r.RoundIdx]
	if !ok {
		if rd := j.s.Rounds[r.RoundIdx]; rd != nil {
			out = j.a.baselinesLeftOut(j.s, rd)
		}
		j.left[r.RoundIdx] = out
	}
	return out[r.AccountID]
}

// baselineDev is how far a grader's baseline is from the median of the round's established
// baselines, its own among them (the kept others' and this one, whether or not its account is
// established), on the exam's score. ok is false while fewer than two others are in OR while the
// round's existing baselines are still disputed: nobody is turned away when no established
// baseline consensus exists yet (otherwise a tie-breaking grader repeating one of the disputed
// baselines could be rejected with 409 before it can resolve the dispute).
func (a *App) baselineDev(s *State, rd *Round, acctID int64, items []float64) (dev float64, ok bool) {
	if a.baselineDisputed(s, rd) {
		return 0, false
	}
	var rows [][]float64
	for _, b := range a.baselinesKept(s, rd) {
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

// baselineConsensusState reports whether the round's kept established baseline cohort is already
// irrevocable AND returns the set of established baseline voters provably guaranteed to remain
// inside the kept cohort (not pushed into oddOut):
//   - never locks before rd.InitSHA locks (on pinned-student seasons), while baselineDisputed is
//     true, or on fewer than 2 corroborated baselines (len(simKept) < 2);
//   - counts ALL live established grade claims on rd (including c.Baseline==false, because clients
//     with a cached local baseline may still attach p.Baseline and enter s.Baselines);
//   - defers locking whenever any other live established grade claim remains on rd (k > 0),
//     because ExamPlan.ScoreItems uses MAD-based outlier clipping (which skips clipping when
//     MAD == 0 and is therefore not monotone in item diffs), so any additional baseline can alter
//     which existing voters stay within tolerance or fall into oddOut.
func (a *App) baselineConsensusState(s *State, rd *Round, acct *Account, est bool, incoming *Baseline) (locked bool, stableVoters map[int64]bool, liveOther map[int64]bool) {
	if rd == nil {
		return false, nil, nil
	}
	if len(a.season.Student) > 0 && rd.InitSHA == "" {
		return false, nil, nil
	}
	if a.baselineDisputed(s, rd) {
		return false, nil, nil
	}
	isEst := est || (acct != nil && acct.Anchor)
	sim := *s
	if isEst && acct != nil && incoming != nil {
		sim.Baselines = maps.Clone(s.Baselines)
		if sim.Baselines == nil {
			sim.Baselines = map[string]*Baseline{}
		}
		sim.Baselines[baselineKey(rd.Idx, acct.ID)] = incoming
	}
	if a.baselineDisputed(&sim, rd) {
		return false, nil, nil
	}
	simAll := a.roundBaselines(&sim, rd)
	simKept := a.baselinesKept(&sim, rd)
	if len(simKept) < baselineQuorum-1 {
		return false, nil, nil
	}
	inSimAll := make(map[int64]bool, len(simAll))
	for _, b := range simAll {
		inSimAll[b.AccountID] = true
	}
	inKept := make(map[int64]bool, len(simKept))
	for _, b := range simKept {
		inKept[b.AccountID] = true
	}
	otherLive := map[int64]bool{}
	t := now()
	for _, c := range s.Claims {
		if c.RoundIdx != rd.Idx || c.Kind != "grade" || !c.live(t) {
			continue
		}
		if (acct != nil && c.AccountID == acct.ID) || inSimAll[c.AccountID] {
			continue
		}
		if ca := s.Accounts[c.AccountID]; ca != nil && (a.established(ca) || ca.Anchor) {
			otherLive[c.AccountID] = true
		}
	}
	if len(otherLive) > 0 {
		return false, nil, otherLive
	}
	return true, inKept, otherLive
}

// baselineConsensusLocked reports whether the round's kept established baseline cohort is already
// irrevocable under any possible baseline submissions from remaining live established claims.
func (a *App) baselineConsensusLocked(s *State, rd *Round, acct *Account, est bool, incoming *Baseline) bool {
	locked, _, _ := a.baselineConsensusState(s, rd, acct, est, incoming)
	return locked
}

// baselineAdmissionRefuses reports whether an incoming baseline measurement must be refused at
// admission with HTTP 409 because it deviates beyond tolerance from an already-locked established
// baseline consensus that neither this report nor any outstanding live baseline-requested claim
// can tie or overturn.
func (a *App) baselineAdmissionRefuses(s *State, rd *Round, acct *Account, est bool, incoming *Baseline) (dev float64, refuse bool) {
	if acct == nil || incoming == nil {
		return 0, false
	}
	dev, ok := a.baselineDev(s, rd, acct.ID, incoming.Items)
	if !ok || within(dev, a.tolerance().Baseline) {
		return dev, false
	}
	if !a.baselineConsensusLocked(s, rd, acct, est, incoming) {
		return dev, false
	}
	if est || acct.Anchor {
		sim := *s
		sim.Baselines = maps.Clone(s.Baselines)
		if sim.Baselines == nil {
			sim.Baselines = map[string]*Baseline{}
		}
		sim.Baselines[baselineKey(rd.Idx, acct.ID)] = incoming
		for _, b := range a.baselinesKept(&sim, rd) {
			if b.AccountID == acct.ID {
				return dev, false
			}
		}
	}
	return dev, true
}

// baselineAgrees says whether a grader's baseline is within the tolerance of the round's: its
// score against the round's baseline, on the exam's aggregate, and matching the agreed InitSHA.
func (a *App) baselineAgrees(rd *Round, b *Baseline) bool {
	if b == nil || len(rd.BaseItems) == 0 || len(b.Items) != len(rd.BaseItems) || b.ExamSHA != rd.ExamSHA || !a.baselineMatchesInit(rd, b) {
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
	var agreeing []*Baseline
	if !j.a.baselineDisputed(j.s, rd) {
		for _, b := range j.a.baselinesKept(j.s, rd) {
			if j.a.baselineAgrees(rd, b) {
				agreeing = append(agreeing, b)
			}
		}
	}
	// Before rd.InitSHA locks, baselines that report different InitSHA values cannot all survive
	// once consensus locks; count only the largest agreeing InitSHA cohort so a replacement
	// baseline is requested immediately rather than letting conflicting baselines fill PerRound.
	if rd.InitSHA == "" && len(agreeing) > 1 {
		byInit := map[string][]*Baseline{}
		for _, b := range agreeing {
			if b.InitSHA != "" {
				byInit[b.InitSHA] = append(byInit[b.InitSHA], b)
			}
		}
		if len(byInit) > 1 {
			bestSHA := ""
			var bestCohort []*Baseline
			for sha, cohort := range byInit {
				if len(cohort) > len(bestCohort) || (len(cohort) == len(bestCohort) && (bestSHA == "" || sha < bestSHA)) {
					bestSHA, bestCohort = sha, cohort
				}
			}
			agreeing = bestCohort
		}
	}
	for _, b := range agreeing {
		who[b.AccountID] = true
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

// baselineDisputed reports whether a round has two or more established baselines whose
// measurements disagree beyond tolerance, or (on a season with a pinned student model before
// rd.InitSHA locks) come from conflicting initial student weights. With two kept baselines, they
// are compared directly against each other (rather than against their midpoint, which would double
// the allowed spread); with three or more, at least two must agree within tolerance of the kept
// median. A round with a disputed baseline must request another independent baseline rather than
// settling on an uncorroborated midpoint, and closes unrated if the disagreement cannot be resolved.
func (a *App) baselineDisputed(s *State, rd *Round) bool {
	all := a.roundBaselines(s, rd)
	if len(all) < 2 {
		return false
	}
	if len(a.season.Student) > 0 && rd.InitSHA == "" {
		byInit := map[string][]*Baseline{}
		for _, b := range all {
			if b.InitSHA != "" {
				byInit[b.InitSHA] = append(byInit[b.InitSHA], b)
			}
		}
		if len(byInit) != 1 {
			return true
		}
		for _, cohort := range byInit {
			if len(cohort) < 2 {
				return true
			}
		}
	}
	if len(all) >= baselineQuorum {
		if best, tied := a.largestBaselineClique(rd, all); len(best) < 2 || tied {
			return true
		}
	}
	kept := a.baselinesKept(s, rd)
	if len(kept) < 2 {
		return true
	}
	if !a.allBaselinesAgree(rd, kept) {
		return true
	}
	if len(kept) == 2 {
		return false
	}
	tol := a.tolerance().Baseline
	mid := medianItems(itemsOf(kept))
	for _, b := range kept {
		if dev, _, err := a.season.Exam.ScoreItems(rd.ExamTable, diffItems(b.Items, mid), rd.Shares); err != nil || !within(dev, tol) {
			return true
		}
	}
	return false
}

// baselineTieLive reports whether an established account without a baseline for this round
// currently holds a live grade job that asks for one.
func (j *judge) baselineTieLive(rd *Round) bool {
	for _, c := range j.s.Claims {
		if c.RoundIdx == rd.Idx && c.Kind == "grade" && c.Baseline && c.live(j.t) && j.est[c.AccountID] &&
			j.s.Baselines[baselineKey(rd.Idx, c.AccountID)] == nil {
			return true
		}
	}
	return false
}

// baselineComing reports whether a round has a baseline, or somebody at work on it was asked
// for one.
func (j *judge) baselineComing(round int) bool {
	rd := j.s.Rounds[round]
	return rd != nil && len(j.measuring(rd)) > 0
}

// baselinesWanted is how many more machines a round would ask for its baseline, as it stands.
func (j *judge) baselinesWanted(rd *Round) int {
	if j.a.baselineDisputed(j.s, rd) {
		if j.baselineTieLive(rd) {
			return 0
		}
		return 1
	}
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

// baselineInitSHA returns the single agreed initial student weights hash behind the round's kept
// baselines (always "", true on seasons without a pinned student model). Before rd.InitSHA locks,
// returns ("", false) when no kept baseline exists, any kept baseline lacks InitSHA, or kept
// baselines disagree on InitSHA.
func (a *App) baselineInitSHA(s *State, rd *Round) (string, bool) {
	if len(a.season.Student) == 0 {
		return "", true
	}
	if rd == nil {
		return "", false
	}
	if rd.InitSHA != "" {
		return rd.InitSHA, true
	}
	kept := a.baselinesKept(s, rd)
	if len(kept) == 0 {
		return "", false
	}
	sha := kept[0].InitSHA
	if sha == "" {
		return "", false
	}
	for _, b := range kept[1:] {
		if b.InitSHA != sha {
			return "", false
		}
	}
	return sha, true
}

// refreshScores sets the round's baseline to the median of the baselines it keeps
// (baselinesKept), item by item, and every grade's score to the exam's aggregate of its item
// scores less that.
func (a *App) refreshScores(tx *Tx, rd *Round) {
	s := tx.s
	hasOpen := false
	for _, e := range s.inRound[rd.Idx] {
		if e.Status == "open" {
			hasOpen = true
			break
		}
	}
	kept := a.baselinesKept(s, rd)
	if len(kept) == 0 || (!hasOpen && a.baselineDisputed(s, rd)) {
		if len(rd.BaseItems) > 0 || rd.BaselineLoss != nil || rd.BaselineN != 0 {
			rd.BaseItems, rd.BaselineLoss, rd.BaselineN = nil, nil, 0
			tx.saveRound(rd)
		}
		for _, e := range s.inRound[rd.Idx] {
			for _, r := range s.results(rd.Idx, e.EditorID) {
				if r.Score != nil || len(r.BySource) > 0 {
					r.Score, r.BySource = nil, nil
					if r.Valid && len(r.Items) > 0 {
						mean := 0.0
						for _, v := range r.Items {
							mean += v
						}
						r.Loss = places5(-mean / float64(len(r.Items)))
					}
					tx.saveResult(r)
				}
			}
		}
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
	out := a.baselinesLeftOut(s, rd)
	rd.BaselineN = 0
	if !a.baselineDisputed(s, rd) {
		for _, b := range s.Baselines {
			if b.RoundIdx == rd.Idx && !out[b.AccountID] && a.baselineAgrees(rd, b) {
				rd.BaselineN++
			}
		}
	}
	tx.saveRound(rd)
	baseInit, baseInitOK := a.baselineInitSHA(s, rd)
	for _, e := range s.inRound[rd.Idx] {
		for _, r := range s.results(rd.Idx, e.EditorID) {
			if !r.Valid || len(r.Items) != len(base) {
				continue
			}
			// Never score a grade against a baseline trained from different initial weights.
			if len(a.season.Student) > 0 && (!baseInitOK || r.InitSHA == "" || r.InitSHA != baseInit) {
				if r.Score != nil || len(r.BySource) > 0 {
					r.Score, r.BySource = nil, nil
					rmean := 0.0
					for _, v := range r.Items {
						rmean += v
					}
					r.Loss = places5(-rmean / float64(len(r.Items)))
					tx.saveResult(r)
				}
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
			if len(rd.BaseItems) > 0 {
				for _, b := range a.baselinesKept(s, rd) {
					out.BaselineOf = append(out.BaselineOf, handle(b.AccountID))
				}
				sort.Strings(out.BaselineOf)
			}
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
