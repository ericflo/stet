package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
)

// --- the organisers' own students, farmed out ---------------------------------------------------
//
// The organisers train students of their own on settled rounds: every entry's and the unedited
// corpus's, at the season's scales or at one it does not run (the scale check, and the finals),
// each trained on what its script keeps and sat on the round's exam. The organisers' machine
// posts a round's students as a batch of tasks. Any of the organisers' machines, rented ones
// included, takes a task when it has no official work (stet verify) and sends back how its
// student did on every item. The organisers' machine then adds them up. Nothing here touches a
// grade, and a task is handed out only to a machine with nothing official to do.

const (
	farmLease       = 45 * 60 // seconds a machine holds a task before it may be handed to another, unless its batch says
	farmLeaseMax    = 24 * 3600
	farmShort       = 3600     // a task held for longer than this goes only to a machine that asks for long ones
	farmTries       = 3        // machines a task is handed to before it is left undone
	farmItemsMax    = 16 << 20 // bytes of a batch's items, compressed
	farmBatchesKept = 60       // the oldest are forgotten past this many
)

// FarmTask is one student a batch needs.
type FarmTask struct {
	Script string    `json:"script"` // sha256 of the script, or "unedited"
	Holder int64     `json:"holder,omitempty"`
	Until  int64     `json:"until,omitempty"`
	Tries  int       `json:"tries"`
	Done   *FarmDone `json:"done,omitempty"`
}

// FarmDone says who trained a task's student, and what its training kept and took. How it
// did on each of the batch's items is kept in the file store, beside them.
type FarmDone struct {
	By       string  `json:"by"`
	At       int64   `json:"at"`
	Hardware string  `json:"hardware,omitempty"`
	Valid    bool    `json:"valid"`
	Reason   string  `json:"reason,omitempty"`
	Kept     float64 `json:"kept_doc_frac,omitempty"`
	Seconds  float64 `json:"train_seconds,omitempty"`
}

// FarmBatch is a round's students for one set of items, at one scale.
type FarmBatch struct {
	ID       string      `json:"id"`
	RoundIdx int         `json:"round"`
	Scale    string      `json:"scale"`
	Digest   string      `json:"items_digest"`
	Items    int         `json:"items"`
	Version  int         `json:"generator_version"`
	At       int64       `json:"at"`
	By       string      `json:"by"`
	Tasks    []*FarmTask `json:"tasks"`
	// A scale the season does not name (the scale check, and finals): its model and budget,
	// for the machines to build it from. And how long a task is held, for a bigger student.
	Spec  json.RawMessage `json:"scale_spec,omitempty"`
	Lease int64           `json:"lease,omitempty"`
}

func (b *FarmBatch) lease() int64 {
	if b.Lease > 0 {
		return b.Lease
	}
	return farmLease
}

func farmBatchKey(id string) string    { return "farm/" + id }
func farmScriptBlob(sha string) string { return "farm/scripts/" + sha + ".zst" }
func farmItemsBlob(id string) string   { return "farm/" + id + "/items.jsonl.zst" }
func farmScoresBlob(id string, i int) string {
	return fmt.Sprintf("farm/%s/%d.json", id, i)
}

var (
	farmBatchRe  = regexp.MustCompile(`^[0-9]{8}-[0-9a-f]{12}$`)
	scaleRe      = regexp.MustCompile(`^[a-z][a-z0-9-]{0,23}$`)
	farmScriptRe = regexp.MustCompile(`^(unedited|[0-9a-f]{64})(#[2-9])?$`) // "#2": the same student, trained again
)

func (tx *Tx) saveFarmBatch(b *FarmBatch) {
	tx.s.Farm[b.ID] = b
	tx.put(farmBatchKey(b.ID), b)
}

// farmBatchID names a batch by its round, items, scale and model.
func farmBatchID(round int, digest, scale string, spec json.RawMessage) string {
	h := sha256.Sum256([]byte(digest + "\x00" + scale + "\x00" + string(spec)))
	return fmt.Sprintf("%08d-%s", round, hex.EncodeToString(h[:])[:12])
}

// bounded is a number a result may hold: finite, and not absurd.
func bounded(v float64) bool { return finite(v) && math.Abs(v) < 1e6 }

// open says whether a task may be handed to a machine now.
func (t *FarmTask) open(at int64) bool {
	return t.Done == nil && t.Tries < farmTries && (t.Holder == 0 || t.Until <= at)
}

// farmWaiting counts the tasks a machine could be handed now: for stet-runpod, which rents
// for them when there is nothing official to do.
func farmWaiting(s *State, at int64) (waiting, underWay, long int) {
	for _, b := range s.Farm {
		for _, t := range b.Tasks {
			switch {
			case t.open(at) && b.lease() > farmShort:
				long++
			case t.open(at):
				waiting++
			case t.Done == nil && t.Holder != 0 && t.Until > at:
				underWay++
			}
		}
	}
	return
}

func (s *State) farmBatchesByAge() []*FarmBatch {
	out := make([]*FarmBatch, 0, len(s.Farm))
	for _, b := range s.Farm {
		out = append(out, b)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].At != out[j].At {
			return out[i].At < out[j].At
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// apiFarmBatch takes a round's batch from the organisers' own machine: the items its students
// sit (zstd JSON lines, base64) and the scripts whose students it wants. Posting the same
// items again adds any scripts not yet in the batch, and is otherwise a no-op.
func (a *App) apiFarmBatch(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signedBy(w, r, 4*farmItemsMax/3+(1<<20), true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	var req struct {
		Round   int             `json:"round"`
		Scale   string          `json:"scale"`
		Digest  string          `json:"items_digest"`
		Items   int             `json:"items"`
		Version int             `json:"generator_version"`
		Scripts []string        `json:"scripts"`
		File    string          `json:"items_file"` // base64 of the zstd JSON lines
		Spec    json.RawMessage `json:"scale_spec"`
		Lease   int64           `json:"lease"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, fail(400, "not a batch: %v", err))
		return
	}
	file, err := base64.StdEncoding.DecodeString(req.File)
	if err != nil || len(file) == 0 || len(file) > farmItemsMax || !sha256Re.MatchString(req.Digest) || req.Items <= 0 ||
		req.Items > 100_000 || !scaleRe.MatchString(req.Scale) || len(req.Scripts) == 0 || len(req.Scripts) > 500 ||
		len(req.Spec) > 2048 || req.Lease < 0 || req.Lease > farmLeaseMax {
		writeErr(w, fail(400, "the batch does not add up"))
		return
	}
	if len(req.Spec) > 0 {
		var spec map[string]float64 // a scale is a handful of numbers
		if json.Unmarshal(req.Spec, &spec) != nil || len(spec) == 0 {
			writeErr(w, fail(400, "a scale is an object of numbers"))
			return
		}
		req.Spec, _ = json.Marshal(spec) // one spelling, whatever the client sent
	}
	seen := map[string]bool{}
	for _, sc := range req.Scripts {
		if seen[sc] || !farmScriptRe.MatchString(sc) {
			writeErr(w, fail(400, "a script is named twice, or is not a sha256"))
			return
		}
		seen[sc] = true
	}
	id := farmBatchID(req.Round, req.Digest, req.Scale, req.Spec)
	var settled bool
	a.st.view(func(s *State) { rd := s.Rounds[req.Round]; settled = rd != nil && rd.Status == "settled" })
	if !settled {
		writeErr(w, fail(409, "round %d has not settled", req.Round))
		return
	}
	if err := a.st.blobs.Put(r.Context(), farmItemsBlob(id), file, "application/zstd"); err != nil {
		slog.Error("keeping a farm batch's items", "batch", id, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	var out *FarmBatch
	added := 0
	err = a.st.update(r.Context(), func(tx *Tx) error {
		b := tx.s.Farm[id]
		if b == nil {
			b = &FarmBatch{ID: id, RoundIdx: req.Round, Scale: req.Scale, Digest: req.Digest, Items: req.Items, Version: req.Version, At: now(), By: acct.Handle,
				Spec: req.Spec, Lease: req.Lease}
		}
		have := map[string]bool{}
		for _, t := range b.Tasks {
			have[t.Script] = true
		}
		for _, sc := range req.Scripts {
			if !have[sc] {
				b.Tasks = append(b.Tasks, &FarmTask{Script: sc})
				added++
			}
		}
		tx.saveFarmBatch(b)
		// Forget the oldest past what is kept; their results have been added up long since.
		for _, old := range tx.s.farmBatchesByAge() {
			if len(tx.s.Farm) <= farmBatchesKept {
				break
			}
			if old.ID != id {
				delete(tx.s.Farm, old.ID)
				tx.put(farmBatchKey(old.ID), nil)
			}
		}
		out = b
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"batch": out.ID, "tasks": len(out.Tasks), "added": added})
}

// apiFarmBatchGet shows a batch to the organisers: every task, and how each student did on
// every item.
func (a *App) apiFarmBatchGet(w http.ResponseWriter, r *http.Request) {
	if _, _, _, err := a.signedBy(w, r, 1<<10, true, organisers); err != nil {
		writeErr(w, err)
		return
	}
	id := r.PathValue("id")
	var b FarmBatch
	var tasks []FarmTask
	found := false
	a.st.view(func(s *State) {
		if x := s.Farm[id]; x != nil {
			found, b = true, *x
			for _, t := range x.Tasks {
				tasks = append(tasks, *t)
			}
		}
	})
	if !found {
		writeErr(w, fail(404, "no such batch"))
		return
	}
	type taskOut struct {
		FarmTask
		Scores []float64 `json:"item_scores,omitempty"`
	}
	out := make([]taskOut, len(tasks))
	for i, t := range tasks {
		out[i].FarmTask = t
		if t.Done != nil && t.Done.Valid {
			raw, err := a.st.blobs.Read(r.Context(), farmScoresBlob(id, i), 8<<20)
			if err == nil {
				err = json.Unmarshal(raw, &out[i].Scores)
			}
			if err != nil {
				slog.Error("reading a farm task's scores", "batch", id, "task", i, "err", err)
				writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
				return
			}
		}
	}
	writeJSON(w, 200, map[string]any{"batch": b.ID, "round": roundID(a.season.ID, b.RoundIdx), "scale": b.Scale, "items_digest": b.Digest,
		"items": b.Items, "generator_version": b.Version, "scale_spec": b.Spec, "lease": b.lease(), "tasks": out})
}

// farmMachine admits the organisers' machines, rented ones included.
func (a *App) farmMachine(w http.ResponseWriter, r *http.Request, limit int64) (*Account, []byte, error) {
	acct, _, body, err := a.signed(w, r, limit, true)
	if err != nil {
		return nil, nil, err
	}
	if !acct.Anchor {
		return nil, nil, fail(403, "the organisers' students are trained by the organisers' machines only")
	}
	return acct, body, nil
}

func (a *App) farmTaskJSON(b *FarmBatch, i int) map[string]any {
	t := b.Tasks[i]
	return map[string]any{"batch": b.ID, "index": i, "round": roundID(a.season.ID, b.RoundIdx), "round_index": b.RoundIdx, "scale": b.Scale,
		"scale_spec": b.Spec, "script": t.Script, "items_digest": b.Digest, "items": b.Items, "generator_version": b.Version, "until": t.Until}
}

// apiFarmClaim hands an organisers' machine a student to train: the one it already holds, or
// the oldest batch's first open task it has time for. A machine says how long it will give one
// (max_minutes, an hour if it does not say), so that a bigger student, held for hours, goes only
// to a machine set aside for it and never keeps one from the official work.
func (a *App) apiFarmClaim(w http.ResponseWriter, r *http.Request) {
	acct, body, err := a.farmMachine(w, r, 1<<12)
	if err != nil {
		writeErr(w, err)
		return
	}
	var ask struct {
		MaxMinutes int64 `json:"max_minutes"`
	}
	json.Unmarshal(body, &ask)
	most := int64(farmShort)
	if ask.MaxMinutes > 0 {
		most = min(ask.MaxMinutes*60, farmLeaseMax)
	}
	var task map[string]any
	err = a.st.update(r.Context(), func(tx *Tx) error {
		at := now()
		var pick *FarmBatch
		pi := -1
		for _, b := range tx.s.farmBatchesByAge() {
			for i, t := range b.Tasks {
				if t.Done == nil && t.Holder == acct.ID && t.Until > at {
					task = a.farmTaskJSON(b, i) // one at a time: the same one again
					return nil
				}
				if pick == nil && t.open(at) && b.lease() <= most {
					pick, pi = b, i
				}
			}
		}
		if pick == nil {
			return nil
		}
		t := pick.Tasks[pi]
		t.Holder, t.Until = acct.ID, at+pick.lease()
		t.Tries++
		tx.saveFarmBatch(pick)
		task = a.farmTaskJSON(pick, pi)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"task": task})
}

// apiFarmItems serves a batch's items to the organisers' machines.
func (a *App) apiFarmItems(w http.ResponseWriter, r *http.Request) {
	if _, _, err := a.farmMachine(w, r, 1<<10); err != nil {
		writeErr(w, err)
		return
	}
	id := r.PathValue("id")
	if !farmBatchRe.MatchString(id) {
		writeErr(w, fail(404, "no such batch"))
		return
	}
	f, size, err := a.st.blobs.Open(r.Context(), farmItemsBlob(id))
	if errors.Is(err, errNoBlob) {
		writeErr(w, fail(404, "no such batch"))
		return
	}
	if err != nil {
		slog.Error("reading a farm batch's items", "batch", id, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	io.Copy(w, f)
}

// farmTaskOf finds the task a machine names, which must be its own.
func farmTaskOf(s *State, acct *Account, id string, i int) (*FarmBatch, *FarmTask, error) {
	b := s.Farm[id]
	if b == nil || i < 0 || i >= len(b.Tasks) {
		return nil, nil, fail(404, "no such task")
	}
	t := b.Tasks[i]
	if t.Done != nil {
		return nil, nil, fail(409, "that task is done")
	}
	if t.Holder != acct.ID {
		return nil, nil, fail(409, "that task is not this machine's")
	}
	return b, t, nil
}

// apiFarmResult takes how a task's student did. A task held past its time is still taken
// from the machine that held it, so long as no other machine has been handed it since.
func (a *App) apiFarmResult(w http.ResponseWriter, r *http.Request) {
	acct, body, err := a.farmMachine(w, r, 8<<20)
	if err != nil {
		writeErr(w, err)
		return
	}
	id := r.PathValue("id")
	i, _ := strconv.Atoi(r.PathValue("i"))
	var req struct {
		FarmDone
		Scores []float64 `json:"item_scores"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		writeErr(w, fail(400, "not a result: %v", err))
		return
	}
	var items int
	a.st.view(func(s *State) {
		var b *FarmBatch
		if b, _, err = farmTaskOf(s, acct, id, i); err == nil {
			items = b.Items
		}
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	d := req.FarmDone
	d.By, d.At = acct.Handle, now()
	d.Hardware, d.Reason = clip(d.Hardware, 200), clip(d.Reason, 500)
	if d.Valid {
		if len(req.Scores) != items || !bounded(d.Kept) || !bounded(d.Seconds) {
			writeErr(w, fail(400, "the result does not add up: %d scores for %d items", len(req.Scores), items))
			return
		}
		for _, v := range req.Scores {
			if !bounded(v) {
				writeErr(w, fail(400, "a score is not a number"))
				return
			}
		}
		raw, _ := json.Marshal(req.Scores)
		if err := a.st.blobs.Put(r.Context(), farmScoresBlob(id, i), raw, "application/json"); err != nil {
			slog.Error("keeping a farm task's scores", "batch", id, "task", i, "err", err)
			writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
			return
		}
	} else {
		d.Kept, d.Seconds = 0, 0
	}
	err = a.st.update(r.Context(), func(tx *Tx) error {
		b, t, err := farmTaskOf(tx.s, acct, id, i)
		if err != nil {
			return err
		}
		t.Done, t.Holder, t.Until = &d, 0, 0
		tx.saveFarmBatch(b)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

// apiFarmRelease hands a task back, for another machine to try.
func (a *App) apiFarmRelease(w http.ResponseWriter, r *http.Request) {
	acct, _, err := a.farmMachine(w, r, 1<<12)
	if err != nil {
		writeErr(w, err)
		return
	}
	id := r.PathValue("id")
	i, _ := strconv.Atoi(r.PathValue("i"))
	err = a.st.update(r.Context(), func(tx *Tx) error {
		b, t, err := farmTaskOf(tx.s, acct, id, i)
		if err != nil {
			return err
		}
		t.Holder, t.Until = 0, 0
		tx.saveFarmBatch(b)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

// apiFarmScriptPut keeps a script for the organisers' students that is no round's entry: a
// reference curator's, or an entry's at a scale the season does not run. The organisers' own
// machine sends it, zstd-compressed; the machines that train it check it against its hash.
func (a *App) apiFarmScriptPut(w http.ResponseWriter, r *http.Request) {
	_, _, body, err := a.signedBy(w, r, a.season.maxScriptBlob(), true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	sha := r.PathValue("sha")
	if !sha256Re.MatchString(sha) {
		writeErr(w, fail(400, "a script is named by its sha256"))
		return
	}
	if len(body) == 0 {
		writeErr(w, fail(400, "not a script"))
		return
	}
	if err := a.st.blobs.Put(r.Context(), farmScriptBlob(sha), body, "application/zstd"); err != nil {
		slog.Error("keeping a farm script", "sha", sha, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	writeJSON(w, 200, map[string]any{"script": sha, "bytes": len(body)})
}

// apiFarmScript serves such a script to the organisers' machines.
func (a *App) apiFarmScript(w http.ResponseWriter, r *http.Request) {
	if _, _, err := a.farmMachine(w, r, 1<<10); err != nil {
		writeErr(w, err)
		return
	}
	sha := r.PathValue("sha")
	if !sha256Re.MatchString(sha) {
		writeErr(w, fail(404, "no such script"))
		return
	}
	f, size, err := a.st.blobs.Open(r.Context(), farmScriptBlob(sha))
	if errors.Is(err, errNoBlob) {
		writeErr(w, fail(404, "no such script"))
		return
	}
	if err != nil {
		slog.Error("reading a farm script", "sha", sha, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	io.Copy(w, f)
}
