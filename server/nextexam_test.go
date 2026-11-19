package main

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

// newWorldS1 is a world whose season's exam is the next exam (testdata/season-s1.json, written by
// tools/export_season.py s1).
func newWorldS1(t *testing.T, anchors ...ed25519.PublicKey) *world {
	t.Helper()
	raw, err := os.ReadFile("testdata/season-s1.json")
	if err != nil {
		t.Fatal(err)
	}
	var season Season
	if err := json.Unmarshal(raw, &season); err != nil {
		t.Fatal(err)
	}
	cfg := testConfig(t, anchors...)
	kv, blobs := newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")}
	app, err := newApp(context.Background(), cfg, &season, kv, blobs)
	if err != nil {
		t.Fatal(err)
	}
	fakeBeacon(t)
	srv := httptest.NewServer(app.routes())
	t.Cleanup(srv.Close)
	return &world{t: t, app: app, srv: srv, kv: kv, blobs: blobs, cfg: cfg}
}

// nextTable is a small exam: items in three pillars, two languages.
func nextTable() []ItemInfo {
	var out []ItemInfo
	for i := 0; i < 12; i++ {
		out = append(out, ItemInfo{Pillar: "reasoning", Part: "arith", Lang: "en", Stratum: "math/answer"})
	}
	for i := 0; i < 6; i++ {
		out = append(out, ItemInfo{Pillar: "writing", Part: "reference", Lang: []string{"en", "de"}[i%2], Stratum: "prose/passage"})
	}
	for i := 0; i < 4; i++ {
		out = append(out, ItemInfo{Pillar: "code", Part: "prog", Lang: "code", Stratum: "code/answer"})
	}
	return out
}

var nextDigest = strings.Repeat("d1", 32)

func (w *world) postExam(by *actor, idx int, table []ItemInfo) (int, string) {
	w.t.Helper()
	enc, _ := zstd.NewWriter(nil)
	file := enc.EncodeAll([]byte("{\"kind\": \"passage\"}\n"), nil)
	var out struct {
		Error string `json:"error"`
	}
	public := enc.EncodeAll([]byte("{\"withheld\": true}\n"), nil)
	code := by.do("POST", fmt.Sprintf("/api/v1/admin/rounds/%s/exam", roundID(w.app.season.ID, idx)), map[string]any{
		"items_digest": nextDigest, "items_file": base64.StdEncoding.EncodeToString(file), "table": table,
		"public_file": base64.StdEncoding.EncodeToString(public),
		"languages": map[string]float64{"en": 60, "de": 40}, "fresh": 6,
	}, &out)
	return code, out.Error
}

// nextResult is a grade on the next exam: each item scores base + gain (the baseline, base).
func (w *world) nextResult(c *claimed, n int, gain, base float64) map[string]any {
	p := w.result(c, 5, 6)
	items := func(x float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = x - float64(i%3)*0.01
		}
		return out
	}
	mean := func(v []float64) float64 {
		s := 0.0
		for _, x := range v {
			s += x
		}
		return s / float64(len(v))
	}
	b, e := items(base), items(base+gain)
	p["exam_sha256"] = nextDigest
	p["baseline"] = map[string]any{"kind": "next", "mean": mean(b), "item_scores": b}
	p["exam"] = map[string]any{"kind": "next", "mean": mean(e), "item_scores": e}
	return p
}

func TestTheNextExamIsGradedOnceItIsPostedAndScoredAgainstTheRoundsBaseline(t *testing.T) {
	pub, priv := newKey()
	w := newWorldS1(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.draw()
	if w.editAll(anchor) != 1 {
		t.Fatal("the anchor should have run the entry")
	}
	w.established("bob")
	// Until the exam is made, there is nothing to grade, and nothing is given up.
	if c := w.claim(bob, false); c.Job != nil {
		t.Fatalf("a grade before the round's exam exists: %+v", c.Job)
	}
	w.pass(w.cfg.ReplicaWait * 4)
	if w.settled(idx) != 0 {
		t.Fatal("the round settled while waiting for its exam")
	}
	// Only the organisers post it, and only a table the season scores.
	table := nextTable()
	if code, _ := w.postExam(bob, idx, table); code != 403 {
		t.Fatalf("a volunteer posted the exam: HTTP %d", code)
	}
	bad := append([]ItemInfo(nil), table...)
	bad[0].Pillar = "astrology"
	if code, _ := w.postExam(anchor, idx, bad); code != 400 {
		t.Fatalf("an exam with a pillar the season does not have: HTTP %d", code)
	}
	if code, msg := w.postExam(anchor, idx, table); code != 200 {
		t.Fatalf("posting the exam: HTTP %d %s", code, msg)
	}
	// Grades start: the job carries the exam's digest, and the grader may fetch the exam.
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" || cb.Job.Round["exam_digest"] != nextDigest {
		t.Fatalf("no grade job with the exam's digest: %+v", cb.Job)
	}
	if code := bob.do("GET", fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)), nil, nil); code != 200 {
		t.Fatalf("the grader could not fetch the exam: HTTP %d", code)
	}
	dave := w.join("dave", nil)
	if code := dave.do("GET", fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)), nil, nil); code != 403 {
		t.Fatalf("an account with no grade job fetched the exam: HTTP %d", code)
	}
	// Carol takes the script too, before either reports.
	carol := w.join("carol", nil)
	w.established("carol")
	ca := w.claim(carol, false)
	if ca.Job == nil || ca.Job.Kind != "grade" {
		t.Fatalf("carol should be grading: %+v", ca.Job)
	}
	// A report that is not one score for each item is refused.
	short := w.nextResult(cb, len(table)-1, 0.1, -3)
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), short, nil); code != 400 {
		t.Fatalf("a report one item short: HTTP %d", code)
	}
	// Bob's baseline scores -3 on every item, his student -2.9: a gain of 0.1 against the round's baseline.
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), w.nextResult(cb, len(table), 0.1, -3), nil); code != 200 {
		t.Fatal("bob's grade was refused")
	}
	// Carol's baseline is a little better: the round's is the mean of the two.
	if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", ca.Job.ID), w.nextResult(ca, len(table), 0.08, -2.98), nil); code != 200 {
		t.Fatal("carol's grade was refused")
	}
	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 1 {
		t.Fatal("the round did not settle")
	}
	e := w.entry(idx, "ref-a")
	// Each grade's score is against the round's baseline (-2.99 on each item): bob's student and
	// carol's both score -2.90, +0.09 on each item.
	var want float64
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		base := make([]float64, len(table))
		for i := range base {
			base[i] = -2.99 - float64(i%3)*0.01
		}
		gains := make([]float64, len(table))
		for i := range gains {
			gains[i] = (-2.9 - float64(i%3)*0.01) - base[i]
		}
		want, _, _ = NextScore(table, gains, rd.Shares)
	})
	if e.Gain == nil || math.Abs(*e.Gain-places5(want)) > 1e-4 {
		t.Fatalf("the entry's gain is %v, want %.5f", e.Gain, want)
	}
	// Settled, the round's published exam is anyone's, unsigned.
	resp, err := http.Get(w.srv.URL + fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("a settled round's published exam: %v %v", err, resp)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	dec, _ := zstd.NewReader(nil)
	if plain, _ := dec.DecodeAll(got, nil); !strings.Contains(string(plain), "withheld") {
		t.Fatalf("the published exam is not the public one: %q", plain)
	}
	// Once settled, what the round held of its exam is in the file store and out of the database.
	w.app.archiveNext(context.Background())
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if len(rd.ExamTable) != 0 || len(rd.BaseItems) != 0 {
			t.Fatal("a settled round still holds its exam's table in the database")
		}
		for _, r := range s.results(idx, e.Editor) {
			if len(r.Items) != 0 {
				t.Fatal("a settled round's grade still holds its item scores")
			}
		}
	})
	if _, _, err := w.blobs.Open(context.Background(), nextScoresBlob(idx)); err != nil {
		t.Fatalf("the round's item scores were not kept: %v", err)
	}
	w.check()
}

func TestAPackSeasonTakesNoPostedExam(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.draw()
	if code, _ := w.postExam(anchor, idx, nextTable()); code != 404 {
		t.Fatalf("season 0 took a posted exam: HTTP %d", code)
	}
}
