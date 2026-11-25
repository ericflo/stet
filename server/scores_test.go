package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"testing"
)

// pillarResult is a grade whose student gains this much on every item of each of testTable's
// parts, over a baseline of -6 on every item. The exam's aggregate of a gain that is the same
// on every item of a pillar is that gain for the pillar.
func (w *world) pillarResult(c *claimed, explain, reference float64) map[string]any {
	p := w.result(c, 5, 6)
	gains := map[string]float64{"explain": explain, "reference": reference}
	var scores []float64
	sum := 0.0
	for _, it := range testTable() {
		scores = append(scores, -6+gains[it.Part])
		sum += -6 + gains[it.Part]
	}
	p["exam"] = map[string]any{"mean": sum / float64(len(scores)), "item_scores": scores}
	return p
}

// runTwiceGradedTwice settles a round with one entry, "tidy", that two people ran, writing two
// scripts, each graded by two more: four grades, no two alike on the exam's pillars. It
// returns the round and the entry's id.
//
// A pillar of testTable in skills counts half and one in comprehension counts half,
// so the grades of the first script come to 0.50(0.40) + 0.50(0.60) = 0.50 and 0.50(0.60) +
// 0.50(0.50) = 0.55, and the script to 0.525; those of the second to 0.45 and 0.54, and it
// to 0.495. The entry's gain is the mean of its two runs': 0.51.
func runTwiceGradedTwice(t *testing.T) (*world, int, string) {
	t.Helper()
	w := newWorld(t)
	alice, bob, carol, dave, erin := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil), w.join("erin", nil)
	id := w.submit(alice, "tidy", "print(1)")
	w.established("bob", "carol", "dave", "erin")
	idx := w.drawWithoutExam()
	one, two := "{\"doc\": 1}\n", "{\"doc\": 2}\n"
	w.run(bob, one)
	w.run(carol, two)
	// Before the exam a script takes a second grade from an established account. Dave and
	// erin take one script each, and then each runner the script the other wrote.
	type job struct {
		by *actor
		c  *claimed
	}
	var jobs []job
	for _, a := range []*actor{dave, erin, bob, carol} {
		c := w.claimEarly(a)
		if c.Job == nil || c.Job.Kind != "grade" {
			t.Fatalf("%s should be grading: %+v", a.handle, c.Job)
		}
		jobs = append(jobs, job{a, c})
	}
	if jobs[0].c.Job.Script.SHA256 == jobs[1].c.Job.Script.SHA256 || jobs[2].c.Job.Script.SHA256 != shaOf(two) || jobs[3].c.Job.Script.SHA256 != shaOf(one) {
		t.Fatal("each script should have gone to two graders, and to neither runner its own")
	}
	w.exam(idx)
	status := ""
	for _, j := range jobs {
		first := j.by == dave || j.by == erin
		var p map[string]any
		switch {
		case j.c.Job.Script.SHA256 == shaOf(one) && first:
			p = w.pillarResult(j.c, 0.40, 0.60)
		case j.c.Job.Script.SHA256 == shaOf(one):
			p = w.pillarResult(j.c, 0.60, 0.50)
		case first:
			p = w.pillarResult(j.c, 0.20, 0.70)
		default:
			p = w.pillarResult(j.c, 0.50, 0.58)
		}
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		if code := j.by.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", j.c.Job.ID), p, &ack); code != 200 {
			t.Fatalf("%s's grade: HTTP %d %s", j.by.handle, code, ack.Error)
		}
		status = ack.Status
	}
	if status != "settled" {
		t.Fatalf("after the fourth grade the entry is %s", status)
	}
	w.check()
	return w, idx, id
}

// The pillars an entry is shown with are made of the same grades as its gain, in the same
// way, so they add up to it.
func TestAnEntrysPillarsAreMadeOfTheGradesItsGainIsMadeOf(t *testing.T) {
	w, idx, id := runTwiceGradedTwice(t)
	e := w.entry(idx, "tidy")
	if !e.Valid || e.NRuns != 2 || e.Gain == nil || math.Abs(*e.Gain-0.51) > 1e-9 {
		t.Fatalf("the entry: %+v", e)
	}
	for _, r := range e.Runs {
		if len(r.Grades) != 2 || !r.Grades[0].Counted || !r.Grades[1].Counted {
			t.Fatalf("run %d should have two grades that count: %+v", r.N, r.Grades)
		}
	}
	// On each pillar, the mean over its two runs of the mean over the run's two grades.
	want := map[string]float64{"explain": (0.40 + 0.60 + 0.20 + 0.50) / 4, "reference": (0.60 + 0.50 + 0.70 + 0.58) / 4}
	if len(e.BySource) != len(want) {
		t.Fatalf("the entry's pillars: %v", e.BySource)
	}
	for k, x := range want {
		if math.Abs(e.BySource[k]-x) > 1e-9 {
			t.Fatalf("the entry's gain on %s is %v, want %v", k, e.BySource[k], x)
		}
	}
	// Weighed as the season's exam weighs its pillars, they come to the entry's gain.
	plan := w.app.season.Exam
	sum := func(by map[string]float64) float64 {
		present := map[string]bool{}
		for k := range by {
			present[k] = true
		}
		total := 0.0
		for k, share := range plan.weights(present) {
			total += share * by[k]
		}
		return total
	}
	if got := sum(e.BySource); math.Abs(got-*e.Gain) > 2e-5 {
		t.Fatalf("the entry's pillars come to %v, and its gain is %v", got, *e.Gain)
	}
	// Which no one grade's do: the pillars of any single grade are another entry's.
	w.app.st.view(func(s *State) {
		rs := s.results(idx, id)
		if len(rs) != 4 {
			t.Fatalf("the entry has %d grades", len(rs))
		}
		for _, r := range rs {
			if math.Abs(sum(r.BySource)-*e.Gain) < 1e-3 || math.Abs(r.BySource["explain"]-e.BySource["explain"]) < 1e-3 {
				t.Fatalf("this test is not testing anything: one grade's pillars are the entry's: %v", r.BySource)
			}
		}
	})
	// The curator's page draws them.
	if _, page := w.page("/editors/" + id[:12]); !strings.Contains(page, "Where the gain comes from") {
		t.Fatal("the curator's page does not draw its gain on each pillar")
	}
}

// Once a round has settled, every score of every grader on every item of its exam is
// anyone's, with what it takes to work the round's scores out again from them.
func TestASettledRoundsItemScoresAreAnyones(t *testing.T) {
	w, idx, id := runTwiceGradedTwice(t)
	rid := roundID(w.app.season.ID, idx)
	get := func(round string) (int, http.Header, []byte) {
		t.Helper()
		resp, err := http.Get(w.srv.URL + "/api/v1/rounds/" + round + "/scores")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header, body
	}
	roundJSON := func() map[string]any {
		var out struct {
			Round map[string]any `json:"round"`
		}
		(&actor{t: t, srv: w.srv}).do("GET", "/api/v1/rounds/"+rid, nil, &out)
		return out.Round
	}
	// Settled a moment ago: they are not where they are served from yet, and nothing says so.
	if code, _, body := get(rid); code != 404 || !strings.Contains(string(body), "only just settled") {
		t.Fatalf("a round's scores before they are published: HTTP %d %s", code, body)
	}
	if r := roundJSON(); r["scores"] != nil {
		t.Fatalf("the round names scores that are not there: %v", r["scores"])
	}
	if code, _, body := get(roundID(w.app.season.ID, idx+7)); code != 404 || !strings.Contains(string(body), "no such round") {
		t.Fatalf("the scores of a round there is not: HTTP %d %s", code, body)
	}
	w.app.tick(context.Background())
	code, header, body := get(rid)
	if code != 200 || header.Get("Content-Type") != "application/json" || !strings.Contains(header.Get("Cache-Control"), "max-age=") {
		t.Fatalf("a settled round's scores: HTTP %d %v", code, header)
	}
	if r := roundJSON(); r["scores"] != "http://api.test/api/v1/rounds/"+rid+"/scores" {
		t.Fatalf("the round says its scores are at %v", r["scores"])
	}
	if _, page := w.page("/rounds/" + rid); !strings.Contains(page, `href="/api/v1/rounds/`+rid+`/scores"`) {
		t.Fatal("the round's page does not link to its item scores")
	}
	var file ExamScores
	if err := json.Unmarshal(body, &file); err != nil {
		t.Fatalf("a settled round's scores are not JSON: %v", err)
	}
	if file.Round != rid || file.Digest != examSHA || len(file.Table) != len(testTable()) || len(file.Baseline) != len(file.Table) || len(file.Grades) != 4 || len(file.GradesOf) != 4 {
		t.Fatalf("the scores of %s: %d items, %d grades, %d of them described", file.Round, len(file.Table), len(file.Grades), len(file.GradesOf))
	}
	if strings.Join(file.BaselineOf, " ") != "bob carol dave erin" || len(file.Baselines) != 4 {
		t.Fatalf("whose baselines the round's is the mean of: %v", file.BaselineOf)
	}
	// The file alone gives the entry's gain: each grade's score is the season's aggregate of
	// its item scores less the round's baseline, a script's the median of its grades that
	// count, and the entry's the median of its runs'.
	runs := map[int][]float64{}
	for gid, of := range file.GradesOf {
		if of.Editor != id || !of.Counted || len(of.Runs) != 1 || (of.Script != shaOf("{\"doc\": 1}\n") && of.Script != shaOf("{\"doc\": 2}\n")) {
			t.Fatalf("grade %s is described as %+v", gid, of)
		}
		score, _, err := w.app.season.Exam.ScoreItems(file.Table, diffItems(file.Grades[gid], file.Baseline), file.Shares)
		if err != nil {
			t.Fatal(err)
		}
		runs[of.Runs[0]] = append(runs[of.Runs[0]], score)
	}
	var gains []float64
	for n, grades := range runs {
		if len(grades) != 2 {
			t.Fatalf("run %d has %d grades in the file", n, len(grades))
		}
		gains = append(gains, median(grades))
	}
	e := w.entry(idx, "tidy")
	if got := median(gains); len(gains) != 2 || math.Abs(got-*e.Gain) > 2e-5 || math.Abs(got-0.51) > 2e-5 {
		t.Fatalf("worked out again from the round's item scores the entry's gain is %v, and the round published %v", got, *e.Gain)
	}
	// Each grade is told apart by who gave it, of which script.
	seen := map[string]bool{}
	for _, of := range file.GradesOf {
		seen[of.By+" "+of.Script[:8]] = true
	}
	if len(seen) != 4 || !seen["bob "+shaOf("{\"doc\": 2}\n")[:8]] || !seen["carol "+shaOf("{\"doc\": 1}\n")[:8]] {
		t.Fatalf("who graded what, by the file: %v", seen)
	}

	// The next round's are sealed with everything else of it until it settles.
	next := w.draw()
	if code, _, body := get(roundID(w.app.season.ID, next)); code != 404 || !strings.Contains(string(body), "has not settled") {
		t.Fatalf("an open round's scores: HTTP %d %s", code, body)
	}
	w.check()
}
