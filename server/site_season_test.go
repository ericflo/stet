package main

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// The site reads in bits per byte, says what the exam is, and shows no multiplier it has no
// curve for.
func TestTheSiteReadsInBitsPerByte(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob, carol := w.join("anchor-one", priv), w.join("bob", nil), w.join("carol", nil)
	id := w.submit(anchor, "ref-a", "print(1)")
	idx := w.drawWithoutExam()
	rid := roundID(w.app.season.ID, idx)
	if w.editAll(anchor) != 1 {
		t.Fatal("the anchor should have run the entry")
	}
	// While the round waits for its exam, its page says so.
	if code, page := w.page("/rounds/" + rid); code != 200 || !strings.Contains(page, "Grading starts when its exam is made") {
		t.Fatalf("the open round does not say it waits for its exam: HTTP %d", code)
	}
	w.established("bob", "carol")
	if code, msg := w.postExam(anchor, idx, testTable()); code != 200 {
		t.Fatalf("posting the exam: HTTP %d %s", code, msg)
	}
	// Both take the script before either reports; their baselines differ a little.
	n := len(testTable())
	var jobs []*claimed
	for _, a := range []*actor{bob, carol} {
		c := w.claim(a, false)
		if c.Job == nil || c.Job.Kind != "grade" {
			t.Fatalf("%s has no grade job: %+v", a.handle, c.Job)
		}
		jobs = append(jobs, c)
	}
	for i, a := range []*actor{bob, carol} {
		if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", jobs[i].Job.ID), w.examResult(jobs[i], n, 0.1-0.02*float64(i), -3+0.02*float64(i)), nil); code != 200 {
			t.Fatalf("%s's grade was refused: HTTP %d", a.handle, code)
		}
	}
	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 1 {
		t.Fatal("the round did not settle")
	}
	// The entry's gain on each pillar, as the exam's aggregate gives it.
	w.app.st.update(t.Context(), func(tx *Tx) error {
		for _, e := range tx.s.inRound[idx] {
			e.BySource = map[string]float64{"reasoning": 0.12, "writing": -0.03, "code": 0.05}
			tx.saveEntry(e)
		}
		return nil
	})

	pages := map[string]bool{ // the page, and whether it says the unit
		"/": true, "/glossary": true, "/enter": true, "/record": true, "/rounds/" + rid: true, "/editors/" + id[:12]: true,
		"/leaderboard": false, "/rounds": false, "/verify": false, "/verifiers": false, "/run": false, "/u/anchor-one": false,
	}
	got := map[string]string{}
	for path, unit := range pages {
		code, page := w.page(path)
		if code != 200 {
			t.Fatalf("%s: HTTP %d", path, code)
		}
		if i := strings.Index(page, "1.00×"); i >= 0 {
			t.Errorf("%s shows a multiplier with no curve: …%s…", path, page[max(0, i-80):min(len(page), i+80)])
		}
		if unit && !strings.Contains(page, "bits per byte") {
			t.Errorf("%s does not say bits per byte", path)
		}
		got[path] = page
	}
	for path, want := range map[string][]string{
		"/glossary":      {"Made for each round after its draw", "7 capabilities, each counting the same", `id="bits"`, "no worth is shown"},
		"/enter":         {"Practice rounds use the generated problems only", "Nobody can know it in advance"},
		"/rounds/" + rid: {"bits per byte better on the exam", "the round's exam, made after the draw", "exam's generated problems"},
		// The gain on each pillar, from zero; the training loss in bits too; no worth without a curve.
		"/editors/" + id[:12]: {"bits per byte better than unedited", `class="bar neg"`, "training loss, bits per token", "on the exam"},
	} {
		for _, s := range want {
			if !strings.Contains(got[path], s) {
				t.Errorf("%s does not say %q", path, s)
			}
		}
	}
	if strings.Contains(got["/editors/"+id[:12]], "Worth") {
		t.Error("a page shows a multiplier with no curve")
	}

	// The API: the unit, the exam's pillars, and no multiplier rather than a wrong one.
	var season map[string]any
	if _, body := w.page("/api/v1/season"); json.Unmarshal([]byte(body), &season) != nil ||
		season["unit"] != "bits per byte" || season["has_multiplier"] != false || len(season["pillars"].([]any)) != len(w.app.season.ExamSources) {
		t.Fatalf("the season says %v, %v, %v", season["unit"], season["pillars"], season["has_multiplier"])
	}
	var board struct {
		Editors     []map[string]any `json:"editors"`
		Provisional []map[string]any `json:"provisional"`
	}
	_, body := w.page("/api/v1/leaderboard")
	json.Unmarshal([]byte(body), &board)
	rows := append(board.Editors, board.Provisional...)
	if len(rows) == 0 {
		t.Fatal("nobody on the leaderboard")
	}
	for _, row := range rows {
		if m, ok := row["multiplier"]; !ok || m != nil || row["rating"] == nil {
			t.Fatalf("a row's multiplier is %v (rating %v), not null", m, row["rating"])
		}
	}
	var ed struct {
		Editor map[string]any `json:"editor"`
	}
	if _, body := w.page("/api/v1/editors/" + id[:12]); json.Unmarshal([]byte(body), &ed) != nil || ed.Editor["multiplier"] != nil {
		t.Fatalf("the editor's multiplier is %v, not null", ed.Editor["multiplier"])
	}
	w.check()
}

// With no reference curve measured for the scale there is no multiplier at all, and once one
// is, a multiplier is read off it.
func TestAMultiplierIsReadOnlyWhereThereIsACurve(t *testing.T) {
	w := newWorld(t)
	if w.app.hasMultiplier() {
		t.Fatal("a multiplier with no curve measured")
	}
	if _, _, ok := w.app.season.multiplier(w.app.curveKey(), 0.1); ok {
		t.Fatal("a multiplier read off no curve")
	}
	season := *w.app.season
	season.Curves = map[string][][2]float64{w.app.curveKey(): {{0.5, 1.95}, {1, 1.84}, {2, 1.75}, {4, 1.68}}}
	w.app.season = &season
	if !w.app.hasMultiplier() {
		t.Fatal("no multiplier once there is a curve")
	}
	if m, capped, ok := w.app.season.multiplier(w.app.curveKey(), 0); !ok || capped || math.Abs(m-1) > 1e-9 {
		t.Fatalf("no gain should read 1×, not %v (capped %v, ok %v)", m, capped, ok)
	}
	if m, capped, ok := w.app.season.multiplier(w.app.curveKey(), 0.09); !ok || capped || math.Abs(m-2) > 1e-9 {
		t.Fatalf("a gain of 0.09 bits per byte should read 2×, not %v (capped %v, ok %v)", m, capped, ok)
	}
}

// Every chart is drawn from whatever it is given, nothing included, and the gain on each
// pillar is drawn as a bar from zero.
func TestTheChartsDrawNothingFromNothingAndTheGainOnEachPillarFromZero(t *testing.T) {
	pillars := []string{"knowledge", "reading", "reasoning", "code"}
	for name, out := range map[string]string{
		"gain":                     string(gainChart(pillars, nil)),
		"gain, no pillar it knows": string(gainChart(pillars, map[string]float64{"astrology": 1})),
		"record":                   string(recordChart(nil)),
		"curve":                    string(curveChart(nil)),
	} {
		if out != "" {
			t.Errorf("the %s chart drew something from nothing", name)
		}
	}
	out := string(gainChart(pillars, map[string]float64{"knowledge": 0.2, "reasoning": -0.05, "code": 0}))
	for _, want := range []string{"bits per byte better than unedited", ">knowledge<", ">+0.200<", `class="bar neg"`, ">−0.050<", ">—<", `class="axis"`} {
		if !strings.Contains(out, want) {
			t.Errorf("the gain chart has no %q", want)
		}
	}
	if !strings.Contains(string(recordChart([]RecordRow{{Record: Record{Rating: 0.1, At: now()}, Name: "x"}})), "rating, bits per byte better than unedited") {
		t.Error("the record chart is not in bits per byte")
	}
	// The trainer reports its loss in natural-log units; the chart says it in bits.
	if got := string(curveChart([][]float64{{1, 9}, {50, 6}, {100, 5.5}, {150, 5.2}})); !strings.Contains(got, "bits per token") || !strings.Contains(got, "loss 7.502") {
		t.Error("the training curve is not in bits per token")
	}
}
