package main

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

var t0 = time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)

func testPolicy() Policy {
	return Policy{MaxWorkers: 4, PerRunner: 2, PerWorker: 4, MaxHourly: 3, Grace: 25 * time.Minute, Silence: 20 * time.Minute}
}

func round(runsWaiting, editing, wanted, grading int) *Work {
	w := &Work{Runners: 1}
	w.Rounds = append(w.Rounds, struct {
		ID             string `json:"id"`
		Entries        int    `json:"entries"`
		RunsWaiting    int    `json:"runs_waiting"`
		EditsUnderWay  int    `json:"edits_under_way"`
		GradesWanted   int    `json:"grades_wanted"`
		GradesUnderWay int    `json:"grades_under_way"`
	}{"s0-o00001", 36, runsWaiting, editing, wanted, grading})
	return w
}

// pool makes workers: "free", or a pod that is "starting", "waiting", "grading", "quiet",
// "stopped" or "never" (started long ago and never asked for work).
func pool(w *Work, states ...string) []*Worker {
	var out []*Worker
	for i, st := range states {
		k := &Key{Handle: fmt.Sprintf("stet-rp-%02d", i+1)}
		wk := &Worker{Key: k}
		created := t0.Add(-time.Hour)
		switch st {
		case "free":
		case "starting":
			wk.Pod = &Pod{Name: k.Handle, Status: "RUNNING", CreatedAt: t0.Add(-5 * time.Minute)}
		case "never":
			wk.Pod = &Pod{Name: k.Handle, Status: "RUNNING", CreatedAt: created}
		case "stopped":
			wk.Pod = &Pod{Name: k.Handle, Status: "EXITED", CreatedAt: created}
		default:
			wk.Pod = &Pod{Name: k.Handle, Status: "RUNNING", CreatedAt: created}
			m := Machine{Handle: k.Handle, Worker: true, LastSeen: t0.Add(-time.Minute).Unix()}
			switch st {
			case "grading":
				m.Busy = "grade"
				m.LastSeen = t0.Add(-40 * time.Minute).Unix() // says nothing while it trains
			case "quiet":
				m.LastSeen = t0.Add(-30 * time.Minute).Unix()
			}
			w.Machines = append(w.Machines, m)
			wk.Machine = &w.Machines[len(w.Machines)-1]
		}
		out = append(out, wk)
	}
	// The pointers above were into a slice that grew: point them at where it ended up.
	for _, wk := range out {
		if wk.Machine != nil {
			wk.Machine = w.machine(wk.Key.Handle)
		}
	}
	return out
}

func handles(es []Ending) string {
	var h []string
	for _, e := range es {
		h = append(h, e.Worker.Key.Handle)
	}
	return strings.Join(h, ",")
}

func TestTheMachinesWantedKeepPaceWithTheRunnersAndClearWhatIsWaiting(t *testing.T) {
	p := testPolicy()
	for _, c := range []struct {
		w    *Work
		want int
	}{
		{&Work{}, 0},            // no round
		{round(0, 0, 0, 3), 0},  // all being graded, nothing more to come
		{round(30, 1, 0, 0), 2}, // a round starting: keep pace with the one runner
		{round(10, 1, 5, 0), 4}, // and the scripts waiting on top: 2 + ceil(5/4)
		{round(0, 0, 3, 0), 1},  // the runs are done; three scripts left
		{round(0, 0, 40, 0), 4}, // never more than --max
		{round(0, 1, 0, 0), 2},  // the last curator running: its script is coming
		{round(5, 0, 0, 0), 2},  // runs to give out and the runner between jobs
		{&Work{Runners: 2, Rounds: round(5, 2, 0, 0).Rounds}, 4}, // two runners
	} {
		if got := p.target(c.w); got != c.want {
			t.Errorf("%+v: %d machines, want %d", c.w.Rounds, got, c.want)
		}
	}
}

func TestMachinesAreRentedForWorkAndEndedWhenThereIsNone(t *testing.T) {
	p := testPolicy()

	// A round starts: two machines for the runner, none rented yet.
	w := round(30, 1, 0, 0)
	d := p.decide(w, pool(w, "free", "free", "free", "free"), 0, t0)
	if d.Launch != 2 || len(d.End) != 0 {
		t.Fatalf("a round starting: %+v", d)
	}
	// Those two are starting: nothing more is rented for the same work.
	w = round(30, 1, 0, 0)
	if d = p.decide(w, pool(w, "starting", "waiting", "free", "free"), 0, t0); d.Launch != 0 || len(d.End) != 0 {
		t.Fatalf("with two on the way: %+v", d)
	}
	// Scripts pile up: more are rented, up to --max, and only on free keys.
	w = round(10, 1, 12, 2)
	if d = p.decide(w, pool(w, "grading", "grading", "free", "free"), 0, t0); d.Launch != 2 || d.Target != 4 {
		t.Fatalf("with scripts piling up: %+v", d)
	}
	w = round(10, 1, 12, 2)
	if d = p.decide(w, pool(w, "grading", "grading", "free"), 0, t0); d.Launch != 1 || !strings.Contains(d.Note, "keys --add") {
		t.Fatalf("with one key free: %+v", d)
	}
	// Past the day's budget: nothing new, and what runs is left to finish.
	p.DailyBudget = 5
	w = round(10, 1, 12, 2)
	if d = p.decide(w, pool(w, "grading", "free", "free", "free"), 5.01, t0); d.Launch != 0 || len(d.End) != 0 || !strings.Contains(d.Note, "budget") {
		t.Fatalf("past the budget: %+v", d)
	}
	p.DailyBudget = 0

	// Some work left, and more machines than it wants: the idle ones are left to stop by
	// themselves, since a script may come to them any moment.
	w = round(0, 0, 1, 2)
	if d = p.decide(w, pool(w, "waiting", "waiting", "grading", "grading"), 0, t0); d.Launch != 0 || len(d.End) != 0 {
		t.Fatalf("a little work left: %+v", d)
	}
	// Nothing left for anyone: whatever holds no job is ended now, and what grades is not.
	w = round(0, 0, 0, 1)
	if d = p.decide(w, pool(w, "waiting", "starting", "grading", "free"), 0, t0); d.Launch != 0 || handles(d.End) != "stet-rp-01,stet-rp-02" {
		t.Fatalf("nothing left: %+v (%s)", d, handles(d.End))
	}
	// Whatever has stopped, never got going, or went quiet holding nothing is ended, work
	// or no work; and the machines it leaves short are rented again.
	w = round(30, 1, 0, 0)
	d = p.decide(w, pool(w, "stopped", "never", "quiet", "grading"), 0, t0)
	if handles(d.End) != "stet-rp-01,stet-rp-02,stet-rp-03" || d.Live != 1 || d.Launch != 0 {
		t.Fatalf("broken machines: %+v (%s)", d, handles(d.End))
	}
	if !strings.Contains(d.End[2].Why, "30m") {
		t.Fatalf("a quiet machine's ending does not say for how long: %q", d.End[2].Why)
	}
}
