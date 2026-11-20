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
	return &Work{Runners: 1, Rounds: []WorkRound{{ID: "s1-o00001", Entries: 36, RunsWaiting: runsWaiting, RunsWanted: runsWaiting, EditsUnderWay: editing, GradesWanted: wanted, GradesUnderWay: grading}}}
}

// pool makes workers: "free", or a pod that is "starting", "waiting", "grading", "running", "training",
// "quiet", "stopped" or "never" (started long ago and never asked for work).
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
			case "running":
				m.Busy = "edit"
				m.LastSeen = t0.Add(-10 * time.Minute).Unix()
			case "training":
				m.Busy = "farm"
				m.LastSeen = t0.Add(-40 * time.Minute).Unix()
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
		{&Work{Waiting: true}, 0},                                // grading only: no round, no work
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

// Machines that run curators as well as grade are wanted for runs and grades alike.
// A machine at a round's job is busy with it for twenty minutes or more at the standard scale:
// it does not count toward what is waiting, which wants machines of its own.
func TestAMachineAtAJobLeavesWhatIsWaitingToAnother(t *testing.T) {
	p := testPolicy()
	p.PerWorker = 1
	w := round(0, 0, 1, 2) // round 1 at 01:23: two scripts being graded, one waiting
	d := p.decide(w, pool(w, "grading", "grading", "free", "free"), 0, t0)
	if d.Target != 3 || d.Launch != 1 || len(d.End) != 0 {
		t.Fatalf("two grading and one waiting: want 3, rent 1; got %+v", d)
	}
	// A curator being run holds its machine as a grade does.
	p.Edits = true
	w = round(0, 1, 1, 0)
	d = p.decide(w, pool(w, "running", "free"), 0, t0)
	if d.Target != 2 || d.Launch != 1 {
		t.Fatalf("one running and one waiting: want 2, rent 1; got %+v", d)
	}
	// Four to a machine, on top of those at work.
	p.Edits, p.PerWorker = false, 4
	w = round(0, 0, 5, 1)
	d = p.decide(w, pool(w, "grading", "free", "free", "free"), 0, t0)
	if d.Target != 3 || d.Launch != 2 {
		t.Fatalf("one grading and five waiting, four to a machine: want 3, rent 2; got %+v", d)
	}
	// Nothing waiting: the machines at work are kept, and one doing nothing is ended at once.
	w = round(0, 0, 0, 1)
	d = p.decide(w, pool(w, "grading", "waiting"), 0, t0)
	if d.Target != 1 || d.Launch != 0 || handles(d.End) != "stet-rp-02" {
		t.Fatalf("one grading, nothing waiting: want 1, end the idle one; got %+v", d)
	}
	// Never more than --max.
	p.PerWorker = 1
	w = round(0, 0, 3, 3)
	d = p.decide(w, pool(w, "grading", "grading", "grading", "free", "free"), 0, t0)
	if d.Target != 4 || d.Launch != 1 {
		t.Fatalf("three grading and three waiting, at most four: want 4, rent 1; got %+v", d)
	}
}

func TestMachinesThatRunCuratorsAreRentedForRunsAndGrades(t *testing.T) {
	p := testPolicy()
	p.Edits, p.PerWorker, p.MaxWorkers = true, 6, 8
	for _, c := range []struct {
		w    *Work
		want int
	}{
		{&Work{}, 0},
		{round(36, 0, 0, 0), 8},   // a round starting: 72 jobs, at most 8 machines
		{round(5, 2, 1, 4), 2},    // 2*5 + 1 = 11 jobs
		{round(0, 3, 0, 5), 0},    // all under way: nothing more for a new machine
		{round(0, 0, 2, 0), 1},    // two scripts left
		{&Work{Waiting: true}, 1}, // an entry waiting, and nobody who runs curators around to have its round drawn
		{&Work{Waiting: true, Runners: 1}, 0},
	} {
		if got := p.target(c.w); got != c.want {
			t.Errorf("%+v waiting %v runners %d: %d machines, want %d", c.w.Rounds, c.w.Waiting, c.w.Runners, got, c.want)
		}
	}
	// A machine running a curator is never ended, and says so.
	w := round(0, 0, 0, 0)
	ws := pool(w, "grading", "waiting")
	ws[0].Machine.Busy = "edit"
	if st := ws[0].state(t0, p); st != "running" {
		t.Fatalf("a machine holding a curator is %q", st)
	}
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-02" {
		t.Fatalf("ended %s", handles(d.End))
	}
}

// The organisers' students are trained by machines with nothing official to do: those waiting
// want machines after the official work, under the same limits, and a machine training one is
// at work however long it says nothing.
func TestTheFarmsStudentsWantMachinesToo(t *testing.T) {
	p := testPolicy()
	p.Edits, p.PerWorker, p.PerFarm, p.MaxWorkers = true, 6, 4, 8
	for _, c := range []struct {
		w    *Work
		want int
	}{
		{&Work{FarmWaiting: 1}, 1},
		{&Work{FarmWaiting: 9}, 3},
		{&Work{FarmUnderWay: 5}, 0},         // all being trained already
		{withFarm(round(5, 0, 1, 0), 9), 5}, // 2 for the round's 11 jobs, 3 for the students
		{withFarm(round(36, 0, 0, 0), 9), 8},
	} {
		if got := p.target(c.w); got != c.want {
			t.Errorf("%+v farm %d: %d machines, want %d", c.w.Rounds, c.w.FarmWaiting, got, c.want)
		}
	}
	p.PerFarm = 0
	if got := p.target(&Work{FarmWaiting: 9}); got != 0 {
		t.Fatalf("--per-farm 0 still rents %d for the students", got)
	}
	p.Edits, p.PerFarm = false, 4
	if got := p.target(&Work{FarmWaiting: 9}); got != 3 {
		t.Fatalf("a machine that only grades trains students too: %d machines, want 3", got)
	}

	w := &Work{FarmUnderWay: 1}
	ws := pool(w, "grading", "waiting")
	ws[0].Machine.Busy = "farm" // and silent for 40 minutes, as it trains
	if st := ws[0].state(t0, p); st != "training" {
		t.Fatalf("a machine training a student is %q", st)
	}
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-02" || d.Live != 1 {
		t.Fatalf("ended %s, %d live", handles(d.End), d.Live)
	}
}

func withFarm(w *Work, waiting int) *Work {
	w.FarmWaiting = waiting
	return w
}

// A bigger student holds a machine for hours: one machine each, and only when the machines
// are rented to take them.
func TestBiggerStudentsWantAMachineEach(t *testing.T) {
	p := testPolicy()
	p.Edits, p.PerWorker, p.PerFarm, p.MaxWorkers = true, 6, 4, 5
	w := &Work{FarmWaiting: 4, FarmLongWaiting: 3}
	if got := p.target(w); got != 1 {
		t.Fatalf("machines not rented for bigger students were wanted for them: %d", got)
	}
	p.LongFarm = true
	if got := p.target(w); got != 4 {
		t.Fatalf("one for the short students and one for each bigger one: %d, want 4", got)
	}
	w.FarmLongWaiting = 12
	if got := p.target(w); got != 5 {
		t.Fatalf("past --max: %d", got)
	}
	// The machines training a bigger student count too: with four at it and two waiting,
	// two more, up to --max.
	busy := pool(w, "training", "training", "training", "training", "free", "free", "free")
	for _, wk := range busy[:4] {
		wk.Machine.Long = true
	}
	w2 := &Work{FarmLongWaiting: 2}
	if d := p.decide(w2, busy, 0, time.Now()); d.Target != 5 || d.Launch != 1 || d.LaunchLong != 1 {
		t.Fatalf("four training bigger students, two waiting: target %d, rent %d (%d for them)", d.Target, d.Launch, d.LaunchLong)
	}
	busy[3].Machine.Long = false // that one trains a short student, and will be free soon
	if d := p.decide(w2, busy, 0, time.Now()); d.Target != 5 {
		t.Fatalf("three training bigger students, two waiting: target %d", d.Target)
	}
	busy[2].Machine.Long = false
	if d := p.decide(w2, busy, 0, time.Now()); d.Target != 4 || d.Launch != 0 {
		t.Fatalf("two training bigger students, two waiting: target %d, rent %d", d.Target, d.Launch)
	}

	// Of the machines rented, those for the bigger students are said, to be chosen for them.
	w.FarmLongWaiting = 3
	if d := p.decide(w, pool(w, "free", "free", "free", "free", "free"), 0, time.Now()); d.Launch != 4 || d.LaunchLong != 3 {
		t.Fatalf("rented %d, %d of them for the bigger students", d.Launch, d.LaunchLong)
	}
	p.LongFarm = false
	if d := p.decide(w, pool(w, "free", "free", "free", "free", "free"), 0, time.Now()); d.LaunchLong != 0 {
		t.Fatalf("rented %d for bigger students the machines do not take", d.LaunchLong)
	}
}
