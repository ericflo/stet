package main

import (
	"encoding/json"
	"fmt"
	"os"
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
			wk.Rental = &Rental{Name: k.Handle, State: "running", Status: "RUNNING", Created: t0.Add(-5 * time.Minute)}
		case "never":
			wk.Rental = &Rental{Name: k.Handle, State: "running", Status: "RUNNING", Created: created}
		case "stopped":
			wk.Rental = &Rental{Name: k.Handle, State: "stopped", Status: "EXITED", Created: created}
		default:
			wk.Rental = &Rental{Name: k.Handle, State: "running", Status: "RUNNING", Created: created}
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
	// A curator is run in a minute or two: the machine running it takes the script next.
	p.Edits = true
	w = round(0, 1, 1, 0)
	d = p.decide(w, pool(w, "running", "free"), 0, t0)
	if d.Target != 1 || d.Launch != 0 {
		t.Fatalf("one running and one waiting: want 1, rent none; got %+v", d)
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
		{round(36, 0, 0, 0), 8},   // a round starting: 36 entries, two to a machine, at most 8
		{round(3, 0, 0, 0), 2},    // three: each is run twice, by two machines
		{round(5, 2, 1, 4), 4},    // five entries (three machines at runsPerMachine=2) and a script
		{round(8, 0, 0, 0), 4},    // eight entries: four curator machines (2 waves instead of 4)
		{round(0, 3, 0, 5), 0},    // all under way: nothing more for a new machine
		{round(0, 0, 2, 0), 1},    // two scripts left
		{&Work{Waiting: true}, 1}, // an entry waiting, and nobody who runs curators around to have its round drawn
		{&Work{Waiting: true, Runners: 1}, 0},
		{&Work{Waiting: true, Paused: 1}, 0}, // the rounds paused: nothing would be drawn for it
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
		{withFarm(round(5, 0, 1, 0), 9), 7}, // 4 for the round's five entries and its script, 3 for the students
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

// A round's early grades want machines only from --early before its exam is due, until it is
// due; once the exam is posted, its grades are wanted as any are, and when the draw is not
// known, when the exam is late, or with --early 0, none is rented ahead.
func TestEarlyGradesWantMachinesShortlyBeforeTheExam(t *testing.T) {
	p := testPolicy()
	p.PerWorker, p.MaxWorkers, p.Edits = 1, 20, true
	p.Early, p.ExamAfter = 45*time.Minute, 2*time.Hour+12*time.Minute
	drawn := time.Unix(1795207805, 0)
	at := func(r WorkRound, after time.Duration) int {
		w := &Work{Runners: 1, Rounds: []WorkRound{r}}
		return p.targetHolding(w, 0, drawn.Add(after), p.earlyWanted(w, drawn.Add(after)))
	}
	r := WorkRound{ID: "s1-o00004", Entries: 8, GradesEarly: 8, DrawnAt: drawn.Unix()}
	for _, c := range []struct {
		r     WorkRound
		after time.Duration
		want  int
	}{
		{r, 30 * time.Minute, 0},
		{r, 86 * time.Minute, 0},
		{r, 87 * time.Minute, 8}, // 45 minutes before 2h12m
		{r, 131 * time.Minute, 8},
		{r, 3 * time.Hour, 0}, // the exam is late: when it will come is not known, so none is rented ahead
		{WorkRound{ID: r.ID, Entries: 8, GradesEarly: 8}, 2 * time.Hour, 0}, // the draw not known
		{WorkRound{ID: r.ID, Entries: 8, GradesEarly: 3, GradesWanted: 2, ExamAt: drawn.Add(2 * time.Hour).Unix(), DrawnAt: drawn.Unix()}, 2 * time.Hour, 2},
	} {
		if got := at(c.r, c.after); got != c.want {
			t.Errorf("%+v at draw+%s: %d machines, want %d", c.r, c.after, got, c.want)
		}
	}
	p.Early = 0
	if got := at(r, 2*time.Hour); got != 0 {
		t.Errorf("with --early 0: %d machines", got)
	}
}

// When a machine may hold several grades before the exam, fewer are rented for them, and those
// rented already are counted for all they were rented to take.
func TestMachinesThatTakeSeveralGradesAreFewer(t *testing.T) {
	p := testPolicy()
	p.PerWorker, p.MaxWorkers, p.Edits, p.MaxHourly, p.Early = 1, 20, true, 16, 90*time.Minute
	work := func() *Work {
		return &Work{EarlyGradesPerMachine: 3, Rounds: []WorkRound{{ID: "s1-o00008", Entries: 12, GradesEarly: 12, ExamDue: t0.Add(time.Hour).Unix()}}}
	}
	// Nothing rented: 12 grades, three to a machine, want four.
	w := work()
	if d := p.decideNext(w, pool(w, "free", "free", "free", "free", "free", "free"), 0, t0, 3); d.Launch != 4 || d.LaunchEarly != 4 {
		t.Fatalf("for 12 grades three to a machine: %+v", d)
	}
	// With one a machine, as before: six, all the keys there are.
	w = work()
	if d := p.decideNext(w, pool(w, "free", "free", "free", "free", "free", "free"), 0, t0, 1); d.Launch != 6 {
		t.Fatalf("for 12 grades one to a machine: %+v", d)
	}
	// Four are starting, each rented for three: nothing more.
	w = work()
	ws := pool(w, "starting", "starting", "starting", "starting", "free", "free")
	for _, wk := range ws[:4] {
		wk.Plan = 3
	}
	if d := p.decideNext(w, ws, 0, t0, 3); d.Launch != 0 || len(d.End) != 0 {
		t.Fatalf("four starting for three each: %+v", d)
	}
	// They hold one each, and will take two more each: nothing more.
	w = work()
	w.Rounds[0].GradesEarly, w.Rounds[0].GradesUnderWay = 8, 4
	ws = pool(w, "grading", "grading", "grading", "grading", "free", "free")
	for _, wk := range ws[:4] {
		wk.Plan, wk.Machine.Holding = 3, 1
	}
	if d := p.decideNext(w, ws, 0, t0, 3); d.Launch != 0 || len(d.End) != 0 {
		t.Fatalf("four holding one of three each: %+v", d)
	}
	// One was rented late, for one: the two grades it will not take want another machine.
	ws[3].Plan = 1
	if d := p.decideNext(w, ws, 0, t0, 3); d.Launch != 1 {
		t.Fatalf("one rented for one: %+v", d)
	}
	// The server's own time for the exam is taken over --exam-after.
	if due := p.examDue(WorkRound{DrawnAt: t0.Unix(), ExamDue: t0.Add(2 * time.Hour).Unix()}, t0); !due.Equal(t0.Add(2 * time.Hour)) {
		t.Fatalf("the exam is due %s", due)
	}
}

// Near an exam, with grades of its round under way, the fastest two machines doing nothing are
// kept for a grade one that fails leaves, rather than ended for nothing waiting.
func TestTheFastestIdleMachinesAreKeptNearAnExam(t *testing.T) {
	p := testPolicy()
	p.PerWorker, p.MaxWorkers, p.Edits = 1, 20, true
	w := &Work{FastFirstSeconds: 120, Rounds: []WorkRound{{ID: "s1-o00008", GradesUnderWay: 6, ExamDue: t0.Add(4 * time.Minute).Unix()}}}
	ws := pool(w, "waiting", "waiting", "waiting", "grading")
	ws[0].Speed, ws[1].Speed, ws[2].Speed = 1.0, 2.3, 1.8
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-01" {
		t.Fatalf("four minutes before the exam, ended %s", handles(d.End))
	}
	// An hour before it, all three are ended, as before.
	w.Rounds[0].ExamDue = t0.Add(time.Hour).Unix()
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-01,stet-rp-02,stet-rp-03" {
		t.Fatalf("an hour before the exam, ended %s", handles(d.End))
	}
}

// /admin/work as server 0.4.4 says it (stet-dc's, from a test world): its new fields are read,
// and two machines holding grades before the exam, one with three and one with one that will
// take two more, leave no grade wanting another machine.
func TestTheWorkAsServer044SaysIt(t *testing.T) {
	raw, err := os.ReadFile("testdata/admin_work_0.4.4.json")
	if err != nil {
		t.Fatal(err)
	}
	var w Work
	if err := json.Unmarshal(raw, &w); err != nil {
		t.Fatal(err)
	}
	r := w.Rounds[0]
	if w.EarlyGradesPerMachine != 3 || w.FastFirstSeconds != 120 || r.ExamDue != 1795252811 || r.GradesEarly != 1 {
		t.Fatalf("read as %+v", w)
	}
	m1, m2 := w.machine("stet-rp-01"), w.machine("stet-rp-02")
	if m1.Holding != 3 || !m1.Ready || m1.PassSeconds != 608.1 || m2.Holding != 1 || m2.Ready || m2.PassSeconds != 0 {
		t.Fatalf("machines read as %+v %+v", m1, m2)
	}
	p := testPolicy()
	p.PerWorker, p.MaxWorkers, p.Edits, p.Early = 1, 20, true, 90*time.Minute
	now := time.Unix(r.ExamDue, 0).Add(-30 * time.Minute)
	created := now.Add(-time.Hour)
	var ws []*Worker
	for _, h := range []string{"stet-rp-01", "stet-rp-02", "stet-rp-03"} {
		ws = append(ws, &Worker{Key: &Key{Handle: h}, Machine: w.machine(h)})
	}
	for _, wk := range ws[:2] {
		wk.Rental, wk.Plan = &Rental{Name: wk.Key.Handle, State: "running", Created: created}, 3
		wk.Machine.LastSeen = now.Unix()
	}
	if d := p.decideNext(&w, ws, 0, now, 3); d.LaunchEarly != 0 || len(d.End) != 0 {
		t.Fatalf("with rp-02 to take two more: %+v", d)
	}
	// Were rp-02 rented for one, the grade still wanted would want a machine.
	ws[1].Plan = 1
	if d := p.decideNext(&w, ws, 0, now, 3); d.LaunchEarly != 1 {
		t.Fatalf("with rp-02 rented for one: %+v", d)
	}
}

// A machine waiting for work before the server's window for the exam's early grades opens for
// its kind, when early grades are all that is wanted, is ended: it would only wait, spending,
// and another is rented in time. One whose window has opened is kept, and so is one when there
// is a curator to run.
func TestAMachineWaitingTooEarlyIsEnded(t *testing.T) {
	p := testPolicy()
	p.PerWorker, p.MaxWorkers, p.Edits, p.Early = 1, 20, true, 2*time.Hour
	work := func() *Work {
		return &Work{EarlyGradesPerMachine: 3, EarlyWindowSeconds: 600,
			Rounds: []WorkRound{{ID: "s1-o00009", Entries: 9, GradesEarly: 6, ExamDue: t0.Add(90 * time.Minute).Unix()}}}
	}
	w := work()
	ws := pool(w, "waiting", "waiting", "free", "free")
	ws[0].Window = t0.Add(40 * time.Minute) // its window opens in 40 minutes
	ws[1].Window = t0.Add(-time.Minute)     // its has
	d := p.decideNext(w, ws, 0, t0, 3)
	if handles(d.End) != "stet-rp-01" || !strings.Contains(d.End[0].Why, "too early") || d.End[0].Failed {
		t.Fatalf("ended %s (%+v)", handles(d.End), d.End)
	}
	// With a curator still to run, nobody is ended for it.
	w = work()
	w.Rounds[0].RunsWanted = 2
	ws = pool(w, "waiting", "free")
	ws[0].Window = t0.Add(40 * time.Minute)
	if d := p.decideNext(w, ws, 0, t0, 3); len(d.End) != 0 {
		t.Fatalf("ended %s with curators to run", handles(d.End))
	}
}

// A machine rented to run curators that may not run them is ended while only curators want
// machines, so that one that may is rented in its place; a grade it could take keeps it.
func TestAMachineThatMayNotRunCuratorsMakesWayForOneThatMay(t *testing.T) {
	p := testPolicy()
	p.PerWorker, p.MaxWorkers, p.Edits = 1, 20, true
	work := func() *Work {
		return &Work{Rounds: []WorkRound{{ID: "s1-o00009", Entries: 4, RunsWanted: 4, RunsWaiting: 4}}}
	}
	const why = "under the season's floor for running curators: CPU 95 MB/s, floor 100"
	w := work()
	ws := pool(w, "waiting", "running", "free", "free")
	ws[0].NoCurators = why
	d := p.decideNext(w, ws, 0, t0, 1)
	if handles(d.End) != "stet-rp-01" || !strings.Contains(d.End[0].Why, "may not run curators ("+why+")") || d.End[0].Failed {
		t.Fatalf("ended %s (%+v)", handles(d.End), d.End)
	}
	if d.Launch != 1 || d.LaunchRuns != 1 {
		t.Fatalf("rented %d, %d of them to run curators, in its place", d.Launch, d.LaunchRuns)
	}
	// A grade of a posted exam wants it: it is kept, and grades.
	w = work()
	w.Rounds[0].GradesWanted = 3
	ws = pool(w, "waiting", "running", "free", "free")
	ws[0].NoCurators = why
	if d := p.decideNext(w, ws, 0, t0, 1); len(d.End) != 0 {
		t.Fatalf("ended %s with a grade for it", handles(d.End))
	}
	// So does an early grade, once its window has opened, and not before.
	w = work()
	w.Rounds[0].GradesEarly, w.Rounds[0].ExamDue, w.EarlyGradesPerMachine, w.EarlyWindowSeconds = 6, t0.Add(30*time.Minute).Unix(), 3, 600
	p.Early = 2 * time.Hour
	ws = pool(w, "waiting", "waiting", "free")
	ws[0].NoCurators, ws[1].NoCurators = why, why
	ws[0].Window, ws[1].Window = t0.Add(-time.Minute), t0.Add(20*time.Minute)
	if d := p.decideNext(w, ws, 0, t0, 1); handles(d.End) != "stet-rp-02" {
		t.Fatalf("ended %s", handles(d.End))
	}
}

// During the 6-16m notice before an on-demand scheduled draw, up to keptNearDraw warm curator-capable
// runners are kept alive (so the runner that triggered the draw isn't killed before the draw fires),
// while extra idle GPUs and grade-only machines wind down immediately; draws far in the future (>20m)
// or quiet/paused periods end all idle GPUs; and a trigger runner checking in right before the next
// draw is scheduled (w.Waiting && w.Runners > 0) is also kept alive.
func TestWarmRunnersAreKeptAcrossImminentDrawNoticeAndEndedWhenQuiet(t *testing.T) {
	p := testPolicy()
	p.Edits, p.MaxWorkers = true, 6
	w := &Work{Runners: 1, NextDrawAt: t0.Add(10 * time.Minute).Unix()}
	ws := pool(w, "waiting", "waiting", "waiting", "free")
	ws[0].Speed, ws[1].Speed, ws[2].Speed = 1.0, 2.5, 1.8
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-01" {
		t.Fatalf("10m before scheduled draw, want only slowest extra ended; ended %s", handles(d.End))
	}
	// A grade-only machine must NEVER be kept warm for a draw, even if it is the fastest.
	ws[1].GradeOnly = true
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-02" {
		t.Fatalf("grade-only machine must not be kept warm near draw; ended %s", handles(d.End))
	}
	ws[1].GradeOnly = false
	// A draw outside the 20m keep-alive window (e.g. 40m away) must NOT keep idle GPUs warm.
	w.NextDrawAt = t0.Add(40 * time.Minute).Unix()
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-01,stet-rp-02,stet-rp-03" {
		t.Fatalf("40m before draw, want all idle GPUs ended; ended %s", handles(d.End))
	}
	// During the brief gap between a trigger runner asking for work (Runners==1) and NextDrawAt
	// appearing (Waiting==true, NextDrawAt==0), keep the trigger runner alive.
	w.NextDrawAt, w.Waiting, w.Runners = 0, true, 1
	wsGap := pool(w, "waiting", "free")
	if d := p.decide(w, wsGap, 0, t0); len(d.End) != 0 {
		t.Fatalf("trigger runner killed during Waiting&&Runners>0 gap: ended %s", handles(d.End))
	}
	// If only a grade-only machine is idle when w.Waiting&&Runners==0, end it and rent a curator runner.
	w.NextDrawAt, w.Waiting, w.Runners = 0, true, 0
	wsGrade := pool(w, "waiting", "free")
	wsGrade[0].GradeOnly = true
	if d := p.decide(w, wsGrade, 0, t0); handles(d.End) != "stet-rp-01" || d.LaunchRuns != 1 {
		t.Fatalf("grade-only machine should make way for draw-trigger runner: end=%s launchRuns=%d", handles(d.End), d.LaunchRuns)
	}
	// Once no draw is scheduled (NextDrawAt == 0, quiet), all idle GPUs end immediately.
	w.NextDrawAt, w.Waiting, w.Runners = 0, false, 1
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-01,stet-rp-02,stet-rp-03" {
		t.Fatalf("when quiet, want all idle GPUs ended; ended %s", handles(d.End))
	}
	// If rounds are paused, even an imminent draw keeps nobody.
	w.NextDrawAt, w.Paused = t0.Add(10*time.Minute).Unix(), t0.Unix()
	if d := p.decide(w, ws, 0, t0); handles(d.End) != "stet-rp-01,stet-rp-02,stet-rp-03" {
		t.Fatalf("when paused, want all idle GPUs ended; ended %s", handles(d.End))
	}
}

// Review 10 P2 #6: when two baseline graders disagree and sit idle ("waiting"), they cannot break
// their own baseline tie, so stet-cloud must rent a fresh machine rather than waiting out --idle.
// Crucially, on a normal undisputed edit-to-grade transition (BaselineDisputed == false,
// FreshGradersWanted == 0), healthy idle GPUs must NOT be evicted or duplicated.
func TestTieBreakBaselineRentsAFreshMachineWhileBaselineGradersIdle(t *testing.T) {
	p := testPolicy()
	p.Edits, p.MaxWorkers = true, 6
	one := 1
	w := &Work{Rounds: []WorkRound{{
		ID: "s3-o00001", Entries: 3, GradesWanted: 1, BaselinesWanted: &one,
		BaselineDisputed: true, FreshGradersWanted: &one,
	}}}
	ws := pool(w, "waiting", "waiting", "free", "free")
	d := p.decide(w, ws, 0, t0)
	if d.Launch == 0 {
		t.Fatalf("stuck: two idle graders cannot break their own baseline tie; Launch=%d End=%v",
			d.Launch, handles(d.End))
	}
	// Once that fresh machine is starting, no additional machines are launched on the next tick.
	wsStarting := pool(w, "waiting", "waiting", "starting", "free")
	if d2 := p.decide(w, wsStarting, 0, t0); d2.Launch != 0 {
		t.Fatalf("with fresh tie-break grader already starting, should not launch more: Launch=%d", d2.Launch)
	}
	// Even at MaxWorkers == 2 (at capacity with 2 idle graders), make room for 1 fresh machine.
	p.MaxWorkers = 2
	wsCap := pool(w, "waiting", "waiting", "free", "free")
	wsCap[0].Speed, wsCap[1].Speed = 1.0, 2.0
	if d3 := p.decide(w, wsCap, 0, t0); d3.Launch != 1 || handles(d3.End) != "stet-rp-01" {
		t.Fatalf("at MaxWorkers=2, want slowest idle grader ended and 1 fresh launched: Launch=%d End=%s", d3.Launch, handles(d3.End))
	}
	// Explicit BaselineDisputed flag launches a fresh machine even while other grades are under way.
	p.MaxWorkers = 6
	wDisp := &Work{Rounds: []WorkRound{{
		ID: "s3-o00001", Entries: 3, GradesWanted: 1, GradesUnderWay: 1, BaselinesWanted: &one, BaselineDisputed: true,
	}}}
	if d4 := p.decide(wDisp, pool(wDisp, "waiting", "waiting", "grading", "free"), 0, t0); d4.Launch != 1 {
		t.Fatalf("explicit BaselineDisputed with other grade under way: want Launch=1, got %d", d4.Launch)
	}
	// Explicit FreshGradersWanted (e.g. two idle graders disagreed on tokens_sha256) launches a 3rd.
	zero := 0
	wTok := &Work{Rounds: []WorkRound{{
		ID: "s3-o00001", Entries: 1, GradesWanted: 1, BaselinesWanted: &zero, FreshGradersWanted: &one,
	}}}
	if d5 := p.decide(wTok, pool(wTok, "waiting", "waiting", "free"), 0, t0); d5.Launch != 1 {
		t.Fatalf("FreshGradersWanted=1 for script tie-break: want Launch=1, got %d", d5.Launch)
	}
	// Normal undisputed edit-to-grade transition: 2 GPUs finished edits and wait for first grade
	// (BaselinesWanted=2, GradesWanted=2, GradesUnderWay=0, BaselineDisputed=false, FreshGradersWanted=0).
	// Must NOT over-rent below MaxWorkers and must NOT evict healthy idle GPUs at MaxWorkers.
	two := 2
	wNormal := &Work{Rounds: []WorkRound{{
		ID: "s3-o00002", Entries: 3, GradesWanted: 2, BaselinesWanted: &two,
	}}}
	p.MaxWorkers = 6
	if d6 := p.decide(wNormal, pool(wNormal, "waiting", "waiting", "free", "free"), 0, t0); d6.Launch != 0 || len(d6.End) != 0 {
		t.Fatalf("normal undisputed round start (Max=6): want Launch=0 End=[], got Launch=%d End=%s", d6.Launch, handles(d6.End))
	}
	p.MaxWorkers = 2
	if d7 := p.decide(wNormal, pool(wNormal, "waiting", "waiting", "free", "free"), 0, t0); d7.Launch != 0 || len(d7.End) != 0 {
		t.Fatalf("normal undisputed round start (Max=2): want Launch=0 End=[], got Launch=%d End=%s", d7.Launch, handles(d7.End))
	}
	// Review 11 P2 #3: with 6 configured keys, --max 6, 4 busy and 2 idle graders who already
	// measured the disputed baseline (ZERO free keys!), policy MUST NOT evict an idle grader
	// (which would just free its barred key to be re-rented next tick) and MUST report what key
	// resource the operator needs.
	p.MaxWorkers = 6
	wsNoSpare := pool(w, "grading", "grading", "grading", "grading", "waiting", "waiting")
	d8 := p.decide(w, wsNoSpare, 0, t0)
	if len(d8.End) != 0 || d8.Launch != 0 || !strings.Contains(d8.Note, "stet-cloud keys --add") {
		t.Fatalf("all 6 keys committed on dispute: want End=[], Launch=0, actionable Note; got End=%s Launch=%d Note=%q",
			handles(d8.End), d8.Launch, d8.Note)
	}
	// Even with a 7th free key, if that key ALREADY participated in the current open round
	// (LastSeen >= DrawnAt), it cannot be treated as a fresh spare key to justify eviction.
	wDrawn := &Work{Rounds: []WorkRound{{
		ID: "s3-o00001", Entries: 3, GradesWanted: 1, BaselinesWanted: &one,
		BaselineDisputed: true, FreshGradersWanted: &one, DrawnAt: t0.Add(-15 * time.Minute).Unix(),
	}}}
	wsUsedSpare := pool(wDrawn, "grading", "grading", "grading", "grading", "waiting", "waiting", "free")
	wsUsedSpare[6].Machine = &Machine{Handle: wsUsedSpare[6].Key.Handle, LastSeen: t0.Add(-5 * time.Minute).Unix()}
	d9 := p.decide(wDrawn, wsUsedSpare, 0, t0)
	if len(d9.End) != 0 || d9.Launch != 0 || !strings.Contains(d9.Note, "stet-cloud keys --add") {
		t.Fatalf("7th free key already active in current round: want End=[], Launch=0, Note; got End=%s Launch=%d Note=%q",
			handles(d9.End), d9.Launch, d9.Note)
	}
	// With a genuinely fresh 7th spare key (LastSeen < DrawnAt), evict 1 slowest idle grader to
	// make room and launch 1 fresh machine; quietestExcluding never selects the evicted key.
	wsFreshSpare := pool(wDrawn, "grading", "grading", "grading", "grading", "waiting", "waiting", "free")
	wsFreshSpare[4].Speed, wsFreshSpare[5].Speed = 1.0, 2.0
	wsFreshSpare[6].Machine = &Machine{Handle: wsFreshSpare[6].Key.Handle, LastSeen: t0.Add(-2 * time.Hour).Unix()}
	d10 := p.decide(wDrawn, wsFreshSpare, 0, t0)
	if len(d10.End) != 1 || d10.End[0].Worker != wsFreshSpare[4] || d10.Launch != 1 {
		t.Fatalf("with 1 fresh 7th spare key: want slowest idle evicted and Launch=1; got End=%s Launch=%d",
			handles(d10.End), d10.Launch)
	}
	// Simulate ending the evicted worker, then confirm quietestExcluding selects ONLY the spare key.
	wsFreshSpare[4].Rental = nil
	evicted := map[string]bool{d10.End[0].Worker.Key.Handle: true}
	picked := quietestExcluding(wsFreshSpare, evicted)
	if len(picked) != 1 || picked[0].Key.Handle != wsFreshSpare[6].Key.Handle {
		t.Fatalf("quietestExcluding must select fresh spare key %s, got %v", wsFreshSpare[6].Key.Handle, picked)
	}
	// Gate 4 P2 #4 Variant A: waitingIdle == 0 because both disputed graders self-ended after --idle,
	// leaving 2 free keys that ALREADY participated in this round (LastSeen >= DrawnAt -> freshFree == 0).
	// Policy MUST clamp Launch to 0 rather than renting a barred key in an idle-burn loop.
	wsBothSelfEnded := pool(wDrawn, "grading", "grading", "grading", "grading", "free", "free")
	wsBothSelfEnded[4].Machine = &Machine{Handle: wsBothSelfEnded[4].Key.Handle, LastSeen: t0.Add(-5 * time.Minute).Unix()}
	wsBothSelfEnded[5].Machine = &Machine{Handle: wsBothSelfEnded[5].Key.Handle, LastSeen: t0.Add(-4 * time.Minute).Unix()}
	d11 := p.decide(wDrawn, wsBothSelfEnded, 0, t0)
	if d11.Launch != 0 || !strings.Contains(d11.Note, "stet-cloud keys --add") {
		t.Fatalf("disputed baseline with only barred free keys (waitingIdle=0): want Launch=0 & keys note; got Launch=%d Note=%q",
			d11.Launch, d11.Note)
	}
	// Gate 4 P2 #4 Variant B: GradesWanted=2 on a disputed round with waitingIdle=1 and 2 barred free keys (freshFree=0).
	p.PerWorker = 1
	wDisp2 := &Work{Rounds: []WorkRound{{
		ID: "s3-o00001", Entries: 3, GradesWanted: 2, BaselinesWanted: &one,
		BaselineDisputed: true, FreshGradersWanted: &one, DrawnAt: t0.Add(-15 * time.Minute).Unix(),
	}}}
	wsVarB := pool(wDisp2, "grading", "grading", "grading", "waiting", "free", "free")
	wsVarB[4].Machine = &Machine{Handle: wsVarB[4].Key.Handle, LastSeen: t0.Add(-5 * time.Minute).Unix()}
	wsVarB[5].Machine = &Machine{Handle: wsVarB[5].Key.Handle, LastSeen: t0.Add(-4 * time.Minute).Unix()}
	d12 := p.decide(wDisp2, wsVarB, 0, t0)
	if d12.Launch != 0 || len(d12.End) != 0 || !strings.Contains(d12.Note, "stet-cloud keys --add") {
		t.Fatalf("disputed baseline Variant B (freshFree=0): want Launch=0 End=[] & keys note; got Launch=%d End=%s Note=%q",
			d12.Launch, handles(d12.End), d12.Note)
	}
	// Gate 4 P2 #6: when pol.MaxWorkers is lowered to 5 to reserve 1 slot for a pending examiner
	// while 6 workers are already live (4 grading + 2 waiting) and runs are waiting, evict 1 slowest
	// idle worker immediately so the examiner is never starved.
	p.MaxWorkers = 5
	wNeedEx := &Work{Rounds: []WorkRound{{ID: "s3-o00003", Entries: 4, RunsWanted: 4, ExamAt: 0}}}
	wsOver := pool(wNeedEx, "grading", "grading", "grading", "grading", "waiting", "waiting")
	wsOver[4].Speed, wsOver[5].Speed = 1.0, 2.0
	d13 := p.decide(wNeedEx, wsOver, 0, t0)
	if len(d13.End) != 1 || d13.End[0].Worker != wsOver[4] || d13.Launch != 0 {
		t.Fatalf("over --max with reserved examiner slot: want slowest idle worker evicted and Launch=0; got End=%s Launch=%d",
			handles(d13.End), d13.Launch)
	}
	// Review 12 P2 #8: when the server explicitly sends fresh_graders_wanted=0 alongside
	// baseline_disputed=true (because an idle organiser machine already online is eligible to
	// break the tie), policy MUST honor that 0 — neither evicting an idle worker nor renting a
	// fresh machine.
	p.MaxWorkers = 3
	p.PerWorker = 6
	wEligibleIdle := &Work{Rounds: []WorkRound{{
		ID: "s3-o00004", Entries: 3, GradesWanted: 1, BaselinesWanted: &one,
		BaselineDisputed: true, FreshGradersWanted: &zero,
	}}}
	wsThreeIdle := pool(wEligibleIdle, "waiting", "waiting", "waiting", "free")
	d14 := p.decide(wEligibleIdle, wsThreeIdle, 0, t0)
	if len(d14.End) != 0 || d14.Launch != 0 {
		t.Fatalf("explicit fresh_graders_wanted=0 on disputed baseline must not evict or rent: End=%s Launch=%d",
			handles(d14.End), d14.Launch)
	}
}
