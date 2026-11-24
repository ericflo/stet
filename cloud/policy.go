package main

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// --- how many machines the work wants ---------------------------------------------------------
//
// A rented machine runs curators, in stet's container sandbox, and grades. A grade holds its
// machine for twenty minutes or so at the standard scale, and every script waiting wants a
// machine for every --per-machine of them. A curator is run in a minute or two, and the
// round's exam, which its grades wait for, is made hours after the draw: the curators want
// only a machine for every runsPerMachine entries, and never fewer than runsEach, since an
// account runs an entry once.
//
// With --no-edits they only grade, and curators are run by the organisers' own machines.
// Scripts to grade then come from those runs, as fast as the runners make them: the
// machines wanted are enough to keep pace with the runners while curators are still to
// run, and enough to clear the scripts already waiting.
//
// A grader can take a grade before the round's exam is posted, train, and sit the exam once
// it is: the grades it could take want machines from --early before the exam is due, each
// kind rented just in time to have trained by then, so that the round settles minutes after
// it is posted. Rented sooner, a machine would only wait.
//
// An entry waiting for a round has it drawn early only once somebody who runs curators has
// asked for work. When nobody has, one machine is rented to ask, unless the organisers have
// paused the rounds: none is drawn then, and it would only wait to stop.
//
// A machine with nothing official to do trains the organisers' students (a few minutes
// each), so those waiting want machines too, after the official work and under the same
// limits: one for every --per-farm of them. A bigger student holds a machine for hours,
// and is handed only to one rented to take it (--long-farm): one machine each.
//
// Letting machines go needs no decision here. A rented machine stops by itself once it has
// had nothing to do for a while, and ends its own pod. What is decided here is ending, at
// once, the ones that are doing nothing when nothing at all is waiting, and those that
// have gone quiet or stopped.

type Policy struct {
	Edits       bool    // the machines run curators as well as grade
	MaxWorkers  int     // never more machines than this
	PerRunner   float64 // grading only: machines that keep pace with one curator runner
	PerWorker   int     // jobs waiting that justify one more machine
	PerFarm     int     // the organisers' students waiting that justify one more machine; 0: none
	LongFarm    bool    // the machines take the organisers' bigger students too, one machine to each
	MaxHourly   float64 // never rent past this many dollars an hour, all machines together
	DailyBudget float64 // no new machine once this much has been spent today (UTC); 0: no limit
	// Rent graders this long before a round's exam is due, ExamAfter its draw (0: only once
	// it is posted).
	Early, ExamAfter time.Duration
	Grace            time.Duration
	Silence          time.Duration
}

// target is how many machines the work justifies, before money is counted.
func (p Policy) target(w *Work) int {
	now := time.Now()
	return p.targetHolding(w, 0, now, p.earlyWanted(w, now))
}

// targetHolding counts too the machines rented now that train a bigger student (held): each
// is busy with it for hours, and so is none of the machines the students still waiting want.
// early is the machines the grades taken before an exam want (earlyMachines).
func (p Policy) targetHolding(w *Work, held int, now time.Time, early int) int {
	n := p.official(w, early)
	if p.PerFarm > 0 {
		n += (w.FarmWaiting + p.PerFarm - 1) / p.PerFarm
	}
	if p.LongFarm {
		n += w.FarmLongWaiting + held // a machine to each, held for hours
	}
	return min(n, p.MaxWorkers)
}

// runsEach is how many times the season runs an entry's curator, each by another account,
// and runsPerMachine how many entries to run justify one more machine.
const runsEach, runsPerMachine = 2, 4

// runners is how many of the machines the rounds' work justifies run curators: one for
// every runsPerMachine entries to run, and never fewer than runsEach; or one to ask for
// work, so that a waiting entry's round is drawn. None with --no-edits.
func (p Policy) runners(w *Work) int {
	if !p.Edits {
		return 0
	}
	runs := 0
	for _, r := range w.Rounds {
		runs += r.RunsWanted
	}
	switch {
	case len(w.Rounds) == 0 && w.Waiting && w.Runners == 0 && w.Paused == 0:
		return 1 // to ask for work, so that the waiting entry's round is drawn
	case runs > 0:
		return max(runsEach, (runs+runsPerMachine-1)/runsPerMachine)
	}
	return 0
}

// official is how many machines the rounds' own work justifies: early is what the grades taken
// before an exam want, a machine for each grade unless machines take several (earlyMachines).
func (p Policy) official(w *Work, early int) int {
	per := max(p.PerWorker, 1)
	if p.Edits {
		grades := early
		for _, r := range w.Rounds {
			grades += r.GradesWanted
		}
		return (grades+per-1)/per + p.runners(w)
	}
	scripts, curators := early, 0
	for _, r := range w.Rounds {
		scripts += r.GradesWanted
		curators += r.RunsWaiting + r.EditsUnderWay
	}
	n := (scripts + per - 1) / per
	if curators > 0 {
		n += int(math.Ceil(p.PerRunner * float64(max(w.Runners, 1))))
	}
	return n
}

// examDue is when a round's exam is expected, about: when the server says it is, or else
// --exam-after its draw. Zero once it is posted, when that is not known, and once that time
// has passed with no exam: when a late exam will come is not known (in Season 2 it is posted
// at the draw, and one that is late waits on the examiner), so nothing is rented to take its
// grades before it, nor kept waiting for them, until it is posted.
func (p Policy) examDue(r WorkRound, now time.Time) time.Time {
	var due time.Time
	switch {
	case r.ExamAt != 0:
		return time.Time{}
	case r.ExamDue > 0:
		due = time.Unix(r.ExamDue, 0)
	case r.DrawnAt == 0 || p.ExamAfter <= 0:
		return time.Time{}
	default:
		due = time.Unix(r.DrawnAt, 0).Add(p.ExamAfter)
	}
	if !now.Before(due) {
		return time.Time{}
	}
	return due
}

// early is how many of a round's grades want a machine before its exam is posted: those a
// grader could take now, once the exam is due within --early. A machine's first grade of a
// round trains the round's baseline and the student before it can sit the exam, and the
// boot and the corpus take a while too: which kind is rented, and when, so that it has
// trained by the time the exam is posted, is forExam's. None when the exam is posted (its
// grades are wanted then), or when the draw is not known.
func (p Policy) early(r WorkRound, now time.Time) int {
	due := p.examDue(r, now)
	if p.Early <= 0 || r.GradesEarly <= 0 || due.IsZero() || now.Before(due.Add(-p.Early)) {
		return 0
	}
	return r.GradesEarly
}

// earlyWanted is the grades the rounds want taken before their exams now.
func (p Policy) earlyWanted(w *Work, now time.Time) int {
	n := 0
	for _, r := range w.Rounds {
		n += p.early(r, now)
	}
	return n
}

// earlyMachines is how many machines the grades taken before an exam want: grades is how many
// are wanted that no machine holds, and each new machine is taken to take next of them (one,
// unless the server lets a machine hold several before the exam). Machines rented already to
// take several take theirs first: one that is starting, or waiting, all it was rented for, and
// one that is grading the rest of them. Those starting or waiting are counted among the
// machines wanted, as they are among those rented, so that they are not rented again.
func (p Policy) earlyMachines(grades, next int, workers []*Worker, now time.Time, ending map[*Worker]bool) (machines, renting int) {
	if grades <= 0 {
		return 0, 0
	}
	next = max(next, 1)
	idle := 0
	for _, wk := range workers {
		if wk.Plan <= 1 || ending[wk] {
			continue
		}
		switch wk.state(now, p) {
		case "grading":
			holding := 1
			if wk.Machine != nil {
				holding = max(wk.Machine.Holding, 1)
			}
			grades -= max(0, wk.Plan-holding)
		case "starting", "waiting":
			grades -= wk.Plan
			idle++
		}
	}
	renting = (max(grades, 0) + next - 1) / next
	return renting + idle, renting
}

// Worker is one key of the pool, with the machine rented with it if there is one, and what
// the server says of it.
type Worker struct {
	Key     *Key
	Rental  *Rental
	Machine *Machine
	// Plan is how many grades it was rented to take before an exam (0 or 1: one).
	Plan int
	// Grace is how long it may take to first ask for work (0: the policy's).
	Grace time.Duration
	// Silence is how long it may say nothing, holding no job, before it is taken to have gone
	// quiet (0: the policy's). Where a machine cannot end itself, it waits to be ended once
	// its stet stops, and says nothing.
	Silence time.Duration
	// Speed is its card's, against a 3090 (0: not known).
	Speed float64
	// Window is the latest a machine of its kind could be rented and still be in time for
	// the first exam's window for early grades (zero: no window): one waiting for work
	// before then, when nothing else wants it, is ended, and another rented in time.
	Window time.Time
	// NoCurators is why the server would not hand it curators to run, rented to run them
	// ("": it would, or it was not): it grades only (curators.go).
	NoCurators string
}

// gradeFor says whether something other than a curator wants a machine now: a grade of a posted
// exam, an early grade whose window has opened for its kind, or a student to train.
func (p Policy) gradeFor(w *Work, wk *Worker, now time.Time) bool {
	if w.FarmWaiting > 0 || (p.LongFarm && w.FarmLongWaiting > 0) {
		return true
	}
	for _, r := range w.Rounds {
		if r.GradesWanted > 0 {
			return true
		}
	}
	return p.earlyWanted(w, now) > 0 && (wk.Window.IsZero() || !now.Before(wk.Window))
}

// tooEarly are the machines waiting for work whose window for the first exam's early grades
// is still to open, when early grades are all that is wanted: no curator to run, no grade of
// a posted exam, no student. Each would only wait, spending, until then.
func (p Policy) tooEarly(w *Work, workers []*Worker, now time.Time) map[*Worker]bool {
	if p.earlyWanted(w, now) == 0 || w.FarmWaiting > 0 || (p.LongFarm && w.FarmLongWaiting > 0) {
		return nil
	}
	for _, r := range w.Rounds {
		if r.RunsWanted > 0 || r.GradesWanted > 0 {
			return nil
		}
	}
	out := map[*Worker]bool{}
	for _, wk := range workers {
		if !wk.Window.IsZero() && now.Before(wk.Window) && wk.state(now, p) == "waiting" {
			out[wk] = true
		}
	}
	return out
}

// state says what a machine is doing, for deciding and for showing.
//
//	free      no machine
//	starting  a machine, and stet has not yet asked the server for work from it
//	grading   it holds a grade job
//	running   it holds a curator to run
//	training  it trains one of the organisers' students
//	waiting   it asks for work and has none
//	quiet     it asked, and then nothing for longer than any job takes without asking
//	stopped   the machine stopped, or never got going
func (w *Worker) state(now time.Time, p Policy) string {
	switch {
	case w.Rental == nil:
		return "free"
	case w.Rental.State == "stopped":
		return "stopped"
	case w.Machine != nil && w.Machine.Busy == "grade":
		return "grading"
	case w.Machine != nil && w.Machine.Busy == "farm":
		return "training"
	case w.Machine != nil && w.Machine.Busy != "":
		return "running"
	}
	seen := int64(0)
	if w.Machine != nil {
		seen = w.Machine.LastSeen
	}
	switch {
	case seen >= w.Rental.Created.Unix() && now.Sub(time.Unix(seen, 0)) <= w.silence(p):
		return "waiting"
	case seen >= w.Rental.Created.Unix():
		return "quiet"
	case now.Sub(w.Rental.Created) <= w.grace(p):
		return "starting"
	default:
		return "stopped" // it never asked for work: the install failed, or stet would not start
	}
}

func (w *Worker) silence(p Policy) time.Duration {
	if w.Silence > 0 {
		return w.Silence
	}
	return p.Silence
}

func (w *Worker) grace(p Policy) time.Duration {
	if w.Grace > 0 {
		return w.Grace
	}
	return p.Grace
}

// Ending is a machine to end, and why. Failed: it never asked for work, or went quiet; what
// became of it is asked of its market before it is ended.
type Ending struct {
	Worker *Worker
	Why    string
	Failed bool
}

// Decision is one look at the work and the machines.
type Decision struct {
	// Machines the work justifies: those at a grade, and enough for what is waiting.
	Target int
	Live   int // machines rented now
	Launch int // machines to rent now
	// Of those, the machines rented for the organisers' bigger students: chosen for what one
	// of them costs on it, under --long-max-price.
	LaunchLong int
	// Of the rest, the machines rented for grades taken before an exam, due at ExamDue: each
	// kind just in time to have trained by then.
	LaunchEarly int
	ExamDue     time.Time
	// Of the rest, the machines rented to run curators, where they may run: the fastest first,
	// as the rest are, since once its runs are done such a machine takes grades the round waits on.
	LaunchRuns int
	End        []Ending // machines to end now
	// Keys no machine is rented with that the server says still hold a job: ended a moment
	// after they took it, or stopped partway through. What they hold is handed back.
	Release []*Worker
	Note    string // why fewer are rented than wanted, if they are
}

func (p Policy) decide(w *Work, workers []*Worker, spentToday float64, now time.Time) Decision {
	return p.decideNext(w, workers, spentToday, now, 1)
}

// nearExam says whether an exam is due within five minutes, or was posted within the time
// the server keeps a grade still wanted for the fastest card free, with grades of its round
// under way: a fast machine doing nothing is kept then, since one that fails leaves a grade
// for it. In round 6 an idle 4090 was ended minutes before the exam.
func (p Policy) nearExam(w *Work, now time.Time) bool {
	window := max(time.Duration(w.FastFirstSeconds)*time.Second, 2*time.Minute)
	for _, r := range w.Rounds {
		if r.GradesUnderWay+r.GradesWanted == 0 {
			continue
		}
		if due := p.examDue(r, now); !due.IsZero() && due.Sub(now) <= 5*time.Minute {
			return true
		}
		if r.ExamAt != 0 && now.Sub(time.Unix(r.ExamAt, 0)) <= window {
			return true
		}
	}
	return false
}

// keptNearExam is how many idle machines are kept near an exam: the fastest two.
const keptNearExam = 2

// nearDraw says whether an on-demand round is already scheduled to draw shortly (within the
// draw notice window) while no round is open yet: keep up to keptNearDraw warm runners alive
// across the 6-16 minute notice so a runner rented for w.Waiting is not killed right before
// the draw it triggered, and consecutive back-to-back rounds don't churn rentals.
func (p Policy) nearDraw(w *Work, now time.Time) bool {
	if !p.Edits || len(w.Rounds) != 0 || w.Paused != 0 || w.NextDrawAt <= 0 {
		return false
	}
	at := time.Unix(w.NextDrawAt, 0)
	return at.After(now.Add(-2*time.Minute)) && at.Sub(now) <= 20*time.Minute
}

const keptNearDraw = 2

// decideNext decides, where a machine rented now for grades taken before an exam is taken to
// take next of them (forExam's plan for the kind it would rent).
func (p Policy) decideNext(w *Work, workers []*Worker, spentToday float64, now time.Time, next int) Decision {
	// A machine at a grade takes no other until it is done, twenty minutes or so at the
	// standard scale: what is waiting wants machines of its own. One running a curator is
	// done in a minute or two, and counts among the machines the curators want.
	held, working := 0, 0
	for _, wk := range workers {
		switch wk.state(now, p) {
		case "grading":
			working++
		case "training":
			if wk.Machine.Long {
				held++
			}
		}
	}
	ending := p.tooEarly(w, workers, now)
	early, renting := p.earlyMachines(p.earlyWanted(w, now), next, workers, now, ending)
	waiting := p.targetHolding(w, held, now, early)
	d := Decision{Target: min(waiting+working, p.MaxWorkers)}
	free := 0
	var idle []*Worker
	for _, wk := range workers {
		switch st := wk.state(now, p); st {
		case "free":
			free++
			if wk.Machine != nil && wk.Machine.Busy != "" {
				d.Release = append(d.Release, wk)
			}
		case "stopped":
			d.End = append(d.End, Ending{wk, "it stopped, or never asked for work", true})
		case "quiet":
			d.End = append(d.End, Ending{wk, fmt.Sprintf("nothing heard from it for %s, and it holds no job", now.Sub(time.Unix(wk.Machine.LastSeen, 0)).Round(time.Minute)), true})
		case "waiting":
			if ending[wk] {
				d.End = append(d.End, Ending{Worker: wk, Why: fmt.Sprintf("too early for the exam's grades: a machine of its kind is rented again in time, from %s UTC", wk.Window.UTC().Format("15:04"))})
				continue
			}
			if wk.NoCurators != "" && p.runners(w) > 0 && !p.gradeFor(w, wk, now) {
				// Rented to run curators, and it may not: it would only wait, and stand in the
				// place of one that may.
				d.End = append(d.End, Ending{Worker: wk, Why: "it may not run curators (" + wk.NoCurators + "), and no grade wants it yet"})
				continue
			}
			d.Live++
			idle = append(idle, wk)
		default:
			d.Live++
			if st == "waiting" || st == "starting" {
				idle = append(idle, wk)
			}
		}
	}
	if waiting == 0 {
		// Nothing is waiting for anybody: a machine doing nothing would only wait to stop. Near
		// an exam, the fastest that are waiting are kept for a grade a failed machine leaves.
		// Near an imminent scheduled draw (e.g., during the 6-16m on-demand notice window), up
		// to keptNearDraw warm runners are kept so the draw starts without cold-boot churn.
		kept := map[*Worker]bool{}
		if p.nearExam(w, now) {
			var fast []*Worker
			for _, wk := range idle {
				if wk.state(now, p) == "waiting" {
					fast = append(fast, wk)
				}
			}
			sort.SliceStable(fast, func(i, j int) bool { return fast[i].Speed > fast[j].Speed })
			for _, wk := range fast[:min(len(fast), keptNearExam)] {
				kept[wk] = true
			}
		} else if p.nearDraw(w, now) {
			var warm []*Worker
			for _, wk := range idle {
				if wk.NoCurators == "" {
					warm = append(warm, wk)
				}
			}
			sort.SliceStable(warm, func(i, j int) bool {
				wi, wj := warm[i].state(now, p) == "waiting", warm[j].state(now, p) == "waiting"
				if wi != wj {
					return wi
				}
				return warm[i].Speed > warm[j].Speed
			})
			for _, wk := range warm[:min(len(warm), keptNearDraw)] {
				kept[wk] = true
			}
		}
		for _, wk := range idle {
			if kept[wk] {
				continue
			}
			d.End = append(d.End, Ending{Worker: wk, Why: "nothing is waiting for it"})
			d.Live--
		}
		return d
	}
	d.Launch = max(0, min(d.Target, p.MaxWorkers)-d.Live)
	if d.Launch > free {
		d.Launch, d.Note = free, fmt.Sprintf("only %d keys are free: make more with `stet-cloud keys --add N`", free)
	}
	if d.Launch > 0 && p.DailyBudget > 0 && spentToday >= p.DailyBudget {
		d.Launch, d.Note = 0, fmt.Sprintf("about $%.2f spent today, and the daily budget is $%.2f", spentToday, p.DailyBudget)
	}
	if p.LongFarm {
		d.LaunchLong = min(d.Launch, w.FarmLongWaiting)
	}
	for _, r := range w.Rounds {
		if n := p.early(r, now); n > 0 {
			if due := p.examDue(r, now); d.ExamDue.IsZero() || due.Before(d.ExamDue) {
				d.ExamDue = due
			}
		}
	}
	d.LaunchEarly = min(renting, d.Launch-d.LaunchLong)
	d.LaunchRuns = min(p.runners(w), d.Launch-d.LaunchLong-d.LaunchEarly)
	sort.SliceStable(d.End, func(i, j int) bool { return d.End[i].Worker.Key.Handle < d.End[j].Worker.Key.Handle })
	return d
}
