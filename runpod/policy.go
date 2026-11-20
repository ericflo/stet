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
func (p Policy) target(w *Work) int { return p.targetHolding(w, 0, time.Now()) }

// targetHolding counts too the machines rented now that train a bigger student (held): each
// is busy with it for hours, and so is none of the machines the students still waiting want.
func (p Policy) targetHolding(w *Work, held int, now time.Time) int {
	n := p.official(w, now)
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

// official is how many machines the rounds' own work justifies.
func (p Policy) official(w *Work, now time.Time) int {
	per := max(p.PerWorker, 1)
	if p.Edits {
		grades := 0
		for _, r := range w.Rounds {
			grades += r.GradesWanted + p.early(r, now)
		}
		return (grades+per-1)/per + p.runners(w)
	}
	scripts, curators := 0, 0
	for _, r := range w.Rounds {
		scripts += r.GradesWanted + p.early(r, now)
		curators += r.RunsWaiting + r.EditsUnderWay
	}
	n := (scripts + per - 1) / per
	if curators > 0 {
		n += int(math.Ceil(p.PerRunner * float64(max(w.Runners, 1))))
	}
	return n
}

// examDue is when a round's exam is expected, about; zero when that is not known.
func (p Policy) examDue(r WorkRound) time.Time {
	if r.ExamAt != 0 || r.DrawnAt == 0 || p.ExamAfter <= 0 {
		return time.Time{}
	}
	return time.Unix(r.DrawnAt, 0).Add(p.ExamAfter)
}

// early is how many of a round's grades want a machine before its exam is posted: those a
// grader could take now, once the exam is due within --early. A machine's first grade of a
// round trains the round's baseline and the student before it can sit the exam, and the
// boot and the corpus take a while too: which kind is rented, and when, so that it has
// trained by the time the exam is posted, is forExam's. None when the exam is posted (its
// grades are wanted then), or when the draw is not known.
func (p Policy) early(r WorkRound, now time.Time) int {
	due := p.examDue(r)
	if p.Early <= 0 || r.GradesEarly <= 0 || due.IsZero() || now.Before(due.Add(-p.Early)) {
		return 0
	}
	return r.GradesEarly
}

// Worker is one key of the pool, with its pod if it has one and what the server says of it.
type Worker struct {
	Key     *Key
	Pod     *Pod
	Machine *Machine
}

// state says what a machine is doing, for deciding and for showing.
//
//	free      no pod
//	starting  a pod, and stet has not yet asked the server for work from it
//	grading   it holds a grade job
//	running   it holds a curator to run
//	training  it trains one of the organisers' students
//	waiting   it asks for work and has none
//	quiet     it asked, and then nothing for longer than any job takes without asking
//	stopped   the pod stopped, or never got going
func (w *Worker) state(now time.Time, p Policy) string {
	switch {
	case w.Pod == nil:
		return "free"
	case w.Pod.Status == "EXITED" || w.Pod.Status == "ERROR" || w.Pod.Status == "TERMINATED":
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
	case seen >= w.Pod.CreatedAt.Unix() && now.Sub(time.Unix(seen, 0)) <= p.Silence:
		return "waiting"
	case seen >= w.Pod.CreatedAt.Unix():
		return "quiet"
	case now.Sub(w.Pod.CreatedAt) <= p.Grace:
		return "starting"
	default:
		return "stopped" // it never asked for work: the install failed, or stet would not start
	}
}

// Ending is a pod to end, and why.
type Ending struct {
	Worker *Worker
	Why    string
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
	// Of the rest, the machines rented to run curators: the cheapest a grade, since once its
	// runs are done such a machine takes a grade before the exam and holds it until then.
	LaunchRuns int
	End        []Ending // pods to end now
	// Machines no pod is rented for that the server says still hold a job: ended a moment
	// after they took it, or stopped partway through. What they hold is handed back.
	Release []*Worker
	Note    string // why fewer are rented than wanted, if they are
}

func (p Policy) decide(w *Work, workers []*Worker, spentToday float64, now time.Time) Decision {
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
	waiting := p.targetHolding(w, held, now)
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
			d.End = append(d.End, Ending{wk, "it stopped, or never asked for work"})
		case "quiet":
			d.End = append(d.End, Ending{wk, fmt.Sprintf("nothing heard from it for %s, and it holds no job", now.Sub(time.Unix(wk.Machine.LastSeen, 0)).Round(time.Minute))})
		default:
			d.Live++
			if st == "waiting" || st == "starting" {
				idle = append(idle, wk)
			}
		}
	}
	if waiting == 0 {
		// Nothing is waiting for anybody: a machine doing nothing would only wait to stop.
		for _, wk := range idle {
			d.End = append(d.End, Ending{wk, "nothing is waiting for it"})
			d.Live--
		}
		return d
	}
	d.Launch = max(0, min(d.Target, p.MaxWorkers)-d.Live)
	if d.Launch > free {
		d.Launch, d.Note = free, fmt.Sprintf("only %d keys are free: make more with `stet-runpod keys --add N`", free)
	}
	if d.Launch > 0 && p.DailyBudget > 0 && spentToday >= p.DailyBudget {
		d.Launch, d.Note = 0, fmt.Sprintf("about $%.2f spent today, and the daily budget is $%.2f", spentToday, p.DailyBudget)
	}
	if p.LongFarm {
		d.LaunchLong = min(d.Launch, w.FarmLongWaiting)
	}
	for _, r := range w.Rounds {
		if n := p.early(r, now); n > 0 {
			if due := p.examDue(r); d.ExamDue.IsZero() || due.Before(d.ExamDue) {
				d.ExamDue = due
			}
			d.LaunchEarly += n
		}
	}
	d.LaunchEarly = min(d.LaunchEarly, d.Launch-d.LaunchLong)
	d.LaunchRuns = min(p.runners(w), d.Launch-d.LaunchLong-d.LaunchEarly)
	sort.SliceStable(d.End, func(i, j int) bool { return d.End[i].Worker.Key.Handle < d.End[j].Worker.Key.Handle })
	return d
}
