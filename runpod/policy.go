package main

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// --- how many machines the work wants ---------------------------------------------------------
//
// Rented machines only grade: RunPod's containers cannot make the sandbox a curator runs in
// (no user namespaces, no cgroup of their own), so curators are run by the organisers' own
// machines. Scripts to grade come from those runs, as fast as the runners make them. So the
// machines wanted are two things added up: enough to keep pace with the runners while
// there are curators still to run, and enough to clear the scripts already waiting.
//
// Letting machines go needs no decision here. A rented machine stops by itself once it has
// had nothing to do for a while, and ends its own pod. What is decided here is ending, at
// once, the ones that are doing nothing when nothing at all is waiting, and those that
// have gone quiet or stopped.

type Policy struct {
	MaxWorkers  int     // never more machines than this
	PerRunner   float64 // machines that keep pace with one curator runner
	PerWorker   int     // scripts waiting that justify one more machine
	MaxHourly   float64 // never rent past this many dollars an hour, all machines together
	DailyBudget float64 // no new machine once this much has been spent today (UTC); 0: no limit
	Grace       time.Duration
	Silence     time.Duration
}

// target is how many machines the work justifies, before money is counted.
func (p Policy) target(w *Work) int {
	scripts, curators := 0, 0
	for _, r := range w.Rounds {
		scripts += r.GradesWanted
		curators += r.RunsWaiting + r.EditsUnderWay
	}
	n := (scripts + max(p.PerWorker, 1) - 1) / max(p.PerWorker, 1)
	if curators > 0 {
		n += int(math.Ceil(p.PerRunner * float64(max(w.Runners, 1))))
	}
	return min(n, p.MaxWorkers)
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
//	grading   it holds a job (edit, should one ever be handed it)
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
	case w.Machine != nil && w.Machine.Busy != "":
		return w.Machine.Busy
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
	Target int      // machines the work justifies
	Live   int      // machines rented now
	Launch int      // machines to rent now
	End    []Ending // pods to end now
	Note   string   // why fewer are rented than wanted, if they are
}

func (p Policy) decide(w *Work, workers []*Worker, spentToday float64, now time.Time) Decision {
	d := Decision{Target: p.target(w)}
	free := 0
	var idle []*Worker
	for _, wk := range workers {
		switch st := wk.state(now, p); st {
		case "free":
			free++
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
	if d.Target == 0 {
		// Nothing is waiting for anybody: a machine doing nothing would only wait to stop.
		for _, wk := range idle {
			d.End = append(d.End, Ending{wk, "nothing is waiting to be graded"})
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
	sort.SliceStable(d.End, func(i, j int) bool { return d.End[i].Worker.Key.Handle < d.End[j].Worker.Key.Handle })
	return d
}
