package main

import (
	"fmt"
	"sort"
	"strings"
)

// --- where a running round's machines are at work, for its page ------------------------------
//
// While a round runs, its page shows what each entry's runs and grades are doing, on what
// GPUs, and how long they have held a machine: where the round's computing goes. Nothing in
// it says how any of that work came out, and no account is named, only a GPU and whether it
// is a volunteer's: the scores are sealed until the round settles, and who grades what is
// told then too.

// workJob is one job of a running round.
type workJob struct {
	GPU       string
	Volunteer bool    // a volunteer's machine, not one of the organisers'
	State     string  // running | training | waiting | sitting | done | handed back | ran out | turned away | not needed
	Busy      float64 // seconds a machine was at work on it
	Waited    float64 // seconds it waited, trained, for the round's exam
	Live      bool    // still under way
}

// workEntry is the work on one entry of a running round.
type workEntry struct {
	Runs, Grades []workJob
	// How many runs are done and under way; how many grades are done, training (or sitting
	// the exam) and trained and waiting for the exam.
	RunsDone, RunsLive, Graded, Training, Waiting int
	Busy, Waited                                  float64
	BusyPct, WaitPct                              float64 // of the busiest entry's, for its bar
	Says                                          string  // where it has got to, in words
}

type gpuCount struct {
	GPU string
	N   int
}

// roundWork is the work on a running round, entry by entry and in all.
type roundWork struct {
	Entries                    map[string]*workEntry // by editor
	Machines                   int                   // machines holding one of its jobs now
	GPUs                       []gpuCount            // and their GPUs, the most common first
	Curating, Grading, Waiting float64               // seconds machines have spent on each
	Spent                      string                // all of that, in words
	Graded, Training, Trained  int
	ExamDue                    int64 // before the exam is posted, when it is expected (pace.go)
}

// roundWork looks at every job a running round has handed out, as of t.
func (a *App) roundWork(s *State, rd *Round, t int64) roundWork {
	w := roundWork{Entries: map[string]*workEntry{}}
	if !examPosted(rd) {
		w.ExamDue = examDue(s, rd)
	}
	// When each finished job's work arrived, and on what GPU it says it was done.
	ended, gpu := map[int64]int64{}, map[int64]string{}
	for _, e := range s.inRound[rd.Idx] {
		w.Entries[e.EditorID] = &workEntry{}
		for _, run := range e.Runs {
			ended[run.Claim], gpu[run.Claim] = run.At, run.GPU
		}
		for _, r := range s.results(rd.Idx, e.EditorID) {
			ended[r.ClaimID] = r.ReceivedAt
			if r.Summary != nil {
				gpu[r.ClaimID] = readHardware(r.Summary.Hardware).gpuName()
			}
		}
	}
	// Its jobs in the order they were handed out.
	var claims []*Claim
	for _, c := range s.Claims {
		if c.RoundIdx == rd.Idx && w.Entries[c.EditorID] != nil {
			claims = append(claims, c)
		}
	}
	sort.Slice(claims, func(x, y int) bool { return claims[x].ID < claims[y].ID })
	held := map[int64]string{} // the machines at work now, and their GPUs
	for _, c := range claims {
		we := w.Entries[c.EditorID]
		acct := s.Accounts[c.AccountID]
		j := workJob{GPU: gpu[c.ID], Volunteer: acct == nil || !acct.Anchor}
		end := c.EndedAt
		switch {
		case c.live(t):
			j.Live, end = true, t
			if acct != nil {
				j.GPU = readHardware(acct.Hardware).gpuName()
				held[acct.ID] = j.GPU
			}
			switch {
			case c.Kind == "edit":
				j.State = "running"
			case c.TrainedAt == 0:
				j.State = "training"
			case !examPosted(rd):
				j.State = "waiting"
			default:
				j.State = "sitting"
			}
		case c.Status == "done":
			j.State = "done"
			if e, ok := ended[c.ID]; ok {
				end = e
			}
		case c.Status == "released":
			j.State = "handed back"
		case c.Status == "lapsed":
			j.State, end = "ran out", c.ExpiresAt
		case c.Status == "refused":
			j.State = "turned away"
		default:
			j.State = "not needed" // its entry settled first
		}
		if j.GPU == "" && acct != nil {
			j.GPU = readHardware(acct.Hardware).gpuName() // what the machine holding its key has now
		}
		j.GPU = shortGPU(j.GPU)
		if end > c.ClaimedAt {
			total := float64(end - c.ClaimedAt)
			if c.Kind == "grade" && c.TrainedAt > 0 {
				until := end
				if examPosted(rd) && rd.ExamAt > 0 && rd.ExamAt < until {
					until = rd.ExamAt
				}
				j.Waited = min(total, max(0, float64(until-c.TrainedAt)))
			}
			j.Busy = total - j.Waited
		}
		if c.Kind == "edit" {
			we.Runs = append(we.Runs, j)
			w.Curating += j.Busy
			switch {
			case j.Live:
				we.RunsLive++
			case j.State == "done":
				we.RunsDone++
			}
		} else {
			we.Grades = append(we.Grades, j)
			w.Grading += j.Busy
			switch {
			case j.State == "waiting":
				we.Waiting++
				w.Trained++
			case j.Live:
				we.Training++
				w.Training++
			case j.State == "done":
				we.Graded++
				w.Graded++
			}
		}
		w.Waiting += j.Waited
		we.Busy += j.Busy
		we.Waited += j.Waited
	}
	top := 0.0
	for _, we := range w.Entries {
		top = max(top, we.Busy+we.Waited)
		we.Says = we.says()
	}
	for _, we := range w.Entries {
		if top > 0 {
			we.BusyPct, we.WaitPct = 100*we.Busy/top, 100*we.Waited/top
		}
	}
	var spent []string
	for _, p := range []struct {
		s    float64
		what string
	}{{w.Curating, "running curators"}, {w.Grading, "training grades"}, {w.Waiting, "trained and waiting for the exam"}} {
		if p.s >= 60 {
			spent = append(spent, duration(p.s)+" "+p.what)
		}
	}
	w.Spent = strings.Join(spent, ", ")
	w.Machines = len(held)
	n := map[string]int{}
	for _, g := range held {
		if g = shortGPU(g); g == "" {
			g = "a GPU it did not name"
		}
		n[g]++
	}
	for g, k := range n {
		w.GPUs = append(w.GPUs, gpuCount{g, k})
	}
	sort.Slice(w.GPUs, func(x, y int) bool {
		if w.GPUs[x].N != w.GPUs[y].N {
			return w.GPUs[x].N > w.GPUs[y].N
		}
		return w.GPUs[x].GPU < w.GPUs[y].GPU
	})
	return w
}

// says is where an entry has got to, in a few words.
func (we *workEntry) says() string {
	var out []string
	switch {
	case we.RunsLive > 0:
		out = append(out, "being run")
	case we.RunsDone == 0:
		return "waiting to be run"
	case we.RunsDone == 1:
		out = append(out, "run")
	default:
		out = append(out, fmt.Sprintf("run %d times", we.RunsDone))
	}
	var g []string
	if we.Graded > 0 {
		g = append(g, fmt.Sprintf("%d graded", we.Graded))
	}
	if we.Training > 0 {
		g = append(g, fmt.Sprintf("%d training", we.Training))
	}
	if we.Waiting > 0 {
		g = append(g, fmt.Sprintf("%d waiting for the exam", we.Waiting))
	}
	if len(g) > 0 {
		out = append(out, strings.Join(g, ", "))
	}
	return strings.Join(out, " · ")
}

// duration says a number of seconds the way the round's page does: 40 s, 12 min, 1.4 h.
func duration(seconds float64) string {
	switch {
	case seconds < 60:
		return fmt.Sprintf("%.0f s", seconds)
	case seconds < 5940:
		return fmt.Sprintf("%.0f min", seconds/60)
	}
	return trimZero(fmt.Sprintf("%.1f", seconds/3600)) + " h"
}

// Word is a job's state in words, and Chip the kind of dot it is drawn with.
func (j workJob) Word() string {
	switch j.State {
	case "running":
		return "running the curator"
	case "waiting":
		return "trained, waiting for the exam"
	case "sitting":
		return "sitting the exam"
	}
	return j.State
}

func (j workJob) Chip() string {
	switch j.State {
	case "running", "training", "sitting":
		return "live"
	case "waiting":
		return "wait"
	case "done":
		return "done"
	case "ran out", "turned away":
		return "bad"
	}
	return ""
}

// Chip is the dot an entry is drawn with: at work, waiting for something, or done with.
func (we *workEntry) Chip() string {
	switch {
	case we.RunsLive+we.Training > 0:
		return "live"
	case we.RunsDone > 0 && we.Waiting == 0 && we.Graded > 0:
		return "done"
	}
	return "wait"
}

// shortGPU is a GPU's name without its maker's brand, which every one of them shares: RTX 5080.
func shortGPU(name string) string {
	for _, brand := range []string{"NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "} {
		if strings.HasPrefix(name, brand) {
			return strings.TrimPrefix(name, brand)
		}
	}
	return name
}
