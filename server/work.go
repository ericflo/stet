package main

import (
	"net/http"
	"sort"
)

// --- what is waiting, for the organisers' machines ----------------------------------------------
//
// The organisers rent machines to grade when there is more to grade than their own can get
// through (stet-runpod), and let them go when there is not. This says how much is waiting
// for such a machine, and which of the organisers' machines are at work, so that one is
// rented when it is wanted and not otherwise. It names no script, no grade and no loss:
// counts, which a sealed round's page does not show, and so for the organisers only.

// WorkRound is what an open round is waiting for.
type WorkRound struct {
	ID      string `json:"id"`
	Entries int    `json:"entries"` // open entries
	// RunsWaiting counts entries that somebody who is around could be handed to run now and
	// nobody is running: each is a script to grade soon.
	RunsWaiting int `json:"runs_waiting"`
	// RunsWanted counts entries a runner of the organisers' which had just started could be
	// handed to run now, whoever else is around: the most such machines that could be put
	// to running curators at once.
	RunsWanted    int `json:"runs_wanted"`
	EditsUnderWay int `json:"edits_under_way"`
	// GradesWanted counts scripts a score is waiting on that a grader of the organisers'
	// which had just started could be handed now: the most such machines that could be put
	// to work at once.
	GradesWanted   int `json:"grades_wanted"`
	GradesUnderWay int `json:"grades_under_way"`
}

// WorkMachine is one of the organisers' accounts, as the judge sees it now.
type WorkMachine struct {
	Handle   string `json:"handle"`
	Worker   bool   `json:"worker,omitempty"`
	LastSeen int64  `json:"last_seen,omitempty"`
	Busy     string `json:"busy,omitempty"` // edit or grade: the job it holds
}

// Work is the answer to GET /api/v1/admin/work.
type Work struct {
	Rounds  []WorkRound `json:"rounds"`
	Runners int         `json:"runners"` // accounts around that may be handed curators to run
	// Waiting says that an entry is waiting for a round. The next one is drawn early for it
	// once somebody who runs curators is around.
	Waiting    bool          `json:"waiting"`
	NextDrawAt int64         `json:"next_draw_at,omitempty"`
	Machines   []WorkMachine `json:"machines"`
}

// work looks at what the open rounds are waiting for, through the judge's own eyes: the
// grades counted are those it would hand a fresh grader of the organisers', one that runs
// no curators and has seen nothing of any round, so that a rented machine is wanted only
// when it would be given something a score is waiting for.
func (a *App) work(s *State) Work {
	j := a.newJudge(s)
	out := Work{Rounds: []WorkRound{}, Machines: []WorkMachine{}}
	fresh := &asker{acct: &Account{ID: -1, Anchor: true, Worker: true}, est: true, gpuMake: "cuda"}
	runner := &asker{acct: &Account{ID: -2, Anchor: true, Worker: true}, est: true, canEdit: true, gpuMake: "cuda"}
	for _, rd := range openRounds(s) {
		states := j.states(rd)
		closing := j.ready(states)
		wr := WorkRound{ID: roundID(a.season.ID, rd.Idx), Entries: len(states)}
		for _, st := range states {
			wr.EditsUnderWay += len(st.LiveEdits)
			if len(st.LiveEdits) == 0 && j.anyone(st, nil) {
				wr.RunsWaiting++
			}
			for _, sc := range st.Scripts {
				wr.GradesUnderWay += len(sc.Live)
			}
			for _, o := range j.offers(st, fresh, closing) {
				if o.kind == "grade" && o.required {
					wr.GradesWanted++
				}
			}
			if j.mayRun(st, runner) {
				wr.RunsWanted++
			}
		}
		out.Rounds = append(out.Rounds, wr)
	}
	for _, q := range j.active() {
		if q.canEdit {
			out.Runners++
		}
	}
	if r := s.roundWithStatus("scheduled"); r != nil {
		out.NextDrawAt = r.DrawAt
	}
	out.Waiting = a.waiting(s)
	busy := map[int64]string{}
	for _, c := range s.Claims {
		if e := s.entry(c.RoundIdx, c.EditorID); c.live(j.t) && e != nil && e.Status == "open" {
			busy[c.AccountID] = c.Kind
		}
	}
	for _, acct := range s.Accounts {
		if acct.Anchor {
			out.Machines = append(out.Machines, WorkMachine{Handle: acct.Handle, Worker: acct.Worker, LastSeen: acct.LastSeen, Busy: busy[acct.ID]})
		}
	}
	sort.Slice(out.Machines, func(x, y int) bool { return out.Machines[x].Handle < out.Machines[y].Handle })
	return out
}

func (a *App) apiAdminWork(w http.ResponseWriter, r *http.Request) {
	if _, _, _, err := a.signedBy(w, r, 1<<10, true, organisers); err != nil {
		writeErr(w, err)
		return
	}
	var out Work
	a.st.view(func(s *State) { out = a.work(s) })
	writeJSON(w, 200, out)
}
