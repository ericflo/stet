package main

import (
	"net/http"
	"sort"
)

// --- what is waiting, for the organisers' machines ----------------------------------------------
//
// The organisers rent machines to grade when there is more to grade than their own can get
// through (stet-cloud), and let them go when there is not. This says how much is waiting
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
	// GradesEarly counts, before the round's exam is posted, the scripts a score will wait
	// on that a grader of the organisers' which trains before the exam (stet verify, from
	// 0.4.3) could be handed now: it trains, waits for the exam, and sits it as soon as it
	// is posted. ExamAt is when the exam was posted (0: not yet).
	GradesEarly int   `json:"grades_early"`
	ExamAt      int64 `json:"exam_at,omitempty"`
	DrawnAt     int64 `json:"drawn_at"` // the exam is made then, from hours of the day before it
	// ExamDue is when the exam is expected, before it is posted: the draw, and the median time
	// the latest rounds took from their draw to their exam (pace.go). Absent once it is posted,
	// and before any round has had one.
	ExamDue int64 `json:"exam_due,omitempty"`
	// BaselinesWanted counts how many more machines the round would ask to measure its
	// unedited baseline with their next grade (exam.go): the season's number, less those
	// that have measured it or hold a grade that asked them to. Such a machine's first
	// grade of the round is two training passes; any other's is one.
	BaselinesWanted int `json:"baselines_wanted"`
}

// WorkMachine is one of the organisers' accounts, as the judge sees it now.
type WorkMachine struct {
	Handle   string `json:"handle"`
	Worker   bool   `json:"worker,omitempty"`
	LastSeen int64  `json:"last_seen,omitempty"`
	Busy     string `json:"busy,omitempty"` // edit or grade: the job it holds; farm: one of the organisers' own students it trains
	// The student it trains is one of the bigger ones, held for hours: stet-cloud counts the
	// machine as one of those it rents for them.
	Long bool `json:"long,omitempty"`
	// Holding counts the grade jobs it holds (Busy is "grade" while there is one). Before an
	// exam a machine may hold up to EarlyGradesPerMachine, trained and waiting for it.
	Holding int `json:"holding,omitempty"`
	// Ready says that every grade it holds is trained and waits only for the exam: before the
	// exam it may be handed another, and after it the grades are sat in seconds.
	Ready bool `json:"ready,omitempty"`
	// PassSeconds is how long it last took to train a grade's student, as it said when it last
	// asked for work (stet 0.4.4 and later). Absent until it has graded.
	PassSeconds float64 `json:"pass_seconds,omitempty"`
	// Curators says whether it could be handed curators to run, as it last asked for work, and
	// CuratorsWhy, when it could not, why (curatorsWhy). The word CPU in it names a part of the
	// floor the machine's processor fell short of.
	Curators    bool   `json:"curators"`
	CuratorsWhy string `json:"curators_why,omitempty"`
	// Failed counts its jobs handed back for something other than stopping, and those it let run
	// out with no report; Refused those whose result the server would not take; Trouble is the
	// latest of either, in words. Each counts the jobs it took within the last day and since the
	// organisers last ended a machine with its key (Account.EndedAt): a key moves from one
	// rented machine to the next.
	Failed  int    `json:"failed,omitempty"`
	Refused int    `json:"refused,omitempty"`
	Trouble string `json:"trouble,omitempty"`
	// Harness is the stet it said it runs, the last time it asked for work.
	Harness string `json:"harness,omitempty"`
}

// Work is the answer to GET /api/v1/admin/work.
type Work struct {
	Rounds  []WorkRound `json:"rounds"`
	Runners int         `json:"runners"` // accounts around that may be handed curators to run
	// Waiting says that the next round would be drawn early, for an entry waiting for one,
	// once somebody who runs curators is around (onDemand). It is false when rounds keep to
	// the clock (STET_ROUND_MIN_GAP unset), while a round is open, and when the draw is
	// already as early as it can be: then a runner rented for it would only wait.
	Waiting    bool          `json:"waiting"`
	NextDrawAt int64         `json:"next_draw_at,omitempty"`
	Machines   []WorkMachine `json:"machines"`
	// FarmWaiting counts the organisers' own students a machine could be handed now, and
	// FarmUnderWay those being trained: work for a machine with nothing official to do.
	FarmWaiting  int `json:"farm_waiting"`
	FarmUnderWay int `json:"farm_under_way"`
	// FarmLongWaiting counts bigger students, held for hours, for a machine set aside for
	// them (stet verify with STET_FARM_MAX_MINUTES): not counted in FarmWaiting.
	FarmLongWaiting int `json:"farm_long_waiting"`
	// Paused is when the organisers paused the rounds (0: they are not): none is drawn.
	Paused int64 `json:"paused,omitempty"`
	// EarlyGradesPerMachine is the most grades one machine may hold before an exam, trained
	// and waiting for it; FastFirstSeconds how long after an exam is posted the grades scores
	// wait for are kept for the faster machines around (pace.go). They are counted in
	// GradesWanted all the same.
	EarlyGradesPerMachine int `json:"early_grades_per_machine,omitempty"`
	FastFirstSeconds      int `json:"fast_first_seconds,omitempty"`
	// EarlyWindowSeconds: a machine the organisers rent is handed grades before an exam only
	// once they, and the further ones it could take, would keep it busy until no sooner than
	// this before the exam is due (pace.go). Absent when that is off.
	EarlyWindowSeconds int `json:"early_window_seconds,omitempty"`
}

// work looks at what the open rounds are waiting for, through the judge's own eyes: the
// grades counted are those it would hand a fresh grader of the organisers', one that runs
// no curators and has seen nothing of any round, so that a rented machine is wanted only
// when it would be given something a score is waiting for.
func (a *App) work(s *State) Work {
	j := a.newJudge(s)
	out := Work{Rounds: []WorkRound{}, Machines: []WorkMachine{}, EarlyGradesPerMachine: earlyGradesPerMachine, FastFirstSeconds: fastFirstSeconds,
		EarlyWindowSeconds: int(a.cfg.EarlyWindow.Seconds())}
	fresh := &asker{acct: &Account{ID: -1, Anchor: true, Worker: true}, est: true, gpuMake: "cuda"}
	early := &asker{acct: &Account{ID: -3, Anchor: true, Worker: true}, est: true, gpuMake: "cuda", early: true}
	runner := &asker{acct: &Account{ID: -2, Anchor: true, Worker: true}, est: true, canEdit: true, gpuMake: "cuda"}
	for _, rd := range openRounds(s) {
		states := j.states(rd)
		closing := j.closing(states)
		wr := WorkRound{ID: roundID(a.season.ID, rd.Idx), Entries: len(states), ExamAt: rd.ExamAt, DrawnAt: rd.DrawnAt, ExamDue: examDue(s, rd),
			BaselinesWanted: j.baselinesWanted(rd)}
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
			if !examPosted(rd) {
				// Every grade a score will wait on: before the exam, one taken alongside
				// another is not held back, and nothing is gained by taking them in turn.
				for _, sc := range st.Scripts {
					if j.mayGrade(st, sc, early.acct) {
						wr.GradesEarly += max(0, j.short(sc)-len(sc.Live))
					}
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
	if r := s.roundWithStatus("scheduled"); r != nil && s.Meta.Paused == 0 {
		out.NextDrawAt = r.DrawAt
	}
	out.Paused = s.Meta.Paused
	soon, _ := a.earlyDraw(s)
	out.Waiting = soon != nil
	out.FarmWaiting, out.FarmUnderWay, out.FarmLongWaiting = farmWaiting(s, now())
	busy, long := map[int64]string{}, map[int64]bool{}
	holding, untrained := map[int64]int{}, map[int64]bool{}
	// Training one of the organisers' own students counts as at work, so that a rented machine is
	// not ended partway through one; an official job it holds says so instead.
	at := now()
	for _, b := range s.Farm {
		for _, t := range b.Tasks {
			if t.Done == nil && t.Holder != 0 && t.Until > at {
				busy[t.Holder] = "farm"
				long[t.Holder] = long[t.Holder] || b.lease() > farmShort
			}
		}
	}
	for _, c := range s.Claims {
		if e := s.entry(c.RoundIdx, c.EditorID); c.live(j.t) && e != nil && e.Status == "open" {
			busy[c.AccountID] = c.Kind
			if c.Kind == "grade" {
				holding[c.AccountID]++
			}
			untrained[c.AccountID] = untrained[c.AccountID] || c.Kind != "grade" || c.TrainedAt == 0
		}
	}
	failed, refused, trouble := troubles(s, at)
	for _, acct := range s.Accounts {
		if acct.Anchor {
			out.Machines = append(out.Machines, WorkMachine{Handle: acct.Handle, Worker: acct.Worker, LastSeen: acct.LastSeen, Busy: busy[acct.ID], Long: busy[acct.ID] == "farm" && long[acct.ID],
				Holding: holding[acct.ID], Ready: holding[acct.ID] > 0 && !untrained[acct.ID], PassSeconds: acct.PassSeconds,
				Curators: acct.RunnerSeen != 0, CuratorsWhy: acct.CuratorsWhy, Failed: failed[acct.ID], Refused: refused[acct.ID], Trouble: trouble[acct.ID],
				Harness: acct.Harness})
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

// stopped are the reasons a job is handed back for that say nothing against the machine: it
// was stopped, the round ended, the organisers ended it, or it did not say (before the server
// kept reasons).
var stopped = map[string]bool{"": true, "interrupted": true, "the verifier stopped": true, "the round ended before its exam was posted": true, endedReason: true}

// troubles counts, for each of the organisers' accounts, the jobs it took since its key's
// last machine was ended (and within a day) that failed or whose result was refused, and says
// what the latest of them was (WorkMachine).
func troubles(s *State, t int64) (failed, refused map[int64]int, trouble map[int64]string) {
	failed, refused, trouble = map[int64]int{}, map[int64]int{}, map[int64]string{}
	latest := map[int64]int64{}
	for id, c := range s.Claims {
		acct := s.Accounts[c.AccountID]
		if acct == nil || !acct.Anchor || c.ClaimedAt <= acct.EndedAt || c.ClaimedAt < t-86400 {
			continue
		}
		var why string
		switch {
		case c.Status == "refused" || (c.Status == "released" && c.Reason == "result refused"):
			refused[c.AccountID]++
			why = article(c.Kind) + "'s result was refused"
		case c.Status == "lapsed":
			failed[c.AccountID]++
			why = article(c.Kind) + " ran out with no result, and was not handed back"
		case c.Status == "released" && !stopped[c.Reason]:
			failed[c.AccountID]++
			why = article(c.Kind) + " was handed back: " + c.Reason
		default:
			continue
		}
		if id > latest[c.AccountID] {
			latest[c.AccountID], trouble[c.AccountID] = id, clip(why, 200)
		}
	}
	return failed, refused, trouble
}
