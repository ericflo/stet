package main

import (
	"math"
	"sort"
)

// --- several grades before the exam, the fastest machines after it ------------------------------
//
// A round's exam is posted some time after its draw: minutes, now that the examiner picks it
// from hours of writing harvested and committed beforehand (it was two hours, when the writing
// was harvested after the draw), and longer if the examiner is late. A grade handed out before
// it trains then and sits the exam the moment it is posted. A machine asked for the round's unedited
// baseline (exam.go) trains that student once, with its first grade, and then the script's;
// every other grade, on any machine, is a single pass, and sitting the exam takes seconds.
// So a machine that has trained a grade may hold
// more than one: while every grade it holds is trained and waits only for the exam, it is
// handed one more a score waits for (never a replica), up to earlyGradesPerMachine, as long
// as a pass on it, and a margin, still fits before the exam is due. A slow machine that
// finished early does not start a pass it could not finish, and one that has not graded yet,
// and so cannot say how long a pass takes it, takes no second grade.
//
// Once the exam is posted, the grades that scores still wait for decide when the round
// settles. For the first fastFirstSeconds they go to the fastest machines: a machine is
// handed one only while there are more of them than faster machines that could take them
// and are free, or about to be (all they hold is trained, and sitting the exam takes
// seconds). The rest wait for those. After that anyone may take them.
//
// A machine the organisers rent is paid for by the hour, and one that has trained its grades
// long before the exam only waits for it. So it is handed a grade before the exam only once
// that grade, and the further ones it could still take after it up to earlyGradesPerMachine,
// would keep it busy until no sooner than EarlyWindow before the exam is due (tooEarly). Until
// then it is handed curators to run, if any, and nothing else, and stet-cloud ends it and rents
// another of its kind in time. Its pass is its own, or else its GPU's, or else the slowest
// GPU's seen at the scale: taken too short, a slow card new to the server would be turned
// away until too late, and its round would wait; taken too long, it only waits a while. With
// no pass seen at all it is handed grades as before.
//
// All of these are the organisers' way of finishing a round soon after its exam, and touch
// only who grades what and when: never how many grades a score wants, or whose count.

const (
	earlyFetch            = 120 // seconds: fetching a round's corpus and making the scripts, before a machine's first grade of it
	earlyGradesPerMachine = 3   // grades one machine may hold before the exam, trained and waiting
	earlyMargin           = 120 // seconds: fetching a script and setting up, on top of a pass
	fastFirstSeconds      = 120 // after the exam, how long a grade a score waits for is kept for faster machines
	fastSeen              = 90  // seconds: a machine that asked this recently is there (an idle one asks every minute)
	examDueRounds         = 5   // the latest rounds whose draw-to-exam time says when the next exam is due
)

// examDue is when a round's exam is expected, in unix seconds: its draw, and the median time
// the latest rounds took from their draw to their exam. 0 once it is posted, or before any
// round has had one.
func examDue(s *State, rd *Round) int64 {
	if examPosted(rd) || rd.DrawnAt == 0 {
		return 0
	}
	var done []*Round
	for _, r := range s.Rounds {
		if r.Idx != rd.Idx && r.DrawnAt > 0 && r.ExamAt > r.DrawnAt {
			done = append(done, r)
		}
	}
	if len(done) == 0 {
		return 0
	}
	sort.Slice(done, func(x, y int) bool { return done[x].Idx > done[y].Idx })
	var gaps []float64
	for _, r := range done[:min(len(done), examDueRounds)] {
		gaps = append(gaps, float64(r.ExamAt-r.DrawnAt))
	}
	return rd.DrawnAt + int64(median(gaps))
}

// more reports whether a machine whose grades are all trained and waiting for the exam
// (held of them) may be handed one more of this round's: see above. It is handed only a
// grade a score waits for (claim).
func (j *judge) more(rd *Round, q *asker, held int, pass float64) bool {
	due := examDue(j.s, rd)
	return q.est && q.early && !examPosted(rd) && held < earlyGradesPerMachine && pass > 0 && due > 0 &&
		float64(j.t)+pass+earlyMargin <= float64(due)
}

// tooEarly reports whether a machine the organisers rent, holding `held` of this round's
// grades already (all trained), would be done too soon with one more and the further ones it
// could take after it: see above. pass is what its hello said; 0 when it did not.
func (j *judge) tooEarly(rd *Round, q *asker, held int, pass float64) bool {
	window := j.a.cfg.EarlyWindow.Seconds()
	if !q.acct.Worker || window <= 0 || examPosted(rd) {
		return false
	}
	due := examDue(j.s, rd)
	if pass <= 0 {
		pass = j.passOf(q.acct, rd.Scale)
	}
	if math.IsInf(pass, 1) {
		pass = 0 // a GPU the server has no pass for: the slowest it has
		for _, p := range j.passes[rd.Scale] {
			pass = max(pass, p)
		}
	}
	if due == 0 || pass <= 0 {
		return false
	}
	work := pass // the student; the round's unedited one, if it was asked for, was trained with the first
	if held == 0 {
		work = pass + earlyFetch
		if j.asksBaseline(rd, q.acct) {
			work += pass
		}
	}
	work += float64(max(0, earlyGradesPerMachine-held-1)) * pass
	return float64(j.t)+work < float64(due)-window
}

// fastFirst is what of a round's offers to a machine it may be handed, the first while after
// the round's exam: the grades scores wait for are kept for the faster machines while there
// are no more of them than such machines that are free or about to be. See above.
func (j *judge) fastFirst(rd *Round, q *asker, offers []offer) []offer {
	if !examPosted(rd) || j.t >= rd.ExamAt+fastFirstSeconds || !q.est {
		return offers
	}
	var needed, rest []offer
	for _, o := range offers {
		if o.kind == "grade" && o.required {
			needed = append(needed, o)
		} else {
			rest = append(rest, o)
		}
	}
	if len(needed) == 0 {
		return offers
	}
	mine, faster := j.passOf(q.acct, rd.Scale), 0
	for _, m := range j.active() {
		if m.acct.ID == q.acct.ID || !m.est || m.acct.LastSeen < j.t-fastSeen || j.training[m.acct.ID] || !(j.passOf(m.acct, rd.Scale) < mine) {
			continue
		}
		for _, o := range needed {
			if !o.sc.wroteAlone(m.acct) && j.mayGrade(o.st, o.sc, m.acct) {
				faster++
				break
			}
		}
	}
	if len(needed) > faster {
		return offers
	}
	return rest
}

// passOf is how long a training pass takes a machine like this account's, in seconds: the
// median over the server's records of its kind of GPU at this scale, the grades that settled
// entries kept (the latest few) and what the machines of that kind last said. A kind with
// none, or a machine that does not say its GPU, counts as the slowest.
func (j *judge) passOf(acct *Account, scale string) float64 {
	if j.passes == nil {
		j.passes = passTimes(j.s, j.a.cfg.Scale, j.a.season.Digest)
	}
	if p, ok := j.passes[scale][readHardware(acct.Hardware).gpuName()]; ok {
		return p
	}
	return math.Inf(1)
}

const passesKept = 10 // settled grades of each kind of GPU counted, the latest

// passTimes is the median pass by scale and GPU. What the machines last said is at the scale
// rounds are drawn at now, the server's, scoped to seasonDigest so timings persisted on returning
// accounts from an earlier season never contaminate a new season's scheduler.
func passTimes(s *State, scale, seasonDigest string) map[string]map[string]float64 {
	type sample struct {
		round int
		at    int64
		secs  float64
	}
	seen := map[string]map[string][]sample{}
	add := func(scale, gpu string, x sample) {
		if gpu == "" || !finite(x.secs) || x.secs <= 0 {
			return
		}
		if seen[scale] == nil {
			seen[scale] = map[string][]sample{}
		}
		seen[scale][gpu] = append(seen[scale][gpu], x)
	}
	for _, e := range s.Entries {
		rd := s.Rounds[e.RoundIdx]
		if rd == nil || e.Detail == nil || e.Detail.Train == nil {
			continue
		}
		add(rd.Scale, readHardware(e.Detail.Hardware).gpuName(), sample{e.RoundIdx, e.SettledAt, e.Detail.Train.Seconds})
	}
	for _, scales := range seen {
		for gpu, xs := range scales {
			sort.Slice(xs, func(a, b int) bool {
				if xs[a].round != xs[b].round {
					return xs[a].round > xs[b].round
				}
				return xs[a].at > xs[b].at
			})
			scales[gpu] = xs[:min(len(xs), passesKept)]
		}
	}
	for _, acct := range s.Accounts {
		if seasonDigest != "" && acct.PassSeason != seasonDigest {
			continue
		}
		add(scale, readHardware(acct.Hardware).gpuName(), sample{secs: acct.PassSeconds})
	}
	out := map[string]map[string]float64{}
	for scale, scales := range seen {
		out[scale] = map[string]float64{}
		for gpu, xs := range scales {
			var v []float64
			for _, x := range xs {
				v = append(v, x.secs)
			}
			out[scale][gpu] = median(v)
		}
	}
	return out
}
