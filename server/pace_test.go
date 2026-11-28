package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

// rented is a machine the organisers rent, as stet-cloud's are: one of theirs, established.
func (w *world) rented(anchor *actor, handle string) *actor {
	w.t.Helper()
	pub, priv := newKey()
	if code := w.vouch(anchor, pub); code != 201 {
		w.t.Fatalf("vouching for %s: HTTP %d", handle, code)
	}
	return w.join(handle, priv)
}

// examAfter makes a round's exam have come that long after its draw.
func (w *world) examAfter(idx int, d time.Duration) {
	w.app.st.update(context.Background(), func(tx *Tx) error {
		r := tx.s.Rounds[idx]
		r.DrawnAt = r.ExamAt - int64(d.Seconds())
		tx.saveRound(r)
		return nil
	})
}

// examAgo makes the open round's exam have been posted that long ago.
func (w *world) examAgo(d time.Duration) {
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, r := range tx.s.Rounds {
			if r.Status == "open" && r.ExamAt != 0 {
				r.ExamAt -= int64(d.Seconds())
				tx.saveRound(r)
			}
		}
		return nil
	})
}

// waiting asks for work as stet 0.4.4 does: a verifier that trains before the exam, says how
// long its last pass took (0: it has not graded yet), and names the grades it holds that are
// trained and wait for the exam.
func (w *world) waiting(a *actor, gpu string, pass float64, trained ...*claimed) *claimed {
	w.t.Helper()
	h := w.hello(false)
	h["early"] = true
	h["hardware"] = map[string]any{"gpu": gpu, "device": "cuda", "backend": "cuda"}
	if pass > 0 {
		h["pass_seconds"] = pass
		h["pass_scale"] = w.app.cfg.Scale
	}
	ids := []int64{}
	for _, c := range trained {
		ids = append(ids, c.Job.ID)
	}
	h["trained"] = ids
	return w.claimWith(a, h)
}

func (w *world) work(anchor *actor) Work {
	w.t.Helper()
	var out Work
	if code := anchor.do("GET", "/api/v1/admin/work", nil, &out); code != 200 {
		w.t.Fatalf("the work: HTTP %d", code)
	}
	return out
}

func (w *world) machine(work Work, handle string) WorkMachine {
	w.t.Helper()
	for _, m := range work.Machines {
		if m.Handle == handle {
			return m
		}
	}
	w.t.Fatalf("no machine %s in %+v", handle, work.Machines)
	return WorkMachine{}
}

func TestAMachineThatHasTrainedTakesAnotherGradeWhileOneMoreFits(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	bob, carol, dave := w.rented(anchor, "bob"), w.rented(anchor, "carol"), w.rented(anchor, "dave")
	w.submit(alice, "one", "print('one')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	w.examAfter(1, 2*time.Hour) // the last exam came two hours after its draw
	for _, name := range []string{"two", "three", "four", "five"} {
		w.submit(w.join("owner-"+name, nil), name, "print('"+name+"')")
	}
	idx := w.drawWithoutExam()
	if n := w.editAll(anchor); n != 5 {
		t.Fatalf("runs: %d", n)
	}
	work := w.work(anchor)
	if work.EarlyGradesPerMachine != 3 || work.FastFirstSeconds != 120 || len(work.Rounds) != 1 {
		t.Fatalf("the settings: %+v", work)
	}
	if r := work.Rounds[0]; r.ExamDue < r.DrawnAt+7200-5 || r.ExamDue > r.DrawnAt+7200+5 || r.GradesEarly != 5 {
		t.Fatalf("the exam is due two hours after the draw, and five grades are wanted before it: %+v", r)
	}

	// Bob's first grade. He has not graded before, so he cannot say how long a pass takes him,
	// and having trained it he is handed no second.
	b1 := w.waiting(bob, "RTX 4090", 0)
	if b1.Job == nil || b1.Job.Kind != "grade" {
		t.Fatalf("bob's first grade: %+v", b1)
	}
	if again := w.waiting(bob, "RTX 4090", 0); again.Job == nil || again.Job.ID != b1.Job.ID {
		t.Fatalf("until he says it is trained, he is handed it again: %+v", again.Job)
	}
	if more := w.waiting(bob, "RTX 4090", 0, b1); more.Job != nil {
		t.Fatalf("without a pass to go by, no second grade: %+v", more.Job)
	}
	m := w.machine(w.work(anchor), "bob")
	if m.Busy != "grade" || m.Holding != 1 || !m.Ready || m.PassSeconds != 0 {
		t.Fatalf("bob holds one grade, trained: %+v", m)
	}

	// Once he says a pass takes him ten minutes, he takes two more, and no fourth.
	b2 := w.waiting(bob, "RTX 4090", 600, b1)
	if b2.Job == nil || b2.Job.Kind != "grade" || b2.Job.ID == b1.Job.ID {
		t.Fatalf("bob's second grade: %+v", b2.Job)
	}
	m = w.machine(w.work(anchor), "bob")
	if m.Holding != 2 || m.Ready || m.PassSeconds != 600 {
		t.Fatalf("bob holds two grades, one of them still training: %+v", m)
	}
	if again := w.waiting(bob, "RTX 4090", 600, b1); again.Job == nil || again.Job.ID != b2.Job.ID {
		t.Fatalf("while the second trains, he is handed it again: %+v", again.Job)
	}
	b3 := w.waiting(bob, "RTX 4090", 600, b1, b2)
	if b3.Job == nil || b3.Job.ID == b2.Job.ID || b3.Job.ID == b1.Job.ID {
		t.Fatalf("bob's third grade: %+v", b3.Job)
	}
	if more := w.waiting(bob, "RTX 4090", 600, b1, b2, b3); more.Job != nil {
		t.Fatalf("three is the most one machine holds: %+v", more.Job)
	}
	m = w.machine(w.work(anchor), "bob")
	if m.Holding != 3 || !m.Ready {
		t.Fatalf("bob holds three grades, all trained: %+v", m)
	}
	if r := w.work(anchor).Rounds[0]; r.GradesEarly != 2 {
		t.Fatalf("two grades are left to hand out: %+v", r)
	}

	// Carol is as fast, but the exam is due in less than a pass and the margin: one grade, no
	// more. A replica is never a second grade, so she is handed nothing.
	c1 := w.waiting(carol, "RTX 4090", 0)
	if c1.Job == nil || c1.Job.Kind != "grade" {
		t.Fatalf("carol's first grade: %+v", c1.Job)
	}
	w.age(2*time.Hour - 10*time.Minute) // the exam is due in ten minutes
	if more := w.waiting(carol, "RTX 4090", 600, c1); more.Job != nil {
		t.Fatalf("a pass of ten minutes no longer fits before the exam: %+v", more.Job)
	}
	if more := w.waiting(carol, "RTX 4090", 400, c1); more.Job == nil || more.Job.Kind != "grade" {
		t.Fatalf("one of under eight minutes does: %+v", more)
	}
	if more := w.waiting(dave, "RTX 4090", 60); more.Job == nil {
		t.Fatalf("dave's first grade, a replica: %+v", more)
	} else if again := w.waiting(dave, "RTX 4090", 60, more); again.Job != nil {
		t.Fatalf("every grade a score waits for is handed out, and a replica is never a second: %+v", again.Job)
	}

	// Bob's machine restarts and has lost its students: he is handed the first again to train.
	if again := w.waiting(bob, "RTX 4090", 600); again.Job == nil || again.Job.ID != b1.Job.ID {
		t.Fatalf("after a restart, the first grade again: %+v", again.Job)
	}
	if m := w.machine(w.work(anchor), "bob"); m.Ready || m.Holding != 3 {
		t.Fatalf("bob is training again: %+v", m)
	}

	// The exam: every grade is sat and reported, and the round settles.
	w.exam(idx)
	if r := w.work(anchor).Rounds[0]; r.ExamDue != 0 || r.ExamAt == 0 {
		t.Fatalf("once the exam is posted, it is no longer due: %+v", r)
	}
	for _, c := range []*claimed{b1, b2, b3} {
		if code, status := w.report(bob, c, 5.5); code != 200 {
			t.Fatalf("bob's grade %d: HTTP %d %s", c.Job.ID, code, status)
		}
	}
	w.check()
}

func TestJustAfterTheExamTheGradesAScoreWaitsForGoToTheFastestMachines(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	fast, quick, slow := w.rented(anchor, "fast"), w.rented(anchor, "quick"), w.rented(anchor, "slow")
	w.submit(alice, "one", "print('one')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	w.examAfter(1, 2*time.Hour)
	for _, name := range []string{"two", "three"} {
		w.submit(w.join("owner-"+name, nil), name, "print('"+name+"')")
	}
	idx := w.drawWithoutExam()
	if n := w.editAll(anchor); n != 3 {
		t.Fatalf("runs: %d", n)
	}
	w.age(2*time.Hour - 4*time.Minute) // the exam is due in four minutes

	// Fast trains one grade before the exam, and has no time for another. Quick has graded
	// before, says how long a pass takes it, and asks as a verifier that does not train
	// before the exam: nothing for it yet.
	f1 := w.waiting(fast, "RTX 5090", 0)
	if f1.Job == nil || f1.Job.Kind != "grade" {
		t.Fatalf("fast's grade before the exam: %+v", f1.Job)
	}
	if more := w.waiting(fast, "RTX 5090", 300, f1); more.Job != nil {
		t.Fatalf("five minutes and the margin do not fit in four: %+v", more.Job)
	}
	h := w.hello(false)
	h["pass_seconds"], h["pass_scale"], h["hardware"] = 600.0, w.app.cfg.Scale, map[string]any{"gpu": "RTX 4090", "device": "cuda", "backend": "cuda"}
	if c := w.claimWith(quick, h); c.Job != nil {
		t.Fatalf("quick does not train before the exam: %+v", c.Job)
	}
	w.exam(idx)
	if r := w.work(anchor).Rounds[0]; r.GradesWanted != 2 {
		t.Fatalf("two scripts wait for a grade: %+v", r)
	}

	// Slow asks first. Fast is sitting the exam and quick is waiting for work, both faster:
	// the two grades are kept for them, and slow is told to ask again when that is over.
	var out struct {
		Job        any `json:"job"`
		RetryAfter int `json:"retry_after"`
	}
	sh := w.hello(false)
	sh["early"], sh["pass_seconds"], sh["pass_scale"], sh["hardware"] = true, 1500.0, w.app.cfg.Scale, map[string]any{"gpu": "RTX 3080", "device": "cuda", "backend": "cuda"}
	if code := slow.do("POST", "/api/v1/jobs/claim", sh, &out); code != 200 || out.Job != nil || out.RetryAfter < 5 || out.RetryAfter > 60 {
		t.Fatalf("slow is held back: HTTP %d %+v", code, out)
	}
	if r := w.work(anchor).Rounds[0]; r.GradesWanted != 2 {
		t.Fatalf("the grades held back are wanted all the same: %+v", r)
	}

	// Fast reports its grade, and takes one of the two.
	if code, status := w.report(fast, f1, 5.5); code != 200 {
		t.Fatalf("fast's grade: HTTP %d %s", code, status)
	}
	if f2 := w.waiting(fast, "RTX 5090", 300); f2.Job == nil || f2.Job.Kind != "grade" {
		t.Fatalf("the fastest takes a grade: %+v", f2.Job)
	}
	// One grade is left, and quick is free and faster than slow: still held back.
	if code := slow.do("POST", "/api/v1/jobs/claim", sh, &out); code != 200 || out.Job != nil {
		t.Fatalf("slow is held back while quick is free: %+v", out)
	}
	// Two minutes after the exam, anyone may take it.
	w.examAgo(2 * time.Minute)
	if code := slow.do("POST", "/api/v1/jobs/claim", sh, &out); code != 200 || out.Job == nil {
		t.Fatalf("after the window, slow takes the last grade: %+v", out)
	}
	w.check()
}

func TestARentedMachineIsHandedGradesOnlyOnceTheyWouldKeepItBusyUntilNearTheExam(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.season.Baselines.PerRound = 4
	w.app.cfg.EarlyWindow = 10 * time.Minute
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	bob, carol, dave := w.rented(anchor, "bob"), w.rented(anchor, "carol"), w.join("dave", nil)
	w.submit(alice, "one", "print('one')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	w.examAfter(1, 2*time.Hour)
	for _, name := range []string{"two", "three", "four", "five"} {
		w.submit(w.join("owner-"+name, nil), name, "print('"+name+"')")
	}
	w.established("dave")
	w.drawWithoutExam()
	if n := w.editAll(anchor); n != 5 {
		t.Fatalf("runs: %d", n)
	}
	if work := w.work(anchor); work.EarlyWindowSeconds != 600 {
		t.Fatalf("the window is the server's to say: %+v", work)
	}

	// The exam is two hours off. Bob's pass is 8 minutes: his first grade and the two he could
	// take after it would be done 34 minutes from now, long before it. Nothing for him yet.
	if c := w.waiting(bob, "RTX 4090", 480); c.Job != nil {
		t.Fatalf("a rented machine that would only wait: %+v", c.Job)
	}
	// A rented machine whose GPU has no pass yet is taken to be as slow as the slowest seen:
	// with nothing slower than Bob's, it waits too. A machine nobody rents by the hour does not.
	if c := w.waiting(carol, "Mystery GPU", 0); c.Job != nil {
		t.Fatalf("an unknown GPU, as slow as the slowest seen: %+v", c.Job)
	}
	if c := w.waiting(dave, "RTX 4090", 480); c.Job == nil {
		t.Fatal("a volunteer's machine takes grades as soon as there are any")
	}

	// A slow card the server has seen (30 minutes a pass) opens the window for one it has not:
	// its three would take 122 minutes, past the window, so it is handed one at once.
	w.waiting(w.rented(anchor, "slowpoke"), "RTX 3060", 1800)
	if c := w.waiting(carol, "Mystery GPU", 0); c.Job == nil {
		t.Fatal("an unknown GPU, as slow as a 3060")
	}

	// 43 minutes before the exam the three would end 9 minutes before it: in.
	w.age(2*time.Hour - 43*time.Minute)
	b1 := w.waiting(bob, "RTX 4090", 480)
	if b1.Job == nil || b1.Job.Kind != "grade" {
		t.Fatalf("bob, his window open: %+v", b1.Job)
	}
	// Trained at once (a test's clock stands still): the two left would end 27 minutes early.
	if c := w.waiting(bob, "RTX 4090", 480, b1); c.Job != nil {
		t.Fatalf("a second grade that would only wait: %+v", c.Job)
	}
	w.age(17 * time.Minute) // 26 minutes before the exam: they would end 10 minutes before it
	if c := w.waiting(bob, "RTX 4090", 480, b1); c.Job == nil || c.Job.ID == b1.Job.ID {
		t.Fatalf("bob's second grade, in its window: %+v", c.Job)
	}
	w.check()
}

// Review 27 P2 #6 (server defense-in-depth): persisted Account.PassSeconds from an earlier season
// (or legacy unkeyed accounts) MUST NOT leak into passTimes or /api/v1/admin/work after rollover.
func TestReview27PassTimesScopedBySeasonDigest(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	bob := w.rented(anchor, "bob")

	// Seed a returning Season 2 account with legacy unkeyed PassSeconds=600 (or old season digest).
	var bobID int64
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, acct := range tx.s.Accounts {
			if acct.Pubkey == bob.pub {
				bobID = acct.ID
				acct.Hardware = []byte(`{"gpu":"RTX 4090","device":"cuda","backend":"cuda"}`)
				acct.PassSeconds = 600
				acct.PassSeason = "old-s2-digest"
				tx.touch(acct)
				break
			}
		}
		return nil
	})
	w.app.st.view(func(s *State) {
		pt := passTimes(s, w.app.season.Digest)
		if _, ok := pt[w.app.cfg.Scale]["RTX 4090"]; ok {
			t.Fatalf("stale season PassSeconds leaked into passTimes: %+v", pt)
		}
	})
	if m := w.machine(w.work(anchor), "bob"); m.PassSeconds != 0 {
		t.Fatalf("stale season PassSeconds leaked into /admin/work: %+v", m)
	}

	// Once bob reports a current-season pass time (1800s), PassSeason and PassScale stamp the active season/scale.
	w.waiting(bob, "RTX 4090", 1800)
	w.app.st.view(func(s *State) {
		if got := s.Accounts[bobID].PassSeason; got != w.app.season.Digest {
			t.Fatalf("expected PassSeason=%q, got %q", w.app.season.Digest, got)
		}
		if got := s.Accounts[bobID].PassScale; got != w.app.cfg.Scale {
			t.Fatalf("expected PassScale=%q, got %q", w.app.cfg.Scale, got)
		}
		pt := passTimes(s, w.app.season.Digest)
		if got := pt[w.app.cfg.Scale]["RTX 4090"]; got != 1800 {
			t.Fatalf("expected current-season pass time 1800, got %v", got)
		}
	})
	if m := w.machine(w.work(anchor), "bob"); m.PassSeconds != 1800 {
		t.Fatalf("expected /admin/work PassSeconds=1800, got %+v", m)
	}
}

// Review 32 Gate 4 P2 #1: PassSeconds MUST be scoped by PassScale so timings reported from
// another scale (mixed-scale open rounds or running verifiers across a server scale restart) and
// legacy persisted accounts lacking PassScale never contaminate passTimes, /admin/work, or
// j.more / j.tooEarly on rounds of a different scale.
func TestReview32PassSecondsScopedByScaleAcrossMixedRoundsAndRestarts(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.cfg.Scale = "standard"
	w.app.cfg.EarlyWindow = 300 * time.Second
	anchor := w.join("anchor-one", priv)
	bob := w.rented(anchor, "bob")

	// 1. Legacy persisted account with matching PassSeason but empty PassScale MUST NOT leak.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, acct := range tx.s.Accounts {
			if acct.Pubkey == bob.pub {
				acct.Hardware = []byte(`{"gpu":"RTX 4090","device":"cuda","backend":"cuda"}`)
				acct.PassSeconds = 30
				acct.PassSeason = w.app.season.Digest
				acct.PassScale = ""
				tx.touch(acct)
			}
		}
		return nil
	})
	w.app.st.view(func(s *State) {
		pt := passTimes(s, w.app.season.Digest)
		if _, ok := pt["standard"]["RTX 4090"]; ok {
			t.Fatalf("legacy empty-PassScale PassSeconds leaked into standard passTimes: %+v", pt)
		}
	})
	if m := w.machine(w.work(anchor), "bob"); m.PassSeconds != 0 {
		t.Fatalf("legacy empty-PassScale PassSeconds leaked into /admin/work: %+v", m)
	}

	// 2. Claiming with pass_scale="nano" (30s) against a "standard" server MUST bucket under "nano"
	//    (not "standard"), report 0 in /admin/work for "standard", AND return active server scale.
	body := w.hello(false)
	body["early"] = true
	body["hardware"] = map[string]any{"gpu": "RTX 4090", "device": "cuda", "backend": "cuda"}
	body["pass_seconds"] = 30.0
	body["pass_scale"] = "nano"
	var raw map[string]any
	if code := bob.do("POST", "/api/v1/jobs/claim", body, &raw); code != 200 {
		t.Fatalf("claim failed: %d %+v", code, raw)
	}
	if raw["scale"] != "standard" {
		t.Fatalf("claim response must include active server scale=\"standard\", got %v", raw["scale"])
	}
	w.app.st.view(func(s *State) {
		pt := passTimes(s, w.app.season.Digest)
		if _, ok := pt["standard"]["RTX 4090"]; ok {
			t.Fatalf("nano pass_seconds contaminated standard passTimes: %+v", pt)
		}
		if got := pt["nano"]["RTX 4090"]; got != 30 {
			t.Fatalf("expected nano passTimes=30, got %+v", pt)
		}
	})
	if m := w.machine(w.work(anchor), "bob"); m.PassSeconds != 0 {
		t.Fatalf("nano pass_seconds leaked into standard /admin/work: %+v", m)
	}

	// 3. Open a pre-exam "standard" round with examDue 1500s away and a prior standard baseline
	//    timing of 1000s. Sending nano pass_seconds=30s MUST NOT trigger tooEarly on the standard
	//    round (3*30s + 120s = 210s < 1200s would falsely say tooEarly, whereas standard's 1000s
	//    is well inside the window and hands out the grade!).
	w.submit(w.join("owner-ed", nil), "ed-one", "print(1)")
	// Seed a settled standard round with 1500s draw->exam gap and 1000s RTX 4090 train time.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		rd0 := &Round{Idx: 1, Scale: "standard", Status: "settled", DrawnAt: now() - 3600, ExamAt: now() - 2100}
		tx.saveRound(rd0)
		edID := ""
		for id := range tx.s.Editors {
			edID = id
		}
		e0 := &Entry{
			RoundIdx: 1, EditorID: edID, Status: "settled", SettledAt: now() - 1000,
			Detail: &EntryDetail{
				Train:    &TrainInfo{Seconds: 1000},
				Hardware: json.RawMessage(`{"gpu":"RTX 4090","device":"cuda","backend":"cuda"}`),
			},
		}
		tx.saveEntry(e0)
		return nil
	})
	w.drawWithoutExam()
	// Produce script so a pre-exam grade is available on the open standard round.
	if n := w.editAll(anchor); n != 1 {
		t.Fatalf("expected 1 edit run on open standard round, got %d", n)
	}
	// Now bob asks with pass_seconds=30, pass_scale="nano": MUST NOT be turned away as tooEarly!
	var out2 struct {
		Job *Job `json:"job"`
	}
	if code := bob.do("POST", "/api/v1/jobs/claim", body, &out2); code != 200 {
		t.Fatalf("claim 2 failed: %d", code)
	}
	if out2.Job == nil || out2.Job.Kind != "grade" {
		t.Fatalf("nano pass_seconds=30 falsely triggered tooEarly on standard round; got job=%+v", out2.Job)
	}
}

// Review 33 P2 #5: An older still-compatible 0.6.0 verifier sending `pass_seconds` WITHOUT
// `pass_scale` (e.g. a 30s nano timing sent to a standard server) MUST be treated as unknown for
// scale-specific scheduling and /admin/work, falling back to settled timings whose scale is known.
func TestReview33UntaggedPassSecondsIgnoredByScheduler(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.cfg.Scale = "standard"
	w.app.cfg.EarlyWindow = 300 * time.Second
	anchor := w.join("anchor-one", priv)
	bob := w.rented(anchor, "bob")

	w.submit(w.join("owner-ed", nil), "ed-one", "print(1)")
	w.app.st.update(context.Background(), func(tx *Tx) error {
		rd0 := &Round{Idx: 1, Scale: "standard", Status: "settled", DrawnAt: now() - 3600, ExamAt: now() - 2100}
		tx.saveRound(rd0)
		edID := ""
		for id := range tx.s.Editors {
			edID = id
		}
		tx.saveEntry(&Entry{
			RoundIdx: 1, EditorID: edID, Status: "settled", SettledAt: now() - 1000,
			Detail: &EntryDetail{
				Train:    &TrainInfo{Seconds: 1000},
				Hardware: json.RawMessage(`{"gpu":"RTX 4090","device":"cuda","backend":"cuda"}`),
			},
		})
		return nil
	})
	w.drawWithoutExam()
	if n := w.editAll(anchor); n != 1 {
		t.Fatalf("expected 1 edit run on open standard round, got %d", n)
	}

	// Exact 0.6.0 client payload: pass_seconds=30.0 with NO pass_scale.
	untagged := w.hello(false)
	untagged["early"] = true
	untagged["hardware"] = map[string]any{"gpu": "RTX 4090", "device": "cuda", "backend": "cuda"}
	untagged["pass_seconds"] = 30.0
	var out struct {
		Job *Job `json:"job"`
	}
	if code := bob.do("POST", "/api/v1/jobs/claim", untagged, &out); code != 200 {
		t.Fatalf("untagged claim failed: HTTP %d", code)
	}
	if out.Job == nil || out.Job.Kind != "grade" {
		t.Fatalf("untagged 30s pass_seconds MUST fall back to settled 1000s standard timing and hand out grade; got job=%+v", out.Job)
	}
	if m := w.machine(w.work(anchor), "bob"); m.PassSeconds != 0 {
		t.Fatalf("untagged pass_seconds MUST NOT leak into /admin/work PassSeconds: got %+v", m)
	}
	w.app.st.view(func(s *State) {
		pt := passTimes(s, w.app.season.Digest)
		if got := pt["standard"]["RTX 4090"]; got != 1000 {
			t.Fatalf("untagged 30s pass_seconds contaminated standard passTimes: want 1000, got %v", got)
		}
	})
}
