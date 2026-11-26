package main

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// --- a few machines measure a round's baseline -------------------------------------------------

// toGrade is a world with a round of n entries open, each run once by the organisers' machine:
// n scripts, each waiting for one grade. It returns the organisers' actor and the round.
func toGrade(t *testing.T, n int) (*world, *actor, int) {
	t.Helper()
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("entry-%d", i)
		w.submit(w.join("owner-"+name, nil), name, fmt.Sprintf("print(%d)", i))
	}
	idx := w.draw()
	if got := w.editAll(anchor); got != n {
		t.Fatalf("runs: %d of %d", got, n)
	}
	return w, anchor, idx
}

// volunteers joins established volunteers.
func (w *world) volunteers(handles ...string) []*actor {
	w.t.Helper()
	var out []*actor
	for _, h := range handles {
		out = append(out, w.join(h, nil))
	}
	w.established(handles...)
	return out
}

// asked has a verifier take a grade job, and fails the test unless the job says what it
// should about the round's baseline.
func (w *world) asked(a *actor, baseline bool) *claimed {
	w.t.Helper()
	c := w.claim(a, false)
	if c.Job == nil || c.Job.Kind != "grade" {
		w.t.Fatalf("%s should be grading: %+v %s", a.handle, c.Job, c.Message)
	}
	if c.Job.Baseline == nil || *c.Job.Baseline != baseline {
		w.t.Fatalf("%s's grade job should say baseline: %v, and says %v", a.handle, baseline, c.Job.Baseline)
	}
	return c
}

// send reports a grade whose student scores -loss, with the machine's own baseline (-baseline)
// or, with baseline 0, without one.
func (w *world) send(a *actor, c *claimed, loss, baseline float64) (int, string) {
	w.t.Helper()
	p := w.result(c, loss, baseline)
	if baseline == 0 {
		delete(p, "baseline")
	}
	var ack struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
	if code != 200 {
		return code, ack.Error
	}
	return code, ack.Status
}

// wanted is how many more machines the open round would ask for its baseline.
func (w *world) wanted(anchor *actor) int {
	w.t.Helper()
	work := w.work(anchor)
	if len(work.Rounds) != 1 {
		w.t.Fatalf("the work: %+v", work)
	}
	return work.Rounds[0].BaselinesWanted
}

// lapse lets a job run out: no report, and not handed back.
func (w *world) lapse(cs ...*claimed) {
	w.t.Helper()
	ctx := context.Background()
	w.app.st.update(ctx, func(tx *Tx) error {
		for _, c := range cs {
			cl := tx.s.Claims[c.Job.ID]
			cl.ExpiresAt = now() - 10
			tx.saveClaim(cl)
		}
		return nil
	})
	w.app.tick(ctx)
	w.app.st.view(func(s *State) {
		for _, c := range cs {
			if s.Claims[c.Job.ID].Status != "lapsed" {
				w.t.Fatalf("a job whose time ran out: %+v", s.Claims[c.Job.ID])
			}
		}
	})
}

// Season 1 had every grader train the round's unedited student, and a third of all training
// was that. Four machines' measurements make a round's baseline as well as forty: the first
// four established graders are asked for it, and the fifth trains the script's student alone.
// Its grade comes without a baseline and is scored against the mean of the four.
func TestFourMachinesMeasureARoundsBaselineAndEveryGradeIsScoredAgainstIt(t *testing.T) {
	w, anchor, idx := toGrade(t, 5)
	if w.app.season.Baselines.PerRound != 4 {
		t.Fatalf("the season asks %d machines for a round's baseline; this test is written for 4", w.app.season.Baselines.PerRound)
	}
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	if n := w.wanted(anchor); n != 4 {
		t.Fatalf("a round nobody has graded wants its four baselines: %d", n)
	}
	var four []*claimed
	for i, a := range g[:4] {
		four = append(four, w.asked(a, true))
		if n := w.wanted(anchor); n != 3-i {
			t.Fatalf("after %d promises the round wants %d more", i+1, n)
		}
	}
	fifth := w.asked(g[4], false)
	// A job read again says what it said when it was handed out.
	if again := w.asked(g[4], false); again.Job.ID != fifth.Job.ID {
		t.Fatalf("the fifth grader was handed another job: %+v", again.Job)
	}
	if again := w.asked(g[0], true); again.Job.ID != four[0].Job.ID {
		t.Fatalf("the first grader was handed another job: %+v", again.Job)
	}

	// The fifth has one pass to train and is done first. Its grade is taken, with nothing
	// yet to score it against, and the round waits for the baseline it was promised.
	if code, status := w.send(g[4], fifth, 5.5, 0); code != 200 || status != "pending" {
		t.Fatalf("a grade from a job that asked for no baseline: HTTP %d %s", code, status)
	}
	// A job that asked for the baseline is not reported without it.
	if code, msg := w.send(g[0], four[0], 5.5, 0); code != 400 || !strings.Contains(msg, "baseline") {
		t.Fatalf("a result that leaves out the baseline its job asked for: HTTP %d %s", code, msg)
	}
	bases := []float64{6.00, 6.02, 5.98, 6.04} // their mean is 6.01
	for i, a := range g[:4] {
		want := "pending"
		if i == 3 {
			want = "settled"
		}
		if code, status := w.send(a, four[i], 5.6, bases[i]); code != 200 || status != want {
			t.Fatalf("%s's grade: HTTP %d %s", a.handle, code, status)
		}
	}
	w.check()
	var rd struct {
		Round struct {
			BaselineLoss float64 `json:"baseline_loss"`
			Verifiers    int     `json:"baseline_verifiers"`
		} `json:"round"`
	}
	anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &rd)
	if math.Abs(rd.Round.BaselineLoss-6.01) > 1e-9 || rd.Round.Verifiers != 4 {
		t.Fatalf("the round's baseline is the mean of the four that measured it: %+v", rd.Round)
	}
	e := w.entry(idx, fifth.Job.Editor.Name)
	if !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.51) > 1e-9 || e.Loss == nil || math.Abs(*e.Loss-5.5) > 1e-9 {
		t.Fatalf("a grade with no baseline of its own is scored against the round's: %+v", e)
	}
	if gr := e.Runs[0].Grades; len(gr) != 1 || gr[0].Handle != "frank" || !gr[0].Counted || gr[0].Gain == nil || math.Abs(*gr[0].Gain-0.51) > 1e-9 {
		t.Fatalf("the grade behind it: %+v", gr)
	}
	// The four are scored against the same mean, not each against its own.
	if e := w.entry(idx, four[1].Job.Editor.Name); e.Gain == nil || math.Abs(*e.Gain-0.41) > 1e-9 {
		t.Fatalf("carol's grade against the round's baseline: %+v", e.Gain)
	}
}

// The first baseline in is no more the round's standard than the next. A machine whose unedited
// student is far out and that reports first is kept, nobody after it is turned away for
// differing from it, and once three have measured it is the odd one out: the round's baseline
// is the median of the others', and every grade is scored against that.
func TestABaselineThatIsFarOutAndFirstInIsNotTheRoundsStandard(t *testing.T) {
	w, anchor, idx := toGrade(t, 4)
	g := w.volunteers("bob", "carol", "dave", "erin")
	var jobs []*claimed
	for _, a := range g {
		jobs = append(jobs, w.asked(a, true))
	}
	bases := []float64{7.5, 6.00, 6.02, 5.98} // bob's is 1.5 bits per byte out, and first
	for i, a := range g {
		if code, status := w.send(a, jobs[i], 5.9, bases[i]); code != 200 {
			t.Fatalf("%s's grade, baseline %.2f: HTTP %d %s", a.handle, bases[i], code, status)
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			want := map[int]float64{0: 7.5, 1: 6.75, 2: 6.01, 3: 6.0}[i]
			if rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-want) > 1e-9 {
				t.Fatalf("after %d baselines the round's is %v, not %.2f", i+1, rd.BaselineLoss, want)
			}
		})
	}
	// What bob's machine made of its script is no more to be relied on than its baseline: the
	// script wants a grade as if it had none, and the round waits for it.
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].Status != "open" {
			t.Fatalf("the round settled on the grade of a machine whose baseline is the odd one out: %s", s.Rounds[idx].Status)
		}
	})
	again := w.asked(g[1], true)
	if again.Job.Editor.Name != jobs[0].Job.Editor.Name {
		t.Fatalf("carol was handed %s, not the script bob graded (%s)", again.Job.Editor.Name, jobs[0].Job.Editor.Name)
	}
	if code, status := w.send(g[1], again, 5.9, 6.0); code != 200 || status != "settled" {
		t.Fatalf("carol's grade of the script bob graded: HTTP %d %s", code, status)
	}
	w.check()
	var rd struct {
		Round struct {
			BaselineLoss float64 `json:"baseline_loss"`
			Verifiers    int     `json:"baseline_verifiers"`
		} `json:"round"`
	}
	anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &rd)
	if math.Abs(rd.Round.BaselineLoss-6.0) > 1e-9 || rd.Round.Verifiers != 3 {
		t.Fatalf("the round's baseline is the median of the three that agree: %+v", rd.Round)
	}
	for i := range g {
		if e := w.entry(idx, jobs[i].Job.Editor.Name); !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.1) > 1e-9 {
			t.Fatalf("the script %s graded gained 0.1, and settled at %+v", g[i].handle, e.Gain)
		}
	}
	verdict := ""
	for _, gr := range w.entry(idx, jobs[0].Job.Editor.Name).Runs[0].Grades {
		if gr.Handle == "bob" {
			verdict = gr.Verdict
		}
	}
	if b := w.account("bob"); verdict != "baseline differs" || b.Strikes != 0 || b.Agreed != 2*w.cfg.EstablishAfter {
		t.Fatalf("bob's grade is shown as %q, and counted for or against him: %+v", verdict, b)
	}
}

// The server requires train.init_sha256 and script.tokens_sha256 on valid Season 3 grades,
// refuses to let a single bad first grade poison Round.InitSHA, locks consensus once two
// independent established graders agree, refuses subsequent mismatches with HTTP 409, and
// voids earlier bad-init grades as "other data" at settlement.
func TestServerComparesInitSHA256AndTokensSHA256(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	owner := w.join("owner-entry", nil)
	owner2 := w.join("owner-entry-2", nil)
	w.submit(owner, "entry-0", "print(0)")
	w.submit(owner2, "entry-1", "print(1)") // left unedited until after 409 checks so round stays open
	idx := w.draw()
	// Confirm entry A with two matching edit scripts while holding entry B's edit replicas in flight
	// so every grade claim goes to entry A and the round stays open through the post-consensus 409 checks.
	g := w.volunteers("bob", "carol", "dave", "frank", "grace", "heidi", "erin")
	erin, heidi := g[6], g[5]
	cEdit0 := w.claim(anchor, true)
	if cEdit0.Job == nil || cEdit0.Job.Kind != "edit" {
		t.Fatalf("anchor did not get edit job 0: %+v", cEdit0.Job)
	}
	if code, msg := w.edit(anchor, cEdit0, "{\"d\": 0}\n"); code != 200 {
		t.Fatalf("anchor edit 0 failed: HTTP %d %s", code, msg)
	}
	entryA := cEdit0.Job.Editor
	var heldJobs []struct {
		a *actor
		c *claimed
	}
	for _, r := range []*actor{anchor, erin, heidi} {
		c := w.claim(r, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("%s did not get edit job: %+v", r.handle, c.Job)
		}
		if c.Job.Editor == entryA {
			if code, msg := w.edit(r, c, "{\"d\": 0}\n"); code != 200 {
				t.Fatalf("%s confirm edit failed: HTTP %d %s", r.handle, code, msg)
			}
			break
		}
		heldJobs = append(heldJobs, struct {
			a *actor
			c *claimed
		}{r, c})
	}

	initA := strings.Repeat("a", 64)
	initB := strings.Repeat("b", 64)
	tokA := strings.Repeat("1", 64)
	tokB := strings.Repeat("2", 64)

	sendWith := func(a *actor, c *claimed, initSHA, tokSHA string) (int, string) {
		t.Helper()
		p := w.result(c, 5.5, 6.0)
		tr := map[string]any{"precision": "bf16", "seconds": 10.0}
		if initSHA != "" {
			tr["init_sha256"] = initSHA
		}
		sc := map[string]any{"script_sha256": c.Job.Script.SHA256}
		if tokSHA != "" {
			sc["tokens_sha256"] = tokSHA
		}
		p["train"], p["script"] = tr, sc
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		if code != 200 {
			return code, ack.Error
		}
		return code, ack.Status
	}

	// Established bob & carol claim the two initial grade replicas; unestablished ivan
	// claims a verification grade on the same script.
	ivan, judy := w.join("ivan", nil), w.join("judy", nil)
	cBad := w.asked(g[0], true)
	c1 := w.asked(g[1], true)
	c3 := w.asked(ivan, true)

	// Missing either integrity digest on a valid Season 3 grade is refused with HTTP 400.
	if code, msg := sendWith(g[0], cBad, "", tokA); code != 400 || !strings.Contains(msg, "init_sha256") {
		t.Fatalf("missing init_sha256: HTTP %d %q, want 400", code, msg)
	}
	if code, msg := sendWith(g[0], cBad, initA, ""); code != 400 || !strings.Contains(msg, "tokens_sha256") {
		t.Fatalf("missing tokens_sha256: HTTP %d %q, want 400", code, msg)
	}

	// First grader (bob) reports a bad initB: accepted initially without locking rd.InitSHA,
	// so honest peers reporting initA are NOT poisoned!
	if code, st := sendWith(g[0], cBad, initB, tokA); code != 200 {
		t.Fatalf("first grade with initB failed: HTTP %d %s", code, st)
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].InitSHA != "" {
			t.Fatalf("single grader prematurely locked Round.InitSHA to %q", s.Rounds[idx].InitSHA)
		}
	})

	// Second grader (carol) reports honest initA -> init_sha256 differs from bob's initB, so
	// judge.same() treats them as a tie and automatically asks a third established grader (dave)!
	if code, st := sendWith(g[1], c1, initA, tokA); code != 200 {
		t.Fatalf("first honest grade failed: HTTP %d %s", code, st)
	}
	c2 := w.asked(g[2], true)
	if code, st := sendWith(g[2], c2, initA, tokA); code != 200 {
		t.Fatalf("second honest grade failed: HTTP %d %s", code, st)
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].InitSHA != initA {
			t.Fatalf("two agreeing graders did not lock Round.InitSHA: got %q, want %q", s.Rounds[idx].InitSHA, initA)
		}
	})

	// Subsequent mismatches on init_sha256 or tokens_sha256 are refused with HTTP 409.
	if code, msg := sendWith(ivan, c3, initB, tokA); code != 409 || !strings.Contains(msg, "initial weights") {
		t.Fatalf("mismatched init_sha256: HTTP %d %q, want 409 mentioning initial weights", code, msg)
	}
	c4 := w.asked(judy, true)
	if code, msg := sendWith(judy, c4, initA, tokB); code != 409 || !strings.Contains(msg, "tokenized training stream") {
		t.Fatalf("mismatched tokens_sha256: HTTP %d %q, want 409 mentioning tokenized training stream", code, msg)
	}

	// Finish the second entry honestly so the round settles cleanly.
	for _, h := range heldJobs {
		if code, msg := w.edit(h.a, h.c, "{\"d\": 1}\n"); code != 200 {
			t.Fatalf("%s held edit failed: HTTP %d %s", h.a.handle, code, msg)
		}
	}
	if len(heldJobs) == 1 {
		c := w.claim(heidi, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("heidi did not get remaining edit job: %+v", c.Job)
		}
		if code, msg := w.edit(heidi, c, "{\"d\": 1}\n"); code != 200 {
			t.Fatalf("heidi remaining edit failed: HTTP %d %s", code, msg)
		}
	}
	if code, st := sendWith(g[3], w.claim(g[3], false), initA, tokA); code != 200 {
		t.Fatalf("frank grade on second entry failed: HTTP %d %s", code, st)
	}
	if code, st := sendWith(g[4], w.claim(g[4], false), initA, tokA); code != 200 {
		t.Fatalf("grace grade on second entry failed: HTTP %d %s", code, st)
	}

	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 2 {
		t.Fatalf("round settled %d entries, want 2", w.settled(idx))
	}

	var differs []Differ
	var bobVerdict string
	w.app.st.view(func(s *State) {
		differs = s.Rounds[idx].Differs
		for _, r := range s.Results {
			if r.RoundIdx == idx && s.Accounts[r.AccountID].Handle == "bob" {
				bobVerdict = r.Verdict
			}
		}
	})
	if bobVerdict != "void" {
		t.Fatalf("bob's bad-init first grade should be voided at settlement, got verdict=%q", bobVerdict)
	}
	if len(differs) != 2 || differs[0].What != "init" || differs[1].What != "tokens" {
		t.Fatalf("expected [init, tokens] in Round.Differs, got %+v", differs)
	}
	if code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); code != 200 ||
		!strings.Contains(page, "initial weights differed") || !strings.Contains(page, "tokenized script differed") {
		t.Fatalf("round page (HTTP %d) did not render init/tokens differ labels", code)
	}
}

// Whichever order they come in, the one that is far out is left out: last in, with two or more
// to say so, it is turned away as it arrives.
func TestABaselineThatIsFarOutAndLastInIsTurnedAway(t *testing.T) {
	w, _, idx := toGrade(t, 4)
	g := w.volunteers("bob", "carol", "dave", "erin")
	var jobs []*claimed
	for _, a := range g {
		jobs = append(jobs, w.asked(a, true))
	}
	// With one other in, a baseline that differs from it is kept: nothing says which is wrong.
	for i, base := range []float64{6.0, 6.3, 6.02} {
		if code, status := w.send(g[i], jobs[i], 5.9, base); code != 200 {
			t.Fatalf("%s's grade, baseline %.2f: HTTP %d %s", g[i].handle, base, code, status)
		}
	}
	if code, msg := w.send(g[3], jobs[3], 5.9, 7.5); code != 409 || !strings.Contains(msg, "baseline") {
		t.Fatalf("a baseline 1.5 bits per byte from three others': HTTP %d %s", code, msg)
	}
	w.app.st.view(func(s *State) {
		// Carol's was kept as it came, and is left out now that three are in: 6.0 and 6.02 remain.
		if rd := s.Rounds[idx]; rd.BaselineN != 2 || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.01) > 1e-9 {
			t.Fatalf("the round's baseline: %d machines, %v", rd.BaselineN, rd.BaselineLoss)
		}
	})
	w.check()
}

func TestMedianItems(t *testing.T) {
	for _, c := range []struct {
		rows [][]float64
		want []float64
	}{
		{nil, nil},
		{[][]float64{{1, 5}}, []float64{1, 5}},
		{[][]float64{{1, 5}, {3, 1}}, []float64{2, 3}},
		{[][]float64{{9, 5}, {1, 1}, {2, 7}}, []float64{2, 5}},
		{[][]float64{{9, 5}, {1, 1}, {2, 7}, {4, 100}}, []float64{3, 6}},
	} {
		got := medianItems(c.rows)
		if len(got) != len(c.want) {
			t.Fatalf("medianItems(%v) = %v", c.rows, got)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("medianItems(%v) = %v, not %v", c.rows, got, c.want)
			}
		}
	}
}

// A promise is a job's: handed back or run out, it is gone, and the next grader is asked. A
// machine that was ever asked is asked again, since it may have the student trained.
func TestABaselineSomebodyWasAskedForAndDidNotSendGoesToTheNextGrader(t *testing.T) {
	w, anchor, _ := toGrade(t, 8)
	g := w.volunteers("bob", "carol", "dave", "erin", "frank", "gina", "harry")
	bob, carol, dave, erin, frank, gina, harry := g[0], g[1], g[2], g[3], g[4], g[5], g[6]
	w.asked(bob, true)
	w.asked(carol, true)
	cd, ce := w.asked(dave, true), w.asked(erin, true)
	cf := w.asked(frank, false)

	// Erin hands her job back, and the baseline she was asked for is the next grader's to give.
	w.release(erin, ce)
	if n := w.wanted(anchor); n != 1 {
		t.Fatalf("with a promise handed back the round wants one more: %d", n)
	}
	w.asked(gina, true)
	if n := w.wanted(anchor); n != 0 {
		t.Fatalf("the promise was taken up: %d", n)
	}
	// Dave's job runs out with no word from him.
	w.lapse(cd)
	if n := w.wanted(anchor); n != 1 {
		t.Fatalf("with a promise run out the round wants one more: %d", n)
	}
	w.asked(harry, true)
	// Frank's job is the one he was handed: it does not start asking for a baseline halfway.
	if again := w.asked(frank, false); again.Job.ID != cf.Job.ID {
		t.Fatalf("frank was handed another job: %+v", again.Job)
	}
	// Erin was asked once, and may have the unedited student trained: she is asked again,
	// though four others are measuring it.
	if again := w.asked(erin, true); again.Job.ID == ce.Job.ID {
		t.Fatal("erin was handed the job she handed back")
	}
	// Bob has sent his, and his later grades of the round keep carrying it.
	if code, status := w.send(bob, w.asked(bob, true), 5.5, 6.0); code != 200 {
		t.Fatalf("bob's grade: HTTP %d %s", code, status)
	}
	w.asked(bob, true)
	// A machine that was not asked, and has the round's baseline on its disk all the same,
	// may send it: it is checked like any other, and is one more for the round's.
	if code, status := w.send(frank, cf, 5.5, 6.02); code != 200 {
		t.Fatalf("a baseline nobody asked for: HTTP %d %s", code, status)
	}
	w.app.st.view(func(s *State) {
		if rd := openRounds(s)[0]; rd.BaselineN != 2 || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.01) > 1e-9 {
			t.Fatalf("the round's baseline: %d machines, %v", rd.BaselineN, rd.BaselineLoss)
		}
	})
	w.check()
}

// An account that is not established is always asked: its baseline is how its machine is
// checked against the others', and is no part of the round's, so it is not one of the four.
func TestAVerifierNotYetEstablishedAlwaysMeasuresTheBaselineAndIsNotOneOfTheFour(t *testing.T) {
	w, anchor, idx := toGrade(t, 6)
	nina, olga := w.join("nina", nil), w.join("olga", nil)
	w.standing("olga") // one grade that agreed: no newcomer, and not established
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	cn := w.asked(nina, true)
	if n := w.wanted(anchor); n != 4 {
		t.Fatalf("a newcomer's baseline is not one of the round's four: %d", n)
	}
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(g[4], false)
	co := w.asked(olga, true) // four are measuring it, and she is asked all the same
	if code, msg := w.send(olga, co, 5.5, 0); code != 400 {
		t.Fatalf("a grade without the baseline that checks its machine: HTTP %d %s", code, msg)
	}
	// Nina's unedited student is a little off the others': inside the tolerance, and no part
	// of the round's baseline.
	if code, status := w.send(nina, cn, 5.5, 6.1); code != 200 {
		t.Fatalf("nina's grade: HTTP %d %s", code, status)
	}
	if n := w.wanted(anchor); n != 0 {
		t.Fatalf("four established machines are measuring it: %d", n)
	}
	for i, a := range g[:4] {
		if code, status := w.send(a, four[i], 5.5, 6.0); code != 200 {
			t.Fatalf("%s's grade: HTTP %d %s", a.handle, code, status)
		}
	}
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.0) > 1e-9 {
			t.Fatalf("the round's baseline is the established graders': %v", rd.BaselineLoss)
		}
	})
	if code, status := w.send(g[4], cf, 5.5, 0); code != 200 {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	if code, status := w.send(olga, co, 5.5, 6.0); code != 200 {
		t.Fatalf("olga's grade: HTTP %d %s", code, status)
	}
	w.check()
}

// One machine, alone, is the first to be asked in every round, and is asked with every grade
// after that: its rounds settle as they always did.
func TestARoundWithOneGraderStillSettles(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(w.join("alice", nil), "tidy", "print(1)")
	w.submit(w.join("amir", nil), "neat", "print(2)")
	idx := w.draw()
	if w.editAll(anchor) != 2 {
		t.Fatal("the organisers' machine should have run both entries")
	}
	for i, want := range []string{"pending", "settled"} {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "grade" || c.Job.Baseline == nil || !*c.Job.Baseline {
			t.Fatalf("grade %d of a machine alone asks for the baseline: %+v", i, c.Job)
		}
		if code, status := w.send(anchor, c, 5.5, 6.0); code != 200 || status != want {
			t.Fatalf("grade %d: HTTP %d %s", i, code, status)
		}
	}
	for _, name := range []string{"tidy", "neat"} {
		if e := w.entry(idx, name); !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.5) > 1e-9 {
			t.Fatalf("%s: %+v", name, e)
		}
	}
	w.check()
}

// The four machines asked for a round's baseline all vanish, and a fifth, which was not, has
// reported. Its grade has nothing to be scored against: the round does not settle on it, and
// the next grade job handed out, whoever takes it, asks for the baseline.
func TestARoundWhoseBaselineNeverCameAsksTheNextGraderForIt(t *testing.T) {
	w, anchor, idx := toGrade(t, 5)
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	frank := g[4]
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(frank, false)
	if code, status := w.send(frank, cf, 5.5, 0); code != 200 || status != "pending" {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	w.lapse(four...)
	if n := w.wanted(anchor); n != 4 {
		t.Fatalf("with every promise gone the round wants its four again: %d", n)
	}
	if w.settled(idx) != 0 {
		t.Fatal("an entry settled on a grade with no baseline to score it against")
	}
	// Frank is still there. He was not asked before, and is now: his next grade measures the
	// baseline, and the grade he sent without one is scored against it.
	for n := 0; n < 4; n++ {
		c := w.asked(frank, true)
		want := "pending"
		if n == 3 {
			want = "settled"
		}
		if code, status := w.send(frank, c, 5.7, 6.0); code != 200 || status != want {
			t.Fatalf("frank's grade %d: HTTP %d %s", n+2, code, status)
		}
	}
	if e := w.entry(idx, cf.Job.Editor.Name); !e.Valid || e.Void || e.Gain == nil || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("the grade sent before the round had a baseline: %+v", e)
	}
	w.check()
}

// A grade that came with no baseline is not left as the only word on its script while the
// round has none: with nobody measuring it, the script wants a grade again, as if it had
// none, and whoever comes next is asked for the baseline whichever script they are handed.
func TestAGradeWithNoBaselineIsNotLeftAsTheRoundsOnlyWord(t *testing.T) {
	w, anchor, idx := toGrade(t, 5)
	g := w.volunteers("bob", "carol", "dave", "erin", "frank", "gina")
	frank, gina := g[4], g[5]
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(frank, false)
	if code, status := w.send(frank, cf, 5.5, 0); code != 200 {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	// While somebody is measuring the baseline, frank's script has the grade it waits for.
	if n := w.work(anchor).Rounds[0].GradesWanted; n != 0 {
		t.Fatalf("every script has its grade, done or under way: %d wanted", n)
	}
	w.idle(gina, false, "every script has its grade, done or under way")
	w.lapse(four...)
	// Now nobody is: frank's script wants a grade like the four that lost theirs.
	if n := w.work(anchor).Rounds[0].GradesWanted; n != 5 {
		t.Fatalf("with no baseline coming, every script wants a grade: %d wanted", n)
	}
	// The first grade gina sends brings the round its baseline, frank's grade is scored
	// against it, and his script wants nothing more unless she already holds it.
	graded := 0
	for {
		c := w.claim(gina, false)
		if c.Job == nil {
			break
		}
		if c.Job.Baseline == nil || !*c.Job.Baseline {
			t.Fatalf("gina's grade %d does not ask for the baseline: %+v", graded+1, c.Job)
		}
		if code, status := w.send(gina, c, 5.52, 6.0); code != 200 {
			t.Fatalf("gina's grade %d: HTTP %d %s", graded+1, code, status)
		}
		graded++
	}
	if graded < 4 || w.settled(idx) != 5 {
		t.Fatalf("the round did not settle once it had a baseline: %d grades, %d entries settled", graded, w.settled(idx))
	}
	e := w.entry(idx, cf.Job.Editor.Name)
	if !e.Valid || e.Gain == nil || len(e.Runs) != 1 || len(e.Runs[0].Grades) != graded-3 {
		t.Fatalf("frank's entry after %d grades by gina: %+v", graded, e)
	}
	for _, gr := range e.Runs[0].Grades {
		if gr.Gain == nil || gr.Verdict != "agreed" {
			t.Fatalf("a grade of frank's script: %+v", gr)
		}
	}
	w.check()
}

// If nobody at all is left to measure it, the round ends as one that nobody was there to
// grade: its entries are void, which is held against none of them.
func TestARoundNobodyIsLeftToMeasureIsGivenUpNotScoredAgainstNothing(t *testing.T) {
	w, _, idx := toGrade(t, 5)
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(g[4], false)
	if code, status := w.send(g[4], cf, 5.5, 0); code != 200 {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	w.lapse(four...)
	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 5 {
		t.Fatalf("the round stayed open with nobody there: %d entries settled", w.settled(idx))
	}
	e := w.entry(idx, cf.Job.Editor.Name)
	if !e.Void || e.Valid || len(e.Dropped) != 1 || !strings.Contains(e.Dropped[0].Reason, "baseline") {
		t.Fatalf("an entry whose only grade had no baseline to be scored against: %+v", e)
	}
	// The grade is published as what it was: an established one, with nothing to score it by.
	if gr := e.Dropped[0].Grades; len(gr) != 1 || gr[0].Handle != "frank" || gr[0].Verdict != "no baseline" || gr[0].Gain != nil || gr[0].Counted {
		t.Fatalf("the grade that had no baseline: %+v", gr)
	}
	if code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); code != 200 || !strings.Contains(page, "no baseline") {
		t.Fatalf("the round's page: HTTP %d", code)
	}
	w.app.st.view(func(s *State) {
		if rd := s.Rounds[idx]; rd.Status != "settled" || rd.BaselineLoss != nil {
			t.Fatalf("the round: %s, baseline %v", rd.Status, rd.BaselineLoss)
		}
		for _, en := range s.inRound[idx] {
			if failing(s, en.EditorID) || roundsRun(s, en.EditorID) != 0 {
				t.Fatalf("a round nobody could score is held against %s", en.EditorID[:12])
			}
		}
	})
	w.check()
}

// A machine the organisers rent is handed grades before an exam only once they would keep it
// busy until near it (pace.go). One that is not asked for the baseline has a pass less to do
// for its first grade, and is kept waiting that much longer.
func TestARentedMachineNotAskedForTheBaselineHasAPassLessToDo(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.cfg.EarlyWindow = 10 * time.Minute
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	w.submit(alice, "one", "print('one')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	w.examAfter(1, 2*time.Hour)
	for _, name := range []string{"two", "three", "four", "five"} {
		w.submit(w.join("owner-"+name, nil), name, "print('"+name+"')")
	}
	w.drawWithoutExam()
	if n := w.editAll(anchor); n != 5 {
		t.Fatalf("runs: %d", n)
	}
	// The exam is two hours off and the window is ten minutes. At 30 minutes a pass, the three
	// grades of a machine asked for the baseline are 122 minutes' work, and it is handed one
	// now. Without the unedited student they are 92, and the machine would only wait.
	var first *actor
	var held *claimed
	for _, handle := range []string{"rented-1", "rented-2", "rented-3", "rented-4"} {
		m := w.rented(anchor, handle)
		c := w.waiting(m, "RTX 3060", 1800)
		if c.Job == nil || c.Job.Kind != "grade" || c.Job.Baseline == nil || !*c.Job.Baseline {
			t.Fatalf("%s is asked for the baseline, and has work enough to start: %+v", handle, c.Job)
		}
		if first == nil {
			first, held = m, c
		}
	}
	fifth := w.rented(anchor, "rented-5")
	if c := w.waiting(fifth, "RTX 3060", 1800); c.Job != nil {
		t.Fatalf("a rented machine with one student to train, not two, was started too soon: %+v", c.Job)
	}
	// One of the four hands its job back. The fifth would be asked now, and so has the work.
	w.release(first, held)
	if c := w.waiting(fifth, "RTX 3060", 1800); c.Job == nil || c.Job.Baseline == nil || !*c.Job.Baseline {
		t.Fatalf("the fifth machine, asked for the baseline in the first's place: %+v", c.Job)
	}
	w.check()
}
