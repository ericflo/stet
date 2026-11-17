package main

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// What a round keeps to itself until it settles, and what it does when the people who are
// there cannot finish it.

const scriptOne, scriptTwo = "{\"doc\": 1}\n", "{\"doc\": 2}\n"

func TestARoundIsNotHeldOpenForEverByAnEntryNobodyThereMayRun(t *testing.T) {
	// The round's only entry is its only runner's own, and no organisers' machine runs
	// editors. Bob and carol keep asking, and are told there is nothing for them.
	w := newWorld(t)
	bob, carol := w.join("bob", nil), w.join("carol", nil)
	mine := w.submit(bob, "his", "print('bob')")
	w.established("bob", "carol")
	idx := w.draw()
	ask := func(minutes int) {
		t.Helper()
		for i := 0; i < minutes; i += 5 {
			w.pass(5 * time.Minute)
			w.idle(bob, true, "nothing is his to run")
			w.idle(carol, false, "nothing has been run")
		}
	}
	ask(40)
	if w.settled(idx) != 0 {
		t.Fatal("the entry was given up before the round had waited its while")
	}
	ask(10)
	e := w.entry(idx, "his")
	if e.Status != "settled" || !e.Void || e.Valid || !strings.Contains(e.Reason, "may not run this entry") {
		t.Fatalf("the entry its only runner owns: %+v", e)
	}
	w.app.st.view(func(s *State) {
		if r := s.Rounds[idx]; r.Status != "settled" {
			t.Fatal("the round is still open, and no later one can be drawn")
		}
		if en := s.entry(idx, mine); !en.Unattended || failing(s, mine) || roundsRun(s, mine) != 0 {
			t.Fatalf("it was held against the entry: %+v", en)
		}
	})
	w.check()

	// The same when the verifiers who are there took the entry and did not finish it. Each
	// is offered a job once, so neither is asked again, and the entry's reason says what
	// happened: somebody was there.
	w = newWorld(t)
	alice := w.join("alice", nil)
	bob, carol = w.join("bob", nil), w.join("carol", nil)
	hers := w.submit(alice, "hers", "print('alice')")
	w.established("bob", "carol")
	idx = w.draw()
	for _, v := range []*actor{bob, carol} {
		c := w.claim(v, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("%s should be running the entry: %+v", v.handle, c.Job)
		}
		w.release(v, c)
	}
	ask(40)
	if w.settled(idx) != 0 {
		t.Fatal("the entry was given up before the round had waited its while")
	}
	ask(10)
	e = w.entry(idx, "hers")
	if e.Status != "settled" || !e.Void || e.Valid || !strings.Contains(e.Reason, "took this entry to run twice and did not finish") {
		t.Fatalf("the entry two verifiers handed back: %+v", e)
	}
	w.app.st.view(func(s *State) {
		if en := s.entry(idx, hers); !en.Unattended || failing(s, hers) {
			t.Fatalf("it was held against the entry: %+v", en)
		}
	})
	w.check()

	// A round that only graders have come to still waits for somebody who runs editors.
	w = newWorld(t)
	alice, carol = w.join("alice", nil), w.join("carol", nil)
	w.submit(alice, "hers", "print('alice')")
	w.established("carol")
	idx = w.draw()
	for i := 0; i < 24; i++ {
		w.pass(5 * time.Minute)
		w.idle(carol, false, "nothing has been run")
	}
	if w.settled(idx) != 0 {
		t.Fatal("a round no runner has come to threw its entry away")
	}
}

func TestARunnerInTheMiddleOfAnEntryIsNotShownItsOtherRun(t *testing.T) {
	w := newWorld(t)
	alice, bob, carol := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	second := w.join("carol-two", nil) // another key of carol's: the same GitHub identity
	id := w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol", "carol-two")
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, h := range []string{"carol", "carol-two"} {
			c := tx.s.byHandle[h]
			c.GithubID = 77
			tx.saveAccount(c)
		}
		return nil
	})
	idx := w.draw()
	cb, cc := w.claim(bob, true), w.claim(carol, true)
	if cb.Job == nil || cc.Job == nil || cb.Job.Kind != "edit" || cc.Job.Kind != "edit" {
		t.Fatalf("both should be running the entry: %+v %+v", cb.Job, cc.Job)
	}
	if code, status := w.edit(bob, cb, scriptOne); code != 200 {
		t.Fatalf("bob's run: HTTP %d %s", code, status)
	}
	// Carol still holds the entry. Asking as a machine that only grades gets her nothing:
	// not bob's script, and no second job beside the one she has.
	c := w.claim(carol, false)
	if c.Job != nil || !strings.Contains(c.Message, "holds an editor it took to run") {
		t.Fatalf("a runner in the middle of an entry was handed %+v (%q)", c.Job, c.Message)
	}
	// Nor does her other key, which holds no job at all.
	w.idle(second, false, "the other key of somebody who is running the entry")
	// Asking as the runner she is, she is handed the job she holds.
	if c := w.claim(carol, true); c.Job == nil || c.Job.ID != cc.Job.ID {
		t.Fatalf("her own job again: %+v", c.Job)
	}
	// And had a script of the entry reached her all the same, a run from her would not be
	// taken: it could be that script.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		tx.newClaim(&Claim{RoundIdx: idx, EditorID: id, AccountID: tx.s.byHandle["carol"].ID, Kind: "grade", Script: shaOf(scriptOne),
			ClaimedAt: now(), ExpiresAt: now(), Status: "released"})
		return nil
	})
	if code, msg := w.edit(carol, cc, scriptOne); code != 409 || !strings.Contains(msg, "handed one of this entry's scripts") {
		t.Fatalf("a run from somebody who had been handed the entry's script: HTTP %d %s", code, msg)
	}
	w.app.st.view(func(s *State) {
		if e := s.entry(idx, id); len(e.Runs) != 1 || e.Runs[0].By != s.byHandle["bob"].ID {
			t.Fatalf("the entry's runs: %+v", e.Runs)
		}
		if c := s.Claims[cc.Job.ID]; c.Status != "released" {
			t.Fatalf("her job afterwards: %+v", c)
		}
	})
	w.check()
}

func TestAScriptIsSealedWhileItsEntryMayStillBeRun(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	zed, mallory, nell := w.join("zed", nil), w.join("mallory", nil), w.join("nell", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.submit(zed, "neat", "print('v2')")
	w.established("bob")
	w.draw()
	// The organisers' machine runs one entry, and bob takes the other.
	ca := w.run(anchor, scriptOne)
	cb := w.claim(bob, true)
	if cb.Job == nil || cb.Job.Kind != "edit" || cb.Job.Editor.ID == ca.Job.Editor.ID {
		t.Fatalf("bob should be giving the other entry its first run: %+v", cb.Job)
	}
	// Bob is at work, and could be handed the first entry to run when he is done. Until
	// then its script goes to nobody whose key anyone could have made.
	w.idle(mallory, false, "an entry that may yet be run again")
	if code, status := w.edit(bob, cb, scriptTwo); code != 200 {
		t.Fatalf("bob's first run: HTTP %d %s", code, status)
	}
	c := w.claim(bob, true)
	if c.Job == nil || c.Job.Kind != "edit" || c.Job.Editor.ID != ca.Job.Editor.ID {
		t.Fatalf("bob should be giving the first entry its second run: %+v", c.Job)
	}
	w.idle(mallory, false, "an entry somebody is running")
	if code, status := w.edit(bob, c, scriptOne); code != 200 {
		t.Fatalf("bob's second run: HTTP %d %s", code, status)
	}
	// That entry has its two runs. Its script may go to anybody now. The other entry's may
	// not: the organisers' machine could still run it.
	sha := shaOf(scriptOne)
	cm := w.claim(mallory, false)
	if cm.Job == nil || cm.Job.Kind != "grade" || cm.Job.Editor.ID != ca.Job.Editor.ID || cm.Job.Script.SHA256 != sha {
		t.Fatalf("mallory should be grading the entry that has both its runs: %+v", cm.Job)
	}

	// The bytes go to whoever holds the script as a job, in a signed request, and to nobody else.
	if code, _ := (&actor{t: t, srv: w.srv}).fetch(sha); code != 403 {
		t.Fatalf("a sealed script, asked for with no key: HTTP %d", code)
	}
	if code, _ := nell.fetch(sha); code != 403 {
		t.Fatalf("a sealed script, asked for by an account that does not hold it: HTTP %d", code)
	}
	if code, _ := mallory.fetch(shaOf(scriptTwo)); code != 403 {
		t.Fatalf("a job on one script opened another: HTTP %d", code)
	}
	code, blob := mallory.fetch(sha)
	if raw, err := unzstd(blob, 1<<20); code != 200 || err != nil || string(raw) != scriptOne {
		t.Fatalf("the script, asked for by the account grading it: HTTP %d %v", code, err)
	}

	// She hands the job back. A key that could do that and ask again would be shown every
	// script of the round, so she is offered no more of this one.
	w.release(mallory, cm)
	if code, _ := mallory.fetch(sha); code != 403 {
		t.Fatalf("a job handed back still opens its script: HTTP %d", code)
	}
	w.run(anchor, scriptTwo)
	if c := w.claim(mallory, false); c.Job != nil || !strings.Contains(c.Message, "handed it back") {
		t.Fatalf("a key that handed one of the round's scripts back was handed %+v (%q)", c.Job, c.Message)
	}
	if c := w.claim(nell, false); c.Job == nil || c.Job.Kind != "grade" {
		t.Fatalf("another newcomer should be grading: %+v", c.Job)
	}
	w.check()
}

func TestAnEntryTakesNoMoreRunsOnceItsScriptHasGoneToANewcomer(t *testing.T) {
	pub, priv := newKey()
	// With nobody else there, the second runner to arrive runs the entry.
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob")
	w.draw()
	w.run(anchor, scriptOne)
	if c := w.claim(bob, true); c.Job == nil || c.Job.Kind != "edit" {
		t.Fatalf("bob should be the entry's second runner: %+v", c.Job)
	}

	w = newWorld(t, pub)
	anchor, alice, bob = w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	nell := w.join("nell", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob")
	idx := w.draw()
	w.run(anchor, scriptOne)
	// Nobody who is around could give the entry a second run, so a newcomer may grade it.
	cn := w.claim(nell, false)
	if cn.Job == nil || cn.Job.Kind != "grade" {
		t.Fatalf("nell should be grading: %+v", cn.Job)
	}
	// Bob arrives. A key anyone could have made has been handed what a run of the entry
	// writes, so a run from him could be a copy of it. He grades the script instead.
	cb := w.claim(bob, true)
	if cb.Job == nil || cb.Job.Kind != "grade" || cb.Job.Script.SHA256 != shaOf(scriptOne) {
		t.Fatalf("bob should be grading the one run there is: %+v", cb.Job)
	}
	if code, status := w.report(nell, cn, 5.5); code != 200 {
		t.Fatalf("nell's grade: HTTP %d %s", code, status)
	}
	if code, status := w.report(bob, cb, 5.5); code != 200 || status != "settled" {
		t.Fatalf("bob's grade: HTTP %d %s", code, status)
	}
	e := w.entry(idx, "tidy")
	if e.Status != "settled" || !e.Valid || e.NRuns != 1 || e.Confirmed || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("the entry: %+v", e)
	}
	w.check()
}

func TestARunIsCreditedWhenItsRoundSettles(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob")
	w.draw()
	// What the verifiers' page is made of.
	board := func() map[string][2]float64 {
		var out struct {
			Verifiers []VerifierRow `json:"verifiers"`
		}
		(&actor{t: t, srv: w.srv}).do("GET", "/api/v1/verifiers", nil, &out)
		rows := map[string][2]float64{}
		for _, v := range out.Verifiers {
			rows[v.Handle] = [2]float64{float64(v.Runs), v.GPUHours}
		}
		return rows
	}
	before := board()
	w.run(bob, scriptOne)
	w.run(anchor, scriptOne)
	// The round is open. Nothing public says who has run anything, or for how long.
	after := board()
	for _, h := range []string{"bob", "anchor-one"} {
		if after[h] != before[h] {
			t.Fatalf("%s's row moved while the round was open: %v, was %v", h, after[h], before[h])
		}
		if a := w.account(h); a.Runs != 0 || a.GPUSeconds != 0 {
			t.Fatalf("%s was credited with a run of an open round: %+v", h, a)
		}
	}
	// Both runs wrote the same script, so it is the entry's only one and is graded twice.
	if _, status := w.grade(bob, true, 5.5); status != "pending" {
		t.Fatalf("one grade of a script both runs wrote, with a second grader around: %s", status)
	}
	if _, status := w.grade(anchor, true, 5.5); status != "settled" {
		t.Fatalf("the grade that settles it: %s", status)
	}
	if a := w.account("anchor-one"); a.Runs != 1 || a.GPUSeconds < 3 {
		t.Fatalf("the organisers' account after the round: %+v", a)
	}
	if b := w.account("bob"); b.Runs != 1 || b.GPUSeconds < 3 {
		t.Fatalf("bob after the round: %+v", b)
	}
	if after := board(); after["bob"][0] != 1 || after["anchor-one"][0] != 1 {
		t.Fatalf("the board after the round: %v", after)
	}
	w.check()

	// A run that an older server recorded was credited when it arrived, and is not again.
	w = newWorld(t, pub)
	anchor, alice = w.join("anchor-one", priv), w.join("alice", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	w.run(anchor, scriptOne)
	w.app.st.update(context.Background(), func(tx *Tx) error {
		e, a := tx.s.entry(idx, id), tx.s.byHandle["anchor-one"]
		e.Runs[0].Owed = false
		a.Runs = 1
		tx.saveEntry(e)
		tx.saveAccount(a)
		return nil
	})
	if _, status := w.grade(anchor, true, 5.5); status != "settled" {
		t.Fatalf("the grade that settles it: %s", status)
	}
	if a := w.account("anchor-one"); a.Runs != 1 {
		t.Fatalf("a run was credited twice: %+v", a)
	}
	w.check()
}

func TestAnEntryGivenUpEarlyStaysSealedUntilItsRoundSettles(t *testing.T) {
	w := newWorld(t)
	carol, zed, bob := w.join("carol", nil), w.join("zed", nil), w.join("bob", nil)
	tidy := w.submit(carol, "tidy", "print('v1')")
	w.submit(zed, "neat", "print('v2')")
	w.established("bob", "carol")
	idx := w.draw()
	w.run(bob, scriptOne)
	w.run(bob, scriptTwo)
	// Carol owns tidy, so the only script she may be handed is neat's. She takes it, and
	// is slow.
	cc := w.claim(carol, false)
	if cc.Job == nil || cc.Job.Kind != "grade" || cc.Job.Editor.Name != "neat" {
		t.Fatalf("carol should be grading neat: %+v", cc.Job)
	}
	w.pass(w.cfg.ReplicaWait + 5*time.Minute)
	// Nobody may grade tidy's script: its runner wrote it, and carol owns the entry. Its
	// run is given up, and the round stays open for carol's job.
	w.app.st.view(func(s *State) {
		if e := s.entry(idx, tidy); e.Status != "settled" || !e.Void || s.Rounds[idx].Status != "open" {
			t.Fatalf("the test is not testing anything: %+v in a round that is %s", e, s.Rounds[idx].Status)
		}
	})
	// Nothing public says so. While its round is open the entry reads as any other does.
	var out struct {
		Entries []map[string]any `json:"entries"`
	}
	(&actor{t: t, srv: w.srv}).do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &out)
	if len(out.Entries) != 2 {
		t.Fatalf("the round's entries: %v", out.Entries)
	}
	for _, e := range out.Entries {
		if len(e) != 7 || e["status"] != "open" || e["stage"] != "grading" {
			t.Fatalf("an entry of an open round says more than that it is being graded: %v", e)
		}
	}
	if e := w.entry(idx, "tidy"); e.Void || e.Reason != "" || e.Gain != nil || e.Detail != nil || len(e.Runs) != 0 || len(e.Dropped) != 0 {
		t.Fatalf("an entry given up early says so while its round is open: %+v", e)
	}
	// When the round settles, it says what happened.
	if code, status := w.report(carol, cc, 5.5); code != 200 || status != "settled" {
		t.Fatalf("carol's grade: HTTP %d %s", code, status)
	}
	e := w.entry(idx, "tidy")
	if e.Status != "settled" || !e.Void || !strings.Contains(e.Reason, "no second verifier was there") || len(e.Dropped) != 1 {
		t.Fatalf("the entry once its round has settled: %+v", e)
	}
	w.check()
}

func TestARoundDrawnByAnOlderServerCountsItsWaitFromNow(t *testing.T) {
	ctx := context.Background()
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	w.run(anchor, scriptOne)
	// As an older server left it: drawn two hours ago, with no note of when it was last
	// moved on, and the organisers' machine stopped for the upgrade.
	w.age(2 * time.Hour)
	w.app.st.update(ctx, func(tx *Tx) error {
		rd := tx.s.Rounds[idx]
		rd.ActiveAt = 0
		tx.saveRound(rd)
		return nil
	})
	w.app.tick(ctx)
	if w.settled(idx) != 0 {
		t.Fatal("a round carried over was given up the moment the new server looked at it")
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].ActiveAt == 0 {
			t.Fatal("the round still has no note of when it was last moved on")
		}
	})
	// From there it waits its while like any other, and the machine that comes back
	// finishes it.
	w.pass(w.cfg.ReplicaWait - time.Minute)
	if w.settled(idx) != 0 {
		t.Fatal("the round was given up before it had waited its while")
	}
	if _, status := w.grade(anchor, true, 5.5); status != "settled" {
		t.Fatalf("the organisers' grade: %s", status)
	}
	w.check()
}

func TestARunThatWasGivenUpDoesNotUseUpOneOfAnEntrysRuns(t *testing.T) {
	w := newWorld(t)
	alice, bob, carol, dave := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol", "dave")
	idx := w.draw()
	cb, cc := w.claim(bob, true), w.claim(carol, true)
	if code, status := w.edit(bob, cb, ""); code != 200 {
		t.Fatalf("bob's run: HTTP %d %s", code, status)
	}
	if code, status := w.edit(carol, cc, scriptOne); code != 200 {
		t.Fatalf("carol's run: HTTP %d %s", code, status)
	}
	// Everybody goes. Only bob could have graded carol's script, so her run is given up,
	// and what is left is bob's word that the editor failed.
	w.pass(w.cfg.ReplicaWait + time.Minute)
	w.app.st.view(func(s *State) {
		if e := s.inRound[idx][0]; e.Status != "open" || len(e.Runs) != 2 || e.Runs[0].Dropped || !e.Runs[1].Dropped {
			t.Fatalf("the test is not testing anything: %+v", e)
		}
	})
	// Dave comes, and is handed the entry: the run that was given up is not one of its two.
	cd := w.claim(dave, true)
	if cd.Job == nil || cd.Job.Kind != "edit" {
		t.Fatalf("dave should be running the entry: %+v", cd.Job)
	}
	if code, status := w.edit(dave, cd, ""); code != 200 {
		t.Fatalf("dave's run: HTTP %d %s", code, status)
	}
	// Two people found that the editor fails. Three runs is the most an entry is given.
	e := w.entry(idx, "tidy")
	if e.Status != "settled" || e.Valid || e.Void || e.NRuns != 2 || len(e.Dropped) != 1 {
		t.Fatalf("the entry: %+v", e)
	}
	w.check()
}

// A verifier turned away for another corpus or another unedited loss is told it will be
// listed with the round. That holds for an established one as much as a newcomer: theirs
// are the machines whose baselines decide scores.
func TestAnEstablishedVerifierTurnedAwayIsListedWithTheRound(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.established("bob")
	for _, name := range []string{"ref-a", "ref-b"} {
		w.submit(anchor, name, "print('"+name+"')")
	}
	idx := w.draw()
	w.editAll(anchor)
	if code, status := w.report(anchor, w.claim(anchor, true), 5.5); code != 200 {
		t.Fatal(status)
	}
	c := w.claim(bob, false)
	if c.Job == nil {
		t.Fatal("no work for an established volunteer")
	}
	var out struct {
		Error string `json:"error"`
	}
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), w.result(c, 5.5, 7.5), &out); code != 409 || !strings.Contains(out.Error, "published") {
		t.Fatalf("a grade with another unedited loss: HTTP %d %s", code, out.Error)
	}
	// Bob goes away, and the organisers' machine finishes the round alone.
	w.pass(2 * w.cfg.ActiveWindow)
	for n := 0; n < 10; n++ {
		c := w.claim(anchor, true)
		if c.Job == nil {
			break
		}
		if code, status := w.report(anchor, c, 5.5); code != 200 {
			t.Fatal(status)
		}
	}
	w.pass(2 * w.cfg.ReplicaWait)
	listed := false
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].Status != "settled" {
			t.Fatalf("the round did not settle: %s", s.Rounds[idx].Status)
		}
		for _, d := range s.Rounds[idx].Differs {
			listed = listed || (d.Handle == "bob" && d.What == "baseline" && d.Est)
		}
	})
	if !listed {
		t.Fatal("an established verifier was told its difference would be published with the round, and it is not there")
	}
	if _, body := w.page("/rounds/" + roundID(w.app.season.ID, idx)); !strings.Contains(body, "/u/bob") || !strings.Contains(body, "established; ") {
		t.Fatal("the round's page does not name the established verifier that was turned away")
	}
	w.check()
}

// "Nobody runs or grades their own entry" has one exception, for the organisers' machines,
// because a season has to start somewhere. It holds only while nobody else could do the
// job: with volunteers around, the organisers' machine keeps its hands off the organisers'
// own entries.
func TestAnOrganisersMachineTakesItsOwnEntryOnlyWhenNobodyElseCan(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob, carol := w.join("anchor-one", priv), w.join("bob", nil), w.join("carol", nil)
	w.established("bob", "carol")
	w.submit(anchor, "ref-a", "print('a')")
	idx := w.draw()
	w.claim(bob, false) // both are around, and both run editors
	w.claim(carol, false)
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, h := range []string{"bob", "carol"} {
			tx.s.byHandle[h].RunnerSeen = now()
			tx.saveAccount(tx.s.byHandle[h])
		}
		return nil
	})
	w.idle(anchor, true, "the organisers' own entry, with two volunteers around who could run it")
	w.run(bob, scriptOne)
	w.run(carol, scriptOne)
	w.idle(anchor, true, "a script of the organisers' own entry, with volunteers around who could grade it")
	for _, who := range []*actor{bob, carol} {
		if c := w.claim(who, true); c.Job != nil {
			if code, status := w.report(who, c, 5.5); code != 200 {
				t.Fatal(status)
			}
		}
	}
	w.pass(2 * w.cfg.ReplicaWait)
	me := w.account("anchor-one").ID
	w.app.st.view(func(s *State) {
		e := s.entry(idx, s.inRound[idx][0].EditorID)
		if e.Status != "settled" || !e.Valid {
			t.Fatalf("the entry did not settle with a score: %s %s", e.Status, e.Reason)
		}
		for _, run := range e.Runs {
			if run.By == me {
				t.Fatal("the organisers' machine ran the organisers' own entry with volunteers around")
			}
		}
		for _, r := range s.results(idx, e.EditorID) {
			if r.AccountID == me {
				t.Fatal("the organisers' machine graded the organisers' own entry with volunteers around")
			}
		}
	})
	w.check()

	// With the volunteers gone, it is the only machine there is, and the season goes on.
	w.submit(anchor, "ref-b", "print('b')")
	w.pass(2 * w.cfg.ActiveWindow)
	next := w.draw()
	for n := 0; n < 20; n++ {
		c := w.claim(anchor, true)
		if c.Job == nil {
			break
		}
		if c.Job.Kind == "edit" {
			if code, status := w.edit(anchor, c, scriptTwo); code != 200 {
				t.Fatal(status)
			}
		} else if code, status := w.report(anchor, c, 5.5); code != 200 {
			t.Fatal(status)
		}
	}
	if w.settled(next) != 2 {
		t.Fatal("alone, the organisers' machine did not run and grade the organisers' own entries")
	}
	w.check()
}

// "Confirmed" says that more than one person answers for a score. Two runs that one
// account graded alone are two runs and one person's word, and the entry says so.
func TestAnEntryOneAccountGradedAloneIsNotCalledConfirmed(t *testing.T) {
	w := newWorld(t)
	alice, bob, carol, dave := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.submit(alice, "steady", "print('steady')")
	w.established("bob", "carol", "dave")
	idx := w.draw()
	w.run(bob, scriptOne)
	w.run(carol, scriptTwo)
	// The two runners leave, and dave is the only grader there is.
	w.pass(2 * w.cfg.ActiveWindow)
	w.grade(dave, false, 5.5)
	if _, status := w.grade(dave, false, 5.5); status != "settled" {
		t.Fatalf("the only grader there is graded both scripts: %s", status)
	}
	w.check()
	e := w.entry(idx, "steady")
	if !e.Valid || e.NRuns != 2 || e.Confirmed || !e.OneGrader || e.RunsDiffer {
		t.Fatalf("two runs, both graded by one account: %+v", e)
	}
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); !strings.Contains(page, ">one grader<") || strings.Contains(page, ">confirmed<") {
		t.Fatal("the round's page calls an entry one account graded alone confirmed")
	}
}

// With two graders and other work to do, an entry's scripts go to different graders, so
// that the usual entry has two people's grades behind it without anybody grading twice.
func TestAnEntrysScriptsGoToDifferentGraders(t *testing.T) {
	// Jobs that are equally wanted are handed out at random, so once proves nothing.
	for try := 0; try < 24; try++ {
		w := newWorld(t)
		alice, zed := w.join("alice", nil), w.join("zed", nil)
		bob, carol, dave, erin := w.join("bob", nil), w.join("carol", nil), w.join("dave", nil), w.join("erin", nil)
		w.submit(alice, "steady", "print('steady')")
		w.submit(zed, "other-one", "print('other')")
		w.established("bob", "carol", "dave", "erin")
		idx := w.draw()
		n := 0
		for _, who := range []*actor{bob, carol} {
			for range 2 {
				n++
				w.run(who, fmt.Sprintf("{\"doc\": %d}\n", n))
			}
		}
		first, _ := w.grade(dave, false, 5.5)
		second, _ := w.grade(dave, false, 5.5)
		if first.Job.Editor.ID == second.Job.Editor.ID {
			t.Fatal("one grader was handed both of an entry's scripts while another entry's were waiting")
		}
		w.grade(erin, false, 5.5)
		if _, status := w.grade(erin, false, 5.5); status != "settled" {
			t.Fatalf("four scripts, four grades: %s", status)
		}
		w.check()
		for _, name := range []string{"steady", "other-one"} {
			if e := w.entry(idx, name); !e.Valid || e.NRuns != 2 || !e.Confirmed || e.OneGrader {
				t.Fatalf("%s, graded by two people: %+v", name, e)
			}
		}
	}
}

// A strike says that a grade is wrong. One volunteer's grade against another's does not
// show which of them is, and neither does the midpoint of two that disagree.
func TestAStrikeNeedsSomethingSolidToBeMeasuredAgainst(t *testing.T) {
	// The organisers' machine runs the entry. The established volunteers named grade its
	// script, and rita, who is not established, replicates it at a gain of 0.5.
	start := func(established map[string]float64) (*world, string, entryJ) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		anchor, alice, rita := w.join("anchor-one", priv), w.join("alice", nil), w.join("rita", nil)
		w.submit(alice, "steady", "print('steady')")
		var names []string
		graders := map[string]*actor{}
		for _, name := range []string{"bob", "carol"} {
			if _, ok := established[name]; ok {
				graders[name] = w.join(name, nil)
				names = append(names, name)
			}
		}
		w.established(names...)
		idx := w.draw()
		w.run(anchor, scriptOne)
		jobs := map[string]*claimed{}
		for _, name := range names {
			if jobs[name] = w.claim(graders[name], false); jobs[name].Job == nil {
				t.Fatalf("no grade for %s", name)
			}
		}
		cr := w.claim(rita, false)
		if cr.Job == nil {
			t.Fatal("no script for a volunteer who is not established to replicate")
		}
		if code, status := w.report(rita, cr, 5.5); code != 200 {
			t.Fatal(status)
		}
		for _, name := range names {
			if code, status := w.report(graders[name], jobs[name], 6.0-established[name]); code != 200 {
				t.Fatal(status)
			}
		}
		w.pass(2 * w.cfg.ReplicaWait)
		w.check()
		e := w.entry(idx, "steady")
		if e.Status != "settled" {
			t.Fatalf("the round did not settle: %+v", e)
		}
		return w, roundID(w.app.season.ID, idx), e
	}
	verdict := func(e entryJ) string {
		for _, g := range e.Runs[0].Grades {
			if g.Handle == "rita" {
				return g.Verdict
			}
		}
		return "not listed"
	}

	// One volunteer's grade is all there is, and rita's differs from it.
	w, round, e := start(map[string]float64{"carol": 0.9})
	if r := w.account("rita"); verdict(e) != "unresolved" || r.Strikes != 0 || r.Agreed != 0 || e.Detail.Disputed != 1 {
		t.Fatalf("against one volunteer's lone grade: %s %+v", verdict(e), r)
	}
	if _, page := w.page("/rounds/" + round); !strings.Contains(page, ">differs, unresolved<") {
		t.Fatal("the round's page does not show the difference nobody could resolve")
	}

	// Two established grades disagree and nobody breaks the tie. Rita's is the same as one
	// of them, and differs from their midpoint.
	w, _, e = start(map[string]float64{"bob": 0.5, "carol": 0.9})
	if r := w.account("rita"); verdict(e) != "unresolved" || r.Strikes != 0 || r.Agreed != 0 {
		t.Fatalf("against the midpoint of two grades that disagree: %s %+v", verdict(e), r)
	}

	// Two established grades agree, and hers differs from both: that is a strike.
	w, _, e = start(map[string]float64{"bob": 0.9, "carol": 0.9})
	if r := w.account("rita"); verdict(e) != "outlier" || r.Strikes != 1 {
		t.Fatalf("against two established grades that agree: %s %+v", verdict(e), r)
	}
}

// Whoever has seen one of an entry's scripts cannot run it. So the one runner left who
// could give an entry a third run is not handed its scripts while its first two runs may
// yet differ and someone else can grade them: otherwise runs that differ settle on two.
func TestTheRunnerWhoCouldGiveAThirdRunIsKeptClearOfTheScripts(t *testing.T) {
	w := newWorld(t)
	alice, bob, carol, dave := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.submit(alice, "flaky", "print('sometimes')")
	w.established("bob", "carol", "dave")
	idx := w.draw()
	cb, cc := w.claim(bob, true), w.claim(carol, true)
	w.idle(dave, true, "a third run before the first two are known to differ")
	if code, status := w.edit(bob, cb, ""); code != 200 { // the editor fails on bob's machine
		t.Fatal(status)
	}
	w.edit(carol, cc, scriptOne)
	// One run failed, and the other wrote a script nobody has graded. Bob can grade it, and
	// he has had his run.
	w.idle(dave, true, "the grade a score is waiting for, which someone who has run the entry can give")
	g := w.claim(bob, true)
	if g.Job == nil || g.Job.Kind != "grade" {
		t.Fatalf("bob should be grading carol's script: %+v", g.Job)
	}
	w.idle(dave, true, "a second grade of the script, while the entry may yet want a run from him")
	if code, status := w.report(bob, g, 5.5); code != 200 || status != "pending" {
		t.Fatalf("one run failed and one did not: HTTP %d %s", code, status)
	}
	// Now the runs are known to differ, and dave is the one who can settle it.
	third := w.claim(dave, true)
	if third.Job == nil || third.Job.Kind != "edit" {
		t.Fatalf("dave should be running it a third time: %+v", third.Job)
	}
	w.edit(dave, third, scriptTwo)
	if _, status := w.grade(carol, true, 5.5); status != "settled" {
		t.Fatalf("the grade of the third run: %s", status)
	}
	w.check()
	if e := w.entry(idx, "flaky"); !e.Valid || e.NRuns != 3 || !e.Confirmed || e.RunsDiffer || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("a failed run and two that agree: %+v", e)
	}

	// With nobody else to give the grade, he gives it: a round does not wait for a run
	// that may never be wanted.
	w = newWorld(t)
	alice, bob, carol = w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	w.submit(alice, "steady", "print('steady')")
	w.established("bob", "carol")
	w.app.cfg.RunsPerEntry = 1
	w.draw()
	w.run(bob, scriptOne)
	if _, status := w.grade(carol, true, 5.5); status != "settled" {
		t.Fatalf("the only other verifier there is: %s", status)
	}
	w.check()
}
