package main

import (
	"context"
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
	if _, status := w.grade(bob, true, 5.5); status != "settled" {
		t.Fatalf("the grade that settles it: %s", status)
	}
	if a := w.account("anchor-one"); a.Runs != 1 || a.GPUSeconds != 3 {
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
