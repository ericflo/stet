package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// While a round runs, its page says where its machines' time is going: each entry's runs and
// grades, on what GPUs, how long, and how much of it trained grades spent waiting for the
// exam. It says nothing of how any of it came out, and names no account.
func TestARunningRoundsPageShowsWhereItsMachinesAreAtWork(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	bob, carol := w.rented(anchor, "bob"), w.rented(anchor, "carol")
	w.submit(alice, "one", "print('one')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	w.examAfter(1, 2*time.Hour)
	w.submit(w.join("owner-two", nil), "two", "print('two')")
	idx := w.drawWithoutExam()
	if n := w.editAll(anchor); n != 2 {
		t.Fatalf("runs: %d", n)
	}
	// Bob trained a grade in ten minutes and has waited five for the exam since; Carol is
	// training one.
	b := w.waiting(bob, "NVIDIA GeForce RTX 5080", 0)
	w.waiting(bob, "NVIDIA GeForce RTX 5080", 0, b)
	c := w.waiting(carol, "NVIDIA GeForce RTX 3080 Ti", 0)
	if b.Job == nil || c.Job == nil || b.Job.Kind != "grade" || c.Job.Kind != "grade" {
		t.Fatalf("two grades before the exam: %+v %+v", b.Job, c.Job)
	}
	at := now()
	w.app.st.update(context.Background(), func(tx *Tx) error {
		cb, cc := tx.s.Claims[b.Job.ID], tx.s.Claims[c.Job.ID]
		cb.ClaimedAt, cb.TrainedAt, cc.ClaimedAt = at-900, at-300, at-120
		tx.saveClaim(cb)
		tx.saveClaim(cc)
		return nil
	})
	var work roundWork
	w.app.st.view(func(s *State) { work = w.app.roundWork(s, s.Rounds[idx], at) })
	if work.Machines != 2 || len(work.GPUs) != 2 || work.GPUs[0] != (gpuCount{"RTX 3080 Ti", 1}) || work.GPUs[1] != (gpuCount{"RTX 5080", 1}) {
		t.Fatalf("the machines at work: %d %+v", work.Machines, work.GPUs)
	}
	if work.Trained != 1 || work.Training != 1 || work.Graded != 0 || work.Waiting != 300 || work.Grading != 600+120 || work.ExamDue == 0 {
		t.Fatalf("the round's work: %+v", work)
	}
	var waiting, training *workJob
	for _, we := range work.Entries {
		if len(we.Runs) != 1 || we.Runs[0].State != "done" || we.RunsDone != 1 {
			t.Fatalf("each entry was run once: %+v", we.Runs)
		}
		for i, j := range we.Grades {
			switch j.State {
			case "waiting":
				waiting = &we.Grades[i]
			case "training":
				training = &we.Grades[i]
			}
		}
	}
	if waiting == nil || waiting.GPU != "RTX 5080" || waiting.Busy != 600 || waiting.Waited != 300 || waiting.Volunteer {
		t.Fatalf("bob's grade: %+v", waiting)
	}
	if training == nil || training.GPU != "RTX 3080 Ti" || training.Busy != 120 || training.Waited != 0 {
		t.Fatalf("carol's grade: %+v", training)
	}
	code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx))
	if code != 200 {
		t.Fatalf("the round's page: HTTP %d", code)
	}
	for _, want := range []string{"2 machines</b> at work on it now: 1 × RTX 3080 Ti, 1 × RTX 5080", "trained, waiting for the exam", "trained in 10 min, waiting 5 min", "2 min so far", "sit its exam once it is made"} {
		if !strings.Contains(page, want) {
			t.Fatalf("the page does not say %q", want)
		}
	}
	// Sealed until the round settles: who grades what, and the scripts' hashes.
	for _, never := range []string{"bob", "carol", b.Job.Script.SHA256[:12]} {
		if strings.Contains(page, never) {
			t.Fatalf("a running round's page says %q", never)
		}
	}
	w.check()
}
