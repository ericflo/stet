package main

import (
	"fmt"
	"strings"
	"testing"
)

// The organisers' tools see, for each of their machines, whether it could run curators as it
// last asked and why not, and the jobs it gave up or had refused since its key's last machine
// was ended (/admin/work).
func TestTheOrganisersSeeWhyAMachineRunsNoCuratorsAndWhichOfItsJobsFailed(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	worker := w.rented(anchor, "rented-one")
	alice, carol := w.join("alice", nil), w.join("carol", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.submit(carol, "trim", "print('v2')")
	w.draw()
	w.app.cfg.Dev = false // as in production: the sandbox and the floor are required
	f := w.app.season.Runner
	hello := func(edits bool, cpu float64) map[string]any {
		h := w.hello(edits)
		h["sandbox"] = "container"
		hw := map[string]any{"gpu": "Test GPU", "device": "cuda", "backend": "cuda"}
		if cpu > 0 {
			hw["bench"] = map[string]any{"gpu_tokens_per_second": f.GPUTokensPerSecond, "cpu_mb_per_second": cpu, "cores": f.Cores, "memory_gb": f.MemoryGB, "gpu_memory_gb": f.GPUMemoryGB}
		}
		h["hardware"] = hw
		return h
	}
	reason := func(id int64) (out string) {
		w.app.st.view(func(s *State) { out = s.Claims[id].Reason })
		return out
	}
	look := func(curators bool, why string) WorkMachine {
		t.Helper()
		m := w.machine(w.work(anchor), "rented-one")
		if m.Curators != curators || !strings.Contains(m.CuratorsWhy, why) || (curators && m.CuratorsWhy != "") {
			t.Fatalf("want curators %v, why %q: %+v", curators, why, m)
		}
		return m
	}
	// It asked to run curators on a machine whose processor is under the floor, and was told why.
	slow := f.CPUMBPerSecond - 5
	c := w.claimWith(worker, hello(true, slow))
	if reason, _ := c.Runner["reason"].(string); c.Runner["eligible"] != false || !strings.Contains(reason, fmt.Sprintf("CPU %.0f MB/s, floor %.0f", slow, f.CPUMBPerSecond)) {
		t.Fatalf("a machine under the CPU floor: %+v", c.Runner)
	}
	look(false, fmt.Sprintf("CPU %.0f MB/s, floor %.0f", slow, f.CPUMBPerSecond))
	// It grades without asking, and says what it measured: that is why it did not ask.
	w.claimWith(worker, hello(false, slow))
	look(false, "floor for running curators: CPU")
	// It said nothing of its speed.
	w.claimWith(worker, hello(false, 0))
	look(false, "it did not ask to run curators")
	// Up to the floor, in the sandbox: it runs curators.
	c = w.claimWith(worker, hello(true, f.CPUMBPerSecond))
	if c.Job == nil || c.Job.Kind != "edit" {
		t.Fatalf("a rented machine up to the floor: %+v %+v", c.Job, c.Runner)
	}
	if m := look(true, ""); m.Failed != 0 || m.Refused != 0 || m.Trouble != "" {
		t.Fatalf("nothing has gone wrong yet: %+v", m)
	}
	// It hands the job back because something broke: that is a failure, and why is kept.
	if code := worker.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "RuntimeError: CUDA error: an illegal memory access"}, nil); code != 200 {
		t.Fatalf("release: HTTP %d", code)
	}
	if got := reason(c.Job.ID); got != "RuntimeError: CUDA error: an illegal memory access" {
		t.Fatalf("the release's reason was not kept: %q", got)
	}
	if m := look(true, ""); m.Failed != 1 || m.Refused != 0 || !strings.Contains(m.Trouble, "an edit was handed back: RuntimeError") {
		t.Fatalf("after a failed job: %+v", m)
	}
	// It takes the other entry's run, and the organisers end it partway: not counted against it,
	// and the machine after it on the same key starts with a clean slate.
	c = w.claimWith(worker, hello(true, f.CPUMBPerSecond))
	if c.Job == nil || c.Job.Kind != "edit" {
		t.Fatalf("the other entry's run: %+v", c.Job)
	}
	var out struct{ Jobs int }
	if code := anchor.do("POST", "/api/v1/admin/workers/"+worker.pub+"/release", nil, &out); code != 200 || out.Jobs != 1 {
		t.Fatalf("ending the machine: HTTP %d %+v", code, out)
	}
	if got := reason(c.Job.ID); got != endedReason {
		t.Fatalf("a job handed back as its machine was ended: %q", got)
	}
	if m := w.machine(w.work(anchor), "rented-one"); m.Failed != 0 || m.Refused != 0 || m.Trouble != "" {
		t.Fatalf("a key's next machine is not charged with the last one's failures: %+v", m)
	}
	w.check()
}

// What counts against a machine, and from when.
func TestAMachinesFailuresAreItsJobsGivenUpOrRefusedSinceItsKeyLastMoved(t *testing.T) {
	const at = int64(1_000_000)
	s := &State{Accounts: map[int64]*Account{
		1: {ID: 1, Anchor: true, EndedAt: at - 600},
		2: {ID: 2, Anchor: true},
		3: {ID: 3}, // a volunteer's: not the organisers' business here
	}, Claims: map[int64]*Claim{}}
	add := func(acct int64, ago int64, status, reason string) {
		id := int64(len(s.Claims) + 1)
		s.Claims[id] = &Claim{ID: id, AccountID: acct, Kind: "grade", ClaimedAt: at - ago, Status: status, Reason: reason}
	}
	add(1, 900, "released", "OSError: no space left on device") // before its key's machine was ended
	add(1, 500, "released", "interrupted")                      // stopped, which says nothing against it
	add(1, 400, "released", "the verifier stopped")
	add(1, 300, "released", "the round ended before its exam was posted")
	add(1, 250, "released", endedReason)
	add(1, 200, "done", "")
	add(1, 150, "lapsed", "")
	add(1, 100, "released", "FetchError: the corpus could not be fetched")
	add(2, 90000, "refused", "") // more than a day ago
	add(2, 500, "refused", "")   // a grade that differed
	add(2, 400, "released", "")  // handed back before reasons were kept
	add(2, 300, "released", "result refused")
	add(3, 100, "released", "RuntimeError: boom")
	failed, refused, trouble := troubles(s, at)
	if failed[1] != 2 || refused[1] != 0 || trouble[1] != "a grade was handed back: FetchError: the corpus could not be fetched" {
		t.Fatalf("the first machine: failed %d refused %d %q", failed[1], refused[1], trouble[1])
	}
	if failed[2] != 0 || refused[2] != 2 || trouble[2] != "a grade's result was refused" {
		t.Fatalf("the second machine: failed %d refused %d %q", failed[2], refused[2], trouble[2])
	}
	if failed[3] != 0 || trouble[3] != "" {
		t.Fatalf("a volunteer's account is counted: %d %q", failed[3], trouble[3])
	}
}
