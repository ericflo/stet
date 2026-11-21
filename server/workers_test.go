package main

import (
	"crypto/ed25519"
	"encoding/hex"
	"path/filepath"
	"strings"
	"testing"
)

// A machine the organisers rent grades the organisers' own entries as one of their machines
// does, and that is all it does: it administers nothing, and posts no round's exam.
func TestARentedMachineGradesForTheOrganisersAndAdministersNothing(t *testing.T) {
	pubA, privA := newKey()
	pubW, privW := newKey()
	cfg := testConfig(t, pubA)
	cfg.RunsPerEntry = 1
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	anchor := w.join("anchor-one", privA)
	if code := w.vouch(anchor, pubW); code != 201 {
		t.Fatalf("an organiser vouching for a rented machine: HTTP %d", code)
	}
	worker := w.join("rented-one", privW)
	var out struct {
		Error string `json:"error"`
	}

	// To the judge it is one of the organisers' machines, established from the start.
	var me struct {
		Account struct {
			Anchor      bool `json:"anchor"`
			Worker      bool `json:"worker"`
			Established bool `json:"established"`
		} `json:"account"`
	}
	if code := worker.do("GET", "/api/v1/me", nil, &me); code != 200 || !me.Account.Anchor || !me.Account.Worker || !me.Account.Established {
		t.Fatalf("a rented machine's account: HTTP %d %+v", code, me.Account)
	}
	me.Account.Anchor, me.Account.Worker = false, false
	if code := anchor.do("GET", "/api/v1/me", nil, &me); code != 200 || !me.Account.Anchor || me.Account.Worker {
		t.Fatalf("an organisers' own machine's account: HTTP %d %+v", code, me.Account)
	}

	// It administers nothing.
	for _, path := range []string{"/api/v1/admin/accounts/anchor-one", "/api/v1/admin/work"} {
		if code := worker.do("GET", path, nil, &out); code != 403 {
			t.Fatalf("a rented machine asked for %s: HTTP %d %s", path, code, out.Error)
		}
	}
	if code := worker.do("POST", "/api/v1/admin/draw", nil, &out); code != 403 {
		t.Fatalf("a rented machine brought the draw forward: HTTP %d", code)
	}
	if code, _ := w.postExam(worker, 1, testTable()); code != 403 {
		t.Fatalf("a rented machine posted a round's exam: HTTP %d", code)
	}

	// The organisers' own machine runs both reference curators; the rented one grades.
	w.submit(anchor, "ref-a", "print('a')")
	w.submit(anchor, "ref-b", "print('b')")
	w.claim(anchor, true)
	w.claim(worker, false)
	idx := w.draw()
	work := func() Work {
		t.Helper()
		var got Work
		if code := anchor.do("GET", "/api/v1/admin/work", nil, &got); code != 200 || len(got.Rounds) != 1 {
			t.Fatalf("what is waiting: HTTP %d %+v", code, got)
		}
		return got
	}
	busy := func(got Work, handle string) string {
		for _, m := range got.Machines {
			if m.Handle == handle {
				return m.Busy
			}
		}
		t.Fatalf("%s is not among the organisers' machines: %+v", handle, got.Machines)
		return ""
	}
	if got := work(); got.Rounds[0].Entries != 2 || got.Rounds[0].RunsWaiting != 2 || got.Rounds[0].GradesWanted != 0 || got.Runners != 1 {
		t.Fatalf("before anything has run: %+v", got)
	}
	w.idle(worker, false, "nothing has been run yet")

	first := w.run(anchor, "{\"doc\": 1}\n")
	if got := work(); got.Rounds[0].RunsWaiting != 1 || got.Rounds[0].GradesWanted != 1 || busy(got, "anchor-one") != "" {
		t.Fatalf("after one run: %+v", got)
	}
	// While it grades, nothing more is wanted of another such machine, and the work says
	// which machine is at it.
	c := w.claim(worker, false)
	if c.Job == nil || c.Job.Kind != "grade" || c.Job.Editor.Name != first.Job.Editor.Name {
		t.Fatalf("the rented machine should be grading the first run's script: %+v", c.Job)
	}
	if got := work(); got.Rounds[0].GradesWanted != 0 || got.Rounds[0].GradesUnderWay != 1 || busy(got, "rented-one") != "grade" {
		t.Fatalf("while the rented machine grades: %+v", got)
	}
	if code, status := w.report(worker, c, 5.5); code != 200 {
		t.Fatal(status)
	}
	w.run(anchor, "{\"doc\": 2}\n")
	w.grade(worker, false, 5.6)
	if w.settled(idx) != 2 {
		t.Fatal("the round did not settle")
	}
	for _, name := range []string{"ref-a", "ref-b"} {
		e := w.entry(idx, name)
		if !e.Valid || len(e.Runs) != 1 || e.Runs[0].By != "anchor-one" || len(e.Runs[0].Grades) != 1 || e.Runs[0].Grades[0].Handle != "rented-one" || !e.Runs[0].Grades[0].Counted {
			t.Fatalf("%s: %+v", name, e.Runs)
		}
		// Run once, by the organisers: their machines are one person, however they come by them.
		if e.Confirmed || e.NRuns != 1 {
			t.Fatalf("%s: confirmed %v, %d runs", name, e.Confirmed, e.NRuns)
		}
	}
	var after Work
	if code := anchor.do("GET", "/api/v1/admin/work", nil, &after); code != 200 || len(after.Rounds) != 0 || busy(after, "rented-one") != "" {
		t.Fatalf("after the round settled: HTTP %d %+v", code, after)
	}

	// Listed among the verifiers once it has been at work, as one of the organisers'.
	var board struct {
		Verifiers []VerifierRow `json:"verifiers"`
	}
	if code := anchor.do("GET", "/api/v1/verifiers", nil, &board); code != 200 {
		t.Fatalf("the verifier board: HTTP %d", code)
	}
	found := false
	for _, v := range board.Verifiers {
		found = found || (v.Handle == "rented-one" && v.Worker && v.Trusted)
	}
	if !found {
		t.Fatalf("the rented machine is not on the board: %+v", board.Verifiers)
	}
	w.check()
}

// A rented machine's key that nobody has used is not listed among the verifiers.
func TestAnUnusedRentedMachineIsNotOnTheBoard(t *testing.T) {
	pubW, privW := newKey()
	pubA, privA := newKey()
	w := newWorld(t, pubA)
	w.vouch(w.join("anchor-one", privA), pubW)
	w.join("rented-two", privW)
	var board struct {
		Verifiers []VerifierRow `json:"verifiers"`
	}
	if code := (&actor{t: t, srv: w.srv}).do("GET", "/api/v1/verifiers", nil, &board); code != 200 || len(board.Verifiers) != 1 || board.Verifiers[0].Handle != "anchor-one" {
		t.Fatalf("HTTP %d %+v", code, board.Verifiers)
	}
}

// The container sandbox is for the organisers' rented machines. A volunteer's machine runs
// curators in bubblewrap; a rented one may run them in the container sandbox, and the work
// says how many a runner like it could take.
func TestOnlyARentedMachineRunsCuratorsInTheContainerSandbox(t *testing.T) {
	pubA, privA := newKey()
	pubW, privW := newKey()
	w := newWorld(t, pubA)
	anchor := w.join("anchor-one", privA)
	w.vouch(anchor, pubW)
	worker := w.join("rented-one", privW)
	alice, bob, carol := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.submit(carol, "trim", "print('v2')")
	w.established("bob")
	var work Work
	if code := anchor.do("GET", "/api/v1/admin/work", nil, &work); code != 200 || work.Waiting {
		t.Fatalf("rounds keep to the clock here, so nothing is drawn early for the two entries waiting: HTTP %d %+v", code, work)
	}
	w.draw()
	w.app.cfg.Dev = false // as in production: the sandbox and the floor are required
	f := w.app.season.Runner
	hello := func(sandbox string) map[string]any {
		h := w.hello(true)
		h["sandbox"] = sandbox
		h["hardware"] = map[string]any{"gpu": "Test GPU", "device": "cuda", "backend": "cuda",
			"bench": map[string]any{"gpu_tokens_per_second": f.GPUTokensPerSecond, "cpu_mb_per_second": f.CPUMBPerSecond, "cores": f.Cores, "memory_gb": f.MemoryGB, "gpu_memory_gb": f.GPUMemoryGB}}
		return h
	}
	// Nobody who runs curators has asked yet, and a runner that started now could take both.
	if code := anchor.do("GET", "/api/v1/admin/work", nil, &work); code != 200 || work.Rounds[0].RunsWanted != 2 || work.Rounds[0].RunsWaiting != 0 {
		t.Fatalf("a round nobody is running yet: HTTP %d %+v", code, work)
	}
	c := w.claimWith(bob, hello("container"))
	if reason, _ := c.Runner["reason"].(string); (c.Job != nil && c.Job.Kind == "edit") || c.Runner["eligible"] != false || !strings.Contains(reason, "rented machines") {
		t.Fatalf("a volunteer asked to run curators in the container sandbox: %+v %+v", c.Job, c.Runner)
	}
	c = w.claimWith(worker, hello("container"))
	if c.Job == nil || c.Job.Kind != "edit" || c.Runner["eligible"] != true {
		t.Fatalf("a rented machine in the container sandbox: %+v %+v", c.Job, c.Runner)
	}
	if code, status := w.edit(worker, c, "{\"doc\": 1}\n"); code != 200 {
		t.Fatal(status)
	}
	// Each entry is run by two machines. A runner that started now could give the first its
	// second run and the other its first; the one that ran the first, only the other.
	if code := anchor.do("GET", "/api/v1/admin/work", nil, &work); code != 200 || work.Rounds[0].RunsWanted != 2 || work.Rounds[0].RunsWaiting != 1 {
		t.Fatalf("after one run: HTTP %d %+v", code, work)
	}
}

// vouch has an organiser vouch for a rented machine's key, as stet-runpod does.
func (w *world) vouch(by *actor, worker ed25519.PublicKey) int {
	w.t.Helper()
	key, at := hex.EncodeToString(worker), now()
	sig := hex.EncodeToString(ed25519.Sign(by.priv, grantMessage(key, at)))
	return by.do("POST", "/api/v1/admin/workers", map[string]any{"key": key, "at": at, "sig": sig}, nil)
}

// An organiser vouches for a machine they rent, with nothing changed in the server's
// configuration. The grant is kept signed, and counts only while the key that made it is
// one of the organisers' own.
func TestAnOrganiserVouchesForARentedMachineWithoutAChangeToTheServer(t *testing.T) {
	pubA, privA := newKey()
	pubB, privB := newKey()
	pubW, privW := newKey()
	pubX, _ := newKey()
	w := newWorld(t, pubA, pubB)
	one, two, eve := w.join("anchor-one", privA), w.join("anchor-two", privB), w.join("eve", nil)
	worker := w.join("rented-one", privW) // an account before anybody vouches for it: nobody's machine
	isWorker := func(wd *world, a *actor) bool {
		t.Helper()
		var me struct {
			Account struct {
				Anchor, Worker, Established bool
			} `json:"account"`
		}
		if code := wd.as(a).do("GET", "/api/v1/me", nil, &me); code != 200 {
			t.Fatalf("/me: HTTP %d", code)
		}
		return me.Account.Anchor && me.Account.Worker && me.Account.Established
	}
	if isWorker(w, worker) {
		t.Fatal("a machine nobody vouched for is the organisers'")
	}

	// Only an organiser's own key may vouch, and only with a grant it signed, for now, for a
	// key that is not already the organisers' own.
	key, at := hex.EncodeToString(pubW), now()
	post := func(by *actor, body map[string]any) int {
		var out struct {
			Error string `json:"error"`
		}
		return by.do("POST", "/api/v1/admin/workers", body, &out)
	}
	good := hex.EncodeToString(ed25519.Sign(privA, grantMessage(key, at)))
	if code := post(eve, map[string]any{"key": key, "at": at, "sig": good}); code != 403 {
		t.Fatalf("an entrant vouched: HTTP %d", code)
	}
	if code := post(two, map[string]any{"key": key, "at": at, "sig": good}); code != 400 {
		t.Fatalf("a grant signed by another organiser than the one who sent it: HTTP %d", code)
	}
	stale := hex.EncodeToString(ed25519.Sign(privA, grantMessage(key, at-7200)))
	if code := post(one, map[string]any{"key": key, "at": at - 7200, "sig": stale}); code != 400 {
		t.Fatalf("a grant two hours old: HTTP %d", code)
	}
	if code := w.vouch(one, pubB); code != 400 {
		t.Fatalf("vouching for an organisers' own key: HTTP %d", code)
	}
	if isWorker(w, worker) {
		t.Fatal("a refused grant made a machine the organisers'")
	}
	if code := post(one, map[string]any{"key": key, "at": at, "sig": good}); code != 201 || !isWorker(w, worker) {
		t.Fatalf("the organiser's grant: HTTP %d, worker %v", code, isWorker(w, worker))
	}
	// A rented machine vouches for nothing.
	if code := w.vouch(worker, pubX); code != 403 {
		t.Fatalf("a rented machine vouched for another: HTTP %d", code)
	}
	var list struct {
		Workers []WorkerView `json:"workers"`
	}
	if code := two.do("GET", "/api/v1/admin/workers", nil, &list); code != 200 || len(list.Workers) != 1 ||
		list.Workers[0].Handle != "rented-one" || list.Workers[0].By != "anchor-one" || !list.Workers[0].Counts {
		t.Fatalf("the grants: HTTP %d %+v", code, list)
	}
	if code := eve.do("GET", "/api/v1/admin/workers", nil, nil); code != 403 {
		t.Fatalf("an entrant listed the grants: HTTP %d", code)
	}
	w.check()

	// It holds across a restart, and lapses with the key that made it.
	again := startWorld(t, w.cfg, w.kv, w.blobs)
	if !isWorker(again, worker) {
		t.Fatal("the grant did not survive a restart")
	}
	cfg := w.cfg
	cfg.AnchorKeys = map[string]bool{hex.EncodeToString(pubB): true}
	without := startWorld(t, cfg, w.kv, w.blobs)
	if isWorker(without, worker) {
		t.Fatal("a grant by a key that is no longer the organisers' still counts")
	}

	// Taken back, it is nobody's machine.
	if code := again.as(two).do("DELETE", "/api/v1/admin/workers/"+key, nil, nil); code != 200 || isWorker(again, worker) {
		t.Fatalf("taking the grant back: HTTP %d", code)
	}
	if code := again.as(two).do("DELETE", "/api/v1/admin/workers/"+key, nil, nil); code != 404 {
		t.Fatalf("taking it back twice: HTTP %d", code)
	}
	again.check()
}

// A rented machine the organisers end a moment after it took a job holds nothing from then on:
// stet-runpod tells the server it is gone, and the job is handed back as if the machine had
// handed it back itself, for another to be given. Only the organisers say so, and only of a
// machine they vouch for.
func TestTheJobsOfAMachineTheOrganisersEndedAreHandedBack(t *testing.T) {
	pubA, privA := newKey()
	pubW, privW := newKey()
	cfg := testConfig(t, pubA)
	cfg.RunsPerEntry = 1
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	anchor := w.join("anchor-one", privA)
	if code := w.vouch(anchor, pubW); code != 201 {
		t.Fatalf("vouching for a rented machine: HTTP %d", code)
	}
	worker := w.join("rented-one", privW)
	alice := w.join("alice", nil)
	w.submit(anchor, "ref-a", "print('a')")
	w.claim(anchor, true)
	w.claim(worker, false)
	w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	c := w.claim(worker, false)
	if c.Job == nil || c.Job.Kind != "grade" {
		t.Fatalf("the rented machine should be grading: %+v", c.Job)
	}

	release := "/api/v1/admin/workers/" + hex.EncodeToString(pubW) + "/release"
	var out struct {
		Handle   string `json:"handle"`
		Jobs     int    `json:"jobs"`
		Students int    `json:"students"`
	}
	if code := alice.do("POST", release, nil, &out); code != 403 {
		t.Fatalf("a volunteer handed back a rented machine's jobs: HTTP %d", code)
	}
	if code := worker.do("POST", release, nil, &out); code != 403 {
		t.Fatalf("a rented machine handed back its own jobs this way: HTTP %d", code)
	}
	if code := anchor.do("POST", "/api/v1/admin/workers/"+alice.pub+"/release", nil, &out); code != 404 {
		t.Fatalf("handing back the jobs of a machine nobody vouches for: HTTP %d", code)
	}
	if code := anchor.do("POST", release, nil, &out); code != 200 || out.Handle != "rented-one" || out.Jobs != 1 || out.Students != 0 {
		t.Fatalf("handing back the ended machine's grade: HTTP %d %+v", code, out)
	}
	var work Work
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	if len(work.Rounds) != 1 || work.Rounds[0].GradesUnderWay != 0 || work.Rounds[0].GradesWanted != 1 {
		t.Fatalf("the grade is not wanted again: %+v", work.Rounds)
	}
	for _, m := range work.Machines {
		if m.Handle == "rented-one" && m.Busy != "" {
			t.Fatalf("the ended machine is still at work: %+v", m)
		}
	}
	if code, _ := w.report(worker, c, 5.5); code != 409 {
		t.Fatalf("a grade from the ended machine was taken: HTTP %d", code)
	}
	// Again, it has nothing to hand back.
	if code := anchor.do("POST", release, nil, &out); code != 200 || out.Jobs != 0 {
		t.Fatalf("handing back again: HTTP %d %+v", code, out)
	}
	w.check()
}
