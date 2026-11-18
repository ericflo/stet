package main

import (
	"bytes"
	"encoding/hex"
	"path/filepath"
	"testing"
)

// A machine the organisers rent grades the organisers' own entries as one of their machines
// does, and that is all it does: it administers nothing, and the held-out exam is never
// shared with it.
func TestARentedMachineGradesForTheOrganisersAndAdministersNothing(t *testing.T) {
	pubA, privA := newKey()
	pubW, privW := newKey()
	cfg := testConfig(t, pubA)
	cfg.WorkerKeys = map[string]bool{hex.EncodeToString(pubW): true}
	cfg.RunsPerEntry = 1
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	anchor, worker := w.join("anchor-one", privA), w.join("rented-one", privW)
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
	for _, path := range []string{"/api/v1/admin/sealed", "/api/v1/admin/sealed/mine", "/api/v1/admin/accounts/anchor-one", "/api/v1/admin/work"} {
		if code := worker.do("GET", path, nil, &out); code != 403 {
			t.Fatalf("a rented machine asked for %s: HTTP %d %s", path, code, out.Error)
		}
	}
	if code := worker.do("POST", "/api/v1/admin/draw", nil, &out); code != 403 {
		t.Fatalf("a rented machine brought the draw forward: HTTP %d", code)
	}
	if code := worker.do("PUT", "/api/v1/admin/files/exam-s0.jsonl.zst", []byte("x"), &out); code != 403 {
		t.Fatalf("a rented machine published a file: HTTP %d", code)
	}

	// The held-out exam is not shared with it: it is not on the list of keys that want a
	// copy, and no copy sealed to it is kept.
	w.app.season.HeldoutSHA256 = "ab" + string(bytes.Repeat([]byte("cd"), 31))
	var list struct {
		Organisers []struct {
			Key string `json:"key"`
		} `json:"organisers"`
	}
	if code := anchor.do("GET", "/api/v1/admin/sealed", nil, &list); code != 200 || len(list.Organisers) != 1 || list.Organisers[0].Key != anchor.pub {
		t.Fatalf("the keys the held-out exam is shared with: HTTP %d %+v", code, list)
	}
	sealed := append([]byte("stet-sealed-1\n"), bytes.Repeat([]byte{7}, 200)...)
	if code := anchor.do("PUT", "/api/v1/admin/sealed/"+worker.pub, sealed, &out); code != 403 {
		t.Fatalf("a copy was kept for a rented machine's key: HTTP %d %s", code, out.Error)
	}
	w.app.season.HeldoutSHA256 = ""

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
	cfg := testConfig(t)
	cfg.WorkerKeys = map[string]bool{hex.EncodeToString(pubW): true}
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	w.join("rented-two", privW)
	var board struct {
		Verifiers []VerifierRow `json:"verifiers"`
	}
	if code := (&actor{t: t, srv: w.srv}).do("GET", "/api/v1/verifiers", nil, &board); code != 200 || len(board.Verifiers) != 0 {
		t.Fatalf("HTTP %d %+v", code, board.Verifiers)
	}
}
