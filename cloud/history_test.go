package main

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// readLines reads a file of JSON lines into v's kind, one each.
func readLines[T any](t *testing.T, path string) []T {
	t.Helper()
	fh, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	var out []T
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		var v T
		if err := json.Unmarshal(sc.Bytes(), &v); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		out = append(out, v)
	}
	return out
}

// Each look is kept: what the server said, whole; what the fleet decided and said; each
// market's offers, how long they took; each machine tried; every call to a market or the
// server, timed, without its query; and the markets' accounts and offers, now and then.
func TestEachLookIsKept(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 2)
	ctx := context.Background()
	st.work = *round(8, 0, 0, 0)
	f.o.Edits = true
	if err := tick(ctx, f); err != nil || len(rp.created) != 2 {
		t.Fatalf("rented %d (%v)", len(rp.created), err)
	}
	looks := readLines[lookRecord](t, f.looksPath())
	if len(looks) != 1 {
		t.Fatalf("%d looks kept", len(looks))
	}
	l := looks[0]
	if len(l.Rounds) != 1 || l.Rounds[0].ID != "s1-o00001" || l.Rounds[0].RunsWanted != 8 || !strings.Contains(string(l.Work), `"runs_wanted":8`) {
		t.Fatalf("the work: %+v %s", l.Rounds, l.Work)
	}
	if l.Target != 4 || l.Launch != 2 || l.LaunchRuns != 2 || l.Rented != 2 {
		t.Fatalf("decided %d to rent of %d wanted, %d to run curators; rented %d", l.Launch, l.Target, l.LaunchRuns, l.Rented)
	}
	if m := l.Markets["runpod"]; m == nil || m.Offers == 0 || m.Error != "" {
		t.Fatalf("RunPod's offers: %+v", m)
	}
	if len(l.Rents) != 2 || l.Rents[0].Outcome != "rented" || l.Rents[0].For != "curators" || l.Rents[0].Market != "runpod" || l.Rents[0].ID == "" {
		t.Fatalf("tried: %+v", l.Rents)
	}
	who := map[string]int{}
	for _, c := range l.Calls {
		who[c.Who]++
		if strings.Contains(c.Path, "?") || strings.Contains(c.Path, "key") {
			t.Fatalf("a call kept with more than its path: %+v", c)
		}
	}
	if who["stet"] == 0 || who["runpod"] == 0 {
		t.Fatalf("calls: %v", who)
	}
	if l.Server != "0123456789abcdef0123456789abcdef01234567" || l.CLI != "0.4.6" { // the fleet's commit: test binaries have none
		t.Fatalf("builds: server %q, cli %q, fleet %q", l.Server, l.CLI, l.Fleet)
	}
	if len(l.Said) == 0 || !strings.Contains(strings.Join(l.Said, "\n"), "rented stet-rp-") {
		t.Fatalf("said: %q", l.Said)
	}
	// What each machine was rented for, the round, and when it was to report.
	led, _ := f.readLedger()
	for _, r := range led.Pods {
		if r.For != "curators" || r.Round != "s1-o00001" || r.Report.Before(r.Start) {
			t.Fatalf("its record: %+v", r)
		}
	}
	if raw, _ := os.ReadFile(f.eventsPath()); !strings.Contains(string(raw), "; for curators of s1-o00001; expected to report by ") {
		t.Fatalf("the rented event:\n%s", raw)
	}
	// The accounts, once an hour; the offers, every ten minutes.
	if bs := readLines[balanceRecord](t, f.balancesPath()); len(bs) != 1 || bs[0].Market != "runpod" {
		t.Fatalf("balances: %+v", bs)
	}
	offers := filepath.Join(f.historyDir(), "offers-"+time.Now().UTC().Format("2006-01-02")+".jsonl")
	if os := readLines[offerKind](t, offers); len(os) == 0 || os[0].Market != "runpod" || os[0].Min <= 0 || os[0].N == 0 {
		t.Fatalf("offers: %+v", os)
	}
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if bs := readLines[balanceRecord](t, f.balancesPath()); len(bs) != 1 {
		t.Fatalf("balances kept again within the hour: %d", len(bs))
	}
	if looks := readLines[lookRecord](t, f.looksPath()); len(looks) != 2 {
		t.Fatalf("%d looks kept", len(looks))
	}
}

// A look that went no further is kept with why.
func TestALookThatFailedIsKept(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	st.srv.Close()
	if err := tick(context.Background(), f); err == nil {
		t.Fatal("no error with the server away")
	}
	looks := readLines[lookRecord](t, f.looksPath())
	if len(looks) != 1 || !strings.Contains(looks[0].Error, "could not be reached") || len(looks[0].Calls) == 0 || looks[0].Calls[0].Error == "" {
		t.Fatalf("kept as %+v", looks)
	}
}

// A round the work no longer has open is kept whole, as the server tells it.
func TestASettledRoundIsKept(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	ctx := context.Background()
	st.drawnAt = 1795262400
	st.work = *round(0, 0, 0, 1)
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	st.work = Work{}
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(f.historyDir(), "rounds", "s1-o00001.json"))
	if err != nil || !strings.Contains(string(raw), `"drawn_at": 1795262400`) {
		t.Fatalf("kept %s (%v)", raw, err)
	}
	lines := readLines[map[string]any](t, filepath.Join(f.historyDir(), "rounds.jsonl"))
	if len(lines) != 1 || lines[0]["round"] != "s1-o00001" || lines[0]["drawn_at"].(float64) != 1795262400 {
		t.Fatalf("rounds.jsonl: %+v", lines)
	}
}

// A machine that asks for work before the next look has its boot counted all the same.
func TestAFastBootIsCounted(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	ctx := context.Background()
	st.work = *round(30, 1, 0, 0)
	if err := tick(ctx, f); err != nil || len(rp.created) != 1 {
		t.Fatalf("rented %d (%v)", len(rp.created), err)
	}
	pod := rp.pods["pod1"]
	pod.CreatedAt = time.Now().Add(-90 * time.Second)
	for k, h := range st.handles {
		_ = k
		st.work.Machines = append(st.work.Machines, Machine{Handle: h, Worker: true, LastSeen: time.Now().Add(-50 * time.Second).Unix()})
	}
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if bs := f.bootsOn("runpod"); len(bs) != 1 || bs[0] > 1 {
		t.Fatalf("boots: %v", bs)
	}
}

// The days before today's are kept compressed.
func TestOldDaysAreCompressed(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	old := filepath.Join(f.historyDir(), "looks-2026-11-18.jsonl")
	os.WriteFile(old, []byte(`{"t":"2026-11-18T00:00:00Z"}`+"\n"), 0o600)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("the old day is still there: %v", err)
	}
	if _, err := os.Stat(old + ".gz"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(f.looksPath()); err != nil {
		t.Fatalf("today's: %v", err)
	}
}
