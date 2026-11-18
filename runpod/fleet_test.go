package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeRunPod is RunPod as far as stet-runpod can tell: a catalog, and pods made and ended.
type fakeRunPod struct {
	mu      sync.Mutex
	pods    map[string]*Pod
	noRoom  map[string]bool // GPU kinds it has none of
	created []NewPod
	n       int
	srv     *httptest.Server
}

func newFakeRunPod(t *testing.T) *fakeRunPod {
	f := &fakeRunPod{pods: map[string]*Pod{}, noRoom: map[string]bool{}}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeRunPod) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if r.Header.Get("Authorization") != "Bearer rp-key" {
		w.WriteHeader(401)
		return
	}
	switch {
	case r.Method == "GET" && r.URL.Path == "/catalog/gpus":
		if r.URL.Query().Get("cloud") != "SECURE" || r.URL.Query().Get("minCudaVersion") != "13.0" {
			w.WriteHeader(422)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"gpus": []map[string]any{
			{"id": "NVIDIA RTX A5000", "name": "RTX A5000", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"secure": 0.27}, "availability": "LOW"},
			{"id": "NVIDIA RTX A4000", "name": "RTX A4000", "manufacturer": "NVIDIA", "memory": 16, "price": map[string]any{"secure": 0.25}, "availability": "NONE"},
			{"id": "Tesla V100-PCIE-16GB", "name": "V100", "manufacturer": "NVIDIA", "memory": 16, "price": map[string]any{"secure": 0.19}, "availability": "HIGH"},
			{"id": "NVIDIA B300 SXM6 AC MIG 1g.34gb", "name": "B300 MIG", "manufacturer": "NVIDIA", "memory": 34, "price": map[string]any{"secure": 0.2}, "availability": "HIGH"},
			{"id": "NVIDIA GeForce RTX 3080", "name": "RTX 3080", "manufacturer": "NVIDIA", "memory": 10, "price": map[string]any{"secure": 0.2}, "availability": "LOW"},
			{"id": "NVIDIA RTX PRO 4000 Blackwell", "name": "RTX PRO 4000", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"secure": 0.57}, "availability": "LOW"},
			{"id": "AMD Instinct MI300X OAM", "name": "MI300X", "manufacturer": "AMD", "memory": 192, "price": map[string]any{"secure": 0.1}, "availability": "HIGH"},
		}})
	case r.Method == "GET" && r.URL.Path == "/pods":
		var list []Pod
		for _, p := range f.pods {
			list = append(list, *p)
		}
		json.NewEncoder(w).Encode(map[string]any{"pods": list, "pagination": map[string]any{"hasNextPage": false}})
	case r.Method == "POST" && r.URL.Path == "/pods":
		var spec NewPod
		json.NewDecoder(r.Body).Decode(&spec)
		if f.noRoom[spec.GPU.ID] {
			w.WriteHeader(400)
			io.WriteString(w, `{"detail":"There are no longer any instances available with the requested specifications."}`)
			return
		}
		f.created = append(f.created, spec)
		f.n++
		price := map[string]float64{"NVIDIA RTX A5000": 0.27, "NVIDIA GeForce RTX 3080": 0.2, "NVIDIA RTX PRO 4000 Blackwell": 0.57}[spec.GPU.ID]
		p := &Pod{ID: fmt.Sprintf("pod%d", f.n), Name: spec.Name, Status: "RUNNING", Cost: price, CreatedAt: time.Now()}
		p.GPU = &struct {
			ID     string `json:"id"`
			Count  int    `json:"count"`
			Memory int    `json:"memory"`
		}{ID: spec.GPU.ID, Count: 1}
		f.pods[p.ID] = p
		w.WriteHeader(201)
		json.NewEncoder(w).Encode(p)
	case r.Method == "DELETE" && strings.HasPrefix(r.URL.Path, "/pods/"):
		id := strings.TrimPrefix(r.URL.Path, "/pods/")
		if f.pods[id] == nil {
			w.WriteHeader(404)
			return
		}
		delete(f.pods, id)
		w.WriteHeader(204)
	default:
		w.WriteHeader(404)
	}
}

// fakeStet is the Stet server: what is waiting, accounts, and the season's site.
type fakeStet struct {
	mu      sync.Mutex
	work    Work
	workers map[string]bool // public keys the server takes as rented machines
	signed  []string        // who signed each request
	srv     *httptest.Server
}

func newFakeStet(t *testing.T) *fakeStet {
	f := &fakeStet{workers: map[string]bool{}}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		f.signed = append(f.signed, r.Method+" "+r.URL.Path+" "+r.Header.Get("X-Stet-Key"))
		switch r.URL.Path {
		case "/api/v1/admin/work":
			json.NewEncoder(w).Encode(f.work)
		case "/api/v1/season":
			io.WriteString(w, `{"site": "https://www.stet.test"}`)
		case "/api/v1/accounts":
			var body struct{ Handle string }
			json.NewDecoder(r.Body).Decode(&body)
			worker := f.workers[r.Header.Get("X-Stet-Key")]
			json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{"handle": body.Handle, "anchor": worker, "worker": worker}})
		default:
			w.WriteHeader(404)
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func testFleet(t *testing.T, rp *fakeRunPod, st *fakeStet, keys int) (*Fleet, *bytes.Buffer) {
	t.Helper()
	o := &Options{Home: t.TempDir(), Cloud: "SECURE", Image: "ubuntu:24.04", Disk: 40, MinCUDA: "13.0", MinRAM: 12, MinCPU: 4, MinGPUMemory: 10, Idle: 10, Every: time.Minute, Policy: testPolicy()}
	var out bytes.Buffer
	op, _ := newKey("ericflo-wsl")
	f := &Fleet{o: o, rp: newRunPod(rp.srv.URL, "rp-key"), st: newStet(st.srv.URL), op: op, out: &out, now: time.Now}
	f.rp.sleep = func(context.Context, time.Duration) error { return nil }
	if _, err := f.addKeys(keys, "stet-rp"); err != nil {
		t.Fatal(err)
	}
	for _, k := range f.keys {
		st.workers[k.Public] = true
	}
	return f, &out
}

func TestScaleRentsTheCheapestThatFitsAndEndsWhatIsNotWanted(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 4)
	ctx := context.Background()
	rp.noRoom["NVIDIA GeForce RTX 3080"] = true // the cheapest that fits has none free

	// A round has started: two machines for the one runner.
	st.work = *round(30, 1, 0, 0)
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 2 {
		t.Fatalf("rented %d machines, want 2:\n%s", len(rp.created), out)
	}
	for i, spec := range rp.created {
		// Not the V100 (no bf16), the MIG slice, the AMD card, or the A4000 (none free);
		// the 3080 had none, so the next cheapest.
		if spec.GPU.ID != "NVIDIA RTX A5000" || spec.Cloud != "SECURE" || spec.GPU.MinCUDAVersion != "13.0" || spec.Name != fmt.Sprintf("stet-rp-%02d", i+1) {
			t.Fatalf("pod %d: %+v", i, spec)
		}
		// It is given its own key, whole, and the script that runs it.
		raw, _ := base64.StdEncoding.DecodeString(spec.Env["STET_WORKER_KEY"])
		var k Key
		if json.Unmarshal(raw, &k); k.Handle != spec.Name || k.Private != f.keys[i].Private {
			t.Fatalf("pod %d was given the key %+v", i, k)
		}
		if spec.Env["STET_INSTALL"] != "https://www.stet.test/install.sh" || spec.Env["STET_API"] != st.srv.URL || spec.Env["STET_IDLE_MINUTES"] != "10" {
			t.Fatalf("pod %d's environment: %v", i, spec.Env)
		}
		if strings.Join(spec.Entrypoint, " ") != "/bin/bash -c" || spec.Cmd[0] != workerScript || len(spec.Ports) != 0 {
			t.Fatalf("pod %d runs %v %v, ports %v", i, spec.Entrypoint, len(spec.Cmd), spec.Ports)
		}
	}
	if strings.Contains(out.String(), f.keys[0].Private) {
		t.Fatal("a private key was printed")
	}
	// Each machine's account was made with its own key, and what is waiting was asked with
	// the organisers' own.
	signedBy := strings.Join(st.signed, "\n")
	if !strings.Contains(signedBy, "GET /api/v1/admin/work "+f.op.Public) || !strings.Contains(signedBy, "POST /api/v1/accounts "+f.keys[1].Public) {
		t.Fatalf("requests:\n%s", signedBy)
	}

	// Looked at again with the same work, nothing more is rented for it.
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 2 {
		t.Fatalf("rented again for the same work: %d", len(rp.created))
	}

	// Scripts pile up; past --max-hourly, the dearer kind is not rented.
	f.o.MaxHourly = 0.27*2 + 0.57 - 0.01
	rp.noRoom["NVIDIA RTX A5000"] = true
	st.work = *round(10, 1, 12, 0)
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 2 || !strings.Contains(out.String(), "--max-hourly") {
		t.Fatalf("rented past --max-hourly:\n%s", out)
	}
	f.o.MaxHourly = 3
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 4 || rp.created[2].GPU.ID != "NVIDIA RTX PRO 4000 Blackwell" {
		t.Fatalf("with room in the budget: %d rented", len(rp.created))
	}

	// The round is done. The two machines that ask for work and have none are ended now;
	// the one grading is not.
	st.work = *round(0, 0, 0, 1)
	now := time.Now().Unix()
	st.work.Machines = []Machine{
		{Handle: "stet-rp-01", Worker: true, LastSeen: now},
		{Handle: "stet-rp-02", Worker: true, LastSeen: now, Busy: "grade"},
		{Handle: "stet-rp-03", Worker: true, LastSeen: now},
	}
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	left := []string{}
	for _, p := range rp.pods {
		left = append(left, p.Name)
	}
	if len(left) != 1 || left[0] != "stet-rp-02" {
		t.Fatalf("left running: %v\n%s", left, out)
	}

	// What was spent is written down: three pods ended, one still running.
	l, err := f.readLedger()
	if err != nil || len(l.Pods) != 1 {
		t.Fatalf("live pods in the ledger: %+v %v", l, err)
	}
	raw, _ := os.ReadFile(f.spentLog())
	if n := strings.Count(string(raw), "\n"); n != 3 {
		t.Fatalf("ended pods written down: %d\n%s", n, raw)
	}
	if spent, err := f.spent(l, startOfDay(time.Now())); err != nil || spent < 0 || spent > 0.01 {
		t.Fatalf("spent so far: %v %v", spent, err)
	}
}

// A key the server does not yet take as a rented machine is never rented for: such a
// machine's grades would be a stranger's, and a stranger is not given work like this.
func TestNothingIsRentedForAKeyTheServerDoesNotKnow(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 2)
	st.workers = map[string]bool{}
	st.work = *round(30, 1, 0, 0)
	err := tick(context.Background(), f)
	if err == nil || !strings.Contains(err.Error(), "STET_WORKER_KEYS") || len(rp.created) != 0 {
		t.Fatalf("rented %d for a key the server does not know (%v)", len(rp.created), err)
	}
}

// --dry-run rents and ends nothing.
func TestADryRunRentsNothing(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	f.o.DryRun = true
	st.work = *round(30, 1, 0, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 0 || strings.Count(out.String(), "would rent") != 2 {
		t.Fatalf("a dry run rented %d:\n%s", len(rp.created), out)
	}
}

// Settings are read wherever they are put: `up 2 --dry-run` is a dry run. (Go's flag
// package alone stops at "2", and once rented two machines that way.)
func TestSettingsAfterTheNumberAreNotIgnored(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	home := t.TempDir()
	op, _ := newKey("stet")
	opPath := home + "/operator.json"
	if err := op.save(opPath); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RUNPOD_API_KEY", "rp-key")
	common := []string{"--home", home, "--api", st.srv.URL, "--runpod-api", rp.srv.URL, "--key", opPath}
	var out, errs bytes.Buffer
	if err := run(append([]string{"keys", "--add", "2"}, common...), &out, &errs); err != nil {
		t.Fatal(err)
	}
	f := &Fleet{o: &Options{Home: home}}
	if err := f.loadKeys(); err != nil || len(f.keys) != 2 {
		t.Fatalf("keys: %d %v", len(f.keys), err)
	}
	for _, k := range f.keys {
		st.workers[k.Public] = true
	}
	out.Reset()
	if err := run(append([]string{"up", "2", "--dry-run"}, common...), &out, &errs); err != nil {
		t.Fatalf("%v\n%s", err, errs.String())
	}
	if len(rp.created) != 0 || strings.Count(out.String(), "would rent") != 2 {
		t.Fatalf("up 2 --dry-run rented %d:\n%s", len(rp.created), out.String())
	}
	for _, bad := range [][]string{{"up", "2", "3"}, {"up", "2x"}, {"status", "now"}} {
		if err := run(append(bad, common...), &out, &errs); err == nil {
			t.Fatalf("%v was taken", bad)
		}
	}
	if len(rp.created) != 0 {
		t.Fatalf("rented %d", len(rp.created))
	}
}
