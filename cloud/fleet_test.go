package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeRunPod is RunPod as far as stet-cloud can tell: a catalog, and pods made and ended.
type fakeRunPod struct {
	mu             sync.Mutex
	pods           map[string]*Pod
	noRoom         map[string]bool     // GPU kinds it has none of
	logs           map[string][]string // what each pod's container said
	extraCommunity []map[string]any    // extra community offers for a single test
	extraPrices    map[string]float64  // hourly prices for extraCommunity
	created        []NewPod
	n              int
	srv            *httptest.Server
}

func newFakeRunPod(t *testing.T) *fakeRunPod {
	f := &fakeRunPod{pods: map[string]*Pod{}, noRoom: map[string]bool{}, logs: map[string][]string{}}
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
		if r.URL.Query().Get("minCudaVersion") != "13.0" {
			w.WriteHeader(422)
			return
		}
		if r.URL.Query().Get("cloud") == "COMMUNITY" {
			gpus := []map[string]any{
				{"id": "NVIDIA GeForce RTX 3080 Ti", "name": "RTX 3080 Ti", "manufacturer": "NVIDIA", "memory": 12, "price": map[string]any{"community": 0.18}, "availability": "LOW"},
				{"id": "NVIDIA RTX A5000", "name": "RTX A5000", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"community": 0.16}, "availability": "NONE"},
				{"id": "NVIDIA GeForce RTX 3090 Ti", "name": "RTX 3090 Ti", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"community": 0.27}, "availability": "LOW"},
				{"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"community": 0.45}, "availability": "LOW"},
			}
			gpus = append(gpus, f.extraCommunity...)
			json.NewEncoder(w).Encode(map[string]any{"gpus": gpus})
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
	case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/pods/") && strings.HasSuffix(r.URL.Path, "/logs"):
		id := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/pods/"), "/logs")
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range f.logs[id] {
			raw, _ := json.Marshal(LogLine{Source: "container", Line: l, TS: "2026-11-21T08:00:00Z"})
			fmt.Fprintf(w, "data: %s\n\n", raw)
		}
	case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/pods/") && strings.Count(r.URL.Path, "/") == 2:
		p := f.pods[strings.TrimPrefix(r.URL.Path, "/pods/")]
		if p == nil {
			w.WriteHeader(404)
			return
		}
		json.NewEncoder(w).Encode(p)
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
		price := map[string]float64{"NVIDIA RTX A5000": 0.27, "NVIDIA GeForce RTX 3080": 0.2, "NVIDIA RTX PRO 4000 Blackwell": 0.57,
			"NVIDIA GeForce RTX 3080 Ti": 0.18, "NVIDIA GeForce RTX 3090 Ti": 0.27}[spec.GPU.ID]
		if p2, ok := f.extraPrices[spec.GPU.ID]; ok {
			price = p2
		}
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

// fakeStet is the Stet server: what is waiting, accounts, grants, and the season's site.
type fakeStet struct {
	mu       sync.Mutex
	work     Work
	drawnAt  int64             // when the open round was drawn, as GET /api/v1/rounds/{id} says
	rounds   string            // GET /api/v1/rounds
	graded   map[string]string // GET /api/v1/rounds/{id} of settled rounds, by id
	workers  map[string]bool   // public keys the server takes as rented machines
	signed   []string          // who signed each request
	vouches  int               // grants it was sent, each with a good signature
	refusing bool              // the key vouching is not one of the organisers' own
	taken    map[string]bool   // handles somebody else has
	handles  map[string]string // each rented machine's handle, by its public key
	released []string          // machines whose jobs were handed back, by handle
	srv      *httptest.Server
}

func newFakeStet(t *testing.T) *fakeStet {
	f := &fakeStet{workers: map[string]bool{}, handles: map[string]string{}}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		f.signed = append(f.signed, r.Method+" "+r.URL.Path+" "+r.Header.Get("X-Stet-Key"))
		switch r.URL.Path {
		case "/api/v1/admin/work":
			json.NewEncoder(w).Encode(f.work)
		case "/api/v1/admin/pause":
			var req struct {
				Paused bool `json:"paused"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			f.work.Paused = 0
			if req.Paused {
				f.work.Paused = 1795000000
			}
			fmt.Fprintf(w, `{"paused": %d, "paused_by": "stet"}`, f.work.Paused)
		case "/api/v1/season":
			io.WriteString(w, `{"site": "https://www.stet.test", "server": "0123456789abcdef0123456789abcdef01234567", "cli": {"version": "0.4.6"}}`)
		case "/api/v1/rounds":
			if f.rounds == "" {
				w.WriteHeader(404)
			}
			io.WriteString(w, f.rounds)
		case "/api/v1/rounds/s1-o00009":
			io.WriteString(w, f.graded["s1-o00009"])
		case "/api/v1/rounds/s1-o00001":
			fmt.Fprintf(w, `{"entries": [], "round": {"id": "s1-o00001", "drawn_at": %d}}`, f.drawnAt)
		case "/api/v1/admin/workers":
			switch r.Method {
			case "GET":
				var list []map[string]any
				for k := range f.workers {
					list = append(list, map[string]any{"key": k, "counts": true, "by": "stet"})
				}
				json.NewEncoder(w).Encode(map[string]any{"workers": list})
			case "POST":
				var g struct {
					Key string `json:"key"`
					At  int64  `json:"at"`
					Sig string `json:"sig"`
				}
				json.NewDecoder(r.Body).Decode(&g)
				pub, _ := hex.DecodeString(r.Header.Get("X-Stet-Key"))
				sig, _ := hex.DecodeString(g.Sig)
				switch {
				case f.refusing:
					w.WriteHeader(403)
					io.WriteString(w, `{"error": "organisers only"}`)
				case len(pub) != ed25519.PublicKeySize || !ed25519.Verify(pub, grantMessage(g.Key, g.At), sig):
					w.WriteHeader(400)
				default:
					f.workers[g.Key] = true
					f.vouches++
					w.WriteHeader(201)
					io.WriteString(w, "{}")
				}
			}
		case "/api/v1/accounts":
			var body struct{ Handle string }
			json.NewDecoder(r.Body).Decode(&body)
			if f.taken[body.Handle] {
				w.WriteHeader(409)
				io.WriteString(w, `{"error": "the handle \"`+body.Handle+`\" is taken"}`)
				return
			}
			worker := f.workers[r.Header.Get("X-Stet-Key")]
			json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{"handle": body.Handle, "anchor": worker, "worker": worker}})
		default:
			if key, ok := strings.CutPrefix(r.URL.Path, "/api/v1/admin/workers/"); ok && r.Method == "DELETE" && f.workers[key] {
				delete(f.workers, key)
				io.WriteString(w, "{}")
				return
			}
			if key, ok := strings.CutSuffix(strings.TrimPrefix(r.URL.Path, "/api/v1/admin/workers/"), "/release"); ok && r.Method == "POST" && f.workers[key] {
				handle := f.handles[key]
				f.released = append(f.released, handle)
				jobs, students := 0, 0
				for i := range f.work.Machines {
					if m := &f.work.Machines[i]; m.Handle == handle {
						if m.Busy == "farm" {
							students++
						} else if m.Busy != "" {
							jobs++
						}
						m.Busy = ""
					}
				}
				fmt.Fprintf(w, `{"handle": %q, "jobs": %d, "students": %d}`, handle, jobs, students)
				return
			}
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
	rpc := newRunPod(rp.srv.URL, "rp-key")
	rpc.sleep = func(context.Context, time.Duration) error { return nil }
	f := &Fleet{o: o, markets: []Market{newRunPodMarket(rpc, &o.Cloud)}, fixed: true, st: newStet(st.srv.URL), op: op, out: &out, now: time.Now}
	if _, err := f.addKeys(keys, "stet-rp"); err != nil {
		t.Fatal(err)
	}
	for _, k := range f.keys {
		st.workers[k.Public] = true
		st.handles[k.Public] = k.Handle
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

// A key is vouched for with the organisers' own key before a machine is rented with it, and
// nothing is rented for a key the server would not take: such a machine's work would be a
// stranger's.
func TestAKeyIsVouchedForBeforeAMachineIsRentedWithIt(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	st.workers = map[string]bool{}
	st.refusing = true // run with a key that is not one of the organisers' own
	st.work = *round(30, 1, 0, 0)
	err := tick(context.Background(), f)
	if err == nil || !strings.Contains(err.Error(), "organisers' own keys") || len(rp.created) != 0 {
		t.Fatalf("rented %d for keys nobody could vouch for (%v)", len(rp.created), err)
	}
	st.refusing = false
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if st.vouches != 2 || len(rp.created) != 2 || strings.Count(out.String(), "vouched for stet-rp-") != 2 {
		t.Fatalf("vouched %d times, rented %d:\n%s", st.vouches, len(rp.created), out)
	}
	// Vouched for once is enough.
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	if err := tick(context.Background(), f); err != nil || st.vouches != 2 {
		t.Fatalf("vouched again: %d (%v)", st.vouches, err)
	}
}

// Each machine rented takes the key the server has heard from longest ago: the server gives an
// account one run of an entry, so the key whose last machine ran the round's curators would be
// given none of their second runs.
func TestTheQuietestKeyIsRented(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	ctx := context.Background()
	st.work = *round(0, 0, 1, 0)
	now := time.Now().Unix()
	st.work.Machines = []Machine{
		{Handle: "stet-rp-01", Worker: true, LastSeen: now - 60},
		{Handle: "stet-rp-02", Worker: true, LastSeen: now - 3600},
	}
	if err := tick(ctx, f); err != nil || len(rp.created) != 1 || rp.created[0].Name != "stet-rp-03" {
		t.Fatalf("rented %v, want stet-rp-03, never heard from (%v):\n%s", names(rp.created), err, out)
	}
	// stet-rp-03 ran what it could and stopped: stet-rp-02 is next, then stet-rp-01.
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	st.work.Machines = append(st.work.Machines, Machine{Handle: "stet-rp-03", Worker: true, LastSeen: now})
	if err := tick(ctx, f); err != nil || len(rp.created) != 2 || rp.created[1].Name != "stet-rp-02" {
		t.Fatalf("rented %v, want stet-rp-02 next (%v):\n%s", names(rp.created), err, out)
	}
}

func names(pods []NewPod) []string {
	var out []string
	for _, p := range pods {
		out = append(out, p.Name)
	}
	return out
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

// keys --add makes keys and vouches for them; --revoke takes a grant back and forgets the key.
func TestKeysAreVouchedForAndTakenBack(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	home := t.TempDir()
	op, _ := newKey("stet")
	if err := op.save(home + "/operator.json"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RUNPOD_API_KEY", "rp-key")
	common := []string{"--home", home, "--api", st.srv.URL, "--runpod-api", rp.srv.URL, "--key", home + "/operator.json"}
	var out, errs bytes.Buffer
	if err := run(append([]string{"keys", "--add", "2"}, common...), &out, &errs); err != nil {
		t.Fatalf("%v %s", err, errs.String())
	}
	if st.vouches != 2 || len(st.workers) != 2 || strings.Count(out.String(), "vouched for by") != 2 {
		t.Fatalf("keys --add: %d vouched\n%s", st.vouches, out.String())
	}
	out.Reset()
	if err := run(append([]string{"keys", "--revoke", "stet-rp-01"}, common...), &out, &errs); err != nil {
		t.Fatal(err)
	}
	f := &Fleet{o: &Options{Home: home}}
	if err := f.loadKeys(); err != nil || len(f.keys) != 1 || f.keys[0].Handle != "stet-rp-02" || len(st.workers) != 1 {
		t.Fatalf("after taking stet-rp-01 back: %d keys here, %d vouched for", len(f.keys), len(st.workers))
	}
	if err := run(append([]string{"keys", "--revoke"}, common...), &out, &errs); err == nil {
		t.Fatal("keys --revoke with nothing named was taken")
	}
}

// A handle somebody else took on the server is no obstacle: the key takes another name.
func TestAKeyWhoseHandleIsTakenTakesAnother(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 1)
	st.taken = map[string]bool{"stet-rp-01": true}
	st.work = *round(30, 1, 0, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 1 || rp.created[0].Name == "stet-rp-01" || !strings.HasPrefix(rp.created[0].Name, "stet-rp-") || !strings.Contains(out.String(), "somebody else's handle") {
		t.Fatalf("rented %+v\n%s", rp.created, out)
	}
	again := &Fleet{o: f.o}
	if err := again.loadKeys(); err != nil || len(again.keys) != 1 || again.keys[0].Handle != rp.created[0].Name || again.keys[0].Public != f.keys[0].Public {
		t.Fatalf("the pool after the rename: %+v (%v)", again.keys, err)
	}
}

// By default the fastest machine free is rented for grades, in either cloud, and the next
// fastest when RunPod has none of a kind free: a round settles only once its slowest grade
// is in. Machines to run curators are the cheapest a grade.
func TestTheFastestMachineIsRentedFirst(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	f.o.Cloud = "ANY"
	f.o.PerWorker = 4
	st.work = *round(0, 0, 8, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	got := []string{}
	for _, spec := range rp.created {
		got = append(got, strings.ToLower(spec.Cloud)+" "+shortGPU(spec.GPU.ID))
	}
	if len(got) != 2 || got[0] != "community RTX 4090" || got[1] != "community RTX 4090" {
		t.Fatalf("rented %v\n%s", got, out)
	}
	// With the 4090s gone, the 3090 Ti, the next fastest, though the 3080 Ti is cheaper.
	rp.noRoom["NVIDIA GeForce RTX 4090"] = true
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if last := rp.created[len(rp.created)-1]; last.Cloud != "COMMUNITY" || last.GPU.ID != "NVIDIA GeForce RTX 3090 Ti" {
		t.Fatalf("after the fastest ran out: %+v\n%s", last, out)
	}
	gpus, err := f.choices(context.Background(), false, false)
	if err != nil || len(gpus) < 3 {
		t.Fatalf("choices: %+v %v", gpus, err)
	}
	for i := 1; i < len(gpus); i++ {
		a, b := gpus[i-1], gpus[i]
		if speed(b.Card) > speed(a.Card) || speed(b.Card) == speed(a.Card) && b.Price < a.Price {
			t.Fatalf("not the fastest first: %+v", gpus)
		}
	}
}

// --cloud any --cheapest rents the cheapest machine a grade in either cloud, every time, and
// falls back to the next cheapest when RunPod has none of a kind free.
func TestTheCheapestMachineInEitherCloudIsRented(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	everyHostStarts(t)
	f.o.Cloud = "ANY"
	f.o.Cheapest = true
	st.work = *round(30, 1, 0, 0)
	rp.noRoom["NVIDIA GeForce RTX 3080"] = true
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	// Both on the community 3080 Ti at $0.18: the community A5000 is cheaper but has none
	// free, and every secure kind costs more.
	got := []string{}
	for _, spec := range rp.created {
		got = append(got, strings.ToLower(spec.Cloud)+" "+shortGPU(spec.GPU.ID))
	}
	if len(got) != 2 || got[0] != "community RTX 3080 Ti" || got[1] != "community RTX 3080 Ti" {
		t.Fatalf("rented %v\n%s", got, out)
	}
	// With the 3080 Tis gone, the next cheapest a grade: the community 3090 Ti and the secure
	// A5000 are both $0.27 an hour, and the 3090 Ti grades in three quarters of the time.
	rp.noRoom["NVIDIA GeForce RTX 3080 Ti"] = true
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if last := rp.created[len(rp.created)-1]; last.Cloud != "COMMUNITY" || last.GPU.ID != "NVIDIA GeForce RTX 3090 Ti" {
		t.Fatalf("after the cheapest ran out: %+v", last)
	}
	if spec := rp.created[0]; spec.GPU.MinRAMPerGPU != f.o.MinRAM || spec.GPU.MinVCPUPerGPU != f.o.MinCPU {
		t.Fatalf("the host was asked for as %+v", spec.GPU)
	}
	if _, ok := rp.created[0].Env["STET_FARM_MAX_MINUTES"]; ok {
		t.Fatal("a machine not rented for bigger students was told to take them")
	}
	f.o.LongMinutes = 600
	if spec := f.launchFor(&Key{Handle: "stet-rp-09"}, "https://www.stet.test"); spec.Env["STET_FARM_MAX_MINUTES"] != "600" {
		t.Fatalf("a machine rented for bigger students: %v", spec.Env["STET_FARM_MAX_MINUTES"])
	}
	f.o.LongMinutes = 0
	gpus, err := f.choices(context.Background(), false, false)
	if err != nil || len(gpus) < 3 || gpus[0].Price != 0.18 || gpus[0].Trust != "community" {
		t.Fatalf("choices: %+v %v", gpus, err)
	}
	for i := 1; i < len(gpus); i++ {
		if gpus[i].Price/speed(gpus[i].Card) < gpus[i-1].Price/speed(gpus[i-1].Card) {
			t.Fatalf("not the cheapest a grade first: %+v", gpus)
		}
	}

	// With a ceiling a machine, nothing dearer is rented, however much is waiting: with the
	// 3080 Tis gone, the $0.27 kinds are over $0.20, and nothing is rented at all.
	f.o.MaxPrice = 0.20
	n := len(rp.created)
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != n {
		t.Fatalf("rented %+v over the ceiling", rp.created[n:])
	}
	if gpus, _ := f.choices(context.Background(), false, false); len(gpus) == 0 || gpus[len(gpus)-1].Price > 0.20 {
		t.Fatalf("choices under a $0.20 ceiling: %+v", gpus)
	}
}

// down with names ends those machines and no others, however many are named.
func TestDownEndsOnlyTheMachinesNamed(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 4)
	st.work = *round(0, 0, 40, 0)
	f.o.Policy.MaxWorkers = 4
	if err := tick(context.Background(), f); err != nil || len(rp.pods) != 4 {
		t.Fatalf("rented %d (%v)", len(rp.pods), err)
	}
	if err := cmdDown(context.Background(), f, true, []string{"stet-rp-01", "stet-rp-03"}); err != nil {
		t.Fatal(err)
	}
	left := []string{}
	for _, p := range rp.pods {
		left = append(left, p.Name)
	}
	sort.Strings(left)
	if strings.Join(left, ",") != "stet-rp-02,stet-rp-04" {
		t.Fatalf("left %v", left)
	}
}

// logs prints the last lines a machine's pod said, from RunPod's stream of them.
func TestLogsPrintWhatAMachineSaid(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 1)
	rp.pods["pod9"] = &Pod{ID: "pod9", Name: "stet-rp-01", Status: "RUNNING", CreatedAt: time.Now()}
	logs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pods/pod9/logs" {
			rp.serve(w, r)
			return
		}
		for i, line := range []string{"installing stet", "stet verify · stet-rp-01", "edit job 7"} {
			fmt.Fprintf(w, "id: %d\ndata: {\"source\":\"container\",\"line\":%q,\"ts\":\"2026-11-18T23:1%d:00Z\"}\n\n", i, line, i)
		}
	}))
	defer logs.Close()
	f.markets = []Market{newRunPodMarket(newRunPod(logs.URL, "rp-key"), &f.o.Cloud)}
	out.Reset()
	if err := cmdLogs(context.Background(), f, "stet-rp-01", false, 2); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "23:11:00  stet verify · stet-rp-01\n23:12:00  edit job 7\n" {
		t.Fatalf("printed %q", got)
	}
}

// The organisers pause the rounds with their own key, and status says so.
func TestPauseAndResumeTheRounds(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 1)
	if err := cmdPause(context.Background(), f, true); err != nil {
		t.Fatal(err)
	}
	if st.work.Paused == 0 || !strings.Contains(out.String(), "rounds paused") {
		t.Fatalf("not paused:\n%s", out)
	}
	if last := st.signed[len(st.signed)-1]; !strings.HasPrefix(last, "POST /api/v1/admin/pause ") || !strings.HasSuffix(last, f.op.Public) {
		t.Fatalf("paused as %q", last)
	}
	if err := cmdStatus(context.Background(), f); err != nil || !strings.Contains(out.String(), "Rounds paused since") {
		t.Fatalf("status (%v):\n%s", err, out)
	}
	if err := cmdPause(context.Background(), f, false); err != nil || st.work.Paused != 0 || !strings.Contains(out.String(), "rounds resumed") {
		t.Fatalf("resume (%v):\n%s", err, out)
	}
}

// A machine ended a moment after it took a job still holds it, as the server sees it, with no
// pod rented for it: what it holds is handed back on the next look, as the organisers, so that
// nothing waits out its lease. A rented machine at work keeps its job. A machine ended is
// handed back as it is ended: before an exam the server keeps a grade for the machine that
// holds it, however long it is dead.
func TestWhatAnEndedMachineHeldIsHandedBack(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	ctx := context.Background()
	st.work = *round(30, 1, 0, 0)
	if err := tick(ctx, f); err != nil || len(rp.created) != 2 {
		t.Fatalf("rented %d machines (%v)", len(rp.created), err)
	}

	// The round is done: stet-rp-01 grades and is kept, stet-rp-02 waits and is ended (and
	// handed back, holding nothing), and stet-rp-03, which no pod is rented for, still holds
	// one of the organisers' students.
	st.work = *round(0, 0, 0, 1)
	now := time.Now().Unix()
	st.work.Machines = []Machine{
		{Handle: "stet-rp-01", Worker: true, LastSeen: now, Busy: "grade"},
		{Handle: "stet-rp-02", Worker: true, LastSeen: now},
		{Handle: "stet-rp-03", Worker: true, LastSeen: now - 3600, Busy: "farm"},
	}
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if len(rp.pods) != 1 || strings.Join(st.released, " ") != "stet-rp-02 stet-rp-03" {
		t.Fatalf("%d pods left, handed back %v:\n%s", len(rp.pods), st.released, out)
	}
	if !strings.Contains(out.String(), "handed back what stet-rp-03 held when it was ended: 1 of the organisers' students") {
		t.Fatalf("not said:\n%s", out)
	}
	if last := st.signed[len(st.signed)-1]; !strings.HasPrefix(last, "POST /api/v1/admin/workers/"+f.keys[2].Public+"/release ") || !strings.HasSuffix(last, f.op.Public) {
		t.Fatalf("handed back as %q", last)
	}

	// stet-rp-02 took a grade in the moment before it was ended: it is handed back now.
	st.work.Machines[1].Busy = "grade"
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if strings.Join(st.released, " ") != "stet-rp-02 stet-rp-03 stet-rp-02" || !strings.Contains(out.String(), "handed back what stet-rp-02 held when it was ended: 1 job") {
		t.Fatalf("handed back %v:\n%s", st.released, out)
	}
	// Nothing is held any more, so nothing more is handed back.
	if err := tick(ctx, f); err != nil || len(st.released) != 3 {
		t.Fatalf("handed back %v (%v)", st.released, err)
	}
}

// A machine rented for the organisers' bigger students is chosen for what a student costs on
// it, under --long-max-price: a quick card can train one for less than a cheap one's hours
// cost. The machines for everything else are still the cheapest under --max-price.
func TestTheBiggerStudentsGetTheCardThatTrainsOneForLeast(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 4)
	everyHostStarts(t)
	f.o.Cloud, f.o.MaxPrice, f.o.LongMaxPrice = "ANY", 0.22, 0.90
	f.o.LongMinutes, f.o.LongFarm, f.o.PerFarm = 600, true, 4
	rp.noRoom["NVIDIA GeForce RTX 3080 Ti"] = true // a student costs least on it, and it has none free
	st.work = Work{FarmWaiting: 1, FarmLongWaiting: 2}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	got := []string{}
	for _, spec := range rp.created {
		got = append(got, strings.ToLower(spec.Cloud)+" "+shortGPU(spec.GPU.ID))
		if spec.Env["STET_FARM_MAX_MINUTES"] != "600" {
			t.Fatalf("%s was not told to take the bigger students", spec.Name)
		}
	}
	// The 4090 at $0.45 trains a student for less than the 3090 Ti at $0.27, the 3080 at
	// $0.20 or the A5000 at $0.27; the one for the short students is the $0.20 3080.
	if strings.Join(got, ", ") != "community RTX 4090, community RTX 4090, secure RTX 3080" {
		t.Fatalf("rented %v\n%s", got, out)
	}

	// Under a lower ceiling for them, the 4090 is not among them.
	f.o.LongMaxPrice = 0.40
	gpus, err := f.choices(context.Background(), true, false)
	var kinds []string
	for _, c := range gpus {
		kinds = append(kinds, shortGPU(c.Card))
	}
	if err != nil || strings.Join(kinds, ", ") != "RTX 3080 Ti, RTX 3090 Ti, RTX 3080, RTX A5000" {
		t.Fatalf("for the bigger students under $0.40: %v %v", kinds, err)
	}
	// And with no ceiling of their own, --max-price's.
	f.o.LongMaxPrice = 0
	if gpus, _ := f.choices(context.Background(), true, false); len(gpus) == 0 || gpus[0].Price > 0.22 {
		t.Fatalf("for the bigger students with no ceiling of their own: %+v", gpus)
	}
}

// A bare `status` judges the work as the service does, by the settings `scale --follow` was
// started with: without them it would say what its own defaults would rent and end. Settings
// given to status are taken over the service's.
func TestStatusJudgesByTheServicesSettings(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	home := t.TempDir()
	op, _ := newKey("stet")
	opPath := home + "/operator.json"
	if err := op.save(opPath); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RUNPOD_API_KEY", "rp-key")
	common := []string{"--home", home, "--api", st.srv.URL, "--runpod-api", rp.srv.URL, "--key", opPath}
	st.work = Work{FarmLongWaiting: 7}

	var out, errs bytes.Buffer
	if err := run(append([]string{"status"}, common...), &out, &errs); err != nil {
		t.Fatalf("%v\n%s", err, errs.String())
	}
	if !strings.Contains(out.String(), "The work wants 0 machines now (at most 6).") || strings.Contains(out.String(), "service") {
		t.Fatalf("with no service, by its own defaults:\n%s", out.String())
	}

	o := &Options{}
	fs := flags("scale", o)
	fs.Bool("follow", false, "")
	if _, err := parse(fs, append([]string{"--follow", "--max", "5", "--long-farm", "600", "--gpu", "RTX 4090,RTX A5000"}, common...)); err != nil {
		t.Fatal(err)
	}
	if err := keepService(fs, home, time.Unix(1795078800, 0)); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := run(append([]string{"status"}, common...), &out, &errs); err != nil {
		t.Fatalf("%v\n%s", err, errs.String())
	}
	if !strings.Contains(out.String(), "The work wants 5 machines now (at most 5)") ||
		!strings.Contains(out.String(), `Judged as the service judges it: scale --follow `) ||
		!strings.Contains(out.String(), `--max 5`) || !strings.Contains(out.String(), `--gpu "RTX 4090,RTX A5000"`) ||
		strings.Contains(out.String(), "--follow true") || !strings.Contains(out.String(), "(started Nov 19 09:00 UTC)") {
		t.Fatalf("by the service's settings:\n%s", out.String())
	}

	// A setting given to status is taken over the service's.
	out.Reset()
	if err := run(append([]string{"status", "--max", "3"}, common...), &out, &errs); err != nil {
		t.Fatalf("%v\n%s", err, errs.String())
	}
	if !strings.Contains(out.String(), "The work wants 3 machines now (at most 3)") {
		t.Fatalf("with --max 3 given:\n%s", out.String())
	}
}

// Graders are rented before a round's exam is due, for the grades they could take then and
// train for: each kind just in time to have trained by the exam, the cheapest a round first,
// and none sooner. The round's draw is asked of the round itself.
func TestGradersAreRentedBeforeTheExam(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 4)
	f.o.Early, f.o.ExamAfter, f.o.PerWorker, f.o.Settle = 5*time.Hour, 5*time.Hour, 1, 15*time.Minute
	st.work = Work{Runners: 1, Rounds: []WorkRound{{ID: "s1-o00001", Entries: 3, GradesEarly: 3}}}
	// Nothing seen yet: a machine starts in 8 minutes, and with 2 to fetch and script and 1 to
	// sit the exam, the secure kinds that fit, the 3080 ($0.20) and the A5000 ($0.27), each
	// 0.85 of a 3090 (61.2m/pass with passOn3090=52m), are rented 133 minutes before they are
	// to report; the PRO 4000 ($0.57), 0.8, 141.
	st.drawnAt = time.Now().Add(-30 * time.Minute).Unix() // the exam is due in 270 minutes
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 0 || !strings.Contains(out.String(), "waiting for the cheapest a round, RTX 3080 at $0.20 an hour") {
		t.Fatalf("rented %d machines 270 minutes before the exam\n%s", len(rp.created), out)
	}
	st.drawnAt = time.Now().Add(-175 * time.Minute).Unix() // due in 125 minutes
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 3 || rp.created[0].GPU.ID != "NVIDIA GeForce RTX 3080" || rp.created[2].GPU.ID != "NVIDIA GeForce RTX 3080" {
		t.Fatalf("rented %d machines for 3 early grades\n%s", len(rp.created), out)
	}
	if !strings.Contains(strings.Join(st.signed, "\n"), "GET /api/v1/rounds/s1-o00001 \n") {
		t.Fatalf("the round's draw was not asked of the round, unsigned: %v", st.signed)
	}
}

// Machines to run curators are rented the fastest first, as grades are: with the exam posted
// at the draw, such a machine's grades are wanted once its runs are done, and the first four
// to take one also measure the round's baseline. In Season 2's first round two RTX 5070s,
// rented for curators as the cheapest a grade, trained twice and set its last 35 minutes.
func TestCuratorsAreRunOnTheFastest(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	f.o.Cloud, f.o.Edits = "ANY", true
	st.work = *round(4, 0, 0, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	got := []string{}
	for _, spec := range rp.created {
		got = append(got, strings.ToLower(spec.Cloud)+" "+shortGPU(spec.GPU.ID))
	}
	// The 4090 is the fastest that fits, and its community cloud the cheaper; the secure 3080
	// would have been the cheapest a grade.
	if len(got) != 2 || got[0] != "community RTX 4090" || got[1] != "community RTX 4090" {
		t.Fatalf("rented %v\n%s", got, out)
	}
}

// Season 3 curators require 24 GB+ VRAM (--min-curator-gpu-memory) while grading allows 16 GB+
// (--min-gpu-memory). Verify both halves: curator launches skip the cheap 16 GB card, while
// grade launches DO rent the 16 GB card (and start it grade-only with STET_WORKER_EDITS=0).
func TestCuratorRunsRequire24GBVRAMWhileGradingAllowsSmaller(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	rp.extraCommunity = []map[string]any{
		// 16 GB card priced at $0.25/hr (speed 1.3 -> $0.192/grade, cheaper per grade than $0.27 RTX 3090 Ti).
		{"id": "NVIDIA GeForce RTX 4080", "name": "RTX 4080", "manufacturer": "NVIDIA", "memory": 16, "price": map[string]any{"community": 0.25}, "availability": "LOW"},
	}
	rp.extraPrices = map[string]float64{"NVIDIA GeForce RTX 4080": 0.25}
	f, out := testFleet(t, rp, st, 4)
	everyHostStarts(t)
	f.o.Cloud, f.o.Edits, f.o.MinGPUMemory, f.o.MinCuratorGPUMemory = "COMMUNITY", true, 16, 24
	f.o.Cheapest = true // would pick 16 GB RTX 4080 ($0.25) if not filtered for curators
	st.work = *round(4, 0, 0, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 2 {
		t.Fatalf("rented %d\n%s", len(rp.created), out)
	}
	for _, spec := range rp.created {
		if spec.GPU.ID == "NVIDIA GeForce RTX 4080" || spec.GPU.ID == "NVIDIA GeForce RTX 3080 Ti" {
			t.Fatalf("rented <24 GB card %s for curator runs despite MinCuratorGPUMemory=24", spec.GPU.ID)
		}
	}
	// Now request grades only (no curator runs): must pick the cheaper 16 GB RTX 4080 and
	// launch it with STET_WORKER_EDITS=0 so it is tracked as grade-only!
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	before := len(rp.created)
	st.work = *round(0, 0, 4, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != before+1 {
		t.Fatalf("expected 1 grade machine rented, got %d\n%s", len(rp.created)-before, out)
	}
	if got := rp.created[before]; got.GPU.ID != "NVIDIA GeForce RTX 4080" || got.Env["STET_WORKER_EDITS"] != "0" {
		t.Fatalf("grade machine: gpu=%s edits=%q, want 16 GB RTX 4080 with STET_WORKER_EDITS=0", got.GPU.ID, got.Env["STET_WORKER_EDITS"])
	}
}

// Calibration pass times are scoped per season: Season 2's shorter passes must not outrank
// Season 3 cards or distort Season 3 lead times.
func TestCalibrationResetsPassTimesOnNewSeason(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	// Seed a legacy Season 2 calibration.json (short 3090=18m, 4090=8m passes, plus boot times).
	seed := Calibration{
		Rounds: []string{"s2-o00001"},
		Pass:   map[string][]float64{"RTX 3090": {18.0}, "RTX 4090": {8.0}},
		Boot:   []BootSample{{Pod: "pod1", Minutes: 3.0}, {Pod: "pod2", Minutes: 4.0}, {Pod: "pod3", Minutes: 5.0}},
	}
	raw, _ := json.Marshal(seed)
	if err := os.WriteFile(f.calibrationPath(), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	// Server is now on Season 3 with one settled Season 3 round on RTX 5090 (21m pass).
	st.rounds = `{"rounds": [{"id": "s3-o00001", "status": "settled", "scale": "standard"}]}`
	st.graded = map[string]string{
		"s3-o00001": `{"round": {"id": "s3-o00001"}, "entries": [{"detail": {"hardware": {"gpu": "NVIDIA GeForce RTX 5090"}, "train": {"seconds": 1260}}}]}`,
	}
	// Override fakeStet route for /api/v1/rounds/s3-o00001 via direct season id check.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/season":
			io.WriteString(w, `{"season": "s3", "site": "https://www.stet.test"}`)
		case "/api/v1/rounds":
			io.WriteString(w, st.rounds)
		case "/api/v1/rounds/s3-o00001":
			io.WriteString(w, st.graded["s3-o00001"])
		default:
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()
	f.st = newStet(srv.URL)
	f.calibrate(context.Background())

	// Season 2's 3090 (18m) and 4090 (8m) must be cleared; 5090 (21m) must now rank faster than
	// both unseen 4090 and unseen 3090! And Boot samples are preserved.
	p5090, seen5090 := f.pass("NVIDIA GeForce RTX 5090")
	p4090, seen4090 := f.pass("NVIDIA GeForce RTX 4090")
	p3090, seen3090 := f.pass("NVIDIA GeForce RTX 3090")
	if !seen5090 || seen4090 || seen3090 {
		t.Fatalf("seen flags: 5090=%v 4090=%v 3090=%v", seen5090, seen4090, seen3090)
	}
	if p5090 >= p4090 || p5090 >= p3090 {
		t.Fatalf("Season 2 pass leaked: 5090=%s 4090=%s 3090=%s", p5090, p4090, p3090)
	}
	if f.boot() != 5*time.Minute {
		t.Fatalf("boot samples should survive season rollover, got %s", f.boot())
	}
}

// How often a kind starts is learned from what was rented: its class of host's prior, then
// its market's hosts of that class, then the kind's own.
func TestHowOftenAKindStartsIsLearned(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	t0 := time.Now().Add(-3 * time.Hour)
	rec := func(id, gpu, cloud string, asked bool, lived time.Duration) podRecord {
		return podRecord{ID: id, Market: "runpod", GPU: gpu, Cloud: cloud, Cost: 0.18, Start: t0, Alive: t0.Add(lived), End: t0.Add(lived), Tracked: true, Asked: asked}
	}
	var lines []string
	for i, r := range []podRecord{
		rec("a", "RTX 3080 Ti", "COMMUNITY", false, 25*time.Minute), rec("b", "RTX 3080 Ti", "COMMUNITY", false, 25*time.Minute),
		rec("c", "RTX 3080 Ti", "COMMUNITY", false, 25*time.Minute), rec("d", "RTX 3080 Ti", "COMMUNITY", true, time.Hour),
		rec("e", "RTX 5080", "COMMUNITY", false, 3*time.Minute), // ended by the fleet before it could be said
		rec("f", "RTX 4090", "SECURE", true, time.Hour), rec("g", "RTX 4090", "SECURE", true, time.Hour),
	} {
		raw, _ := json.Marshal(r)
		lines = append(lines, string(raw))
		_ = i
	}
	untracked, _ := json.Marshal(podRecord{ID: "old", Market: "runpod", GPU: "RTX 3080 Ti", Cloud: "COMMUNITY", Start: t0, End: t0.Add(time.Hour)})
	lines = append(lines, string(untracked))
	if err := os.WriteFile(f.spentLog(), []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := f.startRates()
	// RunPod's community: 1 of 4 started, over a prior of 0.75 for 4: 4/8 = 0.5. Its 3080 Ti:
	// 1 of 4 over 0.5 for 4: 3/8.
	if r := s.of(Offer{Market: "runpod", Card: "RTX 3080 Ti", Trust: "community"}); math.Abs(r-3.0/8) > 1e-9 {
		t.Errorf("a community 3080 Ti starts %.3f", r)
	}
	if r := s.of(Offer{Market: "runpod", Card: "RTX 5080", Trust: "community"}); math.Abs(r-0.5) > 1e-9 {
		t.Errorf("a community 5080, unseen, starts as the class does: %.3f", r)
	}
	// One taken away by its market three minutes in, before it asked, did not start.
	gone := rec("h", "RTX 5080", "COMMUNITY", false, 3*time.Minute)
	gone.Gone = true
	raw, _ := json.Marshal(gone)
	fh, _ := os.OpenFile(f.spentLog(), os.O_APPEND|os.O_WRONLY, 0o600)
	fh.Write(append(raw, '\n'))
	fh.Close()
	if r := f.startRates().of(Offer{Market: "runpod", Card: "RTX 5080", Trust: "community"}); r >= 0.5 {
		t.Errorf("a community 5080 taken away before it asked: %.3f", r)
	}
	if r := s.of(Offer{Market: "runpod", Card: "RTX 4090", Trust: "secure"}); r <= 0.95 {
		t.Errorf("a secure 4090, seen to start twice: %.3f", r)
	}
	if r := s.of(Offer{Market: "vast", Card: "RTX 4090", Trust: "verified"}); r != 0.9 {
		t.Errorf("a verified Vast host, unseen: %.3f", r)
	}
	if p := (Offer{Price: 0.18, Rate: 3.0 / 8}).rankPrice(); math.Abs(p-0.48) > 1e-9 {
		t.Errorf("ranked as $%.3f", p)
	}
}

// What a training pass takes on each kind is learned from the grades of settled rounds, each
// round once, and kept; how long a machine takes to start, from those seen starting.
func TestWhatMachinesTakeIsLearned(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 1)
	st.rounds = `{"rounds": [{"id": "s1-o00010", "status": "open", "scale": "standard"}, {"id": "s1-o00009", "status": "settled", "scale": "standard"}]}`
	st.graded = map[string]string{"s1-o00009": `{"round": {"id": "s1-o00009"}, "entries": [
		{"detail": {"hardware": {"gpu": "NVIDIA RTX PRO 4000 Blackwell"}, "train": {"seconds": 930}}},
		{"detail": {"hardware": {"gpu": "NVIDIA RTX PRO 4000 Blackwell"}, "train": {"seconds": 950}}},
		{"detail": {"hardware": {"gpu": "NVIDIA GeForce RTX 4090"}, "train": {"seconds": 480}}},
		{"detail": {}}]}`}
	pro := "NVIDIA RTX PRO 4000 Blackwell"
	if p, seen := f.pass(pro); seen || p != time.Duration(float64(passOn3090)/0.8) {
		t.Fatalf("before any was seen: %s %v", p, seen)
	}
	f.calibrate(context.Background())
	f.calibrate(context.Background()) // within ten minutes: not asked again
	if p, seen := f.pass(pro); !seen || p.Round(time.Second) != 15*time.Minute+40*time.Second {
		t.Fatalf("a pass on the PRO 4000: %s %v\n%s", p, seen, out)
	}
	if s := f.speedOf("NVIDIA GeForce RTX 4090"); s != float64(passOn3090)/float64(8*time.Minute) {
		t.Fatalf("a 4090 is %vx a 3090", s)
	}
	// Seen starting, then asking for work: 3, 4 and 5 minutes, and one seen only once it had.
	t0 := time.Now().Add(-time.Hour).Truncate(time.Second)
	for i, took := range []time.Duration{3 * time.Minute, 4 * time.Minute, 5 * time.Minute} {
		w := &Worker{Rental: &Rental{ID: fmt.Sprintf("pod%d", i), Created: t0}, Machine: &Machine{}}
		f.seeStart(w)
		w.Machine.LastSeen = t0.Add(took).Unix()
		f.seeStart(w)
		f.seeStart(w)
	}
	f.seeStart(&Worker{Rental: &Rental{ID: "late", Created: t0}, Machine: &Machine{LastSeen: t0.Add(40 * time.Minute).Unix()}})
	if b := f.boot(); b != 5*time.Minute {
		t.Fatalf("a machine starts in %s: %+v", b, f.cal.Boot)
	}
	if l := f.lead(Offer{Card: pro}); l.Round(time.Second) != 5*time.Minute+2*time.Minute+2*(15*time.Minute+40*time.Second)+time.Minute {
		t.Fatalf("a PRO 4000's lead: %s", l)
	}
	f.calAt = time.Time{}
	f.calibrate(context.Background()) // keeps it, and counts no round twice
	again := &Fleet{o: f.o, st: f.st, out: out, now: time.Now}
	if p, seen := again.pass(pro); !seen || p.Round(time.Second) != 15*time.Minute+40*time.Second || len(again.calibration().Pass["RTX PRO 4000 Blackwell"]) != 2 {
		t.Fatalf("what was kept: %+v", again.calibration())
	}
	if b := again.boot(); b != 5*time.Minute {
		t.Fatalf("what was kept of starting: %s", b)
	}
}

// With a server that lets a machine hold three grades before the exam and says when the exam
// is due, the graders are rented to take three each: fewer machines, and none more while
// those rented are starting.
func TestGradersTakeSeveralGradesBeforeTheExam(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 4)
	f.o.Early, f.o.PerWorker, f.o.Settle, f.o.MaxHourly = 8*time.Hour, 1, 15*time.Minute, 16
	// Nothing seen: the 3080 ($0.20, 0.85 of a 3090) passes in 61.2 minutes and leads by 133.4, so it
	// takes three grades rented 256.7 minutes before the exam.
	due := time.Now().Add(257*time.Minute + 30*time.Second)
	st.work = Work{EarlyGradesPerMachine: 3, FastFirstSeconds: 120,
		Rounds: []WorkRound{{ID: "s1-o00008", Entries: 6, GradesEarly: 6, ExamDue: due.Unix()}}}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 2 || rp.created[0].GPU.ID != "NVIDIA GeForce RTX 3080" || strings.Count(out.String(), "to take 3 grades before the exam") != 2 {
		t.Fatalf("rented %d machines for 6 early grades:\n%s", len(rp.created), out)
	}
	if strings.Contains(strings.Join(st.signed, "\n"), "GET /api/v1/rounds/s1-o00008") {
		t.Fatalf("the draw was asked for, though the server said when the exam is due")
	}
	l, _ := f.readLedger()
	for _, r := range l.Pods {
		if r.Grades != 3 || r.Market != "runpod" {
			t.Fatalf("recorded as %+v", r)
		}
	}
	if err := tick(context.Background(), f); err != nil || len(rp.created) != 2 {
		t.Fatalf("rented %d while two were starting for three each (%v):\n%s", len(rp.created), err, out)
	}
}

// Grace: the policy's until a market's machines have been seen to start, then twice what four
// in five took there, ten minutes at least.
func TestGraceIsLearnedPerMarket(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	f.o.Policy.Grace = 25 * time.Minute
	if g := f.grace("salad"); g != 25*time.Minute {
		t.Fatalf("Salad, unseen: %s", g)
	}
	if g := f.grace("clore"); g != 40*time.Minute {
		t.Fatalf("Clore, unseen (20 minutes a boot): %s", g)
	}
	cal := f.calibration()
	for i, m := range []float64{1.1, 2.4, 3.0, 5.5, 18.3} {
		cal.Boot = append(cal.Boot, BootSample{Pod: fmt.Sprintf("pod%d", i), Minutes: m})
	}
	cal.Boot = append(cal.Boot, BootSample{Pod: "vast:1", Minutes: 1.8}, BootSample{Pod: "vast:2", Minutes: 2.0}, BootSample{Pod: "vast:3", Minutes: 2.2})
	if g := f.grace("runpod"); g != 2*time.Duration(18.3*float64(time.Minute)) {
		t.Fatalf("RunPod, seen: %s", g)
	}
	if g := f.grace("vast"); g != graceAtLeast {
		t.Fatalf("Vast, seen to start in two minutes: %s", g)
	}
}

// everyHostStarts takes every kind of host to start, for a test of how kinds are ordered by
// price and speed alone.
func everyHostStarts(t *testing.T) {
	startPrior, learnStarts = func(string) float64 { return 1 }, false
	t.Cleanup(func() { startPrior, learnStarts = expectedStart, true })
}

// A machine ended for never asking for work is ended with what became of it: its market never
// started its container, or its container ran and its last words say why. The cause is kept in
// its record and counted against its kind, and what was seen of it is its story.
func TestWhyAMachineFailedIsSaidAndKept(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	st.work = *round(30, 1, 0, 0)
	if err := tick(context.Background(), f); err != nil || len(rp.created) != 2 {
		t.Fatalf("rented %d (%v)", len(rp.created), err)
	}
	// pod1 never gets a container; pod2's container runs, and stet cannot be installed. Both
	// are older than any Grace.
	long := time.Now().Add(-40 * time.Minute)
	rp.pods["pod1"].CreatedAt = long
	rp.pods["pod2"].CreatedAt = long
	rp.pods["pod2"].Runtime = &struct {
		Uptime int `json:"uptime"`
	}{Uptime: 2300}
	rp.logs["pod2"] = []string{"[stet-worker 08:01:00] installing stet from https://www.stet.test/install.sh",
		"error: Failed to install: nvidia_cublas: No space left on device (os error 28)",
		"[stet-worker 08:03:00] the install did not work; trying again"}
	out.Reset()
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	said := out.String()
	if !strings.Contains(said, "ended stet-rp-01: it stopped, or never asked for work: RunPod never started its container in 40m0s (no container yet)") ||
		!strings.Contains(said, "ended stet-rp-02: it stopped, or never asked for work: stet could not be installed; last it said: [stet-worker 08:03:00] the install did not work; trying again") {
		t.Fatalf("said:\n%s", said)
	}
	causes := map[string]string{}
	raw, _ := os.ReadFile(f.spentLog())
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var r podRecord
		json.Unmarshal([]byte(line), &r)
		causes[r.ID] = r.Cause
	}
	if causes["pod1"] != causeNoContainer || causes["pod2"] != causeInstall {
		t.Fatalf("kept causes %v", causes)
	}
	if k := f.startRates().kind(Offer{Market: "runpod", Card: "RTX 3080", Trust: "secure"}); k.rented != 2 || k.failures() != "1 no container, 1 install" {
		t.Fatalf("counted as %+v: %s", k, k.failures())
	}
	if _, err := os.Stat(f.logPath("pod2")); err != nil {
		t.Fatalf("pod2's last words were not kept: %v", err)
	}
	out.Reset()
	if err := cmdMachine(f, "stet-rp-02", 3); err != nil {
		t.Fatal(err)
	}
	story := out.String()
	for _, want := range []string{"rented    RTX 3080, secure, $0.20 an hour", "phase     container up", "state     stopped", "ended     it stopped, or never asked for work: stet could not be installed", "No space left on device"} {
		if !strings.Contains(story, want) {
			t.Fatalf("the story lacks %q:\n%s", want, story)
		}
	}
}

// What a machine last said says why it never asked for work.
func TestTheCauseIsReadFromWhatTheMachineSaid(t *testing.T) {
	for want, lines := range map[string][]string{
		causeDriver:    {"[stet-worker 08:00:01] driver 570.86.10 is older than 580: asking Salad for another machine"},
		causeInstall:   {"[stet-worker 08:00:01] installing stet from https://x/install.sh", "[stet-worker 08:04:00] stet could not be installed"},
		causeSilent:    {"[stet-worker 08:00:01] stet 0.4.4", "stet verify · stet-rp-01 · NVIDIA GeForce RTX 3090"},
		causeContainer: {"start container", "something else"},
	} {
		if got := causeOf(lines); got != want {
			t.Errorf("%q: %s, want %s", lines, got, want)
		}
	}
	got := logLines([]byte(`{"items": [{"event_name": "Pulling image", "event_time": "2026-11-21T08:00:01Z"}, {"event_name": "Image pulled", "event_time": "2026-11-21T08:02:11Z"}]}`))
	if strings.Join(got, "|") != "08:00:01  Pulling image|08:02:11  Image pulled" {
		t.Errorf("Salad's system log read as %q", got)
	}
	if got := logLines([]byte(`[{"log_text": "Created Docker Container", "received_at": "2026-11-21T07:24:09.728703"}]`)); len(got) != 1 || got[0] != "07:24:09  Created Docker Container" {
		t.Errorf("Lium's read as %q", got)
	}
}

// A machine that ran a round's curators and waits for work two and a half hours before its exam,
// when the round wants only grades taken before the exam, is ended; they are rented in their
// time. (Round 8 kept five $0.89 4090s waiting so, with trained grades, for 85 minutes.)
func TestACuratorMachineDoneTooEarlyIsEnded(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	f.o.Early, f.o.PerWorker, f.o.MaxHourly = 8*time.Hour, 1, 16
	if err := cmdUp(context.Background(), f, 1); err != nil || len(rp.created) != 1 {
		t.Fatalf("rented %d (%v)", len(rp.created), err)
	}
	// A 3080 (nothing seen: 0.85 of a 3090) takes ~365 minutes from being rented to have three
	// grades trained: with the exam 430 minutes off and a 10-minute window, its kind's time
	// is ~55 minutes off.
	now := time.Now()
	st.work = Work{EarlyGradesPerMachine: 3, EarlyWindowSeconds: 600,
		Rounds:   []WorkRound{{ID: "s1-o00009", Entries: 3, GradesEarly: 3, ExamDue: now.Add(430 * time.Minute).Unix()}},
		Machines: []Machine{{Handle: "stet-rp-01", Worker: true, LastSeen: now.Add(time.Second).Unix()}}}
	out.Reset()
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "ended stet-rp-01: too early for the exam's grades") || len(rp.created) != 1 || len(rp.pods) != 0 {
		t.Fatalf("rented %d, %d left:\n%s", len(rp.created), len(rp.pods), out)
	}
}

// A machine past its grace whose log still grows is still getting there, a slow host's install,
// and has up to twice its grace; one whose log has stopped is ended, as is one past twice it.
func TestAMachineStillInstallingIsGivenMoreTime(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 1)
	ctx := context.Background()
	st.work = *round(30, 1, 0, 0)
	if err := tick(ctx, f); err != nil || len(rp.created) != 1 {
		t.Fatalf("rented %d (%v)", len(rp.created), err)
	}
	g := f.grace("runpod")
	id := "pod1"
	rp.pods[id].Runtime = &struct {
		Uptime int `json:"uptime"`
	}{Uptime: 300}
	install := []string{"[stet-worker 08:01:00] installing stet from https://www.stet.test/install.sh", " Downloaded numpy"}
	step := func(age time.Duration, lines ...string) string {
		t.Helper()
		rp.pods[id].CreatedAt = time.Now().Add(-age)
		install = append(install, lines...)
		rp.logs[id] = install
		out.Reset()
		if err := tick(ctx, f); err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	// Its log is read once it nears the end of its grace. Grown in the look before its grace
	// runs out, it is given more then: what is decided comes seconds after what is seen.
	if said := step(g - 90*time.Second); strings.Contains(said, "ended") {
		t.Fatalf("ended before its grace:\n%s", said)
	}
	if said := step(g-30*time.Second, " Downloaded numba"); strings.Contains(said, "ended") || !strings.Contains(said, "still getting there") {
		t.Fatalf("at the end of its grace, still installing:\n%s", said)
	}
	if said := step(g+time.Minute, " Downloaded scipy"); strings.Contains(said, "ended") {
		t.Fatalf("past its grace, still installing:\n%s", said)
	}
	if said := step(g+2*time.Minute, " Downloaded torch"); strings.Contains(said, "ended") || strings.Contains(said, "still getting there") {
		t.Fatalf("said again, or ended:\n%s", said)
	}
	// Its log stops for a look, a large library downloading: it is kept. Stopped for
	// slowStall, it is ended.
	if said := step(g + 3*time.Minute); strings.Contains(said, "ended") {
		t.Fatalf("its log quiet for a look:\n%s", said)
	}
	if said := step(g + 2*time.Minute + slowStall + time.Second); !strings.Contains(said, "ended stet-rp-01: it stopped, or never asked for work") {
		t.Fatalf("its log stopped:\n%s", said)
	}
	if raw, _ := os.ReadFile(f.eventsPath()); !strings.Contains(string(raw), `"event":"slow"`) {
		t.Fatalf("not in the journal:\n%s", raw)
	}
	// Past twice its grace, it is ended however its log grows.
	if err := tick(ctx, f); err != nil || len(rp.created) != 2 {
		t.Fatalf("rented %d (%v)", len(rp.created), err)
	}
	id = "pod2"
	rp.pods[id].Runtime = &struct {
		Uptime int `json:"uptime"`
	}{Uptime: 300}
	for _, age := range []time.Duration{g - time.Minute, g + time.Minute, 2*g - time.Minute} {
		if said := step(age, " Downloaded "+age.String()); strings.Contains(said, "ended") {
			t.Fatalf("ended at %s, within twice its grace of %s:\n%s", age, g, said)
		}
	}
	if said := step(2*g+time.Minute, " Downloaded b"); !strings.Contains(said, "ended stet-rp-01") {
		t.Fatalf("past twice its grace:\n%s", said)
	}
}

// In Season 2 only four machines a round measure its unedited baseline (baselines_wanted):
// once no round open wants one, a machine rented trains once for its first grade, and its lead
// counts one pass, not two. A server that does not say, or no round open, counts two.
func TestOnlyAMachineMeasuringTheBaselineTrainsTwice(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 1)
	c := Offer{Market: "runpod", Card: "NVIDIA GeForce RTX 4090"}
	p, _ := f.pass(c.Card)
	two := f.lead(c)
	none, four := 0, 4
	for _, x := range []struct {
		rounds []WorkRound
		want   time.Duration
	}{
		{nil, two},
		{[]WorkRound{{ID: "s2-o00001", BaselinesWanted: &four}}, two},
		{[]WorkRound{{ID: "s1-o00014"}}, two}, // a Season 1 server: it does not say
		{[]WorkRound{{ID: "s2-o00001", BaselinesWanted: &none}}, two - p},
		{[]WorkRound{{ID: "s2-o00001", BaselinesWanted: &none}, {ID: "s2-o00002", BaselinesWanted: &four}}, two},
	} {
		st.mu.Lock()
		st.work = Work{Rounds: x.rounds}
		st.mu.Unlock()
		if _, err := f.look(context.Background()); err != nil {
			t.Fatalf("%v\n%s", err, out)
		}
		if l := f.lead(c); l != x.want {
			t.Errorf("%d rounds open: lead %s, want %s", len(x.rounds), l, x.want)
		}
	}
}

// fits rejects slow GPUs whose two-pass baseline grade plus margin exceeds the server's lease,
// while accepting faster cards that finish comfortably inside the lease.
func TestFitsDropsCardsTooSlowForLease(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	f.o.MinGPUMemory = 12
	f.lease = 5 * time.Hour

	// With passOn3090=52m, cards slower than ~0.40x a 3090 (e.g. RTX 2000 Ada at 0.35x) exceed
	// a 5h lease on a 2-pass baseline plus margin; cards >=0.40x (even RTX 3060 12 GB!) fit.
	for _, slow := range []Offer{
		{Maker: "NVIDIA", MemoryGB: 16, Model: "NVIDIA RTX 2000 Ada", Card: "RTX 2000 Ada"},
	} {
		if f.fits(slow) {
			t.Errorf("fits(%s) = true under 5h lease, want false", slow.Card)
		}
	}
	for _, fast := range []Offer{
		{Maker: "NVIDIA", MemoryGB: 12, Model: "NVIDIA GeForce RTX 3060", Card: "RTX 3060"},
		{Maker: "NVIDIA", MemoryGB: 16, Model: "NVIDIA GeForce RTX 4060 Ti", Card: "RTX 4060 Ti"},
		{Maker: "NVIDIA", MemoryGB: 16, Model: "NVIDIA GeForce RTX 5060 Ti", Card: "RTX 5060 Ti"},
		{Maker: "NVIDIA", MemoryGB: 24, Model: "NVIDIA L4", Card: "L4"},
		{Maker: "NVIDIA", MemoryGB: 16, Model: "NVIDIA RTX A4000", Card: "RTX A4000"},
		{Maker: "NVIDIA", MemoryGB: 16, Model: "NVIDIA GeForce RTX 4080 SUPER", Card: "RTX 4080 SUPER"},
		{Maker: "NVIDIA", MemoryGB: 24, Model: "NVIDIA GeForce RTX 3090", Card: "RTX 3090"},
		{Maker: "NVIDIA", MemoryGB: 24, Model: "NVIDIA GeForce RTX 4090", Card: "RTX 4090"},
		{Maker: "NVIDIA", MemoryGB: 32, Model: "NVIDIA GeForce RTX 5090", Card: "RTX 5090"},
	} {
		if !f.fits(fast) {
			t.Errorf("fits(%s) = false under 5h lease, want true", fast.Card)
		}
	}
}

// A 16 GB machine rented for grading is marked GradeOnly=true before it ever checks in, and
// calibration samples from an earlier season (s2) are ignored once the active season is s3.
func Test16GBRentalMarkedGradeOnlyAndSeasonGuardIgnoresS2Calibration(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 2)
	f.o.MinGPUMemory = 16
	f.o.MinCuratorGPUMemory = 24
	f.o.Edits = true
	// Seed a Season 2 calibration (15 min 4090 pass) and point the fleet at Season 3.
	f.cal = &Calibration{Season: "s2", Rounds: []string{"s2-o00001"}, Pass: map[string][]float64{"RTX 4090": {15}}}
	f.st.season = "s3"
	gotPass, seen := f.pass("NVIDIA GeForce RTX 4090")
	wantPass := time.Duration(float64(passOn3090) / speed("RTX 4090"))
	if seen || gotPass != wantPass {
		t.Fatalf("s3 pass(4090) = %s (seen=%v), want default %s (seen=false)", gotPass, seen, wantPass)
	}
	// Rent a 16 GB machine for grading (Curators=false in ledger); even with Machine == nil,
	// look() must mark Worker.GradeOnly == true so policy never counts it as a curator runner.
	rp.noRoom["NVIDIA GeForce RTX 4090"] = true
	rp.noRoom["NVIDIA GeForce RTX 3090 Ti"] = true
	rp.extraCommunity = []map[string]any{
		{"id": "NVIDIA GeForce RTX 4080 SUPER", "name": "RTX 4080 SUPER", "manufacturer": "NVIDIA", "memory": 16, "price": map[string]any{"community": 0.25}, "availability": "HIGH"},
	}
	rp.extraPrices = map[string]float64{"NVIDIA GeForce RTX 4080 SUPER": 0.25}
	f.o.Cloud = "COMMUNITY"
	st.work = Work{Rounds: []WorkRound{{ID: "s3-o00001", GradesWanted: 4}}}
	v, err := f.look(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if n, err := f.launch(context.Background(), v, 1, 0, 0, 0, time.Time{}); err != nil || n != 1 {
		t.Fatalf("launch 16GB grader: n=%d err=%v", n, err)
	}
	v2, err := f.look(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range v2.Workers {
		if w.Rental != nil {
			found = true
			if w.Rental.Card != "RTX 4080 SUPER" {
				t.Fatalf("expected 16 GB RTX 4080 SUPER, got %s", w.Rental.Card)
			}
			if w.Machine != nil {
				t.Fatalf("expected Machine == nil before check-in, got %+v", w.Machine)
			}
			if !w.GradeOnly {
				t.Fatal("16 GB grader rental was not marked GradeOnly=true before check-in")
			}
		}
	}
	if !found {
		t.Fatal("rented 16 GB worker missing from view")
	}
}

// When 24 GB curator GPUs are unavailable, unfilled curator slots fall back to 16 GB graders if
// and only if grades or farm tasks are actually waiting.
func TestUnfilled24GBCuratorSlotsFallBackTo16GBGradersOnlyWhenGradesWait(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 4)
	f.o.Cloud = "COMMUNITY"
	f.o.Edits = true
	f.o.MinGPUMemory = 16
	f.o.MinCuratorGPUMemory = 24
	f.o.PerWorker = 2
	// Disable every 24 GB GPU on RunPod community, leaving only a 16 GB RTX 4080 SUPER.
	rp.noRoom["NVIDIA GeForce RTX 3090 Ti"] = true
	rp.noRoom["NVIDIA GeForce RTX 4090"] = true
	rp.noRoom["NVIDIA RTX A5000"] = true
	rp.extraCommunity = []map[string]any{
		{"id": "NVIDIA GeForce RTX 4080 SUPER", "name": "RTX 4080 SUPER", "manufacturer": "NVIDIA", "memory": 16, "price": map[string]any{"community": 0.25}, "availability": "HIGH"},
	}
	rp.extraPrices = map[string]float64{"NVIDIA GeForce RTX 4080 SUPER": 0.25}
	ctx := context.Background()

	// Case 1: Only curator runs waiting (0 grades, 0 farm) -> do NOT waste slots on 16 GB graders.
	st.work = Work{Rounds: []WorkRound{{ID: "s3-o00001", RunsWanted: 4, GradesWanted: 0}}}
	v, err := f.look(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := f.launch(ctx, v, 2, 0, 0, 2, time.Time{}); err != nil || got != 0 {
		t.Fatalf("with 0 grades waiting: rented %d (%v), want 0", got, err)
	}

	// Case 2: Both runs and grades waiting -> unfilled 24 GB curator slots pass to 16 GB graders!
	st.work = Work{Rounds: []WorkRound{{ID: "s3-o00001", RunsWanted: 4, GradesWanted: 6}}}
	v, err = f.look(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := f.launch(ctx, v, 2, 0, 0, 2, time.Time{}); err != nil || got != 2 {
		t.Fatalf("with 6 grades waiting and no 24GB GPUs: rented %d (%v), want 2 16GB graders", got, err)
	}
	// Every grade-only rental must request only graderDisk (35 GB), NOT the 120 GB curator disk,
	// and explicitly disable STET_WORKER_EDITS so worker.sh installs lean STET_GRADE_ONLY=1!
	for _, pod := range rp.created {
		if pod.Disk != graderDisk {
			t.Fatalf("grade-only pod requested %d GB disk, want %d GB", pod.Disk, graderDisk)
		}
		if pod.Env["STET_WORKER_EDITS"] != "0" {
			t.Fatalf("grade-only pod has STET_WORKER_EDITS=%q, want 0", pod.Env["STET_WORKER_EDITS"])
		}
	}
	if !strings.Contains(workerScript, "STET_GRADE_ONLY=1") {
		t.Fatal("worker.sh does not set STET_GRADE_ONLY=1 for grade-only workers")
	}
}

// Review 15 P2 #4: when edits are enabled, an entry is waiting, no round is open, and no runners
// are online, manual `stet-cloud up 1` must launch a curator-capable runner (STET_WORKER_EDITS=1)
// matching Policy.runners so the server announces the on-demand round instead of stalling on a
// grade-only machine.
func TestManualUpLaunchesCuratorRunnerWhenWaitingEntryNeedsDraw(t *testing.T) {
	ctx := context.Background()

	t.Run("up_1_launches_curator_runner_for_waiting_round", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 2)
		f.o.Edits = true
		f.o.MinGPUMemory = 12
		f.o.MinCuratorGPUMemory = 24
		st.work = Work{Rounds: nil, Waiting: true, Runners: 0, Paused: 0}

		if err := cmdUp(ctx, f, 1); err != nil {
			t.Fatalf("cmdUp 1: %v", err)
		}
		if len(rp.created) != 1 {
			t.Fatalf("expected 1 pod created, got %d", len(rp.created))
		}
		pod := rp.created[0]
		if pod.Env["STET_WORKER_EDITS"] != "1" {
			t.Fatalf("manual up 1 for waiting round set STET_WORKER_EDITS=%q, want 1", pod.Env["STET_WORKER_EDITS"])
		}
		if pod.Disk != f.o.Disk {
			t.Fatalf("manual up 1 curator runner requested %d GB disk, want %d GB", pod.Disk, f.o.Disk)
		}
	})

	t.Run("no_edits_keeps_grade_only_on_manual_up", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 2)
		f.o.Edits = false
		st.work = Work{Rounds: nil, Waiting: true, Runners: 0, Paused: 0}

		if err := cmdUp(ctx, f, 1); err != nil {
			t.Fatalf("cmdUp 1 --no-edits: %v", err)
		}
		if len(rp.created) != 1 || rp.created[0].Env["STET_WORKER_EDITS"] != "0" {
			t.Fatalf("expected grade-only worker with --no-edits, got %+v", rp.created)
		}
	})
}

// Review 16 P2 #2: when the examiner fills the last --max slot alongside an existing working
// RunPod grader, `cmdProbe` rents 0 machines and MUST exit immediately without selecting the
// existing grader from the ledger or running forced cleanup on it.
func TestProbeDoesNotTerminateExistingWorkerWhenExaminerFillsLastSlot(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 2)
	f.o.MaxWorkers = 2
	f.o.Market = "runpod"

	// Existing RunPod worker actively grading on stet-rp-01 (slot 1 of 2).
	rp.pods["pod-grading"] = &Pod{
		ID: "pod-grading", Name: "stet-rp-01",
		Status: "RUNNING", Cost: 0.43, CreatedAt: f.now().Add(-10 * time.Minute),
	}
	st.work = Work{
		Rounds: []WorkRound{{ID: "s3-o00042", Entries: 1, ExamAt: 0}},
		Machines: []Machine{{
			Handle: "stet-rp-01", Worker: true,
			Busy: "grade", LastSeen: f.now().Unix(),
		}},
	}
	if err := f.writeLedger(&ledger{Pods: map[string]*podRecord{
		ledgerKey("runpod", "pod-grading"): {
			ID: "pod-grading", Market: "runpod", Handle: "stet-rp-01",
			GPU: "RTX 4090", Cost: 0.43, Start: f.now().Add(-10 * time.Minute),
		},
	}}); err != nil {
		t.Fatalf("writeLedger: %v", err)
	}

	// Active examiner fills slot 2 of 2 (persisted in examiner.json).
	if err := writeExaminerState(f.o.Home, &examinerState{
		InstanceID: "ex-live", RoundID: "s3-o00042", APIKey: "k", CertPEM: "c",
		Price: 0.62, Created: f.now().Add(-2 * time.Minute),
	}); err != nil {
		t.Fatalf("writeExaminerState: %v", err)
	}

	checkFile := filepath.Join(t.TempDir(), "cage_check.py")
	if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
		t.Fatalf("write check file: %v", err)
	}

	err := cmdProbe(context.Background(), f, checkFile, time.Minute)
	if err == nil || !strings.Contains(err.Error(), "nothing was rented") {
		t.Fatalf("expected 'nothing was rented' error when examiner fills last slot, got %v", err)
	}
	if len(rp.created) != 0 {
		t.Fatalf("expected 0 new pods created, got %d", len(rp.created))
	}
	if len(st.released) != 0 {
		t.Fatalf("probe MUST NOT release existing grader's active job! released=%v", st.released)
	}
	if _, ok := rp.pods["pod-grading"]; !ok {
		t.Fatalf("probe MUST NOT terminate existing working grader!")
	}

	// With one open slot (--max 3), probe rents its own pod and cleans up ONLY that probe pod,
	// leaving pod-grading untouched.
	f.o.MaxWorkers = 3
	if err := cmdProbe(context.Background(), f, checkFile, 0); err != nil {
		t.Fatalf("cmdProbe with open slot: %v", err)
	}
	if len(rp.created) != 1 {
		t.Fatalf("expected 1 probe pod rented, got %d", len(rp.created))
	}
	if len(rp.pods) != 1 || rp.pods["pod-grading"] == nil {
		t.Fatalf("expected only the newly rented probe pod ended and pod-grading preserved, got pods=%+v", rp.pods)
	}
}

// Review 17 P2 #4: probe cleanup MUST end ONLY its exact recorded (market, rental ID) under the
// fleet lock and treat an already-gone probe as complete, NEVER deleting a replacement worker
// that reused the same handle or releasing that replacement's active grading job.
func TestReview17ProbeCleanupNeverDeletesReplacementWithReusedHandle(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	f.o.MaxWorkers = 2
	f.o.Market = "runpod"
	now := f.now()

	// Wrap RunPod market so the first Logs() poll during cmdProbe simulates the probe pod vanishing
	// and a replacement grader immediately reusing the same handle `stet-rp-01` and taking a job.
	var swapped bool
	var probeID string
	rpMkt := f.market("runpod")
	f.markets = []Market{&probeSwapMarket{
		Market: rpMkt,
		onLogs: func(id string) ([]string, error) {
			if !swapped {
				swapped = true
				probeID = id
				// Probe pod vanishes from RunPod; replacement pod-replacement takes handle stet-rp-01.
				delete(rp.pods, id)
				rp.pods["pod-replacement"] = &Pod{
					ID: "pod-replacement", Name: "stet-rp-01",
					Status: "RUNNING", Cost: 0.43, CreatedAt: now.Add(5 * time.Second),
				}
				st.work = Work{
					Rounds: []WorkRound{{ID: "s3-o00044", Entries: 1, ExamAt: 0}},
					Machines: []Machine{{
						Handle: "stet-rp-01", Worker: true,
						Busy: "grade", LastSeen: now.Add(10 * time.Second).Unix(),
					}},
				}
				if l, err := f.readLedger(); err == nil {
					l.Pods[ledgerKey("runpod", "pod-replacement")] = &podRecord{
						ID: "pod-replacement", Market: "runpod", Handle: "stet-rp-01",
						GPU: "RTX 4090", Cost: 0.43, Start: now.Add(5 * time.Second), Alive: now.Add(10 * time.Second),
					}
					_ = f.writeLedger(l)
				}
			}
			return []string{"[probe] done: every case held"}, nil
		},
	}}

	checkFile := filepath.Join(t.TempDir(), "cage_check.py")
	if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
		t.Fatalf("write check file: %v", err)
	}
	if err := cmdProbe(context.Background(), f, checkFile, time.Minute); err != nil {
		t.Fatalf("cmdProbe: %v", err)
	}
	if !swapped || probeID == "" {
		t.Fatalf("expected swap hook to run with rented probe ID")
	}
	if rp.pods["pod-replacement"] == nil {
		t.Fatalf("probe cleanup DELETED replacement worker pod-replacement sharing handle stet-rp-01! pods=%+v", rp.pods)
	}
	if len(st.released) != 0 {
		t.Fatalf("probe cleanup MUST NOT release replacement worker's grading job! released=%v", st.released)
	}
	l, err := f.readLedger()
	if err != nil {
		t.Fatalf("readLedger: %v", err)
	}
	if l.Pods[ledgerKey("runpod", "pod-replacement")] == nil {
		t.Fatalf("replacement worker ledger record must remain intact, got %+v", l.Pods)
	}
	if l.Pods[ledgerKey("runpod", probeID)] != nil {
		t.Fatalf("gone probe pod ledger record should have been retired, got %+v", l.Pods)
	}
}

type probeSwapMarket struct {
	Market
	onLogs    func(id string) ([]string, error)
	onRentals func() ([]Rental, error)
	onEnd     func(id string) error
}

func (m *probeSwapMarket) Logs(ctx context.Context, id string, n int) ([]string, error) {
	if m.onLogs != nil {
		return m.onLogs(id)
	}
	if lm, ok := m.Market.(interface {
		Logs(context.Context, string, int) ([]string, error)
	}); ok {
		return lm.Logs(ctx, id, n)
	}
	return nil, nil
}

func (m *probeSwapMarket) Rentals(ctx context.Context) ([]Rental, error) {
	if m.onRentals != nil {
		return m.onRentals()
	}
	return m.Market.Rentals(ctx)
}

func (m *probeSwapMarket) End(ctx context.Context, id string) error {
	if m.onEnd != nil {
		return m.onEnd(id)
	}
	return m.Market.End(ctx, id)
}

// Review 18 P2 #2:
//
//	(a) If the probe pod is temporarily omitted from Rentals() after logs finish, cmdDownRental
//	    MUST still invoke m.End(ctx, pod.ID) directly so the live rental is actually terminated
//	    before its ledger entry is retired.
//	(b) If m.End(ctx, pod.ID) fails, cmdDownRental MUST leave the ledger entry intact and
//	    cmdProbe MUST return a non-nil error instead of reporting success.
func TestReview18ProbeCleanupCallsEndOnListOmissionAndPropagatesError(t *testing.T) {
	t.Run("omitted_from_Rentals_still_calls_End_and_retires_ledger", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 1)
		f.o.MaxWorkers = 2
		f.o.Market = "runpod"

		var probeID string
		var endedIDs []string
		omitProbe := false
		rpMkt := f.market("runpod")
		f.markets = []Market{&probeSwapMarket{
			Market: rpMkt,
			onLogs: func(id string) ([]string, error) {
				probeID = id
				omitProbe = true // simulate stale/eventually-consistent Rentals() omitting the live pod
				return []string{"[probe] done: every case held"}, nil
			},
			onRentals: func() ([]Rental, error) {
				rs, err := rpMkt.Rentals(context.Background())
				if err != nil || !omitProbe {
					return rs, err
				}
				filtered := rs[:0]
				for _, r := range rs {
					if r.ID != probeID {
						filtered = append(filtered, r)
					}
				}
				return filtered, nil
			},
			onEnd: func(id string) error {
				endedIDs = append(endedIDs, id)
				return rpMkt.End(context.Background(), id)
			},
		}}

		checkFile := filepath.Join(t.TempDir(), "cage_check.py")
		if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
			t.Fatalf("write check file: %v", err)
		}
		if err := cmdProbe(context.Background(), f, checkFile, time.Minute); err != nil {
			t.Fatalf("cmdProbe: %v", err)
		}
		if probeID == "" {
			t.Fatalf("expected probe pod ID recorded")
		}
		if len(endedIDs) != 1 || endedIDs[0] != probeID {
			t.Fatalf("expected End(%q) called despite omission from Rentals(), got endedIDs=%v", probeID, endedIDs)
		}
		if rp.pods[probeID] != nil {
			t.Fatalf("live probe pod %s was NOT deleted on RunPod!", probeID)
		}
		l, err := f.readLedger()
		if err != nil {
			t.Fatalf("readLedger: %v", err)
		}
		if l.Pods[ledgerKey("runpod", probeID)] != nil {
			t.Fatalf("expected ledger entry retired after End succeeded, got %+v", l.Pods)
		}
	})

	t.Run("End_error_keeps_ledger_entry_and_returns_error_from_cmdProbe", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 1)
		f.o.MaxWorkers = 2
		f.o.Market = "runpod"

		var probeID string
		rpMkt := f.market("runpod")
		f.markets = []Market{&probeSwapMarket{
			Market: rpMkt,
			onLogs: func(id string) ([]string, error) {
				probeID = id
				return []string{"[probe] done: every case held"}, nil
			},
			onEnd: func(id string) error {
				return fmt.Errorf("RunPod DELETE 503")
			},
		}}

		checkFile := filepath.Join(t.TempDir(), "cage_check.py")
		if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
			t.Fatalf("write check file: %v", err)
		}
		err := cmdProbe(context.Background(), f, checkFile, time.Minute)
		if err == nil || !strings.Contains(err.Error(), "RunPod DELETE 503") {
			t.Fatalf("expected cmdProbe to return wrapped teardown error, got %v", err)
		}
		if rp.pods[probeID] == nil {
			t.Fatalf("expected pod still running on RunPod after End failed")
		}
		l, rerr := f.readLedger()
		if rerr != nil {
			t.Fatalf("readLedger: %v", rerr)
		}
		if l.Pods[ledgerKey("runpod", probeID)] == nil {
			t.Fatalf("ledger MUST retain probe pod %s when End fails so scale/down can clean it up!", probeID)
		}
	})
}

// Review 19 P1 #2:
//
//	(a) If the provider creates the probe pod and the immediate fleet.json ledger write fails,
//	    launchSomeRentals MUST still return the live pod ID and cmdProbe MUST immediately tear it
//	    down on the provider so zero unpaid-for/leaked probe pods keep running `sleep infinity`.
//	(b) If fleet.json becomes corrupt/unparsable after the probe completes, cmdDownRental MUST
//	    still issue m.End(ctx, pod.ID) directly before returning the ledger repair error.
func TestReview19ProbeTeardownSurvivesLedgerWriteAndReadFailures(t *testing.T) {
	t.Run("launch_ledger_write_failure_still_tears_down_probe_pod", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 1)
		f.o.MaxWorkers = 2
		f.o.Market = "runpod"

		var rentedID string
		var endedIDs []string
		rpMkt := f.market("runpod")
		f.markets = []Market{&probeRentHookMarket{
			Market: rpMkt,
			onRent: func(ctx context.Context, o Offer, l Launch) (*Rental, error) {
				r, err := rpMkt.Rent(ctx, o, l)
				if err == nil && r != nil {
					rentedID = r.ID
					// Make home read-only right before writeLedger runs (after Rent succeeded!).
					if cerr := os.Chmod(f.o.Home, 0o500); cerr != nil {
						t.Errorf("chmod home 0500: %v", cerr)
					}
				}
				return r, err
			},
			onEnd: func(ctx context.Context, id string) error {
				endedIDs = append(endedIDs, id)
				return rpMkt.End(ctx, id)
			},
		}}
		t.Cleanup(func() { _ = os.Chmod(f.o.Home, 0o700) })

		checkFile := filepath.Join(t.TempDir(), "cage_check.py")
		if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
			t.Fatalf("write check file: %v", err)
		}
		err := cmdProbe(context.Background(), f, checkFile, time.Minute)
		_ = os.Chmod(f.o.Home, 0o700)
		if err == nil {
			t.Fatalf("expected non-nil error when launch writeLedger fails")
		}
		if rentedID == "" {
			t.Fatalf("expected provider to have created probe pod")
		}
		if len(endedIDs) != 1 || endedIDs[0] != rentedID {
			t.Fatalf("probe MUST tear down created pod %s even when launch writeLedger fails! ended=%v", rentedID, endedIDs)
		}
		if len(rp.pods) != 0 {
			t.Fatalf("leaked live probe pod after launch writeLedger failure! pods=%+v", rp.pods)
		}
	})

	t.Run("corrupt_ledger_at_probe_completion_still_calls_End_before_reporting_error", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 1)
		f.o.MaxWorkers = 2
		f.o.Market = "runpod"

		var probeID string
		var endedIDs []string
		rpMkt := f.market("runpod")
		f.markets = []Market{&probeSwapMarket{
			Market: rpMkt,
			onLogs: func(id string) ([]string, error) {
				probeID = id
				// Corrupt fleet.json before probe teardown runs.
				if err := os.WriteFile(filepath.Join(f.o.Home, "fleet.json"), []byte("{corrupt"), 0o600); err != nil {
					t.Errorf("corrupt fleet.json: %v", err)
				}
				return []string{"[probe] done: every case held"}, nil
			},
			onEnd: func(id string) error {
				endedIDs = append(endedIDs, id)
				return rpMkt.End(context.Background(), id)
			},
		}}

		checkFile := filepath.Join(t.TempDir(), "cage_check.py")
		if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
			t.Fatalf("write check file: %v", err)
		}
		err := cmdProbe(context.Background(), f, checkFile, time.Minute)
		if err == nil || !strings.Contains(err.Error(), "fleet.json") {
			t.Fatalf("expected ledger repair error after teardown, got %v", err)
		}
		if probeID == "" || len(endedIDs) != 1 || endedIDs[0] != probeID {
			t.Fatalf("cmdDownRental MUST call m.End(%q) even when fleet.json is unparsable! ended=%v", probeID, endedIDs)
		}
		if len(rp.pods) != 0 {
			t.Fatalf("live probe pod was NOT terminated when fleet.json was corrupt! pods=%+v", rp.pods)
		}
	})

	t.Run("unwritable_lock_file_still_calls_End_before_reporting_error", func(t *testing.T) {
		rp, st := newFakeRunPod(t), newFakeStet(t)
		f, _ := testFleet(t, rp, st, 1)
		f.o.MaxWorkers = 2
		f.o.Market = "runpod"

		var probeID string
		var endedIDs []string
		rpMkt := f.market("runpod")
		f.markets = []Market{&probeSwapMarket{
			Market: rpMkt,
			onLogs: func(id string) ([]string, error) {
				probeID = id
				// Make <home>/lock unwritable (0400) so f.lock() fails during teardown (simulating EROFS / EACCES).
				lockPath := filepath.Join(f.o.Home, "lock")
				_ = os.Chmod(lockPath, 0o400)
				return []string{"[probe] done: every case held"}, nil
			},
			onEnd: func(id string) error {
				endedIDs = append(endedIDs, id)
				return rpMkt.End(context.Background(), id)
			},
		}}
		t.Cleanup(func() { _ = os.Chmod(filepath.Join(f.o.Home, "lock"), 0o600) })

		checkFile := filepath.Join(t.TempDir(), "cage_check.py")
		if err := os.WriteFile(checkFile, []byte("print('ok')\n"), 0o600); err != nil {
			t.Fatalf("write check file: %v", err)
		}
		err := cmdProbe(context.Background(), f, checkFile, time.Minute)
		_ = os.Chmod(filepath.Join(f.o.Home, "lock"), 0o600)
		// If running as non-root, lock failed and cmdProbe returns lock cleanup error AFTER calling End.
		if probeID == "" || len(endedIDs) != 1 || endedIDs[0] != probeID {
			t.Fatalf("cmdDownRental MUST call m.End(%q) even when f.lock() fails! ended=%v err=%v", probeID, endedIDs, err)
		}
		if len(rp.pods) != 0 {
			t.Fatalf("live probe pod was NOT terminated when f.lock() failed! pods=%+v", rp.pods)
		}
		if os.Geteuid() != 0 && (err == nil || !strings.Contains(err.Error(), "fleet lock")) {
			t.Fatalf("expected fleet lock cleanup error reported after provider teardown, got %v", err)
		}
	})
}

type probeRentHookMarket struct {
	Market
	onRent func(ctx context.Context, o Offer, l Launch) (*Rental, error)
	onEnd  func(ctx context.Context, id string) error
}

func (m *probeRentHookMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	return m.onRent(ctx, o, l)
}

func (m *probeRentHookMarket) End(ctx context.Context, id string) error {
	return m.onEnd(ctx, id)
}
