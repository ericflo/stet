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
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
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
		if r.URL.Query().Get("minCudaVersion") != "13.0" {
			w.WriteHeader(422)
			return
		}
		if r.URL.Query().Get("cloud") == "COMMUNITY" {
			json.NewEncoder(w).Encode(map[string]any{"gpus": []map[string]any{
				{"id": "NVIDIA GeForce RTX 3080 Ti", "name": "RTX 3080 Ti", "manufacturer": "NVIDIA", "memory": 12, "price": map[string]any{"community": 0.18}, "availability": "LOW"},
				{"id": "NVIDIA RTX A5000", "name": "RTX A5000", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"community": 0.16}, "availability": "NONE"},
				{"id": "NVIDIA GeForce RTX 3090 Ti", "name": "RTX 3090 Ti", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"community": 0.27}, "availability": "LOW"},
				{"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "manufacturer": "NVIDIA", "memory": 24, "price": map[string]any{"community": 0.45}, "availability": "LOW"},
			}})
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
		price := map[string]float64{"NVIDIA RTX A5000": 0.27, "NVIDIA GeForce RTX 3080": 0.2, "NVIDIA RTX PRO 4000 Blackwell": 0.57,
			"NVIDIA GeForce RTX 3080 Ti": 0.18, "NVIDIA GeForce RTX 3090 Ti": 0.27}[spec.GPU.ID]
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
			io.WriteString(w, `{"site": "https://www.stet.test"}`)
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
						if m.Busy == "shadow" {
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
	f := &Fleet{o: o, rp: newRunPod(rp.srv.URL, "rp-key"), st: newStet(st.srv.URL), op: op, out: &out, now: time.Now}
	f.rp.sleep = func(context.Context, time.Duration) error { return nil }
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

// --cloud any rents the cheapest machine in either cloud, every time, and falls back to the
// next cheapest when RunPod has none of a kind free.
func TestTheCheapestMachineInEitherCloudIsRented(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	f.o.Cloud = "ANY"
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
	// With the 3080 Tis gone, the next cheapest: the secure A5000 and the community 3090 Ti
	// tie at $0.27, and secure goes first.
	rp.noRoom["NVIDIA GeForce RTX 3080 Ti"] = true
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if last := rp.created[len(rp.created)-1]; last.Cloud != "SECURE" || last.GPU.ID != "NVIDIA RTX A5000" {
		t.Fatalf("after the cheapest ran out: %+v", last)
	}
	if spec := rp.created[0]; spec.GPU.MinRAMPerGPU != f.o.MinRAM || spec.GPU.MinVCPUPerGPU != f.o.MinCPU {
		t.Fatalf("the host was asked for as %+v", spec.GPU)
	}
	if _, ok := rp.created[0].Env["STET_SHADOW_MAX_MINUTES"]; ok {
		t.Fatal("a machine not rented for bigger students was told to take them")
	}
	f.o.LongMinutes = 600
	if spec := f.spec(&Key{Handle: "stet-rp-09"}, "NVIDIA RTX A5000", "SECURE", "https://www.stet.test"); spec.Env["STET_SHADOW_MAX_MINUTES"] != "600" {
		t.Fatalf("a machine rented for bigger students: %v", spec.Env["STET_SHADOW_MAX_MINUTES"])
	}
	f.o.LongMinutes = 0
	gpus, err := f.choices(context.Background(), false)
	if err != nil || len(gpus) < 3 || gpus[0].Price != 0.18 || gpus[0].Cloud != "COMMUNITY" {
		t.Fatalf("choices: %+v %v", gpus, err)
	}
	for i := 1; i < len(gpus); i++ {
		if gpus[i].Price < gpus[i-1].Price {
			t.Fatalf("not cheapest first: %+v", gpus)
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
	if gpus, _ := f.choices(context.Background(), false); len(gpus) == 0 || gpus[len(gpus)-1].Price > 0.20 {
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
	f.rp = newRunPod(logs.URL, "rp-key")
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
// nothing waits out its lease. A rented machine at work keeps its job.
func TestWhatAnEndedMachineHeldIsHandedBack(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	ctx := context.Background()
	st.work = *round(30, 1, 0, 0)
	if err := tick(ctx, f); err != nil || len(rp.created) != 2 {
		t.Fatalf("rented %d machines (%v)", len(rp.created), err)
	}

	// The round is done: stet-rp-01 grades and is kept, stet-rp-02 waits and is ended, and
	// stet-rp-03, which no pod is rented for, still holds one of the next exam's students.
	st.work = *round(0, 0, 0, 1)
	now := time.Now().Unix()
	st.work.Machines = []Machine{
		{Handle: "stet-rp-01", Worker: true, LastSeen: now, Busy: "grade"},
		{Handle: "stet-rp-02", Worker: true, LastSeen: now},
		{Handle: "stet-rp-03", Worker: true, LastSeen: now - 3600, Busy: "shadow"},
	}
	if err := tick(ctx, f); err != nil {
		t.Fatal(err)
	}
	if len(rp.pods) != 1 || strings.Join(st.released, " ") != "stet-rp-03" {
		t.Fatalf("%d pods left, handed back %v:\n%s", len(rp.pods), st.released, out)
	}
	if !strings.Contains(out.String(), "handed back what stet-rp-03 held when it was ended: 1 of the next exam's students") {
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
	if strings.Join(st.released, " ") != "stet-rp-03 stet-rp-02" || !strings.Contains(out.String(), "handed back what stet-rp-02 held when it was ended: 1 job") {
		t.Fatalf("handed back %v:\n%s", st.released, out)
	}
	// Nothing is held any more, so nothing more is handed back.
	if err := tick(ctx, f); err != nil || len(st.released) != 2 {
		t.Fatalf("handed back %v (%v)", st.released, err)
	}
}

// A machine rented for the next exam's bigger students is chosen for what a student costs on
// it, under --long-max-price: a quick card can train one for less than a cheap one's hours
// cost. The machines for everything else are still the cheapest under --max-price.
func TestTheBiggerStudentsGetTheCardThatTrainsOneForLeast(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 4)
	f.o.Cloud, f.o.MaxPrice, f.o.LongMaxPrice = "ANY", 0.22, 0.90
	f.o.LongMinutes, f.o.LongShadow, f.o.PerShadow = 600, true, 4
	rp.noRoom["NVIDIA GeForce RTX 3080 Ti"] = true // a student costs least on it, and it has none free
	st.work = Work{ShadowWaiting: 1, ShadowLongWaiting: 2}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	got := []string{}
	for _, spec := range rp.created {
		got = append(got, strings.ToLower(spec.Cloud)+" "+shortGPU(spec.GPU.ID))
		if spec.Env["STET_SHADOW_MAX_MINUTES"] != "600" {
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
	gpus, err := f.choices(context.Background(), true)
	var kinds []string
	for _, c := range gpus {
		kinds = append(kinds, shortGPU(c.GPU.ID))
	}
	if err != nil || strings.Join(kinds, ", ") != "RTX 3080 Ti, RTX 3090 Ti, RTX 3080, RTX A5000" {
		t.Fatalf("for the bigger students under $0.40: %v %v", kinds, err)
	}
	// And with no ceiling of their own, --max-price's.
	f.o.LongMaxPrice = 0
	if gpus, _ := f.choices(context.Background(), true); len(gpus) == 0 || gpus[0].Price > 0.22 {
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
	st.work = Work{ShadowLongWaiting: 7}

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
	if _, err := parse(fs, append([]string{"--follow", "--max", "5", "--long-shadow", "600", "--gpu", "RTX 4090,RTX A5000"}, common...)); err != nil {
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
