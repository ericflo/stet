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
	workers  map[string]bool // public keys the server takes as rented machines
	signed   []string        // who signed each request
	vouches  int             // grants it was sent, each with a good signature
	refusing bool            // the key vouching is not one of the organisers' own
	taken    map[string]bool // handles somebody else has
	srv      *httptest.Server
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
	gpus, err := f.choices(context.Background())
	if err != nil || len(gpus) < 3 || gpus[0].Price != 0.18 || gpus[0].Cloud != "COMMUNITY" {
		t.Fatalf("choices: %+v %v", gpus, err)
	}
	for i := 1; i < len(gpus); i++ {
		if gpus[i].Price < gpus[i-1].Price {
			t.Fatalf("not cheapest first: %+v", gpus)
		}
	}
}
