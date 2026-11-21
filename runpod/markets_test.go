package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The markets' answers in testdata/ are their own, recorded on Nov 21, 2026 (offers only, cut
// to the fields read and to a few machines each, among them some that each filter must leave
// out). The account's machines are made up: none were rented when they were recorded.

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// fakeMarket serves a market's API: each path's answer, given that the key is sent as the
// market sends it.
type fakeMarket struct {
	header, value string
	answers       map[string]string // "GET /path" → body; a body starting "404 " is that status
	asked         []string
	bodies        []string
	srv           *httptest.Server
}

func newFakeMarket(t *testing.T, header, value string) *fakeMarket {
	m := &fakeMarket{header: header, value: value, answers: map[string]string{}}
	m.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := r.Method + " " + r.URL.Path
		m.asked = append(m.asked, call)
		body, _ := io.ReadAll(r.Body)
		m.bodies = append(m.bodies, string(body))
		if r.Header.Get(m.header) != m.value {
			w.WriteHeader(401)
			io.WriteString(w, `{"detail": "no such key"}`)
			return
		}
		a, ok := m.answers[call]
		if !ok {
			a, ok = m.answers[r.Method+" *"] // any other path
		}
		if !ok {
			w.WriteHeader(404)
			return
		}
		var status int
		if n, err := fmt.Sscanf(a, "%d ", &status); n == 1 && err == nil && status >= 300 {
			w.WriteHeader(status)
			a = a[4:]
		}
		io.WriteString(w, a)
	}))
	t.Cleanup(m.srv.Close)
	return m
}

var testNeed = Need{MinGPUMemGB: 12, MinCPUs: 6, MinRAMGB: 16, DiskGB: 100, MinCUDA: "13.0"}

func byID(os []Offer) map[string]Offer {
	m := map[string]Offer{}
	for _, o := range os {
		m[o.ID] = o
	}
	return m
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestVastsOffersArePricedAllIn(t *testing.T) {
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["POST /api/v0/bundles/"] = string(fixture(t, "vast_bundles.json"))
	m := newVast(fm.srv.URL, "vast-key")
	offers, err := m.Offers(context.Background(), testNeed)
	if err != nil {
		t.Fatal(err)
	}
	if len(offers) != 7 {
		t.Fatalf("%d offers: %+v", len(offers), offers)
	}
	// The search asks Vast for the need: one NVIDIA GPU of 12 GB or more on a verified host,
	// CUDA 13.0, 6 CPUs, 16 GB, 100 GB of disk.
	var q map[string]any
	if err := json.Unmarshal([]byte(fm.bodies[0]), &q); err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]string{"gpu_ram": `{"gte":12000}`, "cuda_max_good": `{"gte":13}`, "cpu_cores_effective": `{"gte":6}`,
		"cpu_ram": `{"gte":16000}`, "disk_space": `{"gte":100}`, "verified": `{"eq":true}`, "num_gpus": `{"eq":1}`, "type": `"ondemand"`} {
		if got, _ := json.Marshal(q[k]); string(got) != want {
			t.Errorf("the search's %s: %s, want %s", k, got, want)
		}
	}
	o := byID(offers)
	// $0.0533 for the GPU, 100 GB at $0.40 a GB a month for an hour, and 3 GB down.
	if c := o["50121277"]; c.Card != "RTX 3060" || c.MemoryGB != 12 || !near(c.Price, 0.05333333333333334+100*0.39999999999999997/720+3*0.0013020833333333333) || c.Trust != "verified" || c.CUDA != "13" {
		t.Errorf("the cheapest 3060: %+v", c)
	}
	if c := o["44053844"]; c.Trust != "datacenter" || c.Card != "RTX 5060 Ti" || c.MemoryGB != 16 {
		t.Errorf("one in a data centre: %+v", c)
	}
	if c := o["48398491"]; c.Card != "RTX 4080 SUPER" || speed(c.Card) != 1.35 {
		t.Errorf("Vast's name for a 4080 SUPER: %+v (speed %v)", c, speed(c.Card))
	}
	// Vast offers a 2060 with 12 GB; the fleet leaves it (before Ampere).
	f := &Fleet{o: &Options{MinGPUMemory: 12}}
	if c := o["52063421"]; c.Card != "RTX 2060" || f.fits(c) {
		t.Errorf("a 2060 would be rented: %+v", c)
	}
}

func TestVastsMachinesAndAccount(t *testing.T) {
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["GET /api/v1/instances/"] = `{"success": true, "instances": [
		{"id": 101, "label": "stet-rp-01", "actual_status": "running", "intended_status": "running", "cur_state": "running", "gpu_name": "RTX 4090", "dph_total": 0.41, "start_date": 1795240000.5},
		{"id": 102, "label": "stet-rp-02", "actual_status": null, "intended_status": "running", "cur_state": "running", "gpu_name": "RTX 3090", "dph_total": 0.2, "start_date": 1795240100},
		{"id": 103, "label": null, "actual_status": "exited", "intended_status": "running", "cur_state": "running", "gpu_name": "RTX 3090", "dph_total": 0.2, "start_date": 1795240200}],
		"next_token": null}`
	fm.answers["GET /api/v0/users/current/"] = `{"credit": 39.66, "balance": 0, "can_pay": true}`
	m := newVast(fm.srv.URL, "vast-key")
	rs, err := m.Rentals(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := ""
	for _, r := range rs {
		got += fmt.Sprintf("%s %s %s %s %.2f %d; ", r.ID, r.Name, r.Card, r.State, r.Price, r.Created.Unix())
	}
	if want := "101 stet-rp-01 RTX 4090 running 0.41 1795240000; 102 stet-rp-02 RTX 3090 starting 0.20 1795240100; 103  RTX 3090 stopped 0.20 1795240200; "; got != want {
		t.Fatalf("rentals: %s", got)
	}
	a, err := m.Check(context.Background())
	if err != nil || !a.Known || a.Balance != 39.66 {
		t.Fatalf("account: %+v %v", a, err)
	}
	if _, err := newVast(fm.srv.URL, "wrong").Check(context.Background()); err == nil || !strings.Contains(err.Error(), "Vast.ai said 401") {
		t.Fatalf("a wrong key: %v", err)
	}
}

// Vast rents an offer with the worker as the container's only process, its settings in its
// environment, and the key's handle as the label; an offer gone is no room, and the next is
// tried; an instance is destroyed, not stopped (a stopped one's disk is billed).
func TestVastRentsAndEnds(t *testing.T) {
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["PUT /api/v0/asks/50121277/"] = `{"success": true, "new_contract": 9001}`
	fm.answers["PUT /api/v0/asks/61536954/"] = `410 {"success": false, "error": "no_such_ask", "msg": "error 410/3907: no_such_ask Instance type 61536954 is no longer available."}`
	fm.answers["DELETE /api/v0/instances/9001/"] = `{"success": true}`
	m := newVast(fm.srv.URL, "vast-key")
	l := Launch{Name: "stet-rp-07", Image: "ubuntu:24.04", Script: "echo hi", Env: map[string]string{"STET_API": "https://api.stet.test"}, Need: testNeed}
	r, err := m.Rent(context.Background(), Offer{Market: "vast", ID: "50121277", Card: "RTX 3060", Price: 0.11}, l)
	if err != nil || r.ID != "9001" || r.Name != "stet-rp-07" || r.State != "starting" || r.Price != 0.11 {
		t.Fatalf("rented %+v (%v)", r, err)
	}
	var body struct {
		Image, Label, Runtype string
		Disk                  int
		Env                   map[string]string
		Args                  []string
		CancelUnavail         bool `json:"cancel_unavail"`
	}
	if err := json.Unmarshal([]byte(fm.bodies[0]), &body); err != nil {
		t.Fatal(err)
	}
	if body.Image != "ubuntu:24.04" || body.Label != "stet-rp-07" || body.Runtype != "args" || body.Disk != 100 || !body.CancelUnavail ||
		strings.Join(body.Args, "|") != "bash|-c|echo hi" || body.Env["STET_MARKET"] != "vast" || body.Env["STET_API"] != "https://api.stet.test" {
		t.Fatalf("asked for %s", fm.bodies[0])
	}
	if _, err := m.Rent(context.Background(), Offer{Market: "vast", ID: "61536954"}, l); !errors.Is(err, errNoRoom) {
		t.Fatalf("an offer gone: %v", err)
	}
	if err := m.End(context.Background(), "9001"); err != nil {
		t.Fatal(err)
	}
	if err := m.End(context.Background(), "9002"); err != nil {
		t.Fatalf("one already gone: %v", err)
	}
	if last := fm.asked[len(fm.asked)-2]; last != "DELETE /api/v0/instances/9001/" {
		t.Fatalf("ended with %s", last)
	}
}

func TestLiumsOffersAreItsSecureNodesThatFit(t *testing.T) {
	fm := newFakeMarket(t, "X-API-Key", "lium-key")
	fm.answers["GET /executors"] = string(fixture(t, "lium_executors.json"))
	m := newLium(fm.srv.URL, "lium-key")
	offers, err := m.Offers(context.Background(), testNeed)
	if err != nil {
		t.Fatal(err)
	}
	// Of six nodes: a spot node, one with five CPUs, and one that rents seven GPUs at a time
	// are left out; an eight-GPU node with four free shares its CPUs and memory out.
	got := ""
	for _, o := range offers {
		got += fmt.Sprintf("%s %d GB $%.2f %d free CUDA %s %s; ", o.Card, o.MemoryGB, o.Price, o.Stock, o.CUDA, o.Trust)
	}
	if want := "RTX 5060 Ti 16 GB $0.12 1 free CUDA 13.1 secure; RTX A4500 20 GB $0.17 1 free CUDA 13 secure; RTX A6000 48 GB $0.42 4 free CUDA 13.1 secure; "; got != want {
		t.Fatalf("offers: %s", got)
	}
	// A driver older than 580 is left out, by the newest CUDA it takes.
	e := liumExecutor{Tier: "secure", AvailableGPUCount: 1, GPUCount: 1, PricePerGPU: 0.1, MachineName: "NVIDIA GeForce RTX 3090"}
	e.Specs.CPU.Count, e.Specs.RAM.Total, e.Specs.HardDisk.Free, e.Specs.GPU.Driver = 8, 32e6, 500e6, "570.86.10"
	if _, ok := liumOffer(e, testNeed); ok {
		t.Fatal("a node with driver 570 is offered")
	}
	e.Specs.GPU.Driver = "580.95.05"
	if o, ok := liumOffer(e, testNeed); !ok || o.CUDA != "13.0" {
		t.Fatalf("a node with driver 580: %+v %v", o, ok)
	}
}

func TestLiumsMachinesAndAccount(t *testing.T) {
	fm := newFakeMarket(t, "X-API-Key", "lium-key")
	fm.answers["GET /pods"] = `[
		{"id": "6b1f0d4e-0000-4000-8000-000000000001", "pod_name": "stet-rp-03", "gpu_name": "NVIDIA GeForce RTX 3090", "status": "RUNNING", "price": 0.16, "created_at": "2026-11-21T06:00:00.126695"},
		{"id": "6b1f0d4e-0000-4000-8000-000000000002", "pod_name": "stet-rp-04", "gpu_name": "NVIDIA RTX A4500", "status": "PENDING", "price": null, "created_at": "2026-11-21T06:05:00Z"},
		{"id": "6b1f0d4e-0000-4000-8000-000000000003", "pod_name": "stet-rp-05", "gpu_name": "NVIDIA RTX A4500", "status": "CREATION_FAILED", "price": 0.17, "created_at": "2026-11-21T06:06:00Z"}]`
	fm.answers["GET /users/me"] = `{"balance": 100.0, "estimated_balance": 100.0, "email_verified": true}`
	m := newLium(fm.srv.URL, "lium-key")
	rs, err := m.Rentals(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := ""
	for _, r := range rs {
		got += fmt.Sprintf("%s %s %s %.2f; ", r.Name, r.Card, r.State, r.Price)
	}
	if want := "stet-rp-03 RTX 3090 running 0.16; stet-rp-04 RTX A4500 starting 0.00; stet-rp-05 RTX A4500 stopped 0.17; "; got != want || !rs[0].Created.Equal(time.Date(2026, 11, 21, 6, 0, 0, 126695000, time.UTC)) {
		t.Fatalf("rentals: %s", got)
	}
	if a, err := m.Check(context.Background()); err != nil || !a.Known || a.Balance != 100 {
		t.Fatalf("account: %+v %v", a, err)
	}
}

func TestSaladsOffersAreClassesWithMachinesFree(t *testing.T) {
	fm := newFakeMarket(t, "Salad-Api-Key", "salad-key")
	fm.answers["GET /organizations/stet/gpu-classes"] = string(fixture(t, "salad_gpu_classes.json"))
	fm.answers["POST /organizations/stet/availability/sce-gpu-availability"] = `{"available_gpu_batch": 2, "available_gpu_high": 337, "available_gpu_low": 2, "available_gpu_medium": 67, "on_call_gpu": 0}`
	m := newSalad(fm.srv.URL, "salad-key", "stet", "fleet")
	offers, err := m.Offers(context.Background(), testNeed)
	if err != nil {
		t.Fatal(err)
	}
	// Of seven classes: an 8 GB 3060, a 4 GB GTX, a laptop's 5090 and an AMD card are left
	// out without asking how many are free.
	got := ""
	for _, o := range offers {
		got += fmt.Sprintf("%s %d GB $%.3f %d free %s; ", o.Card, o.MemoryGB, o.Price, o.Stock, o.Trust)
	}
	if want := "RTX 4090 24 GB $0.440 337 free community; RTX 3080 Ti 12 GB $0.195 337 free community; RTX 3060 12 GB $0.080 337 free community; "; got != want {
		t.Fatalf("offers: %s", got)
	}
	asked := 0
	for i, call := range fm.asked {
		if strings.Contains(call, "availability") {
			asked++
			var b map[string]any
			json.Unmarshal([]byte(fm.bodies[i]), &b)
			if b["cpu"] != 6.0 || b["memory"] != 16384.0 || b["storage_amount"] != float64(100<<30) {
				t.Fatalf("availability asked for %v", b)
			}
		}
	}
	if asked != 3 {
		t.Fatalf("availability asked %d times, not once for each class that fits", asked)
	}
	// Asked again within ten minutes, Salad is not asked again.
	if _, err := m.Offers(context.Background(), testNeed); err != nil || len(fm.asked) != 4 {
		t.Fatalf("asked again: %v, %d calls", err, len(fm.asked))
	}
}

func TestSaladsMachinesAndAccount(t *testing.T) {
	fm := newFakeMarket(t, "Salad-Api-Key", "salad-key")
	fm.answers["GET /organizations/stet/gpu-classes"] = string(fixture(t, "salad_gpu_classes.json"))
	fm.answers["GET /organizations/stet/projects/fleet/containers"] = `{"items": [
		{"id": "a", "name": "stet-rp-06-x7k2q", "display_name": "stet-rp-06", "priority": "high", "create_time": "2026-11-21T06:00:00Z",
		 "current_state": {"status": "running"}, "container": {"resources": {"gpu_classes": ["ed563892-aacd-40f5-80b7-90c9be6c759b"]}}},
		{"id": "b", "name": "stet-rp-07-p0m3d", "display_name": "stet-rp-07", "priority": "high", "create_time": "2026-11-21T06:01:00Z",
		 "current_state": {"status": "deploying"}, "container": {"resources": {"gpu_classes": ["f51baccc-dc95-40fb-a5d1-6d0ee0db31d2"]}}}]}`
	fm.answers["GET /organizations/stet/quotas"] = `{"container_groups_quotas": {"container_replicas_quota": 50, "container_replicas_used": 2}}`
	m := newSalad(fm.srv.URL, "salad-key", "stet", "fleet")
	rs, err := m.Rentals(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := ""
	for _, r := range rs {
		got += fmt.Sprintf("%s %s %s %s %.2f; ", r.ID, r.Name, r.Card, r.State, r.Price)
	}
	if want := "stet-rp-06-x7k2q stet-rp-06 RTX 4090 running 0.44; stet-rp-07-p0m3d stet-rp-07 RTX 3060 starting 0.08; "; got != want {
		t.Fatalf("rentals: %s", got)
	}
	if a, err := m.Check(context.Background()); err != nil || a.Known || a.Note != "2 of 50 replicas in use" {
		t.Fatalf("account: %+v %v", a, err)
	}
	// A project that is not there: listing its container groups is refused.
	fm.answers["GET /organizations/stet/projects/fleet/containers"] = `404 {"type": "project_not_found_exception", "title": "Not Found", "status": 404}`
	if _, err := m.Check(context.Background()); err == nil || !strings.Contains(err.Error(), `has no project "fleet"`) {
		t.Fatalf("no project: %v", err)
	}
}

func TestClorePricesADayAsAnHourAllIn(t *testing.T) {
	fm := newFakeMarket(t, "auth", "clore-key")
	fm.answers["GET /marketplace"] = string(fixture(t, "clore_marketplace.json"))
	m := newClore(fm.srv.URL, "clore-key")
	offers, err := m.Offers(context.Background(), testNeed)
	if err != nil {
		t.Fatal(err)
	}
	// Of six servers: one rented, one with two GPUs, and one whose CUDA is 12.4 are left out.
	got := ""
	for _, o := range offers {
		got += fmt.Sprintf("%s %s %d GB $%.4f %s; ", o.ID, o.Card, o.MemoryGB, o.Price, o.Region)
	}
	want := fmt.Sprintf("124851 RTX 4090 24 GB $%.4f US; 124838 RTX 3090 24 GB $%.4f AR; 124832 B300 SXM6 PC %d GB $%.4f US; ",
		7.95/24*1.05+0.10, 4.0/24*1.05+0.10, 192, 50.0/24*1.05+0.10)
	if got != want {
		t.Fatalf("offers: %s\nwant   %s", got, want)
	}
}

func TestClorePaysNoHeedToHTTPButToItsCode(t *testing.T) {
	fm := newFakeMarket(t, "auth", "clore-key")
	fm.answers["GET /my_orders"] = `{"orders": [
		{"id": 501, "price": 4.0, "currency": "USD-Blockchain", "ct": 1795240000, "si": 124838, "expired": false, "spot": false, "specs": {"gpu": "1x NVIDIA GeForce RTX 3090"}},
		{"id": 499, "price": 7.95, "currency": "USD-Blockchain", "ct": 1795230000, "si": 124851, "expired": true, "spot": false, "specs": {"gpu": "1x NVIDIA GeForce RTX 4090"}}], "limit": 999999, "code": 0}`
	fm.answers["GET /wallets"] = `{"wallets": [{"name": "bitcoin", "balance": 0}, {"name": "CLORE-Blockchain", "balance": 0}, {"name": "USD-Blockchain", "balance": 100}], "code": 0}`
	m := newClore(fm.srv.URL, "clore-key")
	rs, err := m.Rentals(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 2 || rs[0].ID != "501" || rs[0].Name != "" || rs[0].Card != "RTX 3090" || rs[0].State != "running" || !near(rs[0].Price, 4.0/24*1.05) || rs[1].State != "gone" {
		t.Fatalf("rentals: %+v", rs)
	}
	if a, err := m.Check(context.Background()); err != nil || !a.Known || a.Balance != 100 {
		t.Fatalf("account: %+v %v", a, err)
	}
	// Clore says a key is wrong with HTTP 200 and code 3.
	fm.header, fm.value = "X-Anything", ""
	fm.answers["GET /wallets"] = `{"code": 3}`
	if _, err := m.Check(context.Background()); err == nil || !strings.Contains(err.Error(), "code 3, the key is not valid") {
		t.Fatalf("a wrong key: %v", err)
	}
}

func TestCredentialsAreReadAtEveryLook(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "credentials.env")
	write := func(s string, mode os.FileMode) {
		if err := os.WriteFile(path, []byte(s), mode); err != nil {
			t.Fatal(err)
		}
		os.Chmod(path, mode)
	}
	write("# a comment\nRUNPOD_API_KEY=rp\nVAST_API_KEY=\nLIUM_API_KEY=\"li\"\nexport CLORE_API_KEY=cl\n", 0o600)
	o := &Options{Home: dir, Credentials: path, Cloud: "ANY", RunPodAPI: "http://x", VastAPI: "http://x", LiumAPI: "http://x", SaladAPI: "http://x", CloreAPI: "http://x"}
	f := &Fleet{o: o, out: io.Discard, now: time.Now}
	names := func() string {
		var s []string
		for _, m := range f.markets {
			s = append(s, m.Name())
		}
		return strings.Join(s, ",")
	}
	if err := f.refresh(); err != nil || names() != "runpod,lium,clore" {
		t.Fatalf("markets with keys: %s %v", names(), err)
	}
	if !strings.Contains(f.off["vast"], "VAST_API_KEY") || !strings.Contains(f.off["salad"], "SALAD_API_KEY") {
		t.Fatalf("why the others are off: %v", f.off)
	}
	lium := f.market("lium")
	// A key added later counts at the next look, and a market whose key is as it was keeps
	// its client.
	write("RUNPOD_API_KEY=rp\nVAST_API_KEY=va\nLIUM_API_KEY=li\nCLORE_API_KEY=cl\nSALAD_API_KEY=sa\nSALAD_ORG=stet\nSALAD_PROJECT=fleet\n", 0o600)
	if err := f.refresh(); err != nil || names() != "runpod,vast,lium,salad,clore" || f.market("lium") != lium {
		t.Fatalf("after keys were added: %s %v", names(), err)
	}
	// One set in the environment is taken over the file's.
	t.Setenv("LIUM_API_KEY", "other")
	if err := f.refresh(); err != nil || f.market("lium") == lium {
		t.Fatalf("the environment's key: %v", err)
	}
	// A file others may read is refused: it holds keys that spend money.
	write("RUNPOD_API_KEY=rp\n", 0o644)
	if err := f.refresh(); err == nil || !strings.Contains(err.Error(), "chmod 600") {
		t.Fatalf("a file others may read: %v", err)
	}
	// Only RunPod is armed until markets.json says otherwise.
	if !f.armed("runpod") || f.armed("vast") {
		t.Fatal("armed by default: RunPod alone")
	}
}

// A fleet on RunPod and Vast, with only RunPod armed: Vast's offers are shown and marked, but
// nothing is rented there, and Vast cannot be armed while renting there is not built.
func TestNothingIsRentedOnAMarketNotArmed(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["POST /api/v0/bundles/"] = string(fixture(t, "vast_bundles.json"))
	fm.answers["GET /api/v1/instances/"] = `{"instances": [], "next_token": null}`
	f.markets = append(f.markets, newVast(fm.srv.URL, "vast-key"))
	f.o.Cheapest = true
	st.work = Work{Rounds: []WorkRound{{ID: "s1-o00001", GradesWanted: 12}}}
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	for _, call := range fm.asked {
		if strings.HasPrefix(call, "PUT") || strings.Contains(call, "bundles") {
			t.Fatalf("Vast, not armed, was asked %s:\n%s", call, out)
		}
	}
	if len(rp.created) == 0 {
		t.Fatalf("nothing rented on RunPod:\n%s", out)
	}
	out.Reset()
	if err := cmdOffers(context.Background(), f, false); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "vast    verified   RTX 3060") || !strings.Contains(out.String(), "(not armed)") || !strings.Contains(out.String(), "runpod  secure") {
		t.Fatalf("offers:\n%s", out)
	}
	// A market with no key cannot be armed; Vast can, once its key works.
	f.off = map[string]string{"salad": "SALAD_API_KEY is not set"}
	if err := cmdMarkets(context.Background(), f, []string{"arm", "salad"}, "test"); err == nil || !strings.Contains(err.Error(), "has no key") {
		t.Fatalf("armed Salad: %v", err)
	}
	if err := cmdMarkets(context.Background(), f, []string{"arm", "vast"}, "test"); err == nil || !strings.Contains(err.Error(), "key does not work") {
		t.Fatalf("armed Vast with a key that does not work: %v", err)
	}
	fm.answers["GET /api/v0/users/current/"] = `{"credit": 39.66, "can_pay": true}`
	if err := cmdMarkets(context.Background(), f, []string{"arm", "vast"}, "smoke test passed"); err != nil || !f.armed("vast") {
		t.Fatalf("armed Vast: %v", err)
	}
	if a, err := readArming(f.o.Home); err != nil || a.Armed["vast"].Note != "smoke test passed" || a.Armed["vast"].At == 0 || !f.armed("runpod") {
		t.Fatalf("kept as %+v (%v)", a, err)
	}
}

// By hand, up --market rents on that market alone, armed or not; scale takes no --market.
func TestUpRentsOnOneMarketByHand(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["POST /api/v0/bundles/"] = string(fixture(t, "vast_bundles.json"))
	fm.answers["GET /api/v1/instances/"] = `{"instances": [], "next_token": null}`
	fm.answers["PUT /api/v0/asks/50121277/"] = `{"success": true, "new_contract": 9001}`
	f.markets = append(f.markets, newVast(fm.srv.URL, "vast-key"))
	f.o.Market, f.o.Cheapest, f.o.MaxHourly, f.o.Edits = "vast", true, 3, true
	if err := cmdUp(context.Background(), f, 1); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if len(rp.created) != 0 || !strings.Contains(strings.Join(fm.asked, "\n"), "PUT /api/v0/asks/50121277/") || !strings.Contains(out.String(), "rented stet-rp-01 on Vast.ai: RTX 3060") {
		t.Fatalf("rented on RunPod %d, asked Vast %v:\n%s", len(rp.created), fm.asked, out)
	}
	if !strings.Contains(fm.bodies[len(fm.bodies)-1], `"STET_WORKER_EDITS":"0"`) {
		t.Fatalf("a machine on Vast was let run curators: %s", fm.bodies[len(fm.bodies)-1])
	}
	l, _ := f.readLedger()
	if r := l.Pods["vast:9001"]; r == nil || r.Market != "vast" || r.Handle != "stet-rp-01" {
		t.Fatalf("recorded as %+v", l.Pods)
	}
	var o, e bytes.Buffer
	if err := run([]string{"scale", "--market", "vast"}, &o, &e); err == nil || !strings.Contains(err.Error(), "for up and offers alone") {
		t.Fatalf("scale --market: %v", err)
	}
}

// A market that does not answer is passed over; its machines in the record are taken to be as
// they were, so that their keys are not rented again, and none is taken for ended.
func TestAMarketThatDoesNotAnswerIsPassedOverOnlyWhenNothingIsThere(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	fm := newFakeMarket(t, "auth", "clore-key")
	fm.answers["GET /my_orders"] = `500 {}`
	cl := newClore(fm.srv.URL, "clore-key")
	cl.a.sleep = func(context.Context, time.Duration) error { return nil }
	f.markets = append(f.markets, cl)
	if _, err := f.look(context.Background()); err != nil {
		t.Fatalf("Clore down, with nothing there: %v\n%s", err, out)
	}
	l, _ := f.readLedger()
	l.Pods[ledgerKey("clore", "501")] = &podRecord{ID: "501", Market: "clore", Handle: f.keys[0].Handle, GPU: "RTX 3090", Cost: 0.18, Start: time.Now(), Alive: time.Now()}
	if err := f.writeLedger(l); err != nil {
		t.Fatal(err)
	}
	v, err := f.look(context.Background())
	if err != nil {
		t.Fatalf("Clore down, with a machine there in the record: %v", err)
	}
	if w := v.Workers[0]; w.Rental == nil || w.Rental.ID != "501" || w.state(time.Now(), f.o.Policy) == "free" {
		t.Fatalf("Clore down: %s's key looked free: %+v", f.keys[0].Handle, w.Rental)
	}
	if l, _ := f.readLedger(); l.Pods["clore:501"] == nil {
		t.Fatal("Clore down: its machine was taken for ended")
	}
	// Clore answers again: its order has no name, and is the key's by the record.
	fm.answers["GET /my_orders"] = `{"orders": [{"id": 501, "price": 4.0, "ct": 1795240000, "expired": false, "specs": {"gpu": "1x NVIDIA GeForce RTX 3090"}}], "code": 0}`
	v, err = f.look(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if w := v.Workers[0]; w.Rental == nil || w.Rental.Market != "clore" || w.Rental.ID != "501" {
		t.Fatalf("%s's machine on Clore: %+v", f.keys[0].Handle, w.Rental)
	}
	if v.Workers[1].Rental != nil {
		t.Fatalf("the other key: %+v", v.Workers[1].Rental)
	}
}

// Lium rents with a template of the machine's own, used once, which carries its environment
// and the worker as its command; it pays no more than the price offered, and a node whose
// price rose is no room.
func TestLiumRentsWithATemplateOfItsOwn(t *testing.T) {
	fm := newFakeMarket(t, "X-API-Key", "lium-key")
	fm.answers["POST /templates"] = `{"id": "7d0c1b6e-0000-4000-8000-00000000000a", "name": "stet-stet-rp-03-x"}`
	fm.answers["POST /executors/node-a/rent"] = `{"success": true, "pod_id": "6b1f0d4e-0000-4000-8000-000000000009"}`
	fm.answers["POST /executors/node-b/rent"] = `409 {"detail": {"code": "price_changed", "message": {"price_per_gpu_hour": 0.2}}}`
	fm.answers["DELETE /pods/6b1f0d4e-0000-4000-8000-000000000009"] = `{}`
	m := newLium(fm.srv.URL, "lium-key")
	l := Launch{Name: "stet-rp-03", Script: "echo hi", Env: map[string]string{"STET_API": "https://api.stet.test"}, SSHKey: "ssh-ed25519 AAAA test", Need: testNeed}
	r, err := m.Rent(context.Background(), Offer{Market: "lium", ID: "node-a", Card: "RTX 3090", Price: 0.16}, l)
	if err != nil || r.ID != "6b1f0d4e-0000-4000-8000-000000000009" || r.Name != "stet-rp-03" || r.State != "starting" {
		t.Fatalf("rented %+v (%v)", r, err)
	}
	var tpl struct {
		Image, Tag, Start string
		Env               map[string]string `json:"environment"`
		Once              bool              `json:"one_time_template"`
		Private           bool              `json:"is_private"`
	}
	var raw map[string]any
	json.Unmarshal([]byte(fm.bodies[0]), &raw)
	json.Unmarshal([]byte(fm.bodies[0]), &tpl)
	if fmt.Sprint(raw["volumes"]) != "[/workspace]" || raw["docker_image"] != liumImage || raw["startup_commands"] != liumCommand || !tpl.Once || !tpl.Private ||
		tpl.Env["STET_MARKET"] != "lium" || tpl.Env["STET_WORKER_HOME"] != "/workspace/root" || tpl.Env["STET_API"] != "https://api.stet.test" || tpl.Env["STET_WORKER_SCRIPT"] != "ZWNobyBoaQ==" {
		t.Fatalf("the template: %s", fm.bodies[0])
	}
	var rent map[string]any
	json.Unmarshal([]byte(fm.bodies[1]), &rent)
	if rent["template_id"] != "7d0c1b6e-0000-4000-8000-00000000000a" || rent["pod_name"] != "stet-rp-03" || rent["termination_hours"] != float64(liumHours) ||
		rent["expected_price_per_gpu_hour"] != 0.16 || rent["user_public_key"] != "ssh-ed25519 AAAA test" {
		t.Fatalf("the rent: %s", fm.bodies[1])
	}
	if _, err := m.Rent(context.Background(), Offer{Market: "lium", ID: "node-b", Price: 0.15}, l); !errors.Is(err, errNoRoom) {
		t.Fatalf("a price that rose: %v", err)
	}
	if err := m.End(context.Background(), "6b1f0d4e-0000-4000-8000-000000000009"); err != nil {
		t.Fatal(err)
	}
	if err := m.End(context.Background(), "gone"); err != nil {
		t.Fatalf("one already gone: %v", err)
	}
}

// The worker started from its environment, as Lium (and others) start it, is the script given.
func TestTheWorkerCommandStartsTheScriptFromItsEnvironment(t *testing.T) {
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("no bash")
	}
	dir := t.TempDir()
	cmd := strings.ReplaceAll(workerCommand, "/root/", dir+"/")
	c := exec.Command("sh", "-c", cmd)
	c.Env = []string{"PATH=/usr/bin:/bin", "STET_WORKER_SCRIPT=" + base64.StdEncoding.EncodeToString([]byte("echo started $STET_API"))}
	c.Env = append(c.Env, "STET_API=x")
	out, err := c.CombinedOutput()
	if err != nil || strings.TrimSpace(string(out)) != "started x" {
		t.Fatalf("%q %v", out, err)
	}
}

// Salad rents a container group of one replica on the class offered, at high priority, named
// after the key's handle with more after it, and the worker as its command.
func TestSaladRentsAContainerGroupOfOne(t *testing.T) {
	fm := newFakeMarket(t, "Salad-Api-Key", "salad-key")
	fm.answers["POST /organizations/stet/projects/fleet/containers"] = `{"name": "x", "current_state": {"status": "pending"}}`
	m := newSalad(fm.srv.URL, "salad-key", "stet", "fleet")
	l := Launch{Name: "stet-rp-04", Image: "ubuntu:24.04", Script: "echo hi", Env: map[string]string{"STET_API": "https://api.stet.test"}, Need: testNeed}
	r, err := m.Rent(context.Background(), Offer{Market: "salad", ID: "ed563892-aacd-40f5-80b7-90c9be6c759b", Card: "RTX 4090", Price: 0.44}, l)
	if err != nil || r.Name != "stet-rp-04" || !strings.HasPrefix(r.ID, "stet-rp-04-") || len(r.ID) != len("stet-rp-04-")+5 {
		t.Fatalf("rented %+v (%v)", r, err)
	}
	var g struct {
		Name        string
		DisplayName string `json:"display_name"`
		Replicas    int
		Restart     string `json:"restart_policy"`
		Container   struct {
			Image, Priority string
			Command         []string
			Env             map[string]string `json:"environment_variables"`
			Resources       struct {
				CPU, Memory int
				GPUClasses  []string `json:"gpu_classes"`
				Storage     int64    `json:"storage_amount"`
			}
		}
	}
	json.Unmarshal([]byte(fm.bodies[0]), &g)
	c := g.Container
	if g.DisplayName != "stet-rp-04" || g.Replicas != 1 || g.Restart != "never" || c.Priority != "high" || c.Image != "ubuntu:24.04" ||
		strings.Join(c.Command, "|") != "bash|-c|"+workerInner || c.Env["STET_MARKET"] != "salad" || c.Env["STET_WORKER_SCRIPT"] != "ZWNobyBoaQ==" ||
		c.Resources.CPU != 6 || c.Resources.Memory != 16384 || c.Resources.GPUClasses[0] != "ed563892-aacd-40f5-80b7-90c9be6c759b" || c.Resources.Storage != 100<<30 {
		t.Fatalf("asked for %s", fm.bodies[0])
	}
	fm.answers["DELETE /organizations/stet/projects/fleet/containers/"+r.ID] = ``
	if err := m.End(context.Background(), r.ID); err != nil {
		t.Fatal(err)
	}
	if err := m.End(context.Background(), "gone-12345"); err != nil {
		t.Fatalf("one already gone: %v", err)
	}
}

// Clore takes the worker in pieces of its environment no longer than it allows, an order says
// no ID, so it is found by its server, and an order refused (code 6) is no room.
func TestCloreOrdersAndFindsItsOrder(t *testing.T) {
	fm := newFakeMarket(t, "auth", "clore-key")
	fm.answers["POST /create_order"] = `{"code": 0}`
	now := time.Now().Unix()
	fm.answers["GET /my_orders"] = fmt.Sprintf(`{"code": 0, "orders": [
		{"id": 700, "price": 4.0, "ct": %d, "si": 124838, "expired": true, "specs": {"gpu": "1x NVIDIA GeForce RTX 3090"}},
		{"id": 701, "price": 4.0, "ct": %d, "si": 124838, "expired": false, "specs": {"gpu": "1x NVIDIA GeForce RTX 3090"}}]}`, now-7200, now)
	fm.answers["POST /cancel_order"] = `{"code": 0}`
	m := newClore(fm.srv.URL, "clore-key")
	m.sleep = func(context.Context, time.Duration) error { return nil }
	script := strings.Repeat("echo the worker script, at about its length\n", 120)
	l := Launch{Name: "stet-rp-05", Image: "ubuntu:24.04", Script: script, Env: map[string]string{"STET_API": "https://api.stet.test"}, Need: testNeed}
	r, err := m.Rent(context.Background(), Offer{Market: "clore", ID: "124838", Card: "RTX 3090", Price: 0.275, Ask: 4.0}, l)
	if err != nil || r.ID != "701" || r.Name != "stet-rp-05" || r.State != "starting" {
		t.Fatalf("rented %+v (%v)", r, err)
	}
	var o struct {
		Currency, Image, Type, Command string
		Server                         int64             `json:"renting_server"`
		Hours                          int               `json:"rental_hours"`
		Price                          float64           `json:"required_price"`
		Env                            map[string]string `json:"env"`
	}
	json.Unmarshal([]byte(fm.bodies[0]), &o)
	joined, parts := "", 0
	for i := 0; ; i++ {
		v, ok := o.Env[fmt.Sprintf("STET_WORKER_SCRIPT_%d", i)]
		if !ok {
			break
		}
		if len(v) > cloreEnvValue {
			t.Fatalf("a value of %d characters", len(v))
		}
		joined, parts = joined+v, parts+1
	}
	got, _ := base64.StdEncoding.DecodeString(joined)
	if o.Currency != "USD-Blockchain" || o.Type != "on-demand" || o.Server != 124838 || o.Hours != cloreHours || o.Price != 4.0 ||
		string(got) != script || parts < 2 || !strings.Contains(o.Command, "$STET_WORKER_SCRIPT_0$STET_WORKER_SCRIPT_1") || o.Env["STET_MARKET"] != "clore" {
		t.Fatalf("ordered %s", fm.bodies[0])
	}
	// The command puts the pieces back together and runs them.
	env := []string{"PATH=/usr/bin:/bin"}
	for k, v := range o.Env {
		env = append(env, k+"="+v)
	}
	dir := t.TempDir()
	c := exec.Command("sh", "-c", strings.ReplaceAll(strings.ReplaceAll(o.Command, "exec bash", "exec head -c 44"), "/root/", dir+"/"))
	c.Env = env
	if out, err := c.CombinedOutput(); err != nil || string(out) != "echo the worker script, at about its length\n" {
		t.Fatalf("the command ran %q (%v)", out, err)
	}
	fm.answers["POST /create_order"] = `{"code": 6, "error": "server_already_rented"}`
	if _, err := m.Rent(context.Background(), Offer{Market: "clore", ID: "124851", Ask: 7.95}, l); !errors.Is(err, errNoRoom) {
		t.Fatalf("an order refused: %v", err)
	}
	if err := m.End(context.Background(), "701"); err != nil {
		t.Fatal(err)
	}
}

// With Vast armed and cheapest, a rent Vast refuses otherwise than for want of room (its credit
// gone, say) is said, and the machine is rented on another market; so is it when Vast cannot
// say what it has at all.
func TestOneMarketFailingDoesNotStopTheOthers(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 2)
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["POST /api/v0/bundles/"] = string(fixture(t, "vast_bundles.json"))
	fm.answers["GET /api/v1/instances/"] = `{"instances": [], "next_token": null}`
	fm.answers["PUT *"] = `402 {"success": false, "error": "insufficient_credit", "msg": "not enough credit"}`
	vast := newVast(fm.srv.URL, "vast-key")
	vast.a.sleep = func(context.Context, time.Duration) error { return nil }
	f.markets = append(f.markets, vast)
	if err := (&Arming{Armed: map[string]Armed{"runpod": {}, "vast": {At: 1}}}).save(f.o.Home); err != nil {
		t.Fatal(err)
	}
	f.o.Cheapest = true
	st.work = *round(30, 1, 0, 0)
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 2 || !strings.Contains(out.String(), "on Vast.ai did not work") || !strings.Contains(out.String(), "not enough credit") {
		t.Fatalf("rented %d on RunPod:\n%s", len(rp.created), out)
	}
	// Vast cannot say what it has: RunPod's offers are rented all the same.
	for _, p := range rp.pods {
		delete(rp.pods, p.ID)
	}
	fm.answers["POST /api/v0/bundles/"] = `503 {}`
	out.Reset()
	if err := tick(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	if len(rp.created) != 4 || !strings.Contains(out.String(), "Vast.ai's offers could not be had") {
		t.Fatalf("rented %d on RunPod:\n%s", len(rp.created), out)
	}
}
