package main

import (
	"context"
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A CPU is known by its model, however a market writes it.
func TestACPUIsKnownByItsModel(t *testing.T) {
	for _, same := range [][2]string{
		{"Xeon® E5-2680 v4 ", "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz"},
		{"AMD EPYC 7K62 48-Core Processor", "AMD EPYC 7K62 48-Core Processor "},
		{"Core™ i5-14600KF", "Intel(R) Core(TM) i5-14600KF"},
	} {
		if a, b := cpuKey(same[0]), cpuKey(same[1]); a != b || a == "" {
			t.Errorf("%q is %q, and %q is %q", same[0], a, same[1], b)
		}
	}
	if cpuKey("AMD Ryzen 9 9900X 12-Core Processor") == cpuKey("AMD Ryzen 9 9950X 16-Core Processor") {
		t.Error("two CPUs taken for one")
	}
}

// Vast and Clore say the host each offer is on, and its CPU; an ask gone between the search
// and the rent is no room, however Vast says it.
func TestOffersSayTheirHostAndCPU(t *testing.T) {
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["POST /api/v0/bundles/"] = string(fixture(t, "vast_bundles.json"))
	fm.answers["PUT /api/v0/asks/44053844/"] = `400 {"success": false, "error": "no_such_ask", "msg": "error 404/3603: no_such_ask  Instance type by id 44053844 is not available."}`
	m := newVast(fm.srv.URL, "vast-key")
	os, err := m.Offers(context.Background(), testNeed)
	if err != nil {
		t.Fatal(err)
	}
	if o := byID(os)["44053844"]; o.Host != "157631" || o.CPU != "AMD EPYC 7K62 48-Core Processor" {
		t.Fatalf("Vast's offer: %+v", o)
	}
	l := Launch{Name: "stet-rp-07", Image: "ubuntu:24.04", Script: "echo hi", Need: testNeed}
	if _, err := m.Rent(context.Background(), Offer{Market: "vast", ID: "44053844"}, l); !errors.Is(err, errNoRoom) {
		t.Fatalf("an ask gone: %v", err)
	}
	fc := newFakeMarket(t, "auth", "clore-key")
	fc.answers["GET /marketplace"] = string(fixture(t, "clore_marketplace.json"))
	cs, err := newClore(fc.srv.URL, "clore-key").Offers(context.Background(), Need{MinGPUMemGB: 12, MinCPUs: 6, MinRAMGB: 16, DiskGB: 25, MinCUDA: "12.0"})
	if err != nil {
		t.Fatal(err)
	}
	if o, ok := byID(cs)["124851"]; !ok || o.Host != "124851" || cpuKey(o.CPU) != cpuKey("Xeon® E5-2680 v4") {
		t.Fatalf("Clore's offer: %+v (%v)", o, ok)
	}
}

// A machine rented to run curators that the server says may not run them teaches where they
// may not: its host, and its CPU where the server names the CPU, are not rented for curators
// again for a week; the next is rented elsewhere.
func TestAMachineUnderTheFloorTeachesWhereCuratorsMayNotRun(t *testing.T) {
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, out := testFleet(t, rp, st, 3)
	ctx := context.Background()
	fm := newFakeMarket(t, "Authorization", "Bearer vast-key")
	fm.answers["POST /api/v0/bundles/"] = string(fixture(t, "vast_bundles.json"))
	vast := newVast(fm.srv.URL, "vast-key")
	f.markets = append(f.markets, vast)
	f.o.Edits, f.o.Disk, f.o.Market = true, 100, "vast"
	if err := cmdMarkets(ctx, f, []string{"curators", "vast"}, "cage_check: every case held", false); err != nil {
		t.Fatal(err)
	}
	st.work = Work{Rounds: []WorkRound{{ID: "s1-o00009", Entries: 8, RunsWanted: 8, RunsWaiting: 8}}}
	instances := []string{}
	// rent rents one machine to run curators, as the contract given, and says which ask.
	rent := func(contract string) string {
		t.Helper()
		fm.answers["GET /api/v1/instances/"] = `{"instances": [` + strings.Join(instances, ",") + `], "next_token": null}`
		fm.answers["PUT *"] = `{"success": true, "new_contract": ` + contract + `}`
		v, err := f.look(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := f.launch(ctx, v, 1, 0, 0, 1, time.Time{}); err != nil || got != 1 {
			t.Fatalf("rented %d (%v)\n%s", got, err, out)
		}
		ask := fm.asked[len(fm.asked)-1]
		l, _ := f.readLedger()
		rec := l.Pods["vast:"+contract]
		instances = append(instances, `{"id": `+contract+`, "label": "`+rec.Handle+`", "actual_status": "running", "intended_status": "running", "cur_state": "running", "gpu_name": "RTX 3060", "dph_total": 0.06, "start_date": `+itoa(time.Now().Add(-5*time.Minute).Unix())+`}`)
		return strings.TrimSuffix(strings.TrimPrefix(ask, "PUT /api/v0/asks/"), "/")
	}
	// refuse has the server say the machine on a contract may not run curators.
	no := false
	refuse := func(contract, why string) *podRecord {
		t.Helper()
		l, _ := f.readLedger()
		rec := l.Pods["vast:"+contract]
		st.work.Machines = append(st.work.Machines, Machine{Handle: rec.Handle, Worker: true, LastSeen: time.Now().Unix(), Curators: &no, CuratorsWhy: why})
		fm.answers["GET /api/v1/instances/"] = `{"instances": [` + strings.Join(instances, ",") + `], "next_token": null}`
		v, err := f.look(ctx)
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range v.Workers {
			if w.Key.Handle == rec.Handle && w.NoCurators != why {
				t.Fatalf("%s may not run curators, and the fleet has it as %q", rec.Handle, w.NoCurators)
			}
		}
		l, _ = f.readLedger()
		return l.Pods["vast:"+contract]
	}

	// The cheapest a grade first: the 4080 SUPER on host 153491, an EPYC 7452.
	if ask := rent("9001"); ask != "48398491" {
		t.Fatalf("rented ask %s first", ask)
	}
	rec := refuse("9001", "this machine is under the season's floor for running curators: memory 15 GB, floor 16; CPU 95 MB/s, floor 100")
	if rec.Refused == "" || rec.Host != "153491" || !rec.Curators {
		t.Fatalf("its record: %+v", rec)
	}
	r, err := f.readRefusals()
	if err != nil || r.Hosts["vast:153491"].Handle != rec.Handle || r.CPUs[cpuKey("AMD EPYC 7452 32-Core Processor")].Host != "153491" {
		t.Fatalf("kept as %+v (%v)", r, err)
	}
	if !strings.Contains(out.String(), "may not run curators: this machine is under the season's floor for running curators: memory 15 GB, floor 16; CPU 95 MB/s, floor 100. It grades only; neither its host nor any AMD EPYC 7452 32-Core Processor is rented to run curators for a week") {
		t.Fatalf("not said:\n%s", out)
	}
	if raw, _ := os.ReadFile(f.eventsPath()); !strings.Contains(string(raw), `"event":"no curators"`) {
		t.Fatalf("not in the journal:\n%s", raw)
	}
	// The same CPU elsewhere is not rented for curators either; another is, and none is
	// refused after a week.
	if why, no := r.refused(Offer{Market: "clore", Host: "1", CPU: "AMD EPYC 7452 32-Core Processor"}, time.Now()); !no || !strings.Contains(why, "EPYC 7452") {
		t.Fatalf("the same CPU on Clore: %q", why)
	}
	if _, no := r.refused(Offer{Market: "vast", Host: "1", CPU: "AMD Ryzen 9 9900X"}, time.Now()); no {
		t.Fatal("another CPU refused")
	}
	if _, no := r.refused(Offer{Market: "vast", Host: "153491"}, time.Now().Add(refusedFor+time.Hour)); no {
		t.Fatal("still refused after a week")
	}
	// The next, one of the two 3060s, near enough the same a grade: one on host 34619 (an
	// i5-14600KF), one on host 155716 (a Ryzen 5 7600X), where there is another 3060 too.
	// Refused for too few of the host's cores, not its CPU's speed, its host alone is passed
	// over, and its CPU is not.
	host := map[string]string{"50121277": "34619", "61536954": "155716", "61568486": "155716", "48398491": "153491"}
	cpu := map[string]string{"50121277": "Core™ i5-14600KF", "61536954": "AMD Ryzen 5 7600X 6-Core Processor"}
	second := rent("9002")
	if second != "50121277" && second != "61536954" {
		t.Fatalf("rented ask %s second: %v\n%s", second, fm.asked, out)
	}
	refuse("9002", "this machine is under the season's floor for running curators: CPU 4 cores, floor 6")
	r, _ = f.readRefusals()
	if _, no := r.refused(Offer{Market: "vast", Host: host[second]}, time.Now()); !no || len(r.CPUs) != 1 {
		t.Fatalf("kept as %+v", r)
	}
	if _, no := r.refused(Offer{Market: "vast", Host: "1", CPU: cpu[second]}, time.Now()); no {
		t.Fatal("a CPU refused for what was not the CPU")
	}
	third := rent("9003")
	if h := host[third]; h == "153491" || h == host[second] || third == "" {
		t.Fatalf("rented ask %s, on a host refused (%s)\n%s", third, h, out)
	}
	if !strings.Contains(out.String(), "may not run curators: this machine is under the season's floor for running curators: CPU 4 cores, floor 6. It grades only; its host is not rented to run curators for a week") {
		t.Fatalf("not said:\n%s", out)
	}
}

func itoa(n int64) string { return strconv.FormatInt(n, 10) }
