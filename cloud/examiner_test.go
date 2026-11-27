package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeExaminerProv struct {
	mu      sync.Mutex
	offers  []ExaminerOffer
	insts   map[string]ExaminerInstance
	rented  []ExaminerLaunch
	ended   []string
	nextID  int
	failIDs map[string]error
	endErr  map[string]error
}

func (p *fakeExaminerProv) ExaminerOffers(ctx context.Context, maxPrice float64) ([]ExaminerOffer, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []ExaminerOffer
	for _, o := range p.offers {
		if maxPrice <= 0 || o.Price <= maxPrice {
			out = append(out, o)
		}
	}
	return out, nil
}

func (p *fakeExaminerProv) RentExaminer(ctx context.Context, offerID string, l ExaminerLaunch) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.failIDs[offerID]; err != nil {
		return "", err
	}
	p.nextID++
	id := "ex-" + string(rune('0'+p.nextID))
	p.rented = append(p.rented, l)
	if p.insts == nil {
		p.insts = map[string]ExaminerInstance{}
	}
	p.insts[id] = ExaminerInstance{
		ID: id, Label: l.Label, State: "starting", Status: "loading",
		Price: 0.62, Created: time.Now(),
	}
	return id, nil
}

func (p *fakeExaminerProv) ListExaminers(ctx context.Context) ([]ExaminerInstance, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	var out []ExaminerInstance
	for _, inst := range p.insts {
		out = append(out, inst)
	}
	return out, nil
}

func (p *fakeExaminerProv) EndExaminer(ctx context.Context, id string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.endErr[id]; err != nil {
		return err
	}
	p.ended = append(p.ended, id)
	delete(p.insts, id)
	return nil
}

func TestExaminerFullLifecycleProxyAuthAndTeardown(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	var mu sync.Mutex
	var wantKey string
	var gotAuth []string

	// Fake upstream vLLM server requiring the exact per-instance random 256-bit Bearer key.
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		k := wantKey
		gotAuth = append(gotAuth, r.Header.Get("Authorization"))
		mu.Unlock()
		if r.Header.Get("Authorization") != "Bearer "+k {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"bad key"}`))
			return
		}
		if r.URL.Path == "/v1/models" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": defaultExaminerModel + "@" + defaultExaminerRevision}},
			})
			return
		}
		if r.URL.Path == "/v1/chat/completions" {
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "cmpl-ok", "choices": []any{}})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer vllm.Close()
	vu, _ := url.Parse(vllm.URL)

	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62, CUDA: 13.3, InetDown: 1200}},
	}
	home := t.TempDir()
	var spent []podRecord
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)
	ex.onSpent = func(rec podRecord) { spent = append(spent, rec) }

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	addr, stopProxy, err := ex.Start(ctx)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer stopProxy()
	proxyBase := "http://" + addr.String()

	// 1. Idle (no open round): proxy returns 503 and nothing is rented.
	resp, err := http.Get(proxyBase + "/v1/models")
	if err != nil {
		t.Fatalf("GET idle proxy: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("idle proxy status = %d, want 503", resp.StatusCode)
	}
	if err := ex.Reconcile(ctx, &Work{}, 0, 0); err != nil {
		t.Fatalf("Reconcile idle: %v", err)
	}
	if len(prov.rented) != 0 {
		t.Fatalf("rented %d instances when idle", len(prov.rented))
	}

	// 2. Round opens with ExamAt == 0 -> rents 1x RTX 5090 with pinned image/model/revision and 256-bit key.
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00001", Entries: 4, ExamAt: 0}}}
	if err := ex.Reconcile(ctx, openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}
	if len(prov.rented) != 1 {
		t.Fatalf("expected 1 examiner rented, got %d", len(prov.rented))
	}
	launch := prov.rented[0]
	if launch.Image != defaultExaminerImage || launch.Model != defaultExaminerModel || launch.Revision != defaultExaminerRevision {
		t.Fatalf("unpinned examiner launch: %+v", launch)
	}
	if len(launch.APIKey) != 64 {
		t.Fatalf("expected 64-hex-char 256-bit API key, got len %d", len(launch.APIKey))
	}
	mu.Lock()
	wantKey = launch.APIKey
	mu.Unlock()
	if st := readExaminerState(home); st == nil || st.InstanceID != "ex-1" || st.APIKey != launch.APIKey {
		t.Fatalf("examiner.json not persisted: %+v", st)
	}

	// Still booting -> proxy remains 503.
	resp, _ = http.Get(proxyBase + "/v1/models")
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("booting proxy status = %d, want 503", resp.StatusCode)
	}

	// 3. Container becomes running with mapped IP:port -> Reconcile probes /v1/models and enables proxy!
	prov.mu.Lock()
	inst := prov.insts["ex-1"]
	inst.State, inst.Status = "running", "running"
	inst.PublicIP, inst.HostPort = vu.Hostname(), vu.Port()
	inst.Created = now
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	now = now.Add(4 * time.Minute)
	if err := ex.Reconcile(ctx, openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile ready: %v", err)
	}
	// Proxy forwards /v1/chat/completions to upstream injecting Authorization: Bearer <apiKey>.
	resp, err = http.Post(proxyBase+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("POST proxied chat: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(body), "cmpl-ok") {
		t.Fatalf("proxied chat: HTTP %d %s", resp.StatusCode, string(body))
	}

	// 4. Exam is posted (ExamAt > 0) -> Reconcile destroys Vast instance at once, clears state, records spend.
	now = now.Add(11 * time.Minute)
	postedWork := &Work{Rounds: []WorkRound{{ID: "s3-o00001", Entries: 4, ExamAt: now.Unix()}}}
	if err := ex.Reconcile(ctx, postedWork, 0, 0); err != nil {
		t.Fatalf("Reconcile posted: %v", err)
	}
	if len(prov.ended) != 1 || prov.ended[0] != "ex-1" {
		t.Fatalf("expected ex-1 destroyed after exam posted, ended=%v", prov.ended)
	}
	if readExaminerState(home) != nil {
		t.Fatalf("examiner.json should be removed after teardown")
	}
	if len(spent) != 1 || spent[0].For != "examiner" {
		t.Fatalf("expected examiner spend record, got %+v", spent)
	}
	resp, _ = http.Get(proxyBase + "/v1/models")
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("post-teardown proxy status = %d, want 503", resp.StatusCode)
	}
}

func TestExaminerOrphanCleanupAndTimeoutAndLoopbackGuard(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.65, CUDA: 13.3, InetDown: 900}},
		insts: map[string]ExaminerInstance{
			"orphan-99": {ID: "orphan-99", Label: "stet-examiner-s3-o00000", State: "running", Price: 0.65, Created: now.Add(-10 * time.Minute)},
		},
	}
	home := t.TempDir()
	ex := newExaminer(ExaminerConfig{Addr: "0.0.0.0:8000", Timeout: 30 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)

	// Non-loopback address MUST be refused.
	if _, _, err := ex.Start(context.Background()); err == nil {
		t.Fatalf("expected Start to reject non-loopback 0.0.0.0:8000")
	}

	// Orphan without local examiner.json is destroyed when idle.
	if err := ex.Reconcile(context.Background(), &Work{}, 0, 0); err != nil {
		t.Fatalf("Reconcile orphan idle: %v", err)
	}
	if len(prov.ended) != 1 || prov.ended[0] != "orphan-99" {
		t.Fatalf("expected orphan-99 destroyed, ended=%v", prov.ended)
	}

	// Rent for an open round, then advance past hard 30m Timeout -> destroyed & replaced.
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00002", Entries: 2, ExamAt: 0}}}
	if err := ex.Reconcile(context.Background(), openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}
	if len(prov.rented) != 1 {
		t.Fatalf("expected 1 rented, got %d", len(prov.rented))
	}
	now = now.Add(31 * time.Minute)
	if err := ex.Reconcile(context.Background(), openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile past timeout: %v", err)
	}
	if len(prov.ended) != 2 || prov.ended[1] != "ex-1" {
		t.Fatalf("expected timed-out ex-1 destroyed, ended=%v", prov.ended)
	}
	if len(prov.rented) != 2 {
		t.Fatalf("expected fresh replacement examiner rented after timeout, rented=%d", len(prov.rented))
	}
	if prov.rented[0].Label == prov.rented[1].Label {
		t.Fatalf("replacement examiner must have a distinct Vast label from timed-out VM: %q", prov.rented[1].Label)
	}
}

func TestExaminerConcurrentProbeAndTeardownNoRaceAndNoStaleRevival(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	probeStarted := make(chan struct{})
	releaseProbe := make(chan struct{})
	var once sync.Once

	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			once.Do(func() {
				close(probeStarted)
				<-releaseProbe
			})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": defaultExaminerModel + "@" + defaultExaminerRevision}},
			})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer vllm.Close()
	vu, _ := url.Parse(vllm.URL)

	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
	}
	home := t.TempDir()
	var buf strings.Builder
	f := &Fleet{
		o:   &Options{Home: home},
		st:  &Stet{},
		out: &buf,
		now: func() time.Time { return now },
	}
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, f.say, false)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	addr, stopProxy, err := ex.Start(ctx)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer stopProxy()

	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00003", Entries: 1, ExamAt: 0}}}
	if err := ex.Reconcile(ctx, openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile rent: %v", err)
	}
	prov.mu.Lock()
	inst := prov.insts["ex-1"]
	inst.State, inst.PublicIP, inst.HostPort = "running", vu.Hostname(), vu.Port()
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	// Start background ProbeIfBooting; pause inside /v1/models while Reconcile tears down ex-1
	// AND Fleet.beginLook/endLook concurrently mutate f.cur under -race.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ex.ProbeIfBooting(ctx)
	}()
	<-probeStarted

	_ = f.beginLook()
	postedWork := &Work{Rounds: []WorkRound{{ID: "s3-o00003", Entries: 1, ExamAt: now.Unix()}}}
	if err := ex.Reconcile(ctx, postedWork, 0, 0); err != nil {
		t.Fatalf("Reconcile teardown: %v", err)
	}
	f.endLook(now, nil)

	// Unblock stale /v1/models response from destroyed ex-1: must NOT revive proxy or recreate examiner.json.
	close(releaseProbe)
	wg.Wait()

	if readExaminerState(home) != nil {
		t.Fatalf("stale ProbeIfBooting recreated examiner.json after teardown")
	}
	resp, err := http.Get("http://" + addr.String() + "/v1/models")
	if err != nil {
		t.Fatalf("GET proxy after stale probe: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("stale ProbeIfBooting revived destroyed proxy: HTTP %d, want 503", resp.StatusCode)
	}
}

func TestExaminerTimeoutEndFailureAndZeroCreatedSpend(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
		endErr: map[string]error{"ex-1": errors.New("transient Vast 503")},
	}
	home := t.TempDir()
	var spent []podRecord
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 30 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)
	ex.onSpent = func(rec podRecord) { spent = append(spent, rec) }

	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00004", Entries: 1, ExamAt: 0}}}
	if err := ex.Reconcile(context.Background(), openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}
	// Simulate Vast returning start_date == 0 (Created.IsZero() == true).
	prov.mu.Lock()
	inst := prov.insts["ex-1"]
	inst.Created = time.Time{}
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	// Past timeout, EndExaminer fails transiently -> must return error and NOT rent a duplicate 5090.
	now = now.Add(31 * time.Minute)
	if err := ex.Reconcile(context.Background(), openWork, 0, 0); err == nil {
		t.Fatalf("expected error when EndExaminer fails on timeout")
	}
	if len(prov.rented) != 1 {
		t.Fatalf("transient EndExaminer failure must not rent duplicate VM, rented=%d", len(prov.rented))
	}
	if st := readExaminerState(home); st == nil || st.InstanceID != "ex-1" {
		t.Fatalf("failed teardown must preserve ex-1 in examiner.json, got %+v", st)
	}

	// Next tick EndExaminer succeeds -> ex-1 destroyed, spend Start uses saved.Created (never 1970!).
	prov.mu.Lock()
	delete(prov.endErr, "ex-1")
	prov.mu.Unlock()
	if err := ex.Reconcile(context.Background(), openWork, 0, 0); err != nil {
		t.Fatalf("Reconcile retry timeout: %v", err)
	}
	if len(spent) != 1 {
		t.Fatalf("expected 1 spend record, got %d", len(spent))
	}
	wantStart := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	if !spent[0].Start.Equal(wantStart) {
		t.Fatalf("zero inst.Created should fall back to saved.Created (%s), got %s", wantStart, spent[0].Start)
	}
}
