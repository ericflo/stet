package main

import (
	"context"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeExaminerProv struct {
	mu          sync.Mutex
	offers      []ExaminerOffer
	insts       map[string]ExaminerInstance
	rented      []ExaminerLaunch
	ended       []string
	nextID      int
	failIDs     map[string]error
	endErr      map[string]error
	onList      func()
	offerPrices []float64
}

func testExaminerBudget() ExaminerBudget {
	return ExaminerBudget{MaxWorkers: 6, MaxHourly: 4.00}
}

func (p *fakeExaminerProv) ExaminerOffers(ctx context.Context, maxPrice float64) ([]ExaminerOffer, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.offerPrices = append(p.offerPrices, maxPrice)
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
	price := 0.62
	for _, o := range p.offers {
		if o.ID == offerID {
			price = o.Price
			break
		}
	}
	p.insts[id] = ExaminerInstance{
		ID: id, Label: l.Label, State: "starting", Status: "loading",
		Price: price, Created: time.Now(),
	}
	return id, nil
}

func (p *fakeExaminerProv) ListExaminers(ctx context.Context) ([]ExaminerInstance, error) {
	p.mu.Lock()
	hook := p.onList
	p.mu.Unlock()
	if hook != nil {
		hook()
	}
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

type testClock struct {
	mu sync.Mutex
	t  time.Time
}

func newTestClock(t time.Time) *testClock { return &testClock{t: t} }
func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}
func (c *testClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

func startTLSServerWithCert(t *testing.T, certPEM, keyPEM []byte, handler http.Handler) *httptest.Server {
	t.Helper()
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("X509KeyPair: %v", err)
	}
	srv := httptest.NewUnstartedServer(handler)
	srv.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	srv.StartTLS()
	return srv
}

func TestExaminerFullLifecycleProxyAuthAndTeardown(t *testing.T) {
	// Deliberately use a past date (2020) so this test ALSO proves examinerTLSConfig uses ex.now
	// (never wall-clock time.Now()), preventing future CI certificate-expiry time-bombs.
	clk := newTestClock(time.Date(2020, 6, 1, 12, 0, 0, 0, time.UTC))
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62, CUDA: 13.3, InetDown: 1200}},
	}
	home := t.TempDir()
	var spent []podRecord
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
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
	if err := ex.Reconcile(ctx, &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile idle: %v", err)
	}
	if len(prov.rented) != 0 {
		t.Fatalf("rented %d instances when idle", len(prov.rented))
	}

	// 2. Round opens with ExamAt == 0 -> rents 1x RTX 5090 with pinned image/model/revision,
	//    256-bit API key, per-instance TLS cert+key, owner label, and TTL.
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00001", Entries: 4, ExamAt: 0}}}
	if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}
	if len(prov.rented) != 1 {
		t.Fatalf("expected 1 examiner rented, got %d", len(prov.rented))
	}
	launch := prov.rented[0]
	if launch.Image != defaultExaminerImage || launch.Model != defaultExaminerModel || launch.Revision != defaultExaminerRevision {
		t.Fatalf("unpinned examiner launch: %+v", launch)
	}
	if len(launch.APIKey) != 64 || len(launch.CertPEM) == 0 || len(launch.KeyPEM) == 0 || launch.TTLSeconds != 45*60 {
		t.Fatalf("incomplete security/TTL launch fields: keyLen=%d certLen=%d keyPEMLen=%d ttl=%d",
			len(launch.APIKey), len(launch.CertPEM), len(launch.KeyPEM), launch.TTLSeconds)
	}
	if !examinerBelongsToOwner(launch.Label, ex.ownerID()) {
		t.Fatalf("launch label %q does not belong to owner %q", launch.Label, ex.ownerID())
	}
	if st := readExaminerState(home); st == nil || st.InstanceID != "ex-1" || st.APIKey != launch.APIKey || st.CertPEM != string(launch.CertPEM) {
		t.Fatalf("examiner.json missing state or pinned cert: %+v", st)
	}

	// Fake upstream vLLM HTTPS server using the exact per-instance TLS cert+key & Bearer auth.
	vllm := startTLSServerWithCert(t, launch.CertPEM, launch.KeyPEM, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil {
			w.WriteHeader(http.StatusUpgradeRequired)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+launch.APIKey {
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
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "cmpl-tls-ok", "choices": []any{}})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer vllm.Close()
	vu, _ := url.Parse(vllm.URL)

	// Still booting -> proxy remains 503.
	resp, _ = http.Get(proxyBase + "/v1/models")
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("booting proxy status = %d, want 503", resp.StatusCode)
	}

	// 3. Container becomes running with mapped IP:port -> Reconcile probes https:///v1/models over
	//    peer-verified pinned TLS and enables proxy.
	prov.mu.Lock()
	inst := prov.insts["ex-1"]
	inst.State, inst.Status = "running", "running"
	inst.PublicIP, inst.HostPort = vu.Hostname(), vu.Port()
	inst.Created = clk.Now()
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	clk.Advance(4 * time.Minute)
	if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile ready: %v", err)
	}
	if st := readExaminerState(home); st == nil || !strings.HasPrefix(st.Upstream, "https://") {
		t.Fatalf("examiner upstream must be https://, got %+v", st)
	}
	resp, err = http.Post(proxyBase+"/v1/chat/completions", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("POST proxied chat: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(body), "cmpl-tls-ok") {
		t.Fatalf("proxied chat over pinned TLS: HTTP %d %s", resp.StatusCode, string(body))
	}

	// 4. Exam is posted (ExamAt > 0) -> Reconcile destroys Vast instance at once, clears state, records spend.
	clk.Advance(11 * time.Minute)
	postedWork := &Work{Rounds: []WorkRound{{ID: "s3-o00001", Entries: 4, ExamAt: clk.Now().Unix()}}}
	if err := ex.Reconcile(ctx, postedWork, testExaminerBudget()); err != nil {
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

// Review 11 P1 #4: examiner remote hop MUST refuse plaintext HTTP and MUST reject a MITM HTTPS
// server presenting any certificate other than the pinned per-instance certificate before sending
// the Bearer key or exam content.
func TestExaminerRejectsPlaintextHTTPAndWrongTLSCertificate(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
	}
	home := t.TempDir()
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	addr, stopProxy, err := ex.Start(ctx)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer stopProxy()

	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00009", Entries: 1, ExamAt: 0}}}
	if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}

	// 1. Plaintext HTTP server on the instance's mapped port MUST fail TLS probe (never gets Bearer key).
	plainGotAuth := false
	plainSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			plainGotAuth = true
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": defaultExaminerModel + "@" + defaultExaminerRevision}}})
	}))
	defer plainSrv.Close()
	pu, _ := url.Parse(plainSrv.URL)
	prov.mu.Lock()
	inst := prov.insts["ex-1"]
	inst.State, inst.PublicIP, inst.HostPort = "running", pu.Hostname(), pu.Port()
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile plaintext: %v", err)
	}
	if plainGotAuth {
		t.Fatalf("plaintext HTTP server received Authorization header!")
	}
	resp, _ := http.Get("http://" + addr.String() + "/v1/models")
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("proxy must stay 503 against plaintext upstream, got %d", resp.StatusCode)
	}

	// 2. HTTPS server presenting a DIFFERENT self-signed cert (MITM) MUST fail peer cert pinning!
	wrongCert, wrongKey, err := generateExaminerCert(now, 45*time.Minute)
	if err != nil {
		t.Fatalf("generateExaminerCert: %v", err)
	}
	mitmGotAuth := false
	mitmSrv := startTLSServerWithCert(t, wrongCert, wrongKey, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			mitmGotAuth = true
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": defaultExaminerModel + "@" + defaultExaminerRevision}}})
	}))
	defer mitmSrv.Close()
	mu, _ := url.Parse(mitmSrv.URL)
	prov.mu.Lock()
	inst = prov.insts["ex-1"]
	inst.PublicIP, inst.HostPort = mu.Hostname(), mu.Port()
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile MITM: %v", err)
	}
	if mitmGotAuth {
		t.Fatalf("MITM TLS server with unpinned cert received Authorization header!")
	}
	resp, _ = http.Get("http://" + addr.String() + "/v1/models")
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("proxy must stay 503 against unpinned TLS cert, got %d", resp.StatusCode)
	}
}

func TestExaminerOrphanCleanupAndTimeoutAndLoopbackGuard(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	owner := loadOrCreateExaminerOwner(home)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.65, CUDA: 13.3, InetDown: 900}},
		insts: map[string]ExaminerInstance{
			"orphan-99": {ID: "orphan-99", Label: examinerLabelPrefix + "-" + owner + "-s3-o00000-deadbeef", State: "running", Price: 0.65, Created: now.Add(-10 * time.Minute)},
		},
	}
	ex := newExaminer(ExaminerConfig{Addr: "0.0.0.0:8000", Timeout: 30 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)

	// Non-loopback address MUST be refused.
	if _, _, err := ex.Start(context.Background()); err == nil {
		t.Fatalf("expected Start to reject non-loopback 0.0.0.0:8000")
	}

	// Owned orphan without local examiner.json is destroyed when idle.
	if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile orphan idle: %v", err)
	}
	if len(prov.ended) != 1 || prov.ended[0] != "orphan-99" {
		t.Fatalf("expected owned orphan-99 destroyed, ended=%v", prov.ended)
	}

	// Rent for an open round, then advance past hard 30m Timeout -> destroyed & replaced.
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00002", Entries: 2, ExamAt: 0}}}
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}
	if len(prov.rented) != 1 {
		t.Fatalf("expected 1 rented, got %d", len(prov.rented))
	}
	now = now.Add(31 * time.Minute)
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
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
	if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile rent: %v", err)
	}
	launch := prov.rented[0]
	vllm := startTLSServerWithCert(t, launch.CertPEM, launch.KeyPEM, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	if err := ex.Reconcile(ctx, postedWork, testExaminerBudget()); err != nil {
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
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile open: %v", err)
	}
	prov.mu.Lock()
	inst := prov.insts["ex-1"]
	inst.Created = time.Time{}
	prov.insts["ex-1"] = inst
	prov.mu.Unlock()

	now = now.Add(31 * time.Minute)
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
		t.Fatalf("expected error when EndExaminer fails on timeout")
	}
	if len(prov.rented) != 1 {
		t.Fatalf("transient EndExaminer failure must not rent duplicate VM, rented=%d", len(prov.rented))
	}
	if st := readExaminerState(home); st == nil || st.InstanceID != "ex-1" {
		t.Fatalf("failed teardown must preserve ex-1 in examiner.json, got %+v", st)
	}

	prov.mu.Lock()
	delete(prov.endErr, "ex-1")
	prov.mu.Unlock()
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
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

// Review 11 P2 #7: two deployments sharing one Vast account must scope adoption and cleanup strictly
// to their own persistent owner ID so an idle controller never deletes another deployment's examiner.
func TestExaminerOwnerIsolationPreventsCrossDeploymentDeletion(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
	}
	ex1 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, t.TempDir(), prov, func() time.Time { return now }, nil, false)
	ex2 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, t.TempDir(), prov, func() time.Time { return now }, nil, false)

	// Controller 1 rents an examiner for an open round.
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00005", Entries: 2, ExamAt: 0}}}
	if err := ex1.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("ex1 Reconcile: %v", err)
	}
	if len(prov.insts) != 1 {
		t.Fatalf("expected 1 live examiner, got %d", len(prov.insts))
	}

	// Controller 2 has NO open work and reconciles -> MUST NOT touch Controller 1's active examiner!
	if err := ex2.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("ex2 Reconcile idle: %v", err)
	}
	if len(prov.ended) != 0 || len(prov.insts) != 1 {
		t.Fatalf("controller 2 deleted controller 1's examiner! ended=%v remaining=%d", prov.ended, len(prov.insts))
	}

	// Controller 2's EnforceTimeout also ignores Controller 1's expired instance.
	now = now.Add(2 * time.Hour)
	if err := ex2.EnforceTimeout(context.Background()); err != nil {
		t.Fatalf("ex2 EnforceTimeout: %v", err)
	}
	if len(prov.ended) != 0 || len(prov.insts) != 1 {
		t.Fatalf("controller 2 timed out controller 1's examiner! ended=%v", prov.ended)
	}
}

// Review 11 P2 #8: when deleting an owned live orphan fails transiently on an open round, Reconcile
// must keep the deletion pending and refuse to rent a replacement until removal is confirmed.
func TestExaminerFailedOrphanDeletionBlocksReplacementRental(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	owner := loadOrCreateExaminerOwner(home)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
		insts: map[string]ExaminerInstance{
			"orphan-1": {ID: "orphan-1", Label: examinerLabelPrefix + "-" + owner + "-s3-o00001-11223344", State: "running", Price: 0.62, Created: now.Add(-5 * time.Minute)},
		},
		endErr: map[string]error{"orphan-1": errors.New("transient Vast 500")},
	}
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00006", Entries: 1, ExamAt: 0}}}

	// Transient error deleting orphan -> Reconcile MUST fail and MUST NOT rent a 2nd examiner!
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
		t.Fatalf("expected error when live orphan deletion fails")
	}
	if len(prov.rented) != 0 || len(prov.insts) != 1 {
		t.Fatalf("must not rent replacement while orphan-1 is still live: rented=%d insts=%d", len(prov.rented), len(prov.insts))
	}

	// Next tick deletion succeeds -> orphan destroyed AND 1 replacement rented cleanly.
	delete(prov.endErr, "orphan-1")
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile after orphan delete recovers: %v", err)
	}
	if len(prov.ended) != 1 || prov.ended[0] != "orphan-1" || len(prov.rented) != 1 || len(prov.insts) != 1 {
		t.Fatalf("after recovery: ended=%v rented=%d insts=%d", prov.ended, len(prov.rented), len(prov.insts))
	}
}

// Review 11 P1 #6: a Stet /admin/work outage (503) must NOT disable --examiner-timeout enforcement,
// and the Vast launch script must embed TLS flags + a container-side self-destruct TTL watchdog.
func TestExaminerEnforceTimeoutDuringStetOutageAndSelfDestructPayload(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
	}
	home := t.TempDir()
	var spent []podRecord
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, func() time.Time { return now }, nil, false)
	ex.onSpent = func(rec podRecord) { spent = append(spent, rec) }

	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00007", Entries: 1, ExamAt: 0}}}
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile rent: %v", err)
	}
	// Verify container-side onstart script embeds TLS flags and Vast self-destruct TTL watchdog.
	launch := prov.rented[0]
	script := examinerOnstartScript(launch, launch.Model+"@"+launch.Revision)
	for _, want := range []string{"--ssl-certfile", "--ssl-keyfile", "STET_EXAMINER_TTL_SECONDS", "CONTAINER_API_KEY", "console.vast.ai/api/v0/instances/"} {
		if !strings.Contains(script, want) {
			t.Fatalf("onstart script missing %q:\n%s", want, script)
		}
	}

	// Simulate Stet /admin/work returning 503 for 2 hours: EnforceTimeout destroys ex-1 independently!
	stet503 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer stet503.Close()
	op, _ := newKey("op")
	f := &Fleet{
		o:        &Options{Home: home, Every: time.Minute},
		st:       newStet(stet503.URL),
		op:       op,
		out:      io.Discard,
		now:      func() time.Time { return now },
		examiner: ex,
		fixed:    true,
	}
	now = now.Add(2 * time.Hour)
	if err := tick(context.Background(), f); err == nil {
		t.Fatalf("expected tick to report Stet 503 error")
	}
	if len(prov.ended) != 1 || prov.ended[0] != "ex-1" {
		t.Fatalf("Stet 503 outage must still enforce hard 45m timeout; ended=%v", prov.ended)
	}
	if readExaminerState(home) != nil || len(spent) != 1 {
		t.Fatalf("expected state cleared and spend recorded after outage timeout; state=%v spent=%d", readExaminerState(home), len(spent))
	}
}

type stubWorkerMarket struct {
	mu      sync.Mutex
	name    string
	offers  []Offer
	rentals []Rental
	nextID  int
}

func (m *stubWorkerMarket) Name() string { return m.name }
func (m *stubWorkerMarket) Caps() Caps   { return Caps{Rents: true, SelfEnd: true, Curators: true} }
func (m *stubWorkerMarket) Offers(ctx context.Context, n Need) ([]Offer, error) {
	return m.offers, nil
}
func (m *stubWorkerMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nextID++
	r := Rental{
		Market: m.name, ID: "wk-" + string(rune('0'+m.nextID)), Name: l.Name,
		Card: o.Card, Price: o.Price, State: "running", Status: "running", Created: time.Now(),
	}
	m.rentals = append(m.rentals, r)
	return &r, nil
}
func (m *stubWorkerMarket) Rentals(ctx context.Context) ([]Rental, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Rental(nil), m.rentals...), nil
}
func (m *stubWorkerMarket) End(ctx context.Context, id string) error { return nil }
func (m *stubWorkerMarket) Check(ctx context.Context) (Account, error) {
	return Account{Balance: 100}, nil
}

// Review 11 P1 #5: examiner rentals must count against --max, --max-hourly, status, and daily spend
// from launch, so --max 6 --max-hourly 4 with 12 curators waiting never rents 7 GPUs or > $4.00/h.
func TestExaminerRespectsFleetMaxWorkersAndMaxHourlyInSingleTick(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	op, _ := newKey("op")

	// Fake Stet server: 1 open round with 12 curators waiting and ExamAt == 0.
	stetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/v1/admin/work":
			_ = json.NewEncoder(w).Encode(Work{
				Rounds: []WorkRound{{ID: "s3-o00008", Entries: 12, RunsWanted: 12, ExamAt: 0}},
			})
		case r.URL.Path == "/api/v1/season":
			_ = json.NewEncoder(w).Encode(map[string]any{"site": "https://example.invalid"})
		case r.URL.Path == "/api/v1/admin/workers" && r.Method == "GET":
			_ = json.NewEncoder(w).Encode(map[string]any{"workers": []any{}})
		case r.URL.Path == "/api/v1/admin/workers" && r.Method == "POST":
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
		case r.URL.Path == "/api/v1/accounts":
			_ = json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{"anchor": true, "worker": true}})
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{})
		}
	}))
	defer stetSrv.Close()

	// Fake worker market offering 24 GB GPUs at $0.57/h (matching Review 11 Finding 5's exact repro).
	workerMarket := &stubWorkerMarket{
		name: "runpod",
		offers: []Offer{{
			Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090",
			Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure",
		}},
	}
	exProv := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, func() time.Time { return now }, nil, false)

	var keys []*Key
	for i := 1; i <= 8; i++ {
		k, _ := newKey("stet-rp-0" + string(rune('0'+i)))
		keys = append(keys, k)
	}
	o := &Options{
		Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
		MinCuratorGPUMemory: 24, MinGPUMemory: 12,
		Policy: Policy{MaxWorkers: 6, MaxHourly: 4.00, Edits: true, PerWorker: 6},
	}
	f := &Fleet{
		o: o, st: newStet(stetSrv.URL), op: op, out: io.Discard,
		now: func() time.Time { return now }, fixed: true,
		markets: []Market{workerMarket}, keys: keys, examiner: ex,
	}

	if err := tick(context.Background(), f); err != nil {
		t.Fatalf("tick: %v", err)
	}
	totalGPUs := len(workerMarket.rentals) + len(exProv.insts)
	totalHourly := float64(len(workerMarket.rentals))*0.57 + 0.62*float64(len(exProv.insts))
	if len(exProv.insts) != 1 || len(workerMarket.rentals) != 5 || totalGPUs != 6 {
		t.Fatalf("want 1 examiner + 5 workers = 6 GPUs (<= --max 6), got examiner=%d workers=%d total=%d",
			len(exProv.insts), len(workerMarket.rentals), totalGPUs)
	}
	if totalHourly > 4.00+1e-9 {
		t.Fatalf("total hourly $%.2f exceeded --max-hourly $4.00", totalHourly)
	}
	// View.Hourly and View.Today MUST include the active examiner immediately from launch!
	now = now.Add(30 * time.Minute)
	v, err := f.look(context.Background())
	if err != nil {
		t.Fatalf("look: %v", err)
	}
	if v.ExaminerLive != 1 || v.Hourly < totalHourly-1e-6 || v.Hourly > totalHourly+1e-6 {
		t.Fatalf("View.Hourly must include active examiner ($%.2f): got ExaminerLive=%d Hourly=$%.2f", totalHourly, v.ExaminerLive, v.Hourly)
	}
	// Accrued examiner spend for 30m at $0.62/h = $0.31 must be included in v.Today.
	if v.ExaminerAccrued < 0.30 || v.ExaminerAccrued > 0.32 || v.Today < v.ExaminerAccrued {
		t.Fatalf("View.Today must include active examiner spend ($0.31): ExaminerAccrued=$%.3f Today=$%.3f", v.ExaminerAccrued, v.Today)
	}

	// Verify examiner itself refuses to rent if hourly headroom would exceed --max-hourly.
	ex2 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, t.TempDir(), &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}, func() time.Time { return now }, nil, false)
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00009", Entries: 1, ExamAt: 0}}}
	if err := ex2.Reconcile(context.Background(), openWork, ExaminerBudget{LiveWorkers: 5, MaxWorkers: 6, HourlyNow: 3.50, MaxHourly: 4.00}); err != nil {
		t.Fatalf("Reconcile hourly cap: %v", err)
	}
	if live, _, _ := ex2.ActiveSnapshot(now); live != 0 {
		t.Fatalf("examiner must not rent when $3.50 + $0.62 > --max-hourly $4.00")
	}

	// Gate 4 P2 #5: cmdUp must count v.ExaminerLive against --max so `stet-cloud up 1` with
	// 5 workers + 1 active examiner at --max 6 rents 0 additional machines.
	beforeWorkers := len(workerMarket.rentals)
	if err := cmdUp(context.Background(), f, 1); err != nil {
		t.Fatalf("cmdUp at capacity with active examiner: %v", err)
	}
	if len(workerMarket.rentals) != beforeWorkers {
		t.Fatalf("cmdUp exceeded --max 6 (5 workers + 1 examiner): rented %d more", len(workerMarket.rentals)-beforeWorkers)
	}
}

// Gate 4 P1 #1, P2 #8, P2 #9: verify onstart cert/key materialization creates files with mode 0600,
// legacy examiner.json instances without ownerID are still cleaned up across upgrades, and failed
// deletion of a stopped owned orphan blocks renting a replacement.
func TestExaminerOnstart0600PermsLegacyUpgradeAndStoppedOrphanBlock(t *testing.T) {
	now := time.Date(2020, 6, 1, 12, 0, 0, 0, time.UTC)
	certPEM, keyPEM, err := generateExaminerCert(now, 45*time.Minute)
	if err != nil {
		t.Fatalf("generateExaminerCert: %v", err)
	}
	// 1. Execute the exact Python cert/key materialization snippet from examinerOnstartScript
	//    against a temp directory and assert both files are created with mode 0600!
	tmp := t.TempDir()
	script := examinerOnstartScript(ExaminerLaunch{
		Model: defaultExaminerModel, Revision: defaultExaminerRevision, APIKey: "k",
		CertPEM: certPEM, KeyPEM: keyPEM, TTLSeconds: 2700,
	}, defaultExaminerModel+"@"+defaultExaminerRevision)
	// 0. Validate shell syntax of the complete onstart script with sh -n.
	if out, err := exec.Command("sh", "-n", "-c", script).CombinedOutput(); err != nil {
		t.Fatalf("onstart script failed sh -n syntax check: %v\n%s\n%s", err, out, script)
	}
	// Extract the python3 cert writer lines (between python3 -c ' and closing ').
	startIdx := strings.Index(script, "python3 -c '")
	if startIdx < 0 {
		t.Fatalf("cert writer missing from onstart script")
	}
	rest := script[startIdx+len("python3 -c '"):]
	endIdx := strings.Index(rest, "' >>")
	if endIdx < 0 {
		t.Fatalf("unterminated python3 -c block in onstart script")
	}
	pyCode := strings.ReplaceAll(rest[:endIdx], "/tmp/", tmp+"/")
	cmd := exec.Command("python3", "-c", pyCode)
	cmd.Env = append(os.Environ(),
		"STET_EXAMINER_CERT_HEX="+hex.EncodeToString(certPEM),
		"STET_EXAMINER_KEY_HEX="+hex.EncodeToString(keyPEM),
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("cert materializer failed: %v\n%s", err, out)
	}
	for name, want := range map[string][]byte{"stet-examiner-cert.pem": certPEM, "stet-examiner-key.pem": keyPEM} {
		p := filepath.Join(tmp, name)
		fi, err := os.Stat(p)
		if err != nil {
			t.Fatalf("stat %s: %v", name, err)
		}
		if fi.Mode().Perm() != 0o600 {
			t.Fatalf("%s permissions = %04o, want 0600", name, fi.Mode().Perm())
		}
		got, err := os.ReadFile(p)
		if err != nil || string(got) != string(want) {
			t.Fatalf("%s content mismatch", name)
		}
	}

	// 2. Legacy examiner.json referencing an instance without ownerID in its label is still
	//    discovered and cleanly destroyed when idle (no leaked instance across upgrade).
	homeLegacy := t.TempDir()
	if err := writeExaminerState(homeLegacy, &examinerState{
		InstanceID: "legacy-1", RoundID: "s3-o00001", APIKey: "oldkey", Price: 0.62, Created: now,
	}); err != nil {
		t.Fatalf("writeExaminerState: %v", err)
	}
	provLegacy := &fakeExaminerProv{
		insts: map[string]ExaminerInstance{
			"legacy-1": {ID: "legacy-1", Label: "stet-examiner-s3-o00001-deadbeef", State: "running", Price: 0.62, Created: now},
		},
	}
	exLegacy := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute}, homeLegacy, provLegacy, func() time.Time { return now }, nil, false)
	if err := exLegacy.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile legacy idle: %v", err)
	}
	if len(provLegacy.ended) != 1 || provLegacy.ended[0] != "legacy-1" {
		t.Fatalf("legacy examiner.json instance must be cleaned up across upgrade, ended=%v", provLegacy.ended)
	}

	// 3. Failed deletion of a STOPPED owned orphan (still billing disk on Vast) blocks renting a replacement.
	homeStop := t.TempDir()
	owner := loadOrCreateExaminerOwner(homeStop)
	provStop := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-1", Card: "RTX 5090", Price: 0.62}},
		insts: map[string]ExaminerInstance{
			"stopped-1": {ID: "stopped-1", Label: examinerLabelPrefix + "-" + owner + "-s3-o00001-11223344", State: "stopped", Price: 0.62, Created: now},
		},
		endErr: map[string]error{"stopped-1": errors.New("transient 500")},
	}
	exStop := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute}, homeStop, provStop, func() time.Time { return now }, nil, false)
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00010", Entries: 1, ExamAt: 0}}}
	if err := exStop.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
		t.Fatalf("expected error when deleting stopped owned orphan fails")
	}
	if len(provStop.rented) != 0 {
		t.Fatalf("must not rent replacement while stopped orphan deletion failed, rented=%d", len(provStop.rented))
	}
}

// Review 12 P1 #4: explicit zero ceilings (--max 0 or --max-hourly 0) parsed via real CLI flags
// must prevent renting a paid examiner just as they prevent renting workers, and negative caps
// must be rejected at flag parsing.
func TestExaminerZeroCapsEnforcedFromRealCLIFlags(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	stetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/admin/work":
			_ = json.NewEncoder(w).Encode(Work{
				Rounds: []WorkRound{{ID: "s3-o00020", Entries: 1, RunsWanted: 1, ExamAt: 0}},
			})
		case "/api/v1/season":
			_ = json.NewEncoder(w).Encode(map[string]any{"site": "https://example.invalid"})
		case "/api/v1/admin/workers":
			_ = json.NewEncoder(w).Encode(map[string]any{"workers": []any{}})
		case "/api/v1/accounts":
			_ = json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{"anchor": true, "worker": true}})
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{})
		}
	}))
	defer stetSrv.Close()

	for _, tc := range []struct {
		name string
		args []string
	}{
		{"max_zero", []string{"--max", "0", "--max-hourly", "4"}},
		{"max_hourly_zero", []string{"--max", "6", "--max-hourly", "0"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			var o Options
			fs := flags("scale", &o)
			if _, err := parse(fs, append([]string{"--home", home, "--api", stetSrv.URL, "--examiner-addr", "127.0.0.1:0"}, tc.args...)); err != nil {
				t.Fatalf("parse %v: %v", tc.args, err)
			}
			exProv := &fakeExaminerProv{
				offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			}
			ex := newExaminer(o.Examiner, home, exProv, func() time.Time { return now }, nil, false)
			op, _ := newKey("op")
			k1, _ := newKey("stet-rp-01")
			workerMarket := &stubWorkerMarket{
				name:   "runpod",
				offers: []Offer{{Market: "runpod", ID: "gpu-1", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}},
			}
			f := &Fleet{
				o: &o, st: newStet(stetSrv.URL), op: op, out: io.Discard,
				now: func() time.Time { return now }, fixed: true,
				markets: []Market{workerMarket}, keys: []*Key{k1}, examiner: ex,
			}
			if err := tick(context.Background(), f); err != nil {
				t.Fatalf("tick: %v", err)
			}
			if len(exProv.rented) != 0 || len(workerMarket.rentals) != 0 {
				t.Fatalf("%s rented machines under explicit zero ceiling: examiners=%d workers=%d",
					tc.name, len(exProv.rented), len(workerMarket.rentals))
			}
		})
	}

	// Negative limits and zero --examiner-max-price must be rejected during flag parsing.
	for _, bad := range [][]string{{"--max", "-1"}, {"--max-hourly", "-0.5"}, {"--daily-budget", "-1"}, {"--examiner-max-price", "0"}, {"--examiner-max-price", "-1"}} {
		var o Options
		fs := flags("scale", &o)
		if _, err := parse(fs, bad); err == nil {
			t.Fatalf("expected parse to reject invalid flag %v", bad)
		}
	}
}

// Review 12 P2 #5: standalone reconcileExaminerOnce must serialize with cmdUp via f.lock so both
// cannot spend the last --max slot concurrently, and must defer new rentals when fleet.json or
// spent.jsonl cannot be read while still allowing idle teardown.
func TestStandaloneExaminerSerializesWithUpAndDefersOnAccountingError(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	home := t.TempDir()
	op, _ := newKey("op")
	k1, _ := newKey("stet-rp-01")

	examAt := int64(0)
	var mu sync.Mutex
	stetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		ea := examAt
		mu.Unlock()
		switch r.URL.Path {
		case "/api/v1/admin/work":
			_ = json.NewEncoder(w).Encode(Work{
				Rounds: []WorkRound{{ID: "s3-o00021", Entries: 1, RunsWanted: 1, ExamAt: ea}},
			})
		case "/api/v1/season":
			_ = json.NewEncoder(w).Encode(map[string]any{"site": "https://example.invalid"})
		case "/api/v1/admin/workers":
			_ = json.NewEncoder(w).Encode(map[string]any{"workers": []any{}})
		case "/api/v1/accounts":
			_ = json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{"anchor": true, "worker": true}})
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{})
		}
	}))
	defer stetSrv.Close()

	listEntered := make(chan struct{})
	releaseList := make(chan struct{})
	var listOnce sync.Once
	exProv := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		onList: func() {
			listOnce.Do(func() {
				close(listEntered)
				<-releaseList
			})
		},
	}
	workerMarket := &stubWorkerMarket{
		name:   "runpod",
		offers: []Offer{{Market: "runpod", ID: "gpu-1", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}},
	}

	makeFleet := func(ex *Examiner) *Fleet {
		return &Fleet{
			o: &Options{
				Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
				MinCuratorGPUMemory: 24, MinGPUMemory: 12,
				Policy: Policy{MaxWorkers: 1, MaxHourly: 4.00, Edits: true, PerWorker: 6},
			},
			st: newStet(stetSrv.URL), op: op, out: io.Discard,
			now: func() time.Time { return now }, fixed: true,
			markets: []Market{workerMarket}, keys: []*Key{k1}, examiner: ex,
		}
	}

	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, func() time.Time { return now }, nil, false)
	fStandalone := makeFleet(ex)
	fUp := makeFleet(nil) // separate CLI process (`stet-cloud up 1`) sharing --home

	errCh := make(chan error, 1)
	go func() {
		errCh <- reconcileExaminerOnce(context.Background(), fStandalone)
	}()
	<-listEntered

	upDone := make(chan error, 1)
	go func() {
		upDone <- cmdUp(context.Background(), fUp, 1)
	}()
	// Give cmdUp a moment to contend on f.lock, then release standalone reconciliation.
	time.Sleep(40 * time.Millisecond)
	close(releaseList)

	if err := <-errCh; err != nil {
		t.Fatalf("reconcileExaminerOnce: %v", err)
	}
	if err := <-upDone; err != nil {
		t.Fatalf("cmdUp: %v", err)
	}
	totalLive := len(exProv.insts) + len(workerMarket.rentals)
	if len(exProv.insts) != 1 || len(workerMarket.rentals) != 0 || totalLive != 1 {
		t.Fatalf("concurrent standalone examiner + up exceeded --max 1: examiners=%d workers=%d",
			len(exProv.insts), len(workerMarket.rentals))
	}

	// Now corrupt fleet.json -> teardown on ExamAt > 0 MUST still succeed, while a new rental on
	// an open round MUST be deferred with an accounting error (never treating unknown spend as 0).
	if err := os.WriteFile(filepath.Join(home, "fleet.json"), []byte("{not-json"), 0o600); err != nil {
		t.Fatalf("write corrupt fleet.json: %v", err)
	}
	mu.Lock()
	examAt = now.Unix()
	mu.Unlock()
	if err := reconcileExaminerOnce(context.Background(), fStandalone); err != nil {
		t.Fatalf("idle teardown should succeed even when fleet.json is corrupt: %v", err)
	}
	if len(exProv.insts) != 0 {
		t.Fatalf("expected active examiner torn down on ExamAt > 0, remaining=%d", len(exProv.insts))
	}
	mu.Lock()
	examAt = 0
	mu.Unlock()
	if err := reconcileExaminerOnce(context.Background(), fStandalone); err == nil {
		t.Fatalf("expected accounting error to defer new examiner rental when fleet.json is corrupt")
	}
	if len(exProv.insts) != 0 {
		t.Fatalf("must not rent new examiner when fleet.json is unreadable, got %d", len(exProv.insts))
	}
}

// Review 12 P2 #6: when an examiner self-terminates (e.g. via its container watchdog) and disappears
// from Vast before restart/reconciliation, its accrued spend must be written idempotently to
// spent.jsonl (bounded by Timeout) rather than vanishing to $0.000.
func TestExaminerSelfTerminatedRecordsBoundedIdempotentDailySpend(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
	home := t.TempDir()
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	appendSpent := func(rec podRecord) {
		line, err := json.Marshal(rec)
		if err != nil {
			t.Fatalf("marshal spent: %v", err)
		}
		fh, err := os.OpenFile(filepath.Join(home, "spent.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			t.Fatalf("open spent.jsonl: %v", err)
		}
		_, _ = fh.Write(append(line, '\n'))
		_ = fh.Close()
	}

	ex1 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	ex1.onSpent = appendSpent
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00022", Entries: 1, ExamAt: 0}}}
	if err := ex1.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile rent: %v", err)
	}

	// Simulate 45m runtime followed by container self-destruct AND a 2-hour controller outage.
	clk.Advance(2 * time.Hour)
	prov.mu.Lock()
	delete(prov.insts, "ex-1")
	prov.mu.Unlock()

	// Fresh restarted controller (new Examiner loading examiner.json from disk) reconciles idle.
	// Even BEFORE Reconcile runs, ActiveSnapshot must cap accrued spend at Timeout ($0.465, not 2h $1.24).
	ex2 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	ex2.onSpent = appendSpent
	if _, _, snapAccrued := ex2.ActiveSnapshot(clk.Now()); snapAccrued < 0.464 || snapAccrued > 0.466 {
		t.Fatalf("ActiveSnapshot after 2h outage must cap accrued spend at 45m timeout ($0.465), got $%.4f", snapAccrued)
	}
	if err := ex2.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile after self-destruct: %v", err)
	}
	if readExaminerState(home) != nil {
		t.Fatalf("expected examiner.json cleared after missing-instance reconciliation")
	}

	// Verify spent.jsonl recorded the bounded 45m charge ($0.465 = 0.62 * 0.75), NOT $0.000 and NOT 2h ($1.24).
	f := &Fleet{o: &Options{Home: home}, now: clk.Now}
	spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
	if err != nil {
		t.Fatalf("f.spent: %v", err)
	}
	if spentToday < 0.464 || spentToday > 0.466 {
		t.Fatalf("expected bounded 45m daily spend $0.465 after self-termination, got $%.4f", spentToday)
	}

	// Re-calling EnforceTimeout / recordMissingSaved for the same instance ID must be idempotent.
	ex2.recordMissingSaved(examinerState{InstanceID: "ex-1", RoundID: "s3-o00022", Price: 0.62, Created: clk.Now().Add(-2 * time.Hour)}, "duplicate")
	spentAgain, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
	if err != nil {
		t.Fatalf("f.spent again: %v", err)
	}
	if spentAgain < 0.464 || spentAgain > 0.466 {
		t.Fatalf("expected idempotent spend $0.465 (no double-charge), got $%.4f", spentAgain)
	}
}

// Review 12 P2 #7: when remaining fleet hourly headroom ($0.60/h) is lower than --examiner-max-price
// ($1.50/h) and the fastest tier (>=1500 Mbps) only has a $1.20/h card while the >=500 Mbps tier has
// an affordable $0.50/h card, Reconcile must pass the effective headroom into ExaminerOffers so Vast
// falls through to the affordable slower tier and rents the $0.50/h 5090.
func TestExaminerOfferSearchFallsThroughToAffordableSlowerTier(t *testing.T) {
	now := time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC)
	var rentedOfferID string
	var seenQueries []float64
	vastSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/api/v1/instances/"):
			_ = json.NewEncoder(w).Encode(map[string]any{"instances": []any{}})
		case r.Method == "POST" && r.URL.Path == "/api/v0/bundles/":
			var q map[string]any
			_ = json.NewDecoder(r.Body).Decode(&q)
			minDown := 0.0
			if m, ok := q["inet_down"].(map[string]any); ok {
				minDown, _ = m["gte"].(float64)
			}
			maxDPH := 999.0
			if m, ok := q["dph_total"].(map[string]any); ok {
				maxDPH, _ = m["lte"].(float64)
			}
			seenQueries = append(seenQueries, minDown)
			var offers []vastOffer
			all := []vastOffer{
				{ID: 15001, GPUName: "RTX 5090", NumGPUs: 1, DPHTotal: 1.20, DPHBase: 1.20, InetDown: 2000, CUDAMaxGood: 13.3},
				{ID: 5001, GPUName: "RTX 5090", NumGPUs: 1, DPHTotal: 0.50, DPHBase: 0.50, InetDown: 600, CUDAMaxGood: 13.3},
			}
			for _, o := range all {
				if o.InetDown >= minDown && o.DPHTotal <= maxDPH+1e-9 {
					offers = append(offers, o)
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"offers": offers})
		case r.Method == "PUT" && strings.HasPrefix(r.URL.Path, "/api/v0/asks/"):
			rentedOfferID = strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/v0/asks/"), "/")
			_ = json.NewEncoder(w).Encode(map[string]any{"success": true, "new_contract": 77777})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer vastSrv.Close()

	vm := newVast(vastSrv.URL, "test-vast-key")
	home := t.TempDir()
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, vm, func() time.Time { return now }, nil, false)
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00023", Entries: 1, ExamAt: 0}}}
	// Fleet has $0.60/h headroom ($4.00 cap - $3.40 live).
	budget := ExaminerBudget{LiveWorkers: 4, MaxWorkers: 6, HourlyNow: 3.40, MaxHourly: 4.00}
	if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}
	if rentedOfferID != "5001" {
		t.Fatalf("expected fallthrough to $0.50/h 500 Mbps offer 5001 (queries=%v), got %q", seenQueries, rentedOfferID)
	}
}
