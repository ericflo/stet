package main

import (
	"context"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
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
	listErr     error
	onList      func()
	onRent      func()
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
	hook := p.onRent
	p.mu.Unlock()
	if hook != nil {
		hook()
	}
	p.mu.Lock()
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
	if p.listErr != nil {
		return nil, p.listErr
	}
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
	ex.onSpent = func(rec podRecord) error { spent = append(spent, rec); return nil }

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
	ex.onSpent = func(rec podRecord) error { spent = append(spent, rec); return nil }

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
	ex.onSpent = func(rec podRecord) error { spent = append(spent, rec); return nil }

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
	if live, _, _, err := ex2.ActiveSnapshot(now); err != nil || live != 0 {
		t.Fatalf("examiner must not rent when $3.50 + $0.62 > --max-hourly $4.00: live=%d err=%v", live, err)
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
	appendSpent := func(rec podRecord) error {
		return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec)
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
	// Because the end time is unconfirmed before provider reconciliation (a slow image pull or
	// broken startup could have kept the VM alive past 45m), ActiveSnapshot keeps accruing through
	// now ($1.24 for 2h at $0.62/h) rather than optimistically capping at $0.465 before confirmation.
	ex2 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	ex2.onSpent = appendSpent
	if _, _, snapAccrued, err := ex2.ActiveSnapshot(clk.Now()); err != nil || snapAccrued < 1.239 || snapAccrued > 1.241 {
		t.Fatalf("ActiveSnapshot after 2h outage before provider confirmation must accrue full 2h ($1.24), got $%.4f err=%v", snapAccrued, err)
	}
	if err := ex2.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile after self-destruct: %v", err)
	}
	if readExaminerState(home) != nil {
		t.Fatalf("expected examiner.json cleared after missing-instance reconciliation")
	}

	// Verify spent.jsonl recorded the conservative 2h charge ($1.24 = 0.62 * 2), NOT $0.000.
	f := &Fleet{o: &Options{Home: home}, now: clk.Now}
	spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
	if err != nil {
		t.Fatalf("f.spent: %v", err)
	}
	if spentToday < 1.239 || spentToday > 1.241 {
		t.Fatalf("expected conservative 2h daily spend $1.24 after unconfirmed disappearance, got $%.4f", spentToday)
	}

	// Re-calling EnforceTimeout / recordMissingSaved for the same instance ID must be idempotent.
	ex2.recordMissingSaved(examinerState{InstanceID: "ex-1", RoundID: "s3-o00022", Price: 0.62, Created: clk.Now().Add(-2 * time.Hour)}, "duplicate")
	spentAgain, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
	if err != nil {
		t.Fatalf("f.spent again: %v", err)
	}
	if spentAgain < 1.239 || spentAgain > 1.241 {
		t.Fatalf("expected idempotent spend $1.24 (no double-charge), got $%.4f", spentAgain)
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

// Review 13 P2 #8: a temporary spent.jsonl write failure must propagate an error, retain a
// retryable terminal spend record in examiner.json (still counted in daily accrual), and flush
// the exact $0.31 charge once spent.jsonl becomes writable again — without double-charging.
func TestExaminerTemporarySpendLogWriteErrorRetainsAndFlushesCharge(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
	home := t.TempDir()
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	spentPath := filepath.Join(home, "spent.jsonl")
	prodWriter := func(rec podRecord) error {
		return appendExaminerSpend(spentPath, rec)
	}

	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	ex.onSpent = prodWriter
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00024", Entries: 1, ExamAt: 0}}}
	if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile rent: %v", err)
	}

	// Let the $0.62/h examiner run 30 minutes ($0.31) and disappear, while spent.jsonl is unwritable.
	clk.Advance(30 * time.Minute)
	prov.mu.Lock()
	delete(prov.insts, "ex-1")
	prov.mu.Unlock()
	if err := os.Mkdir(spentPath, 0o700); err != nil {
		t.Fatalf("block spent.jsonl with directory: %v", err)
	}

	// Reconciliation MUST return an error and retain the pending $0.31 charge in examiner.json!
	if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
		t.Fatalf("expected Reconcile error when spent.jsonl is unwritable")
	}
	st := readExaminerState(home)
	if st == nil || st.InstanceID != "" || len(st.PendingSpends) != 1 || st.PendingSpends[0].ID != "ex-1" {
		t.Fatalf("expected examiner.json to retain pending spend for ex-1 with empty InstanceID, got %+v", st)
	}
	// Daily accrual must STILL reflect $0.31 while pending flush (both in-memory and from disk).
	if live, hourly, acc, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 0 || hourly != 0 || acc < 0.309 || acc > 0.311 {
		t.Fatalf("ActiveSnapshot while pending flush: live=%d hourly=%.2f accrued=%.4f err=%v, want 0/0/$0.31", live, hourly, acc, err)
	}
	if live, hourly, acc, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 0 || hourly != 0 || acc < 0.309 || acc > 0.311 {
		t.Fatalf("activeExaminerFromDisk while pending flush: live=%d hourly=%.2f accrued=%.4f err=%v, want 0/0/$0.31", live, hourly, acc, err)
	}

	// Gate 4 P2 #2 & #3: even when a second stray fails deletion while saved ex-1 ends and
	// spent.jsonl is unwritable, saved state clears immediately (no double-count, no End inflation).
	prov2 := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	home2 := t.TempDir()
	spentPath2 := filepath.Join(home2, "spent.jsonl")
	exP := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home2, prov2, clk.Now, nil, false)
	exP.onSpent = func(rec podRecord) error { return appendExaminerSpend(spentPath2, rec) }
	if err := exP.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile rent exP: %v", err)
	}
	owner2 := exP.ownerID()
	prov2.mu.Lock()
	ex1Inst := prov2.insts["ex-1"]
	ex1Inst.Created = clk.Now()
	prov2.insts["ex-1"] = ex1Inst
	prov2.insts["stray-2"] = ExaminerInstance{ID: "stray-2", Label: examinerLabelPrefix + "-" + owner2 + "-s3-o00024-deadbeef", State: "running", Price: 0.62, Created: clk.Now()}
	prov2.endErr = map[string]error{"stray-2": errors.New("transient Vast 500")}
	prov2.mu.Unlock()
	clk.Advance(30 * time.Minute)
	if err := os.Mkdir(spentPath2, 0o700); err != nil {
		t.Fatalf("block spentPath2: %v", err)
	}
	if err := exP.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
		t.Fatalf("expected error while stray-2 fails deletion")
	}
	// ex-1 was confirmed ended -> InstanceID must be cleared so ex-1 is NOT double-counted, while
	// still-live unresolved stray-2 ($0.62/h, up 30m = $0.31) IS counted -> live=1, hourly=$0.62,
	// accrued=$0.62 ($0.31 pending ex-1 + $0.31 live stray-2, NOT $0.93).
	if live, hourly, acc, err := exP.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 || acc < 0.619 || acc > 0.621 {
		t.Fatalf("expected live stray-2 + single-counted pending ex-1: live=%d hourly=%.2f acc=%.4f err=%v", live, hourly, acc, err)
	}
	// Advance clock another 10m and re-reconcile: ex-1's pending End must NOT inflate past 30m ($0.31).
	clk.Advance(10 * time.Minute)
	prov2.mu.Lock()
	delete(prov2.endErr, "stray-2")
	prov2.mu.Unlock()
	if err := exP.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
		t.Fatalf("expected spend log error while spentPath2 still blocked")
	}
	if err := os.Remove(spentPath2); err != nil {
		t.Fatalf("unblock spentPath2: %v", err)
	}
	if err := exP.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("flush spentPath2: %v", err)
	}
	raw2, err := os.ReadFile(spentPath2)
	if err != nil {
		t.Fatalf("read spentPath2: %v", err)
	}
	foundEx1 := false
	for _, ln := range strings.Split(strings.TrimSpace(string(raw2)), "\n") {
		var r podRecord
		if json.Unmarshal([]byte(ln), &r) == nil && r.ID == "ex-1" {
			foundEx1 = true
			if got := r.End.Sub(r.Start); got != 30*time.Minute {
				t.Fatalf("ex-1 terminal duration inflated: got %s, want 30m", got)
			}
		}
	}
	if !foundEx1 {
		t.Fatalf("expected ex-1 record in spentPath2, got %s", raw2)
	}

	// Make spent.jsonl writable again and reconcile with a fresh restarted Examiner!
	if err := os.Remove(spentPath); err != nil {
		t.Fatalf("unblock spent.jsonl: %v", err)
	}
	ex2 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	ex2.onSpent = prodWriter
	if err := ex2.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("Reconcile after spent.jsonl recovered: %v", err)
	}
	if readExaminerState(home) != nil {
		t.Fatalf("expected examiner.json removed once pending spend flushed")
	}
	f := &Fleet{o: &Options{Home: home}, now: clk.Now}
	spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
	if err != nil {
		t.Fatalf("f.spent: %v", err)
	}
	if spentToday < 0.309 || spentToday > 0.311 {
		t.Fatalf("expected recovered $0.31 daily spend, got $%.4f", spentToday)
	}
	// Idempotent on repeat reconciliation.
	if err := ex2.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
		t.Fatalf("third Reconcile: %v", err)
	}
	spentAgain, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
	if err != nil || spentAgain < 0.309 || spentAgain > 0.311 {
		t.Fatalf("expected idempotent $0.31 daily spend, got $%.4f err=%v", spentAgain, err)
	}
}

// Review 13 P2 #9: when `scale` tick() fetches /admin/work cleanly with ExamAt > 0 and then
// fails reading the worker ledger (fleet.json), tick() MUST still tear down the active examiner
// instead of leaving it running until TTL, while still blocking new rentals when ExamAt == 0.
func TestScaleTickTearsDownExaminerWhenExamPostedDespiteUnreadableLedger(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
	home := t.TempDir()
	var mu sync.Mutex
	examAt := int64(0)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		ea := examAt
		mu.Unlock()
		if r.URL.Path == "/api/v1/admin/work" {
			_ = json.NewEncoder(w).Encode(Work{
				Rounds: []WorkRound{{ID: "s3-o00025", Entries: 1, ExamAt: ea, DrawnAt: clk.Now().Unix()}},
			})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	exProv := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
	// Rent an active examiner while ExamAt == 0.
	if err := ex.Reconcile(context.Background(), &Work{Rounds: []WorkRound{{ID: "s3-o00025", Entries: 1, ExamAt: 0}}}, testExaminerBudget()); err != nil {
		t.Fatalf("initial examiner rent: %v", err)
	}
	if len(exProv.insts) != 1 {
		t.Fatalf("expected 1 active examiner, got %d", len(exProv.insts))
	}

	// Corrupt fleet.json so f.readLedger() fails inside f.look().
	if err := os.WriteFile(filepath.Join(home, "fleet.json"), []byte("{corrupt ledger"), 0o600); err != nil {
		t.Fatalf("write corrupt fleet.json: %v", err)
	}
	// Mark exam posted on /admin/work.
	mu.Lock()
	examAt = clk.Now().Unix()
	mu.Unlock()

	op, err := newKey("stet")
	if err != nil {
		t.Fatalf("newKey: %v", err)
	}
	f := &Fleet{
		o: &Options{
			Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
			MinCuratorGPUMemory: 24, MinGPUMemory: 12,
			Policy: Policy{MaxWorkers: 4, MaxHourly: 4.0},
		},
		st:       newStet(srv.URL),
		op:       op,
		examiner: ex,
		now:      clk.Now,
		out:      io.Discard,
		fixed:    true,
	}

	// tick() MUST return the ledger error AND immediately destroy the active examiner!
	if err := tick(context.Background(), f); err == nil {
		t.Fatalf("expected tick() to return fleet.json ledger error")
	}
	if len(exProv.ended) != 1 || len(exProv.insts) != 0 {
		t.Fatalf("tick() with ExamAt > 0 and corrupt fleet.json MUST destroy examiner immediately: ended=%v remaining=%d", exProv.ended, len(exProv.insts))
	}
	if readExaminerState(home) != nil {
		t.Fatalf("expected examiner.json cleared after teardown")
	}

	// Now flip ExamAt back to 0 while fleet.json is still corrupt: tick() MUST NOT rent a new examiner!
	mu.Lock()
	examAt = 0
	mu.Unlock()
	if err := tick(context.Background(), f); err == nil {
		t.Fatalf("expected tick() error when fleet.json is still corrupt")
	}
	if len(exProv.insts) != 0 {
		t.Fatalf("tick() with ExamAt == 0 and corrupt fleet.json must NOT rent a new examiner, got %d", len(exProv.insts))
	}
}

// Review 14 P1 #4: when an owned orphan examiner fails EndExaminer on an ExamAt > 0 tick,
// tick() must count the live unresolved orphan against --max and --max-hourly (and defer new
// worker launches on examiner reconciliation errors) while still allowing idle worker teardown;
// once the orphan is later destroyed, the next tick launches the worker cleanly.
func TestFailedOrphanTeardownBlocksWorkerLaunchInSameTick(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
	home := t.TempDir()
	op, _ := newKey("op")
	k1, _ := newKey("stet-rp-01")

	stetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/admin/work":
			_ = json.NewEncoder(w).Encode(Work{
				Rounds: []WorkRound{{ID: "s3-o00030", Entries: 1, RunsWanted: 1, ExamAt: clk.Now().Unix(), DrawnAt: clk.Now().Unix()}},
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

	owner := loadOrCreateExaminerOwner(home)
	exProv := &fakeExaminerProv{
		insts: map[string]ExaminerInstance{
			"orphan-1": {
				ID: "orphan-1", Label: examinerLabelPrefix + "-" + owner + "-s3-o00029-deadbeef",
				State: "running", Price: 0.62, Created: clk.Now().Add(-10 * time.Minute),
			},
		},
		endErr: map[string]error{"orphan-1": errors.New("transient Vast 500")},
	}
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
	workerMarket := &stubWorkerMarket{
		name: "runpod",
		offers: []Offer{{
			Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090",
			Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure",
		}},
	}
	f := &Fleet{
		o: &Options{
			Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
			MinCuratorGPUMemory: 24, MinGPUMemory: 12,
			Policy: Policy{MaxWorkers: 1, MaxHourly: 1.00, Edits: true, PerWorker: 6},
		},
		st: newStet(stetSrv.URL), op: op, out: io.Discard,
		now: clk.Now, fixed: true, markets: []Market{workerMarket}, keys: []*Key{k1}, examiner: ex,
	}

	// Tick 1: orphan-1 teardown fails -> MUST rent 0 workers (respecting --max 1 and --max-hourly $1.00)!
	if err := tick(context.Background(), f); err != nil {
		t.Fatalf("tick 1: %v", err)
	}
	if len(workerMarket.rentals) != 0 {
		t.Fatalf("failed orphan teardown must block worker launch in same tick: got %d workers", len(workerMarket.rentals))
	}
	if live, hourly, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
		t.Fatalf("ActiveSnapshot must count live unresolved orphan: live=%d hourly=%.2f err=%v", live, hourly, err)
	}
	// Gate 4 P1 #2: unresolved orphan MUST be persisted to examiner.json so a separate CLI process
	// (`stet-cloud up 1`, with fUp.examiner == nil) ALSO sees live=1 and rents 0 workers!
	if live, hourly, _, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
		t.Fatalf("activeExaminerFromDisk must count persisted unresolved orphan: live=%d hourly=%.2f err=%v", live, hourly, err)
	}
	fUp := &Fleet{
		o: &Options{
			Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
			MinCuratorGPUMemory: 24, MinGPUMemory: 12,
			Policy: Policy{MaxWorkers: 1, MaxHourly: 1.00, Edits: true, PerWorker: 6},
		},
		st: newStet(stetSrv.URL), op: op, out: io.Discard,
		now: clk.Now, fixed: true, markets: []Market{workerMarket}, keys: []*Key{k1}, examiner: nil,
	}
	if err := cmdUp(context.Background(), fUp, 1); err != nil {
		t.Fatalf("cmdUp with persisted unresolved orphan: %v", err)
	}
	if len(workerMarket.rentals) != 0 {
		t.Fatalf("separate cmdUp process must NOT rent worker while unresolved orphan is persisted: got %d", len(workerMarket.rentals))
	}

	// Tick 2: orphan-1 teardown succeeds -> unresolved clears on disk and worker launches cleanly!
	exProv.mu.Lock()
	delete(exProv.endErr, "orphan-1")
	exProv.mu.Unlock()
	if err := tick(context.Background(), f); err != nil {
		t.Fatalf("tick 2: %v", err)
	}
	if len(exProv.insts) != 0 || len(workerMarket.rentals) != 1 {
		t.Fatalf("after orphan cleared, want 0 examiners and 1 worker: examiners=%d workers=%d", len(exProv.insts), len(workerMarket.rentals))
	}
	if readExaminerState(home) != nil {
		t.Fatalf("expected examiner.json cleared once unresolved orphan is destroyed")
	}

	// Gate 4 P2 #3: failed teardown of a timed-out saved instance must NOT skip ending or tracking
	// subsequent owned strays in the same Reconcile pass.
	home2 := t.TempDir()
	owner2 := loadOrCreateExaminerOwner(home2)
	cert2, _, err := generateExaminerCert(clk.Now().Add(-2*time.Hour), 45*time.Minute)
	if err != nil {
		t.Fatalf("generateExaminerCert: %v", err)
	}
	if err := writeExaminerState(home2, &examinerState{
		InstanceID: "saved-old", RoundID: "s3-o00030", APIKey: "k2", CertPEM: string(cert2),
		Price: 0.62, Created: clk.Now().Add(-2 * time.Hour),
	}); err != nil {
		t.Fatalf("writeExaminerState: %v", err)
	}
	prov2 := &fakeExaminerProv{
		insts: map[string]ExaminerInstance{
			"saved-old": {ID: "saved-old", Label: examinerLabelPrefix + "-" + owner2 + "-s3-o00030-11111111", State: "running", Price: 0.62, Created: clk.Now().Add(-2 * time.Hour)},
			"stray-ok":  {ID: "stray-ok", Label: examinerLabelPrefix + "-" + owner2 + "-s3-o00029-22222222", State: "running", Price: 0.62, Created: clk.Now().Add(-10 * time.Minute)},
			"stray-err": {ID: "stray-err", Label: examinerLabelPrefix + "-" + owner2 + "-s3-o00028-33333333", State: "running", Price: 0.62, Created: clk.Now().Add(-10 * time.Minute)},
		},
		endErr: map[string]error{
			"saved-old": errors.New("Vast 500 on saved-old"),
			"stray-err": errors.New("Vast 500 on stray-err"),
		},
	}
	ex2 := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home2, prov2, clk.Now, nil, false)
	if err := ex2.Reconcile(context.Background(), &Work{Rounds: []WorkRound{{ID: "s3-o00030", Entries: 1, ExamAt: 0}}}, testExaminerBudget()); err == nil {
		t.Fatalf("expected error when timed-out saved-old and stray-err fail teardown")
	}
	// stray-ok MUST still have been ended despite saved-old failing, AND BOTH saved-old + stray-err
	// MUST be counted (live=2, hourly=$1.24) both in memory and on disk!
	if len(prov2.ended) != 1 || prov2.ended[0] != "stray-ok" {
		t.Fatalf("stray-ok must still be ended when saved-old fails timeout teardown: ended=%v", prov2.ended)
	}
	if live, hourly, _, err := ex2.ActiveSnapshot(clk.Now()); err != nil || live != 2 || hourly < 1.239 || hourly > 1.241 {
		t.Fatalf("ActiveSnapshot must count both failed timed-out saved-old and stray-err: live=%d hourly=%.2f err=%v", live, hourly, err)
	}
	if live, hourly, _, err := activeExaminerFromDisk(home2, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 2 || hourly < 1.239 || hourly > 1.241 {
		t.Fatalf("activeExaminerFromDisk must count both failed timed-out saved-old and stray-err: live=%d hourly=%.2f err=%v", live, hourly, err)
	}
}

// Review 14 P1 #5: examiner.json persistence failures must never silently allow renting an
// unrecorded VM or let a separate CLI process (`stet-cloud up 1`) see a free slot. Covers:
//   - directory blocker / unreadable examiner.json before rent -> 0 VMs rented AND cmdUp fails;
//   - corrupt examiner.json -> cmdUp fails and rents 0 workers;
//   - post-rental persist failure -> unwinds immediately when EndExaminer succeeds, and retains
//     disk PendingRent / in-memory reservation when unwind also fails;
//   - PendingRent reservation written before crash -> blocks cmdUp AND is adopted on recovery.
func TestExaminerStatePersistenceFailureBlocksRentAndSeparateCLIUp(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
	op, _ := newKey("op")
	k1, _ := newKey("stet-rp-01")
	stetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/admin/work":
			_ = json.NewEncoder(w).Encode(Work{
				Rounds: []WorkRound{{ID: "s3-o00031", Entries: 1, RunsWanted: 1, ExamAt: 0}},
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

	makeFleet := func(home string, ex *Examiner, wm Market) *Fleet {
		return &Fleet{
			o: &Options{
				Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
				MinCuratorGPUMemory: 24, MinGPUMemory: 12,
				Policy: Policy{MaxWorkers: 1, MaxHourly: 1.00, Edits: true, PerWorker: 6},
			},
			st: newStet(stetSrv.URL), op: op, out: io.Discard,
			now: clk.Now, fixed: true, markets: []Market{wm}, keys: []*Key{k1}, examiner: ex,
		}
	}

	t.Run("directory_blocker_prevents_rent_and_blocks_separate_up", func(t *testing.T) {
		home := t.TempDir()
		if err := os.Mkdir(examinerStatePath(home), 0o700); err != nil {
			t.Fatalf("mkdir examiner.json blocker: %v", err)
		}
		exProv := &fakeExaminerProv{offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}}}
		wm := &stubWorkerMarket{name: "runpod", offers: []Offer{{Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}}}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
		fEx := makeFleet(home, ex, wm)
		fUp := makeFleet(home, nil, wm)

		if err := reconcileExaminerOnce(context.Background(), fEx); err == nil {
			t.Fatalf("expected reconcileExaminerOnce error when examiner.json is a directory")
		}
		if len(exProv.rented) != 0 || len(exProv.insts) != 0 {
			t.Fatalf("must NOT rent examiner when examiner.json is unwritable: rented=%d insts=%d", len(exProv.rented), len(exProv.insts))
		}
		if err := cmdUp(context.Background(), fUp, 1); err == nil {
			t.Fatalf("expected cmdUp error when examiner.json is unreadable directory")
		}
		if len(wm.rentals) != 0 {
			t.Fatalf("cmdUp must NOT rent worker when examiner.json cannot be read: got %d", len(wm.rentals))
		}
	})

	t.Run("corrupt_examiner_json_blocks_up", func(t *testing.T) {
		home := t.TempDir()
		if err := os.WriteFile(examinerStatePath(home), []byte("{corrupt"), 0o600); err != nil {
			t.Fatalf("write corrupt examiner.json: %v", err)
		}
		wm := &stubWorkerMarket{name: "runpod", offers: []Offer{{Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}}}
		fUp := makeFleet(home, nil, wm)
		if err := cmdUp(context.Background(), fUp, 1); err == nil {
			t.Fatalf("expected cmdUp error when examiner.json is corrupt")
		}
		if len(wm.rentals) != 0 {
			t.Fatalf("cmdUp must NOT rent worker when examiner.json is corrupt, got %d", len(wm.rentals))
		}
	})

	t.Run("post_rental_persist_failure_unwinds_or_keeps_reservation", func(t *testing.T) {
		// Case A: EndExaminer succeeds -> VM is immediately unwound!
		homeA := t.TempDir()
		exProvA := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			onRent: func() {
				_ = os.Remove(examinerStatePath(homeA))
				_ = os.Mkdir(examinerStatePath(homeA), 0o700)
			},
		}
		exA := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, homeA, exProvA, clk.Now, nil, false)
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00031", Entries: 1, ExamAt: 0}}}
		if err := exA.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 1.00}); err == nil {
			t.Fatalf("expected Reconcile error on post-rental persist failure")
		}
		if len(exProvA.insts) != 0 || len(exProvA.ended) != 1 {
			t.Fatalf("expected immediate unwind on post-rental persist failure: insts=%d ended=%v", len(exProvA.insts), exProvA.ended)
		}

		// Case B: EndExaminer ALSO fails -> separate cmdUp is STILL blocked (0 workers)!
		homeB := t.TempDir()
		exProvB := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			endErr: map[string]error{"ex-1": errors.New("Vast 500")},
			onRent: func() {
				_ = os.Remove(examinerStatePath(homeB))
				_ = os.Mkdir(examinerStatePath(homeB), 0o700)
			},
		}
		wmB := &stubWorkerMarket{name: "runpod", offers: []Offer{{Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}}}
		exB := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, homeB, exProvB, clk.Now, nil, false)
		if err := exB.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 1.00}); err == nil {
			t.Fatalf("expected Reconcile error on post-rental persist + unwind failure")
		}
		if live, _, _, err := exB.ActiveSnapshot(clk.Now()); err != nil || live != 1 {
			t.Fatalf("in-memory ActiveSnapshot must still count unwound-failed VM as live=1: live=%d err=%v", live, err)
		}
		fUpB := makeFleet(homeB, nil, wmB)
		if err := cmdUp(context.Background(), fUpB, 1); err == nil {
			t.Fatalf("expected cmdUp error when examiner.json is blocked after failed unwind")
		}
		if len(wmB.rentals) != 0 {
			t.Fatalf("cmdUp must rent 0 workers after failed unwind, got %d", len(wmB.rentals))
		}
	})

	t.Run("pending_rent_reservation_blocks_up_and_adopts_on_recovery", func(t *testing.T) {
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		certPEM, _, err := generateExaminerCert(clk.Now(), 45*time.Minute)
		if err != nil {
			t.Fatalf("generateExaminerCert: %v", err)
		}
		label := examinerLabelPrefix + "-" + owner + "-s3-o00031-cafebabe"
		if err := writeExaminerState(home, &examinerState{
			PendingRent: true, ReservedLabel: label, RoundID: "s3-o00031",
			APIKey: "secretkey", CertPEM: string(certPEM), Price: 0.62, Created: clk.Now(),
		}); err != nil {
			t.Fatalf("write PendingRent state: %v", err)
		}
		wm := &stubWorkerMarket{name: "runpod", offers: []Offer{{Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}}}
		fUp := makeFleet(home, nil, wm)
		if err := cmdUp(context.Background(), fUp, 1); err != nil {
			t.Fatalf("cmdUp with PendingRent reservation: %v", err)
		}
		if len(wm.rentals) != 0 {
			t.Fatalf("PendingRent reservation must occupy --max 1 slot so cmdUp rents 0 workers, got %d", len(wm.rentals))
		}
		// Restarted Examiner finds the matching Vast instance and adopts it without renting a duplicate!
		exProv := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			insts: map[string]ExaminerInstance{
				"ex-adopted": {ID: "ex-adopted", Label: label, State: "starting", Price: 0.62, Created: clk.Now()},
			},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
		if err := ex.Reconcile(context.Background(), &Work{Rounds: []WorkRound{{ID: "s3-o00031", Entries: 1, ExamAt: 0}}}, ExaminerBudget{MaxWorkers: 1, MaxHourly: 1.00}); err != nil {
			t.Fatalf("Reconcile adoption: %v", err)
		}
		if len(exProv.rented) != 0 {
			t.Fatalf("must adopt matching PendingRent VM rather than renting a 2nd VM: rented=%d", len(exProv.rented))
		}
		st := readExaminerState(home)
		if st == nil || st.InstanceID != "ex-adopted" || st.PendingRent {
			t.Fatalf("expected adopted InstanceID=ex-adopted with PendingRent=false, got %+v", st)
		}
	})

	// Gate 4 P2 #4: ambiguous non-errNoRoom RentExaminer failure (e.g. network timeout / 502 after
	// Vast actually created the VM) must retain PendingRent on disk so separate cmdUp sees live=1
	// and the next Reconcile adopts the created VM rather than destroying it as an unadoptable stray.
	t.Run("ambiguous_rent_error_retains_pending_rent_and_adopts_on_next_tick", func(t *testing.T) {
		home := t.TempDir()
		var createdLabel string
		exProv := &fakeExaminerProv{
			offers:  []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			failIDs: map[string]error{"off-5090": errors.New("502 Bad Gateway")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00031", Entries: 1, ExamAt: 0}}}
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 1.00}); err == nil {
			t.Fatalf("expected Reconcile error on 502 from RentExaminer")
		}
		st := readExaminerState(home)
		if st == nil || !st.PendingRent || st.ReservedLabel == "" {
			t.Fatalf("ambiguous 502 must retain PendingRent reservation on disk, got %+v", st)
		}
		createdLabel = st.ReservedLabel

		// Separate cmdUp sees PendingRent (live=1) and rents 0 workers!
		wm := &stubWorkerMarket{name: "runpod", offers: []Offer{{Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}}}
		fUp := makeFleet(home, nil, wm)
		if err := cmdUp(context.Background(), fUp, 1); err != nil {
			t.Fatalf("cmdUp during ambiguous PendingRent: %v", err)
		}
		if len(wm.rentals) != 0 {
			t.Fatalf("cmdUp must NOT rent worker while PendingRent is retained after ambiguous 502, got %d", len(wm.rentals))
		}

		// Vast actually created the VM with createdLabel -> next Reconcile adopts it (0 ended, 0 re-rented)!
		exProv.mu.Lock()
		exProv.insts = map[string]ExaminerInstance{
			"ex-delayed": {ID: "ex-delayed", Label: createdLabel, State: "starting", Price: 0.62, Created: clk.Now()},
		}
		delete(exProv.failIDs, "off-5090")
		exProv.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 1.00}); err != nil {
			t.Fatalf("Reconcile adoption after ambiguous 502: %v", err)
		}
		if len(exProv.ended) != 0 || len(exProv.rented) != 0 {
			t.Fatalf("expected clean adoption with 0 ended and 0 re-rented, ended=%v rented=%d", exProv.ended, len(exProv.rented))
		}
		st2 := readExaminerState(home)
		if st2 == nil || st2.InstanceID != "ex-delayed" || st2.PendingRent {
			t.Fatalf("expected adopted ex-delayed with PendingRent=false, got %+v", st2)
		}
	})
}

// Review 14 P2 #6: if the provider listing omits the hourly rate (Price <= 0) on normal teardown,
// spend recording must fall back to saved.Price (and Vast ListExaminers falls back to dph_base)
// so accrued daily spend ($0.31) is preserved; if no valid rate exists at all (stray with Price=0),
// it must NOT write $0.00 to spent.jsonl and must instead keep a pending spend record & return error.
func TestExaminerMissingProviderRateFallsBackToSavedAndKeepsUnknownPending(t *testing.T) {
	t.Run("saved_rate_fallback_preserves_daily_spend_on_normal_teardown", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }

		if err := ex.Reconcile(context.Background(), &Work{Rounds: []WorkRound{{ID: "s3-o00032", Entries: 1, ExamAt: 0}}}, testExaminerBudget()); err != nil {
			t.Fatalf("Reconcile rent: %v", err)
		}
		// Align instance Created with test clock and advance 30m -> accrued $0.31.
		prov.mu.Lock()
		inst := prov.insts["ex-1"]
		inst.Created = clk.Now()
		inst.Price = 0 // Provider omits hourly rate on final listing!
		prov.insts["ex-1"] = inst
		prov.mu.Unlock()
		clk.Advance(30 * time.Minute)

		if _, _, accBefore, err := ex.ActiveSnapshot(clk.Now()); err != nil || accBefore < 0.309 || accBefore > 0.311 {
			t.Fatalf("expected $0.31 accrued before teardown, got $%.4f err=%v", accBefore, err)
		}
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("Reconcile teardown with missing provider rate: %v", err)
		}
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil {
			t.Fatalf("f.spent: %v", err)
		}
		if spentToday < 0.309 || spentToday > 0.311 {
			t.Fatalf("daily spend must remain $0.31 after teardown with omitted provider rate, got $%.4f", spentToday)
		}
	})

	t.Run("vast_list_examiners_falls_back_to_dph_base", func(t *testing.T) {
		vastSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			lbl := "stet-examiner-0123456789ab-s3-o00032-deadbeef"
			st := "running"
			_ = json.NewEncoder(w).Encode(map[string]any{
				"instances": []map[string]any{{
					"id": 99001, "label": lbl, "actual_status": st,
					"dph_total": 0.0, "dph_base": 0.62, "start_date": 1700000000,
				}},
			})
		}))
		defer vastSrv.Close()
		vm := newVast(vastSrv.URL, "test-key")
		insts, err := vm.ListExaminers(context.Background())
		if err != nil || len(insts) != 1 || insts[0].Price < 0.619 || insts[0].Price > 0.621 {
			t.Fatalf("ListExaminers should fall back to dph_base $0.62 when dph_total==0, got %+v err=%v", insts, err)
		}
	})

	t.Run("unknown_rate_stray_records_conservative_fallback_and_never_deadlocks", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			insts: map[string]ExaminerInstance{
				"stray-zero": {
					ID: "stray-zero", Label: examinerLabelPrefix + "-" + owner + "-s3-o00032-deadbeef",
					State: "running", Price: 0, Created: clk.Now().Add(-30 * time.Minute),
				},
			},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }

		// Ending an unknown-rate stray must NOT write $0.00 and must NOT leave a permanent
		// unflushable PendingSpend that deadlocks future examiner rentals or worker autoscaling:
		// it records conservatively at MaxPrice ($1.50/h * 0.5h = $0.75) and stays healthy.
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("Reconcile unknown-rate stray should succeed with conservative MaxPrice fallback: %v", err)
		}
		if !spentHasExaminer(home, "stray-zero") {
			t.Fatalf("expected conservative spend record written for stray-zero")
		}
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil {
			t.Fatalf("f.spent: %v", err)
		}
		if spentToday < 0.749 || spentToday > 0.751 {
			t.Fatalf("expected conservative $0.75 daily spend (not $0.00!), got $%.4f", spentToday)
		}
		if readExaminerState(home) != nil {
			t.Fatalf("expected examiner.json clean (no unflushable PendingSpends deadlock)")
		}
		// Subsequent round needing an exam MUST rent normally without being deadlocked!
		if err := ex.Reconcile(context.Background(), &Work{Rounds: []WorkRound{{ID: "s3-o00033", Entries: 1, ExamAt: 0}}}, testExaminerBudget()); err != nil {
			t.Fatalf("subsequent Reconcile must not be deadlocked after unknown-rate stray: %v", err)
		}
		if len(prov.rented) != 1 {
			t.Fatalf("expected 1 examiner rented on subsequent round, got %d", len(prov.rented))
		}
	})
}

// Review 15 P1 #2: when RentExaminer times out after creating a contract on Vast and the next
// provider listing (15s later) temporarily omits the instance (eventual consistency), Reconcile
// must keep PendingRent, ReservedLabel, and its $0.62/h budget slot across stale listings rather
// than clearing state and renting a second paid GPU; once the delayed instance appears, it must
// be adopted cleanly so only 1 GPU at $0.62/h ever exists. Confirmed 4xx rejections release the
// reservation immediately, and expiry past examinerPendingRentGrace allows retrying.
func TestPendingRentSurvivesStaleListingAndPreventsDoubleRental(t *testing.T) {
	t.Run("stale_empty_list_between_timeout_and_visibility_prevents_second_rental", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		op, _ := newKey("op")
		k1, _ := newKey("stet-rp-01")
		stetSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/admin/work":
				_ = json.NewEncoder(w).Encode(Work{
					Rounds: []WorkRound{{ID: "s3-o00035", Entries: 1, RunsWanted: 1, ExamAt: 0}},
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

		// Simulate RentExaminer timing out AFTER Vast actually accepted creation under l.Label.
		exProv := &fakeExaminerProv{
			offers:  []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			failIDs: map[string]error{"off-5090": errors.New("context deadline exceeded")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
		wm := &stubWorkerMarket{
			name:   "runpod",
			offers: []Offer{{Market: "runpod", ID: "gpu-57", Card: "RTX 4090", Model: "NVIDIA GeForce RTX 4090", Maker: "NVIDIA", Price: 0.57, MemoryGB: 24, Trust: "secure"}},
		}
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00035", Entries: 1, RunsWanted: 1, ExamAt: 0}}}

		// Poll 1 (T0): RentExaminer times out -> PendingRent reservation written & kept!
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err == nil {
			t.Fatalf("expected timeout error from Poll 1 RentExaminer")
		}
		st1 := readExaminerState(home)
		if st1 == nil || !st1.PendingRent || st1.ReservedLabel == "" {
			t.Fatalf("expected PendingRent persisted after timeout, got %+v", st1)
		}
		origLabel := st1.ReservedLabel

		// Clear failIDs so IF Reconcile erroneously tries to rent a 2nd time, it WOULD create a 2nd VM!
		exProv.mu.Lock()
		delete(exProv.failIDs, "off-5090")
		exProv.mu.Unlock()

		// Poll 2 (T0 + 15s): provider listing succeeds but is STALE (returns empty []).
		// Reconcile MUST keep PendingRent, original ReservedLabel, and $0.62/h budget charge,
		// and MUST NOT rent a second GPU or allow cmdUp/tick to rent a worker!
		clk.Advance(15 * time.Second)
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err != nil {
			t.Fatalf("Poll 2 stale list should wait cleanly without error: %v", err)
		}
		if len(exProv.rented) != 0 {
			t.Fatalf("stale empty listing triggered a second paid rental! rented=%d", len(exProv.rented))
		}
		st2 := readExaminerState(home)
		if st2 == nil || !st2.PendingRent || st2.ReservedLabel != origLabel || st2.Price != 0.62 {
			t.Fatalf("expected original PendingRent reservation preserved across stale list, got %+v", st2)
		}
		if live, hourly, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("ActiveSnapshot during stale-list grace: live=%d hourly=%.2f err=%v, want 1/$0.62", live, hourly, err)
		}
		fUp := &Fleet{
			o: &Options{
				Home: home, Every: time.Minute, Cloud: "secure", Idle: 20,
				MinCuratorGPUMemory: 24, MinGPUMemory: 12,
				Policy: Policy{MaxWorkers: 1, MaxHourly: 0.70, Edits: true, PerWorker: 6},
			},
			st: newStet(stetSrv.URL), op: op, out: io.Discard,
			now: clk.Now, fixed: true, markets: []Market{wm}, keys: []*Key{k1}, examiner: nil,
		}
		if err := cmdUp(context.Background(), fUp, 1); err != nil {
			t.Fatalf("cmdUp during stale-list grace: %v", err)
		}
		if len(wm.rentals) != 0 {
			t.Fatalf("cmdUp exceeded --max 1 / $0.70/h during PendingRent grace: workers=%d", len(wm.rentals))
		}

		// Poll 3 (T0 + 30s): delayed VM finally becomes visible under origLabel -> adopted cleanly!
		clk.Advance(15 * time.Second)
		exProv.mu.Lock()
		exProv.insts = map[string]ExaminerInstance{
			"ex-first": {ID: "ex-first", Label: origLabel, State: "starting", Price: 0.62, Created: clk.Now().Add(-30 * time.Second)},
		}
		exProv.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err != nil {
			t.Fatalf("Poll 3 adoption: %v", err)
		}
		if len(exProv.insts) != 1 || len(exProv.rented) != 0 || len(exProv.ended) != 0 {
			t.Fatalf("want exactly 1 adopted instance (0 extra rents, 0 ended): insts=%d rented=%d ended=%d",
				len(exProv.insts), len(exProv.rented), len(exProv.ended))
		}
		st3 := readExaminerState(home)
		if st3 == nil || st3.InstanceID != "ex-first" || st3.PendingRent {
			t.Fatalf("expected adopted ex-first with PendingRent=false, got %+v", st3)
		}
	})

	t.Run("confirmed_rejection_clears_immediately_and_grace_expiry_allows_retry", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00036", Entries: 1, ExamAt: 0}}}

		// 1. Confirmed provider rejection (errRentRejected) clears PendingRent immediately.
		homeRej := t.TempDir()
		provRej := &fakeExaminerProv{
			offers:  []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			failIDs: map[string]error{"off-5090": errors.Join(errRentRejected, errors.New("400 bad request"))},
		}
		exRej := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, homeRej, provRej, clk.Now, nil, false)
		if err := exRej.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err == nil {
			t.Fatalf("expected error on confirmed rejection")
		}
		if st := readExaminerState(homeRej); st != nil {
			t.Fatalf("confirmed errRentRejected must clear PendingRent immediately, got %+v", st)
		}

		// 2. Review 19 P1 #1: an ambiguous create whose response was lost MUST NOT discard
		//    PendingRent and create a second contract after 3m01s (or even past timeout): a delayed
		//    successful first create stays within --max 1 / $0.70/h and is adopted once visible.
		homeExp := t.TempDir()
		provExp := &fakeExaminerProv{
			offers:  []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			failIDs: map[string]error{"off-5090": errors.New("504 Gateway Timeout")},
		}
		exExp := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, homeExp, provExp, clk.Now, nil, false)
		if err := exExp.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err == nil {
			t.Fatalf("expected 504 error")
		}
		stPend := readExaminerState(homeExp)
		if stPend == nil || !stPend.PendingRent || stPend.ReservedLabel == "" {
			t.Fatalf("expected PendingRent persisted after 504, got %+v", stPend)
		}
		origLabel := stPend.ReservedLabel
		provExp.mu.Lock()
		delete(provExp.failIDs, "off-5090")
		provExp.mu.Unlock()

		// 3m01s later with listing still empty -> MUST keep charging reservation and NOT create a 2nd GPU!
		clk.Advance(3*time.Minute + time.Second)
		if err := exExp.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err != nil {
			t.Fatalf("Reconcile at 3m01s with unresolved PendingRent: %v", err)
		}
		if len(provExp.rented) != 0 {
			t.Fatalf("MUST NOT rent a second examiner while first create outcome is unconfirmed! rented=%d", len(provExp.rented))
		}
		if live, hourly, _, err := exExp.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("unresolved PendingRent must keep holding 1 slot / $0.62/h: live=%d hourly=%.2f err=%v", live, hourly, err)
		}

		// When the delayed first contract finally appears at T0 + 5m, Reconcile adopts it cleanly (still 1 GPU total!).
		clk.Advance(2 * time.Minute)
		provExp.mu.Lock()
		provExp.insts = map[string]ExaminerInstance{
			"ex-delayed-1": {ID: "ex-delayed-1", Label: origLabel, State: "running", Price: 0.62, Created: clk.Now().Add(-5 * time.Minute)},
		}
		provExp.mu.Unlock()
		if err := exExp.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err != nil {
			t.Fatalf("adopt delayed first contract: %v", err)
		}
		if len(provExp.rented) != 0 || len(provExp.insts) != 1 {
			t.Fatalf("expected delayed first contract adopted with 0 extra rentals: rented=%d insts=%d", len(provExp.rented), len(provExp.insts))
		}
		if st := readExaminerState(homeExp); st == nil || st.InstanceID != "ex-delayed-1" || st.PendingRent {
			t.Fatalf("expected adopted ex-delayed-1 with PendingRent=false, got %+v", st)
		}
	})

	t.Run("failed_end_on_materialized_pending_rent_dedups_and_clears_on_vanish", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		lbl := examinerLabelPrefix + "-" + owner + "-s3-o00039-cafebabe"
		if err := writeExaminerState(home, &examinerState{
			PendingRent:   true,
			ReservedLabel: lbl,
			RoundID:       "s3-o00039",
			APIKey:        "k-mat",
			CertPEM:       "dummy-cert",
			Price:         0.62,
			Created:       clk.Now().Add(-30 * time.Second),
		}); err != nil {
			t.Fatalf("writeExaminerState: %v", err)
		}
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			insts: map[string]ExaminerInstance{
				"ex-mat": {ID: "ex-mat", Label: lbl, State: "stopped", Price: 0.62, Created: clk.Now().Add(-30 * time.Second)},
			},
			endErr: map[string]error{"ex-mat": errors.New("Vast 503")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }

		// Case 2 sees stopped VM matching ReservedLabel; EndExaminer fails -> promotes to InstanceID
		// so both memory and disk snapshots count exact 1x GPU ($0.62/h), NOT 2x ($1.24/h)!
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00039", Entries: 1, ExamAt: 0}}}
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err == nil {
			t.Fatalf("expected EndExaminer failure on stopped materialized VM")
		}
		if live, hourly, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("in-memory snapshot double-counted materialized PendingRent: live=%d hourly=%.2f err=%v", live, hourly, err)
		}
		if live, hourly, _, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("disk snapshot double-counted materialized PendingRent: live=%d hourly=%.2f err=%v", live, hourly, err)
		}
		st := readExaminerState(home)
		if st == nil || st.InstanceID != "ex-mat" || st.PendingRent {
			t.Fatalf("expected PendingRent promoted to InstanceID=ex-mat, got %+v", st)
		}

		// VM disappears from listing 30s later while provider still returns 503 on DELETE ->
		// termination is NOT confirmed yet, so ex-mat must remain tracked with zero terminal spend!
		clk.Advance(30 * time.Second)
		prov.mu.Lock()
		delete(prov.insts, "ex-mat")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), &Work{}, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err == nil {
			t.Fatalf("expected error while DELETE still fails on missing ex-mat")
		}
		if live, _, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 {
			t.Fatalf("unconfirmed missing ex-mat must remain live=1, got live=%d err=%v", live, err)
		}

		// Once DELETE succeeds (e.g. Vast answers 404 confirming termination), must record spend
		// once, clear state cleanly (not strand PendingRent!), and allow renting!
		prov.mu.Lock()
		delete(prov.endErr, "ex-mat")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), &Work{}, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err != nil {
			t.Fatalf("Case 1 reconcile after materialized VM confirmed terminated: %v", err)
		}
		if stAfter := readExaminerState(home); stAfter != nil {
			t.Fatalf("expected examiner.json cleared without stranding PendingRent, got %+v", stAfter)
		}
		if live, _, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 0 {
			t.Fatalf("expected 0 live examiners after vanished VM cleared, got live=%d err=%v", live, err)
		}
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil || spentToday < 0.009 || spentToday > 0.012 {
			t.Fatalf("expected 1m single spend (~$0.0103) recorded once, got $%.5f err=%v", spentToday, err)
		}
	})

	t.Run("vast_408_request_timeout_remains_ambiguous_not_rejected", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "PUT" {
				w.WriteHeader(http.StatusRequestTimeout)
				_, _ = w.Write([]byte(`{"msg":"request timeout"}`))
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"instances": []any{}})
		}))
		defer srv.Close()
		vm := newVast(srv.URL, "tok")
		vm.a.http = srv.Client()
		_, err := vm.RentExaminer(context.Background(), "123", ExaminerLaunch{
			Label: "stet-examiner-test", Image: "vllm/vllm-openai@sha256:0000", Model: "m", APIKey: "k", CertPEM: []byte("c"), KeyPEM: []byte("p"),
		})
		if err == nil {
			t.Fatalf("expected error on HTTP 408")
		}
		if errors.Is(err, errRentRejected) || errors.Is(err, errNoRoom) {
			t.Fatalf("HTTP 408 Request Timeout must stay ambiguous (not errRentRejected/errNoRoom): %v", err)
		}
	})
}

// Review 15 P2 #3: when an unresolved owned orphan disappears from the provider before the next
// poll (e.g. Vast finished deleting it after returning an error on the prior EndExaminer call),
// Reconcile and EnforceTimeout must record a bounded, idempotent terminal spend charge in
// spent.jsonl (preserving it in PendingSpends if spent.jsonl is temporarily unwritable) so its
// accrued daily spend never vanishes to $0.00.
func TestDisappearedUnresolvedOrphanRecordsBoundedSpend(t *testing.T) {
	t.Run("records_bounded_charge_when_unresolved_orphan_disappears", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		prov := &fakeExaminerProv{
			insts: map[string]ExaminerInstance{
				"orphan-30m": {
					ID: "orphan-30m", Label: examinerLabelPrefix + "-" + owner + "-s3-o00037-deadbeef",
					State: "running", Price: 0.62, Created: clk.Now().Add(-30 * time.Minute),
				},
			},
			endErr: map[string]error{"orphan-30m": errors.New("transient Vast 503")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }

		// Poll 1: EndExaminer fails -> retained in Unresolved with $0.31 accrued.
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
			t.Fatalf("expected EndExaminer failure on Poll 1")
		}
		if live, _, acc, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || acc < 0.309 || acc > 0.311 {
			t.Fatalf("Poll 1 snapshot: live=%d acc=%.4f err=%v, want 1/$0.31", live, acc, err)
		}

		// Provider omits orphan-30m before Poll 2 while DELETE still errors -> retained in Unresolved!
		prov.mu.Lock()
		delete(prov.insts, "orphan-30m")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
			t.Fatalf("expected error when omitted orphan-30m DELETE still fails")
		}
		if live, _, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 {
			t.Fatalf("omitted orphan-30m with failing DELETE must stay live=1, got live=%d err=%v", live, err)
		}

		// Provider finishes deleting orphan-30m (DELETE now returns 404 -> nil).
		prov.mu.Lock()
		delete(prov.endErr, "orphan-30m")
		prov.mu.Unlock()

		// Poll 2: Unresolved clears AND $0.31 terminal charge is durably written to spent.jsonl!
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("Poll 2 after orphan vanished: %v", err)
		}
		if readExaminerState(home) != nil {
			t.Fatalf("expected examiner.json cleared after vanished orphan spend recorded")
		}
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil {
			t.Fatalf("f.spent: %v", err)
		}
		if spentToday < 0.309 || spentToday > 0.311 {
			t.Fatalf("vanished unresolved orphan spend must remain $0.31 (not $0.00!), got $%.4f", spentToday)
		}
		// Idempotent on repeat reconciliation.
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("Poll 3 idempotent check: %v", err)
		}
		spentAgain, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil || spentAgain < 0.309 || spentAgain > 0.311 {
			t.Fatalf("expected idempotent $0.31 spend, got $%.4f err=%v", spentAgain, err)
		}
	})

	t.Run("preserves_obligation_when_spent_jsonl_unwritable_and_respects_live_observation", func(t *testing.T) {
		// Part A: When the provider confirms an orphan is STILL listed live at T0+2h and refuses
		// deletion, accrual MUST keep counting through the live observation time ($1.24 at $0.62/h,
		// not capped at 45m / $0.465), preserve $1.24 in PendingSpends while spent.jsonl is blocked,
		// and flush $1.24 once unblocked.
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		spentPath := filepath.Join(home, "spent.jsonl")
		prov := &fakeExaminerProv{
			insts: map[string]ExaminerInstance{
				"orphan-2h": {
					ID: "orphan-2h", Label: examinerLabelPrefix + "-" + owner + "-s3-o00038-deadbeef",
					State: "running", Price: 0.62, Created: clk.Now().Add(-2 * time.Hour),
				},
			},
			endErr: map[string]error{"orphan-2h": errors.New("transient Vast 503")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(spentPath, rec) }

		// EnforceTimeout sees orphan-2h live at T0+2h and fails delete -> stores orphan-2h with ObservedLiveAt=T0+2h.
		if err := ex.EnforceTimeout(context.Background()); err == nil {
			t.Fatalf("expected EnforceTimeout delete failure")
		}
		if live, _, acc, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 1 || acc < 1.239 || acc > 1.241 {
			t.Fatalf("live-observed orphan at T0+2h must accrue $1.24: live=%d acc=%.4f err=%v", live, acc, err)
		}
		// Block spent.jsonl, remove orphan-2h from provider, and clear transient 503 so DELETE confirms 404 -> nil.
		if err := os.Mkdir(spentPath, 0o700); err != nil {
			t.Fatalf("block spent.jsonl: %v", err)
		}
		prov.mu.Lock()
		delete(prov.insts, "orphan-2h")
		delete(prov.endErr, "orphan-2h")
		prov.mu.Unlock()

		// EnforceTimeout detects disappearance, fails writing spent.jsonl, and retains full observed 2h ($1.24) in PendingSpends!
		if err := ex.EnforceTimeout(context.Background()); err == nil {
			t.Fatalf("expected spent.jsonl write error to propagate")
		}
		if live, _, acc, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 0 || acc < 1.239 || acc > 1.241 {
			t.Fatalf("disk snapshot while PendingSpends retained: live=%d acc=%.4f err=%v, want 0/$1.240", live, acc, err)
		}
		// Unblock spent.jsonl -> flushes exact $1.24 charge and clears examiner.json.
		if err := os.Remove(spentPath); err != nil {
			t.Fatalf("unblock spent.jsonl: %v", err)
		}
		if err := ex.EnforceTimeout(context.Background()); err != nil {
			t.Fatalf("EnforceTimeout flush: %v", err)
		}
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil || spentToday < 1.239 || spentToday > 1.241 {
			t.Fatalf("expected flushed observed $1.24 spend, got $%.4f err=%v", spentToday, err)
		}

		// Part B: Unobserved offline orphan (persisted at T0, gone by T0+2h, end time uncertain
		// until confirmed via EndExaminer) accrues conservatively through confirmation time ($1.24).
		homeOff := t.TempDir()
		ownerOff := loadOrCreateExaminerOwner(homeOff)
		if err := writeExaminerState(homeOff, &examinerState{
			Unresolved: []ExaminerInstance{{
				ID: "orphan-offline", Label: examinerLabelPrefix + "-" + ownerOff + "-s3-o00038-cafebabe",
				State: "running", Price: 0.62, Created: clk.Now().Add(-2 * time.Hour),
			}},
		}); err != nil {
			t.Fatalf("write offline unresolved state: %v", err)
		}
		spentOff := filepath.Join(homeOff, "spent.jsonl")
		if err := os.Mkdir(spentOff, 0o700); err != nil {
			t.Fatalf("block offline spent.jsonl: %v", err)
		}
		exOff := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, homeOff, &fakeExaminerProv{}, clk.Now, nil, false)
		exOff.onSpent = func(rec podRecord) error { return appendExaminerSpend(spentOff, rec) }
		if err := exOff.EnforceTimeout(context.Background()); err == nil {
			t.Fatalf("expected spent.jsonl write error on offline orphan")
		}
		if live, _, acc, err := activeExaminerFromDisk(homeOff, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 0 || acc < 1.239 || acc > 1.241 {
			t.Fatalf("offline unobserved orphan must accrue conservatively through confirmation time ($1.24): live=%d acc=%.4f err=%v", live, acc, err)
		}
	})
}

// Review 16 P1 #1: once Vast acknowledges a contract ID, a subsequent listing that omits it (stale
// eventual-consistency listing or delayed label visibility) must NOT record terminal spend, clear
// credentials, or rent a second paid GPU. The acknowledged ID, credentials, slot, and hourly rate
// remain intact until it reappears (re-adopted) or EndExaminer confirms termination (on timeout or
// idle cleanup).
func TestOmittedAcknowledgedExaminerPreservesContractWithoutDoubleRental(t *testing.T) {
	t.Run("omitted_contract_re_adopted_without_replacement_or_terminal_charge", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		spentCalls := 0
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error {
			spentCalls++
			return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec)
		}
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00040", Entries: 1, ExamAt: 0}}}
		budget := ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}

		// Poll 1 (T0): create succeeds -> ex-1 acknowledged and persisted.
		if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
			t.Fatalf("Poll 1 rent: %v", err)
		}
		st1 := readExaminerState(home)
		if st1 == nil || st1.InstanceID != "ex-1" || st1.APIKey == "" || st1.CertPEM == "" {
			t.Fatalf("expected persisted ex-1 with credentials, got %+v", st1)
		}
		savedInst := prov.insts["ex-1"]

		// Poll 2 (T0 + 15s): hide ex-1 from provider listing (stale list / delayed label visibility).
		clk.Advance(15 * time.Second)
		prov.mu.Lock()
		delete(prov.insts, "ex-1")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
			t.Fatalf("Poll 2 stale omission: %v", err)
		}
		if len(prov.rented) != 1 || len(prov.ended) != 0 || spentCalls != 0 {
			t.Fatalf("stale omission must NOT rent a replacement, terminate, or write terminal spend: rented=%d ended=%d spentCalls=%d",
				len(prov.rented), len(prov.ended), spentCalls)
		}
		st2 := readExaminerState(home)
		if st2 == nil || st2.InstanceID != "ex-1" || st2.APIKey != st1.APIKey || st2.CertPEM != st1.CertPEM {
			t.Fatalf("expected ex-1 and credentials preserved across omission, got %+v", st2)
		}
		if live, hourly, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("in-memory snapshot during omission: live=%d hourly=%.2f err=%v, want 1/$0.62", live, hourly, err)
		}
		if live, hourly, _, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("disk snapshot during omission: live=%d hourly=%.2f err=%v, want 1/$0.62", live, hourly, err)
		}

		// EnforceTimeout while within TTL and still omitted must ALSO keep ex-1 untouched!
		if err := ex.EnforceTimeout(context.Background()); err != nil {
			t.Fatalf("EnforceTimeout within TTL during omission: %v", err)
		}
		if len(prov.ended) != 0 || spentCalls != 0 || readExaminerState(home) == nil {
			t.Fatalf("EnforceTimeout within TTL must keep omitted contract intact: ended=%d spentCalls=%d", len(prov.ended), spentCalls)
		}

		// Poll 3 (T0 + 30s): restore visibility -> ex-1 is re-adopted cleanly!
		clk.Advance(15 * time.Second)
		prov.mu.Lock()
		prov.insts["ex-1"] = savedInst
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
			t.Fatalf("Poll 3 re-adoption: %v", err)
		}
		if len(prov.rented) != 1 || len(prov.insts) != 1 || len(prov.ended) != 0 || spentCalls != 0 {
			t.Fatalf("expected exact 1 live contract at $0.62/h, 0 terminations, 0 terminal spend: rented=%d insts=%d ended=%d spentCalls=%d",
				len(prov.rented), len(prov.insts), len(prov.ended), spentCalls)
		}
		st3 := readExaminerState(home)
		if st3 == nil || st3.InstanceID != "ex-1" || st3.APIKey != st1.APIKey {
			t.Fatalf("expected original ex-1 re-adopted with original credentials, got %+v", st3)
		}
	})

	t.Run("idle_and_timeout_cleanup_require_confirmed_termination_for_omitted_contract", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			endErr: map[string]error{"ex-1": errors.New("Vast 503")},
		}
		spentCalls := 0
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error {
			spentCalls++
			return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec)
		}
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00041", Entries: 1, ExamAt: 0}}}
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err != nil {
			t.Fatalf("rent: %v", err)
		}
		// Hide ex-1 from listing; idle cleanup calls EndExaminer("ex-1") -> 503 -> must NOT clear or bill!
		clk.Advance(10 * time.Minute)
		prov.mu.Lock()
		delete(prov.insts, "ex-1")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), &Work{}, ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}); err == nil {
			t.Fatalf("expected idle cleanup error when EndExaminer fails on omitted ex-1")
		}
		if st := readExaminerState(home); st == nil || st.InstanceID != "ex-1" || spentCalls != 0 {
			t.Fatalf("unconfirmed idle cleanup must retain ex-1 without terminal spend: st=%+v spentCalls=%d", st, spentCalls)
		}
		// Past timeout, EnforceTimeout also calls EndExaminer("ex-1") -> still 503 -> must NOT clear or bill!
		clk.Advance(40 * time.Minute)
		if err := ex.EnforceTimeout(context.Background()); err == nil {
			t.Fatalf("expected EnforceTimeout error when EndExaminer fails on omitted timed-out ex-1")
		}
		if st := readExaminerState(home); st == nil || st.InstanceID != "ex-1" || spentCalls != 0 {
			t.Fatalf("unconfirmed timeout cleanup must retain ex-1 without terminal spend: st=%+v spentCalls=%d", st, spentCalls)
		}
		// Once EndExaminer succeeds (200/404 -> nil), EnforceTimeout confirms termination, bills conservatively through now (50m = $0.5167), and clears!
		prov.mu.Lock()
		delete(prov.endErr, "ex-1")
		prov.mu.Unlock()
		if err := ex.EnforceTimeout(context.Background()); err != nil {
			t.Fatalf("EnforceTimeout confirmed end: %v", err)
		}
		if readExaminerState(home) != nil || spentCalls != 1 {
			t.Fatalf("expected cleared state and 1 terminal spend after confirmed end: st=%+v spentCalls=%d", readExaminerState(home), spentCalls)
		}
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil || spentToday < 0.515 || spentToday > 0.518 {
			t.Fatalf("expected conservative 50m spend $0.5167 through confirmation time, got $%.4f err=%v", spentToday, err)
		}
	})

	t.Run("condemned_unresolved_saved_examiner_confirms_and_replaces_without_45m_stall", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 26, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
			endErr: map[string]error{"ex-1": errors.New("Vast 503")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00042", Entries: 1, ExamAt: 0}}}
		budget := ExaminerBudget{MaxWorkers: 1, MaxHourly: 0.70}

		// Rent ex-1, then crash it to State="stopped" at T0+2m with transient DELETE 503 -> Unresolved=[ex-1].
		if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
			t.Fatalf("rent ex-1: %v", err)
		}
		clk.Advance(2 * time.Minute)
		prov.mu.Lock()
		inst := prov.insts["ex-1"]
		inst.State = "stopped"
		prov.insts["ex-1"] = inst
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, budget); err == nil {
			t.Fatalf("expected DELETE failure on stopped ex-1")
		}

		// At T0+3m (well within 45m TTL), ex-1 vanishes from ListExaminers while DELETE still returns 503 ->
		// must keep ex-1 in Unresolved (deduplicated live=1) and NOT rent replacement.
		clk.Advance(time.Minute)
		prov.mu.Lock()
		delete(prov.insts, "ex-1")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, budget); err == nil {
			t.Fatalf("expected error while DELETE still fails on missing condemned ex-1")
		}
		stMid := readExaminerState(home)
		if stMid == nil || stMid.InstanceID != "ex-1" || len(stMid.Unresolved) != 1 || stMid.Unresolved[0].ID != "ex-1" {
			t.Fatalf("missing condemned ex-1 must stay in BOTH InstanceID and Unresolved, got %+v", stMid)
		}
		if live, hourly, _, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly < 0.619 || hourly > 0.621 {
			t.Fatalf("expected deduplicated live=1/$0.62, got live=%d hourly=%.2f err=%v", live, hourly, err)
		}

		// DELETE now succeeds (404 -> nil): Case 2 immediately confirms termination and rents replacement ex-2
		// without waiting out the remaining 42 minutes of TTL!
		prov.mu.Lock()
		delete(prov.endErr, "ex-1")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
			t.Fatalf("reconcile after condemned ex-1 confirmed terminated: %v", err)
		}
		stNew := readExaminerState(home)
		if stNew == nil || stNew.InstanceID != "ex-2" || len(stNew.Unresolved) != 0 {
			t.Fatalf("expected immediate replacement ex-2 with empty Unresolved, got %+v", stNew)
		}
		if len(prov.rented) != 2 || len(prov.insts) != 1 {
			t.Fatalf("expected clean replacement (2 total rents, 1 live), got rented=%d insts=%d", len(prov.rented), len(prov.insts))
		}
	})
}

// Review 17 P1 #3: two controllers sharing a fleet home under the fleet lock MUST reload durable
// examiner.json on every reconcile so controller B adopts controller A's active rental instead of
// destroying it as a stray, and a stale background ProbeIfBooting MUST NOT clobber examiner.json
// or revive a proxy after another controller has ended or replaced the instance.
func TestReview17SecondControllerAdoptsExaminerAndStaleProbeNeverClobbers(t *testing.T) {
	t.Run("second_controller_adopts_without_destroying_active_rental", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		cfgA := ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}
		cfgB := ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}
		exA := newExaminer(cfgA, home, prov, clk.Now, nil, false)
		exB := newExaminer(cfgB, home, prov, clk.Now, nil, false)
		f := &Fleet{o: &Options{Home: home}, now: clk.Now}

		// Both controllers reconcile while idle (loading empty examiner.json).
		for name, ex := range map[string]*Examiner{"A": exA, "B": exB} {
			unlock, err := f.lock(time.Minute)
			if err != nil {
				t.Fatalf("lock idle %s: %v", name, err)
			}
			err = ex.Reconcile(context.Background(), &Work{}, testExaminerBudget())
			unlock()
			if err != nil {
				t.Fatalf("idle reconcile %s: %v", name, err)
			}
		}

		// Round opens -> A rents ex-1 under lock.
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00050", Entries: 1, ExamAt: 0}}}
		unlock, err := f.lock(time.Minute)
		if err != nil {
			t.Fatalf("lock A rent: %v", err)
		}
		err = exA.Reconcile(context.Background(), openWork, testExaminerBudget())
		unlock()
		if err != nil {
			t.Fatalf("A rent reconcile: %v", err)
		}
		if len(prov.rented) != 1 || prov.insts["ex-1"].ID != "ex-1" {
			t.Fatalf("expected A to rent ex-1, rented=%d insts=%+v", len(prov.rented), prov.insts)
		}

		// B's next tick under lock MUST reload examiner.json and adopt ex-1 (0 ends, 0 extra rents).
		unlock, err = f.lock(time.Minute)
		if err != nil {
			t.Fatalf("lock B reconcile: %v", err)
		}
		err = exB.Reconcile(context.Background(), openWork, testExaminerBudget())
		unlock()
		if err != nil {
			t.Fatalf("B reconcile: %v", err)
		}
		if len(prov.ended) != 0 {
			t.Fatalf("B MUST NOT destroy A's healthy ex-1 as a stray! ended=%v", prov.ended)
		}
		if len(prov.rented) != 1 || len(prov.insts) != 1 {
			t.Fatalf("B MUST NOT rent a second examiner! rented=%d insts=%+v", len(prov.rented), prov.insts)
		}
		if live, _, _, err := exB.ActiveSnapshot(clk.Now()); err != nil || live != 1 {
			t.Fatalf("B ActiveSnapshot must see adopted ex-1: live=%d err=%v", live, err)
		}
	})

	t.Run("stale_background_probe_does_not_clobber_replacement_state", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		exA := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		exB := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		addrB, closeB, err := exB.Start(ctx)
		if err != nil {
			t.Fatalf("start B proxy: %v", err)
		}
		defer closeB()

		open1 := &Work{Rounds: []WorkRound{{ID: "s3-o00051", Entries: 1, ExamAt: 0}}}
		if err := exA.Reconcile(ctx, open1, testExaminerBudget()); err != nil {
			t.Fatalf("A rent ex-1: %v", err)
		}
		if err := exB.Reconcile(ctx, open1, testExaminerBudget()); err != nil {
			t.Fatalf("B adopt ex-1: %v", err)
		}
		launch1 := prov.rented[0]

		inProbe := make(chan struct{})
		releaseProbe := make(chan struct{})
		vllm1 := startTLSServerWithCert(t, launch1.CertPEM, launch1.KeyPEM, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/v1/models" {
				close(inProbe)
				<-releaseProbe
				_ = json.NewEncoder(w).Encode(map[string]any{
					"data": []map[string]any{{"id": defaultExaminerModel + "@" + defaultExaminerRevision}},
				})
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer vllm1.Close()
		vu, _ := url.Parse(vllm1.URL)
		prov.mu.Lock()
		inst1 := prov.insts["ex-1"]
		inst1.State, inst1.Status, inst1.PublicIP, inst1.HostPort = "running", "running", vu.Hostname(), vu.Port()
		prov.insts["ex-1"] = inst1
		prov.mu.Unlock()

		// Start B's ProbeIfBooting; pause it inside /v1/models while A tears down ex-1 and rents ex-2.
		probeDone := make(chan struct{})
		go func() {
			defer close(probeDone)
			exB.ProbeIfBooting(ctx)
		}()
		<-inProbe

		if err := exA.Reconcile(ctx, &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("A teardown ex-1: %v", err)
		}
		open2 := &Work{Rounds: []WorkRound{{ID: "s3-o00052", Entries: 1, ExamAt: 0}}}
		if err := exA.Reconcile(ctx, open2, testExaminerBudget()); err != nil {
			t.Fatalf("A rent ex-2: %v", err)
		}
		close(releaseProbe)
		<-probeDone

		st := readExaminerState(home)
		if st == nil || st.InstanceID != "ex-2" {
			t.Fatalf("stale B ProbeIfBooting MUST NOT clobber A's ex-2 examiner.json: got %+v", st)
		}
		resp, err := http.Get("http://" + addrB.String() + "/v1/models")
		if err != nil {
			t.Fatalf("GET B proxy: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("stale B ProbeIfBooting MUST NOT activate proxy for destroyed ex-1: got HTTP %d, want 503", resp.StatusCode)
		}
	})

	t.Run("transient_pre_rental_write_error_and_dirty_recovery_never_clobber_peer", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		exA := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		exB := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00053", Entries: 1, ExamAt: 0}}}

		// Ensure owner file exists while writable, then make home temporarily unwritable for A's pre-rental write.
		_ = exA.ownerID()
		if err := os.Chmod(home, 0o500); err != nil {
			t.Fatalf("chmod home 0500: %v", err)
		}
		if err := exA.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
			_ = os.Chmod(home, 0o700)
			t.Fatalf("expected pre-rental examiner.json write error on read-only home")
		}
		if err := os.Chmod(home, 0o700); err != nil {
			t.Fatalf("chmod home 0700: %v", err)
		}

		// Disk recovers -> Controller B rents ex-1 and persists it to examiner.json.
		if err := exB.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("B rent ex-1: %v", err)
		}
		if live, _, _, err := exA.ActiveSnapshot(clk.Now()); err != nil || live != 1 {
			t.Fatalf("A ActiveSnapshot MUST see B's ex-1 (not stale empty dirty state): live=%d err=%v", live, err)
		}
		// Controller A reconciles -> MUST adopt B's ex-1, NEVER delete examiner.json or kill ex-1!
		if err := exA.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("A reconcile after disk recovery: %v", err)
		}
		if len(prov.ended) != 0 || len(prov.rented) != 1 || readExaminerState(home) == nil || readExaminerState(home).InstanceID != "ex-1" {
			t.Fatalf("A clobbered B's ex-1 after pre-rental error! ended=%v rented=%d st=%+v", prov.ended, len(prov.rented), readExaminerState(home))
		}

		// Genuine dirty empty state on A after a failed clearState() on ex-1:
		//  (a) Single-controller: while disk STILL holds A's own stale ex-1, ActiveSnapshot MUST NOT
		//      resurrect destroyed ex-1 (live=0), and once writable Reconcile flushes the delete!
		//  (b) Multi-controller: if peer B moved examiner.json to a NEW ex-2 while A was dirty,
		//      A's reloadState MUST adopt B's ex-2 instead of deleting it!
		prov.mu.Lock()
		inst1 := prov.insts["ex-1"]
		inst1.State = "stopped"
		prov.insts["ex-1"] = inst1
		prov.mu.Unlock()
		if err := os.Chmod(home, 0o500); err != nil {
			t.Fatalf("chmod home 0500 before stopped teardown: %v", err)
		}
		// A ends stopped ex-1, clearState() fails os.Remove(examiner.json) -> A.dirty=true, disk still has ex-1!
		_ = exA.Reconcile(context.Background(), &Work{}, testExaminerBudget())
		if live, _, _, err := exA.ActiveSnapshot(clk.Now()); err != nil || live != 0 {
			_ = os.Chmod(home, 0o700)
			t.Fatalf("failed clearState MUST NOT resurrect destroyed ex-1 into ActiveSnapshot: live=%d err=%v", live, err)
		}
		if err := os.Chmod(home, 0o700); err != nil {
			t.Fatalf("chmod home 0700: %v", err)
		}
		// Peer B reconciles openWork -> sees stale ex-1 gone (Wait: B removes stale ex-1 via idle or rents ex-2).
		if err := exB.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("B idle cleanup of stale ex-1: %v", err)
		}
		if err := exB.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("B rent ex-2: %v", err)
		}
		// Now disk moved from lastDisk={ex-1} to {ex-2}: A MUST adopt B's ex-2, NEVER delete examiner.json or kill ex-2!
		if err := exA.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("A dirty recovery after peer rented ex-2: %v", err)
		}
		if len(prov.rented) != 2 || readExaminerState(home) == nil || readExaminerState(home).InstanceID != "ex-2" {
			t.Fatalf("A dirty recovery clobbered peer B's ex-2! rented=%d ended=%v st=%+v", len(prov.rented), prov.ended, readExaminerState(home))
		}
		if prov.insts["ex-2"].ID != "ex-2" {
			t.Fatalf("peer B's ex-2 MUST remain alive on provider: insts=%+v", prov.insts)
		}

		// Single-controller flush after failed clearState: destroy ex-2 on read-only home, then
		// restore write permission and verify A flushes the removal of examiner.json (never stalls).
		prov.mu.Lock()
		inst2 := prov.insts["ex-2"]
		inst2.State = "stopped"
		prov.insts["ex-2"] = inst2
		prov.mu.Unlock()
		if err := os.Chmod(home, 0o500); err != nil {
			t.Fatalf("chmod home 0500 before ex-2 teardown: %v", err)
		}
		_ = exA.Reconcile(context.Background(), &Work{}, testExaminerBudget())
		if err := os.Chmod(home, 0o700); err != nil {
			t.Fatalf("chmod home 0700: %v", err)
		}
		if err := exA.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("A single-controller dirty flush reconcile: %v", err)
		}
		if st := readExaminerState(home); st != nil {
			t.Fatalf("expected stale examiner.json removed on single-controller dirty flush, got %+v", st)
		}
	})
}

// Review 17 P2 #5: Vast ExaminerOffers MUST pass allocated_storage=80 and check the full 80 GB
// hourly rate (including storage) against maxPrice for both admission and accounting.
func TestReview17ExaminerOffersIncludesAllocatedStorageAndFullRate(t *testing.T) {
	var seenReq map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v0/bundles/" && r.Method == "POST" {
			_ = json.NewDecoder(r.Body).Decode(&seenReq)
			// Offer 1: $0.48/h GPU + $0.72/GB-mo storage -> 8 GB default dph_total=$0.488,
			// but full 80 GB contract rate is $0.48 + 80*0.72/720 = $0.560/h (> $0.50 cap!).
			// Offer 2: $0.40/h GPU + $0.72/GB-mo storage -> full 80 GB rate is $0.480/h (<= $0.50 cap).
			_ = json.NewEncoder(w).Encode(map[string]any{
				"offers": []map[string]any{
					{"id": 101, "gpu_name": "RTX 5090", "num_gpus": 1, "dph_base": 0.48, "dph_total": 0.488, "storage_cost": 0.72, "cuda_max_good": 13.2, "inet_down": 2000},
					{"id": 102, "gpu_name": "RTX 5090", "num_gpus": 1, "dph_base": 0.40, "dph_total": 0.408, "storage_cost": 0.72, "cuda_max_good": 13.2, "inet_down": 2000},
				},
			})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	vm := newVast(srv.URL, "test-vast-key")
	offers, err := vm.ExaminerOffers(context.Background(), 0.50)
	if err != nil {
		t.Fatalf("ExaminerOffers: %v", err)
	}
	gotAlloc, _ := seenReq["allocated_storage"].(float64)
	if int(gotAlloc) != examinerDiskGB {
		t.Fatalf("Vast search query missing allocated_storage=%d: got %#v", examinerDiskGB, seenReq["allocated_storage"])
	}
	if len(offers) != 1 || offers[0].ID != "102" {
		t.Fatalf("expected offer 101 ($0.560/h at 80GB) rejected and only offer 102 accepted, got %+v", offers)
	}
	if offers[0].Price < 0.479 || offers[0].Price > 0.481 {
		t.Fatalf("expected full 80 GB rate $0.480/h on offer 102, got $%.4f/h", offers[0].Price)
	}
}

// Review 17 P2 #6: ending an untracked owner-labelled orphan during Reconcile records its spend to
// today's total; daily budget admission MUST recompute spend immediately before renting so an
// orphan that exhausts --daily-budget blocks renting a new examiner in the same tick.
func TestReview17RecoveringOrphanExhaustsDailyBudgetBeforeRent(t *testing.T) {
	t.Run("via_reconcileExaminerOnce_fleet_budget", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		st := newFakeStet(t)
		st.work = Work{Rounds: []WorkRound{{ID: "s3-o00060", Entries: 1, ExamAt: 0}}}
		exProv := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.40}},
			insts: map[string]ExaminerInstance{
				"ex-stray": {
					ID: "ex-stray", Label: examinerLabelPrefix + "-" + owner + "-s3-o00059-abcd1234",
					State: "running", Price: 1.20, Created: clk.Now().Add(-30 * time.Minute),
				},
			},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }
		opKey, err := newKey("op")
		if err != nil {
			t.Fatalf("newKey: %v", err)
		}
		f := &Fleet{
			o:        &Options{Home: home, Policy: Policy{MaxWorkers: 2, MaxHourly: 2.00, DailyBudget: 0.50}},
			st:       newStet(st.srv.URL),
			op:       opKey,
			out:      io.Discard,
			now:      clk.Now,
			examiner: ex,
		}
		if err := reconcileExaminerOnce(context.Background(), f); err != nil {
			t.Fatalf("reconcileExaminerOnce: %v", err)
		}
		if len(exProv.ended) != 1 || exProv.ended[0] != "ex-stray" {
			t.Fatalf("expected stray orphan ended, ended=%v", exProv.ended)
		}
		if len(exProv.rented) != 0 {
			t.Fatalf("MUST NOT rent new examiner after orphan spend ($0.60) exhausted $0.50 daily budget! rented=%d", len(exProv.rented))
		}
		spentToday, err := f.spent(&ledger{Pods: map[string]*podRecord{}}, startOfDay(clk.Now()))
		if err != nil || spentToday < 0.599 || spentToday > 0.601 {
			t.Fatalf("expected $0.60 logged to spent.jsonl, got $%.4f err=%v", spentToday, err)
		}
	})

	t.Run("direct_Reconcile_without_CurrentSpentToday_callback", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC))
		home := t.TempDir()
		owner := loadOrCreateExaminerOwner(home)
		exProv := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.40}},
			insts: map[string]ExaminerInstance{
				"ex-stray": {
					ID: "ex-stray", Label: examinerLabelPrefix + "-" + owner + "-s3-o00059-abcd1234",
					State: "running", Price: 1.20, Created: clk.Now().Add(-30 * time.Minute),
				},
			},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00060", Entries: 1, ExamAt: 0}}}
		if err := ex.Reconcile(context.Background(), openWork, ExaminerBudget{MaxWorkers: 2, MaxHourly: 2.00, SpentToday: 0.0, DailyBudget: 0.50}); err != nil {
			t.Fatalf("Reconcile: %v", err)
		}
		if len(exProv.ended) != 1 || exProv.ended[0] != "ex-stray" {
			t.Fatalf("expected stray ended, ended=%v", exProv.ended)
		}
		if len(exProv.rented) != 0 {
			t.Fatalf("direct Reconcile MUST also block rental after orphan exhausted $0.50 daily budget: rented=%d", len(exProv.rented))
		}
	})
}

// Review 17 P2 #7:
//
//	(a) A listing outage with a working delete endpoint MUST still attempt EndExaminer on known due
//	    instance IDs in both EnforceTimeout (past TTL) and Reconcile (no demand).
//	(b) When the provider confirms a contract is still live past TTL while refusing deletion,
//	    accrual MUST keep counting through the latest observation time ($2.00 at T0+2h, not $0.75).
func TestReview17ListingOutageTeardownAndLiveAccrualPastTTL(t *testing.T) {
	t.Run("listing_outage_still_ends_timed_out_and_idle_known_ids", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 1.00}},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		ex.onSpent = func(rec podRecord) error { return appendExaminerSpend(filepath.Join(home, "spent.jsonl"), rec) }
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00070", Entries: 1, ExamAt: 0}}}

		// 1. Timeout path under listing outage: rent ex-1, fail ListExaminers only, advance past 45m.
		if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("rent ex-1: %v", err)
		}
		prov.mu.Lock()
		prov.listErr = errors.New("Vast GET /instances 503")
		prov.mu.Unlock()

		// While within TTL and round still open, listing outage must NOT destroy healthy ex-1.
		clk.Advance(5 * time.Minute)
		if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
			t.Fatalf("expected listing error")
		}
		if len(prov.ended) != 0 {
			t.Fatalf("must not destroy healthy in-TTL ex-1 during listing outage, ended=%v", prov.ended)
		}

		// Past 45m TTL, EnforceTimeout MUST call EndExaminer("ex-1") even though ListExaminers fails!
		clk.Advance(45 * time.Minute)
		if err := ex.EnforceTimeout(context.Background()); err == nil {
			t.Fatalf("expected listing error returned alongside direct teardown")
		}
		if len(prov.ended) != 1 || prov.ended[0] != "ex-1" {
			t.Fatalf("EnforceTimeout MUST end known timed-out ex-1 during listing outage! ended=%v", prov.ended)
		}
		if readExaminerState(home) != nil {
			t.Fatalf("examiner.json should be cleared once ex-1 ended, got %+v", readExaminerState(home))
		}

		// 2. Idle / no-demand path under listing outage: rent ex-2, fail listing, Reconcile idle.
		prov.mu.Lock()
		prov.listErr = nil
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("rent ex-2: %v", err)
		}
		clk.Advance(10 * time.Minute)
		prov.mu.Lock()
		prov.listErr = errors.New("Vast GET /instances 503")
		prov.mu.Unlock()
		if err := ex.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
			t.Fatalf("expected listing error returned alongside direct idle teardown")
		}
		if len(prov.ended) != 2 || prov.ended[1] != "ex-2" {
			t.Fatalf("Reconcile idle MUST end known ex-2 during listing outage! ended=%v", prov.ended)
		}
		if readExaminerState(home) != nil {
			t.Fatalf("examiner.json should be cleared after idle teardown, got %+v", readExaminerState(home))
		}
	})

	t.Run("live_machine_refusing_delete_accrues_through_observation_time", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 1.00}},
			endErr: map[string]error{"ex-1": errors.New("Vast DELETE 500")},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00071", Entries: 1, ExamAt: 0}}}
		if err := ex.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("rent ex-1: %v", err)
		}
		// Two hours later, provider still lists $1/h ex-1 as running and refuses deletion.
		clk.Advance(2 * time.Hour)
		if err := ex.EnforceTimeout(context.Background()); err == nil {
			t.Fatalf("expected delete refusal error")
		}
		if live, hourly, acc, err := ex.ActiveSnapshot(clk.Now()); err != nil || live != 1 || hourly != 1.00 || acc < 1.999 || acc > 2.001 {
			t.Fatalf("ActiveSnapshot must accrue full $2.00 at T0+2h (not $0.75): live=%d hourly=%.2f acc=%.4f err=%v", live, hourly, acc, err)
		}
		if live, hourly, acc, err := activeExaminerFromDisk(home, clk.Now(), 45*time.Minute, 1.50); err != nil || live != 1 || hourly != 1.00 || acc < 1.999 || acc > 2.001 {
			t.Fatalf("activeExaminerFromDisk must accrue full $2.00 at T0+2h (not $0.75): live=%d hourly=%.2f acc=%.4f err=%v", live, hourly, acc, err)
		}
	})
}

// Review 18 P1 #1: when examiner.json is unreadable while a round still needs an exam, a newly
// started controller MUST NOT treat the active round's owned live examiner as a stray and destroy
// it; independently expired instances are still ended, and idle (no-demand) cleanup still runs.
func TestReview18UnreadableStateDefersStrayDeletionWhileDemandActive(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
	home := t.TempDir()
	prov := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	exA := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00080", Entries: 1, ExamAt: 0}}}
	if err := exA.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
		t.Fatalf("A rent ex-1: %v", err)
	}
	owner := exA.ownerID()

	// Also add an independently expired owned instance (created 60m ago > 45m timeout).
	prov.mu.Lock()
	prov.insts["ex-old"] = ExaminerInstance{
		ID:      "ex-old",
		Label:   examinerLabelPrefix + "-" + owner + "-s3-o00079-deadbeef",
		State:   "running",
		Price:   0.62,
		Created: clk.Now().Add(-60 * time.Minute),
	}
	prov.mu.Unlock()

	// Make examiner.json unreadable (simulate permission/I/O error on restarted controller B).
	statePath := filepath.Join(home, "examiner.json")
	if err := os.Chmod(statePath, 0o000); err != nil {
		t.Fatalf("chmod examiner.json 0000: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(statePath, 0o600) })
	// Verify test user isn't running as root where chmod 0000 is still readable.
	if _, rerr := os.ReadFile(statePath); rerr == nil {
		// Fall back to corrupt JSON so readExaminerStateStrict deterministically returns loadErr.
		_ = os.Chmod(statePath, 0o600)
		if err := os.WriteFile(statePath, []byte("{corrupt"), 0o600); err != nil {
			t.Fatalf("write corrupt examiner.json: %v", err)
		}
	}

	exB := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
	err := exB.Reconcile(context.Background(), openWork, testExaminerBudget())
	if err == nil {
		t.Fatalf("expected non-nil load error from B.Reconcile when examiner.json is unreadable")
	}
	// Active in-TTL ex-1 MUST remain alive; expired ex-old MUST be destroyed; no replacement rented.
	if _, ok := prov.insts["ex-1"]; !ok {
		t.Fatalf("B destroyed active in-TTL ex-1 while examiner.json was unreadable! ended=%v", prov.ended)
	}
	if _, ok := prov.insts["ex-old"]; ok {
		t.Fatalf("B failed to destroy independently expired ex-old while examiner.json was unreadable")
	}
	if len(prov.ended) != 1 || prov.ended[0] != "ex-old" {
		t.Fatalf("expected only expired ex-old ended, got ended=%v", prov.ended)
	}
	if len(prov.rented) != 1 {
		t.Fatalf("B must not rent while examiner.json is unreadable: rented=%d", len(prov.rented))
	}

	// When no round needs an exam (idle / Case 1), B still destroys ex-1 even if examiner.json is unreadable.
	if err := exB.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err == nil {
		t.Fatalf("expected loadErr returned alongside idle cleanup")
	}
	if _, ok := prov.insts["ex-1"]; ok {
		t.Fatalf("idle Case 1 cleanup must still destroy ex-1 when no round needs an exam")
	}
}

// Review 18 Gate 4 regressions (P1-1, P2-1, P2-2):
//
//	(a) P1-1 Trace A & B: while examiner.json is unreadable (0000 / corrupt), updateUnresolved and
//	    queuePendingSpend MUST NOT overwrite or remove examiner.json on disk; when readability
//	    returns, both controllers recover the intact active ex-1 AND preserve pending obligations.
//	(b) P2-1: under loadErr != nil (e.g. dirty flush failure), a missing Unresolved orphan MUST
//	    still be confirmed via confirmMissingUnresolved / EndExaminer instead of silently dropped.
//	(c) P2-2: under loadErr != nil, ending own timed-out saved.InstanceID MUST take the proxy
//	    offline (HTTP 503), clear in-memory active state (ActiveSnapshot live=0), and bill using
//	    saved.Price / saved.RoundID.
func TestReview18Gate4UnreadableStatePreservesDiskAndHandlesUnresolvedAndOwnTimeout(t *testing.T) {
	t.Run("unreadable_disk_never_clobbered_by_updateUnresolved_or_queuePendingSpend", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		// Peer A runs an idle tick first so exA.loaded == true (Trace A).
		exA := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		if err := exA.Reconcile(context.Background(), &Work{}, testExaminerBudget()); err != nil {
			t.Fatalf("exA idle tick: %v", err)
		}
		// Peer B rents ex-1 and persists it to examiner.json.
		exB := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00082", Entries: 1, ExamAt: 0}}}
		if err := exB.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("exB rent ex-1: %v", err)
		}
		owner := exB.ownerID()
		statePath := filepath.Join(home, "examiner.json")
		origBytes, err := os.ReadFile(statePath)
		if err != nil {
			t.Fatalf("read original examiner.json: %v", err)
		}

		// Add two expired owned instances: ex-fail (delete fails -> updateUnresolved) and
		// ex-spent (delete succeeds but onSpent fails -> queuePendingSpend).
		prov.mu.Lock()
		prov.insts["ex-fail"] = ExaminerInstance{
			ID: "ex-fail", Label: examinerLabelPrefix + "-" + owner + "-s3-o00078-11111111",
			State: "running", Price: 0.62, Created: clk.Now().Add(-60 * time.Minute),
		}
		prov.insts["ex-spent"] = ExaminerInstance{
			ID: "ex-spent", Label: examinerLabelPrefix + "-" + owner + "-s3-o00077-22222222",
			State: "running", Price: 0.62, Created: clk.Now().Add(-60 * time.Minute),
		}
		prov.endErr = map[string]error{"ex-fail": errors.New("Vast DELETE 503")}
		prov.mu.Unlock()

		// Make examiner.json unreadable while home/ stays writable (0700).
		if err := os.Chmod(statePath, 0o000); err != nil {
			t.Fatalf("chmod examiner.json 0000: %v", err)
		}
		t.Cleanup(func() { _ = os.Chmod(statePath, 0o600) })
		usedCorrupt := false
		if _, rerr := os.ReadFile(statePath); rerr == nil {
			usedCorrupt = true
			_ = os.Chmod(statePath, 0o600)
			if err := os.WriteFile(statePath, []byte("{corrupt"), 0o600); err != nil {
				t.Fatalf("write corrupt examiner.json: %v", err)
			}
		}

		// Trace A: long-running exA (e.loaded == true) hits updateUnresolved on ex-fail -> MUST NOT overwrite examiner.json!
		if err := exA.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
			t.Fatalf("expected error from exA.Reconcile")
		}
		// Trace B: brand-new exC (e.loaded == false) with failing onSpent hits queuePendingSpend on ex-spent -> MUST NOT overwrite examiner.json!
		exC := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		exC.onSpent = func(podRecord) error { return errors.New("spent.jsonl disk full") }
		if err := exC.Reconcile(context.Background(), openWork, testExaminerBudget()); err == nil {
			t.Fatalf("expected error from exC.Reconcile")
		}

		// Restore readability of examiner.json and confirm original ex-1 file was NEVER overwritten or removed!
		if usedCorrupt {
			if err := os.WriteFile(statePath, origBytes, 0o600); err != nil {
				t.Fatalf("restore orig examiner.json: %v", err)
			}
		} else {
			if err := os.Chmod(statePath, 0o600); err != nil {
				t.Fatalf("chmod examiner.json 0600: %v", err)
			}
		}
		gotBytes, err := os.ReadFile(statePath)
		if err != nil {
			t.Fatalf("examiner.json was deleted while unreadable: %v", err)
		}
		if string(gotBytes) != string(origBytes) {
			t.Fatalf("examiner.json was OVERWRITTEN while unreadable!\ngot:  %s\nwant: %s", string(gotBytes), string(origBytes))
		}

		// Now exA reconciles with examiner.json readable again -> MUST adopt ex-1 intact AND keep ex-fail tracked!
		prov.mu.Lock()
		delete(prov.endErr, "ex-fail")
		prov.mu.Unlock()
		if err := exA.Reconcile(context.Background(), openWork, testExaminerBudget()); err != nil {
			t.Fatalf("exA recovery reconcile: %v", err)
		}
		if _, ok := prov.insts["ex-1"]; !ok {
			t.Fatalf("exA destroyed ex-1 after readability recovered! ended=%v", prov.ended)
		}
		st := readExaminerState(home)
		if st == nil || st.InstanceID != "ex-1" {
			t.Fatalf("ex-1 not preserved in examiner.json after recovery: %+v", st)
		}
	})

	t.Run("missing_unresolved_and_own_timed_out_instance_under_loadErr", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		var spendRecs []podRecord
		ex.onSpent = func(rec podRecord) error {
			spendRecs = append(spendRecs, rec)
			return nil
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		addr, stop, err := ex.Start(ctx)
		if err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer stop()

		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00083", Entries: 1, ExamAt: 0}}}
		if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
			t.Fatalf("rent ex-1: %v", err)
		}
		launch := prov.rented[0]
		vllm := startTLSServerWithCert(t, launch.CertPEM, launch.KeyPEM, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"id": defaultExaminerModel + "@" + defaultExaminerRevision}},
			})
		}))
		defer vllm.Close()
		vu, _ := url.Parse(vllm.URL)
		prov.mu.Lock()
		inst1 := prov.insts["ex-1"]
		inst1.State, inst1.Status, inst1.PublicIP, inst1.HostPort = "running", "running", vu.Hostname(), vu.Port()
		prov.insts["ex-1"] = inst1
		prov.mu.Unlock()
		if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err != nil {
			t.Fatalf("activate proxy: %v", err)
		}
		// Seed a dirty Unresolved orphan ex-orphan that is omitted from ListExaminers and whose
		// EndExaminer fails, while home/ is read-only so reloadState(true) returns loadErr != nil.
		ex.mu.Lock()
		ex.dirty = true
		ex.state.Unresolved = []ExaminerInstance{{
			ID: "ex-orphan", State: "running", Price: 0.50,
			Created: clk.Now().Add(-10 * time.Minute), ObservedLiveAt: clk.Now(),
		}}
		ex.mu.Unlock()
		prov.mu.Lock()
		prov.endErr = map[string]error{"ex-orphan": errors.New("Vast DELETE 503")}
		prov.mu.Unlock()
		if err := os.Chmod(home, 0o500); err != nil {
			t.Fatalf("chmod home 0500: %v", err)
		}
		t.Cleanup(func() { _ = os.Chmod(home, 0o700) })

		// Advance past 45m TTL so own ex-1 expires while ex-orphan is missing and failing delete.
		clk.Advance(50 * time.Minute)
		if err := ex.Reconcile(ctx, openWork, testExaminerBudget()); err == nil {
			t.Fatalf("expected error from Reconcile under read-only home + unresolved orphan")
		}
		// P2-2: ex-1 must be ended, proxy must be offline (503), and ActiveSnapshot must NOT count ex-1!
		if _, ok := prov.insts["ex-1"]; ok {
			t.Fatalf("expired ex-1 was not ended under loadErr")
		}
		resp, err := http.Get("http://" + addr.String() + "/v1/models")
		if err != nil {
			t.Fatalf("GET proxy: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("proxy stayed online after own timed-out ex-1 ended under loadErr: HTTP %d, want 503", resp.StatusCode)
		}
		// P2-1: missing ex-orphan whose EndExaminer failed MUST remain in Unresolved (live=1 for ex-orphan only, hourly=0.50, NOT ex-1's 0.62!).
		live, hourly, _, err := ex.ActiveSnapshot(clk.Now())
		if err != nil || live != 1 || hourly < 0.499 || hourly > 0.501 {
			t.Fatalf("expected only ex-orphan ($0.50/h) remaining active after ex-1 ended: live=%d hourly=%.3f err=%v", live, hourly, err)
		}
		foundEx1Spend := false
		for _, r := range spendRecs {
			if r.ID == "ex-1" && r.Cost >= 0.619 && r.Cost <= 0.621 {
				foundEx1Spend = true
			}
		}
		if !foundEx1Spend {
			t.Fatalf("expected ex-1 spend billed at saved.Price ($0.62/h), got %+v", spendRecs)
		}
	})
}

// Review 19 P2 #3: standalone `stet-cloud examiner` (`reconcileExaminerOnce`) MUST reconcile owned
// worker rentals with the markets before admitting a new examiner so a running worker missing from
// `fleet.json` (e.g. after a failed ledger write) is counted against `--max` and `--max-hourly`,
// while teardown remains available when market accounting is unavailable.
func TestReview19StandaloneExaminerReconcilesUntrackedLiveWorker(t *testing.T) {
	clk := newTestClock(time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC))
	home := t.TempDir()
	rp, st := newFakeRunPod(t), newFakeStet(t)
	f, _ := testFleet(t, rp, st, 1)
	f.o.Home = home
	f.now = clk.Now
	f.o.Policy.MaxWorkers = 1
	f.o.Policy.MaxHourly = 0.70
	f.o.Policy.DailyBudget = 10.0

	// Put a running $0.43/h worker under registered fleet handle `stet-rp-01` on RunPod while
	// local fleet.json is completely empty (simulating successful create + failed ledger write).
	rp.pods["pod-untracked"] = &Pod{
		ID: "pod-untracked", Name: "stet-rp-01",
		Status: "RUNNING", Cost: 0.43, CreatedAt: clk.Now().Add(-5 * time.Minute),
	}
	st.work = Work{Rounds: []WorkRound{{ID: "s3-o00090", Entries: 1, ExamAt: 0}}}
	exProv := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, exProv, clk.Now, nil, false)
	f.examiner = ex

	if err := reconcileExaminerOnce(context.Background(), f); err != nil {
		t.Fatalf("reconcileExaminerOnce: %v", err)
	}
	if len(exProv.rented) != 0 {
		t.Fatalf("standalone examiner exceeded --max 1 / $0.70/h alongside untracked $0.43/h worker! rented=%d", len(exProv.rented))
	}
	// Verify reconciliation also adopted the live worker into fleet.json.
	l, err := f.readLedger()
	if err != nil || l.Pods[ledgerKey("runpod", "pod-untracked")] == nil {
		t.Fatalf("expected untracked worker adopted into fleet.json: pods=%+v err=%v", l.Pods, err)
	}

	// Now verify teardown still runs when market listing fails (accounting unavailable):
	// remove the worker, rent an examiner, then fail RunPod Rentals() and publish ExamAt > 0.
	delete(rp.pods, "pod-untracked")
	if err := reconcileExaminerOnce(context.Background(), f); err != nil {
		t.Fatalf("reconcileExaminerOnce rent examiner: %v", err)
	}
	if len(exProv.rented) != 1 {
		t.Fatalf("expected examiner rented once worker ended, got rented=%d", len(exProv.rented))
	}
	f.markets = []Market{&probeSwapMarket{
		Market: f.market("runpod"),
		onRentals: func() ([]Rental, error) {
			return nil, errors.New("RunPod API 503")
		},
	}}
	st.work = Work{Rounds: []WorkRound{{ID: "s3-o00090", Entries: 1, ExamAt: clk.Now().Unix()}}}
	if err := reconcileExaminerOnce(context.Background(), f); err != nil {
		t.Fatalf("teardown under market accounting error: %v", err)
	}
	if len(exProv.ended) != 1 || len(exProv.insts) != 0 {
		t.Fatalf("expected live examiner torn down on ExamAt > 0 even when market Rentals() fails: ended=%v insts=%d", exProv.ended, len(exProv.insts))
	}

	// Gate 4 P2-1: a production standalone examiner daemon (f.fixed == false) whose initial
	// f.keys already has 1 key MUST reload newly added worker keys from <home>/workers/*.json
	// on subsequent ticks so a worker running under a newly added key is still counted!
	homeDyn := t.TempDir()
	rpDyn, stDyn := newFakeRunPod(t), newFakeStet(t)
	k1, err := newKey("stet-rp-01")
	if err != nil {
		t.Fatalf("newKey k1: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(homeDyn, "workers"), 0o700); err != nil {
		t.Fatalf("mkdir workers: %v", err)
	}
	if err := k1.save(filepath.Join(homeDyn, "workers", "stet-rp-01.json")); err != nil {
		t.Fatalf("save k1: %v", err)
	}
	opDyn, _ := newKey("op")
	exProvDyn := &fakeExaminerProv{
		offers: []ExaminerOffer{{ID: "off-5090", Card: "RTX 5090", Price: 0.62}},
	}
	exDyn := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, homeDyn, exProvDyn, clk.Now, nil, false)
	fDyn, _ := testFleet(t, rpDyn, stDyn, 1)
	fDyn.fixed = false
	fDyn.o.Home = homeDyn
	fDyn.o.Credentials = ""
	fDyn.now = clk.Now
	fDyn.o.Policy.MaxWorkers = 1
	fDyn.o.Policy.MaxHourly = 0.70
	fDyn.o.Policy.DailyBudget = 10.0
	fDyn.keys = []*Key{k1}
	fDyn.examiner = exDyn
	_ = opDyn
	// Add a brand-new key `stet-rp-02` to disk AFTER daemon startup, and put a live worker under it!
	k2, _ := newKey("stet-rp-02")
	if err := k2.save(filepath.Join(homeDyn, "workers", "stet-rp-02.json")); err != nil {
		t.Fatalf("save k2: %v", err)
	}
	rpDyn.pods["pod-new-key"] = &Pod{
		ID: "pod-new-key", Name: "stet-rp-02",
		Status: "RUNNING", Cost: 0.43, CreatedAt: clk.Now().Add(-3 * time.Minute),
	}
	stDyn.work = Work{Rounds: []WorkRound{{ID: "s3-o00091", Entries: 1, ExamAt: 0}}}
	if err := reconcileExaminerOnce(context.Background(), fDyn); err != nil {
		t.Fatalf("reconcileExaminerOnce with newly added worker key: %v", err)
	}
	if len(exProvDyn.rented) != 0 {
		t.Fatalf("standalone examiner missed worker under newly added key stet-rp-02! rented=%d", len(exProvDyn.rented))
	}
}

// Review 19 P2 #4: examinerOnstartScript MUST:
//   - read STET_EXAMINER_EXPIRE_AT and STET_EXAMINER_TTL_SECONDS from /etc/environment (never
//     falling back to shell default 2700 when delivered via /etc/environment),
//   - arm the self-destruct watchdog BEFORE fallible TLS materialization, and
//   - compute remaining sleep from the absolute STET_EXAMINER_EXPIRE_AT deadline (even after a
//     slow image pull).
func TestReview19OnstartWatchdogUsesEtcEnvironmentAbsoluteExpiryAndArmsBeforeTLS(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
	cert, key, err := generateExaminerCert(time.Now(), 5*time.Minute)
	if err != nil {
		t.Fatalf("generateExaminerCert: %v", err)
	}
	// Helper that rewrites /etc/environment and /tmp/stet-* into an isolated temp dir, captures
	// the computed delay from /tmp/stet-selfdestruct.py without sleeping or calling Vast.
	runSandboxed := func(t *testing.T, launch ExaminerLaunch, etcEnv string) (rc int, watchdogDelay string, watchdogArmed bool, failArmed bool) {
		t.Helper()
		dir := t.TempDir()
		envPath := filepath.Join(dir, "environment")
		if err := os.WriteFile(envPath, []byte(etcEnv), 0o600); err != nil {
			t.Fatalf("write env file: %v", err)
		}
		script := examinerOnstartScript(launch, launch.Model+"@"+launch.Revision)
		// Rewrite paths into temp dir and stub time.sleep + vllm so the script runs fast & safely.
		script = strings.ReplaceAll(script, "/etc/environment", envPath)
		script = strings.ReplaceAll(script, "/tmp/stet-", filepath.Join(dir, "stet-"))
		script = strings.ReplaceAll(script,
			`if delay > 0: time.sleep(delay)`,
			`open(`+strconv.Quote(filepath.Join(dir, "delays.log"))+`, "a").write(f"{sys.argv[1:]!r}:{delay}\n"); sys.exit(0)`)
		script = strings.ReplaceAll(script, "nohup sh "+filepath.Join(dir, "stet-vllm.sh"), "true # skip vllm")
		// Run watchdog synchronously in test so delays.log is guaranteed flushed before return.
		script = strings.ReplaceAll(script, "nohup python3 "+filepath.Join(dir, "stet-selfdestruct.py")+" >", "python3 "+filepath.Join(dir, "stet-selfdestruct.py")+" >")
		script = strings.ReplaceAll(script, "nohup python3 "+filepath.Join(dir, "stet-selfdestruct.py")+" 15 >", "python3 "+filepath.Join(dir, "stet-selfdestruct.py")+" 15 >")
		script = strings.ReplaceAll(script, "2>&1 &", "2>&1")

		cmd := exec.Command("sh", "-c", script)
		cmd.Env = []string{"PATH=" + os.Getenv("PATH")} // deliberately empty STET_EXAMINER_* in process env!
		out, runErr := cmd.CombinedOutput()
		if runErr != nil {
			var ee *exec.ExitError
			if errors.As(runErr, &ee) {
				rc = ee.ExitCode()
			} else {
				t.Fatalf("exec sh: %v (%s)", runErr, string(out))
			}
		}
		raw, _ := os.ReadFile(filepath.Join(dir, "delays.log"))
		for _, ln := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
			if strings.HasPrefix(ln, "[]:") {
				watchdogArmed = true
				watchdogDelay = strings.TrimPrefix(ln, "[]:")
			}
			if strings.HasPrefix(ln, "['15']:") {
				failArmed = true
			}
		}
		return rc, watchdogDelay, watchdogArmed, failArmed
	}

	certHex := hex.EncodeToString(cert)
	keyHex := hex.EncodeToString(key)

	// 1. /etc/environment delivers STET_EXAMINER_TTL_SECONDS=300 (5m) with no exported shell vars:
	//    watchdog MUST compute 300s, NOT 2700s!
	rc, delayStr, armed, _ := runSandboxed(t,
		ExaminerLaunch{Model: defaultExaminerModel, Revision: defaultExaminerRevision, TTLSeconds: 2700},
		fmt.Sprintf("STET_EXAMINER_TTL_SECONDS=300\nSTET_EXAMINER_CERT_HEX=%s\nSTET_EXAMINER_KEY_HEX=%s\nSTET_EXAMINER_API_KEY=k\n", certHex, keyHex),
	)
	if rc != 0 || !armed || delayStr != "300" {
		t.Fatalf("/etc/environment TTL=300: rc=%d armed=%v delay=%q, want rc=0 armed=true delay=\"300\"", rc, armed, delayStr)
	}

	// 2. /etc/environment delivers absolute STET_EXAMINER_EXPIRE_AT = now+120 (e.g. 5m TTL after 3m pull):
	//    watchdog MUST sleep <= 120s (not 300s or 2700s)!
	exp := time.Now().Unix() + 120
	rc, delayStr, armed, _ = runSandboxed(t,
		ExaminerLaunch{Model: defaultExaminerModel, Revision: defaultExaminerRevision, TTLSeconds: 300, ExpireAt: exp},
		fmt.Sprintf("STET_EXAMINER_EXPIRE_AT=%d\nSTET_EXAMINER_TTL_SECONDS=300\nSTET_EXAMINER_CERT_HEX=%s\nSTET_EXAMINER_KEY_HEX=%s\nSTET_EXAMINER_API_KEY=k\n", exp, certHex, keyHex),
	)
	dVal, perr := strconv.Atoi(delayStr)
	if rc != 0 || !armed || perr != nil || dVal < 110 || dVal > 120 {
		t.Fatalf("absolute STET_EXAMINER_EXPIRE_AT: rc=%d armed=%v delay=%q (err=%v), want 110..120", rc, armed, delayStr, perr)
	}

	// 3. Missing TLS material causes setup to fail (rc != 0), YET watchdog MUST ALREADY BE ARMED
	//    AND prompt 15s self-destruct triggered!
	rc, delayStr, armed, failArmed := runSandboxed(t,
		ExaminerLaunch{Model: defaultExaminerModel, Revision: defaultExaminerRevision, TTLSeconds: 300},
		"STET_EXAMINER_TTL_SECONDS=300\n",
	)
	if rc == 0 || !armed || delayStr != "300" || !failArmed {
		t.Fatalf("missing TLS setup failure: rc=%d armed=%v delay=%q failArmed=%v, want rc!=0 armed=true delay=300 failArmed=true", rc, armed, delayStr, failArmed)
	}
}

// Review 18 P2 #3: an offer with missing/zero/non-positive price MUST be rejected by both
// vastMarket.ExaminerOffers and Examiner.Reconcile instead of passing --max-hourly as $0.00/h.
func TestReview18UnpricedOfferRejectedByVastAndReconcile(t *testing.T) {
	t.Run("vast_ExaminerOffers_rejects_unpriced_offers", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/v0/bundles/" && r.Method == "POST" {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"offers": []map[string]any{
						{"id": 201, "gpu_name": "RTX 5090", "num_gpus": 1, "cuda_max_good": 13.2, "inet_down": 2000},
						{"id": 202, "gpu_name": "RTX 5090", "num_gpus": 1, "dph_base": 0.0, "dph_total": 0.0, "cuda_max_good": 13.2, "inet_down": 2000},
						{"id": 203, "gpu_name": "RTX 5090", "num_gpus": 1, "dph_base": -0.5, "dph_total": -0.5, "cuda_max_good": 13.2, "inet_down": 2000},
					},
				})
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		vm := newVast(srv.URL, "test-vast-key")
		offers, err := vm.ExaminerOffers(context.Background(), 0.10)
		if err != nil {
			t.Fatalf("ExaminerOffers: %v", err)
		}
		if len(offers) != 0 {
			t.Fatalf("expected all unpriced/non-positive offers rejected, got %+v", offers)
		}
	})

	t.Run("Reconcile_rejects_non_positive_offer_price", func(t *testing.T) {
		clk := newTestClock(time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC))
		home := t.TempDir()
		prov := &fakeExaminerProv{
			offers: []ExaminerOffer{
				{ID: "off-zero", Card: "RTX 5090", Price: 0},
				{ID: "off-neg", Card: "RTX 5090", Price: -0.25},
				{ID: "off-over", Card: "RTX 5090", Price: 0.62},
			},
		}
		ex := newExaminer(ExaminerConfig{Addr: "127.0.0.1:0", Timeout: 45 * time.Minute, MaxPrice: 1.50}, home, prov, clk.Now, nil, false)
		openWork := &Work{Rounds: []WorkRound{{ID: "s3-o00081", Entries: 1, ExamAt: 0}}}
		budget := ExaminerBudget{MaxWorkers: 2, MaxHourly: 0.10, DailyBudget: 10.0}
		if err := ex.Reconcile(context.Background(), openWork, budget); err != nil {
			t.Fatalf("Reconcile: %v", err)
		}
		if len(prov.rented) != 0 {
			t.Fatalf("Reconcile MUST NOT rent $0/negative offer under $0.10/h cap! rented=%+v", prov.rented)
		}
	})
}
