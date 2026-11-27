package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// --- automated ephemeral Season 3 examiner (1x RTX 5090 vLLM) ----------------------------------
//
// Season 3 synthesizes its 640 fresh reasoning items once at each round draw using pinned
// thinkingmachines/Inkling-2-Mini via vLLM. Running a 1x RTX 5090 continuously would cost ~$450/mo,
// whereas each round needs it only from draw until POST /api/v1/admin/rounds/{id}/exam (~14-18 min
// cold start + synthesis, ~$0.15-$0.20/round).
//
// When stet-cloud runs with --examiner-addr (e.g. 127.0.0.1:8000), it automatically:
//   1. Watches GET /api/v1/admin/work for any open round with ExamAt == 0.
//   2. Rents one verified 1x RTX 5090 on Vast.ai (within --max, --max-hourly, --daily-budget, and
//      --examiner-max-price) running the pinned immutable vLLM Docker digest and pinned
//      Inkling-2-Mini revision with a fresh per-instance 256-bit --api-key, a fresh per-instance
//      ECDSA P-256 TLS certificate, a persistent deployment owner label, and a container-side
//      self-destruct TTL watchdog that survives controller exit.
//   3. Probes https://<ip>:<port>/v1/models over a peer-verified pinned-certificate TLS connection,
//      then enables a local loopback reverse proxy on --examiner-addr that forwards over that same
//      pinned TLS transport injecting Authorization: Bearer <api_key>.
//   4. Destroys its owned Vast.ai instance immediately once every open round has ExamAt > 0 or
//      once hard --examiner-timeout expires (enforced even during Stet server outages, and backed
//      by the container-side self-destruct timer), logging spend to spent.jsonl.

const (
	defaultExaminerImage    = "vllm/vllm-openai@sha256:d24e5fc73f0b859bd76c9d2f6f6fa4da3001afd7637f2610876420f3027ea6c7"
	defaultExaminerModel    = "thinkingmachines/Inkling-2-Mini"
	defaultExaminerRevision = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
	examinerLabelPrefix     = "stet-examiner"
	examinerServerName      = "stet-examiner.internal"
	examinerDiskGB          = 80
)

// ExaminerConfig configures the automated ephemeral vLLM examiner GPU and local auth proxy.
type ExaminerConfig struct {
	Addr     string        // local loopback listen address, e.g. 127.0.0.1:8000 (empty disables)
	Timeout  time.Duration // hard wall-clock TTL for an examiner GPU (default 45m)
	MaxPrice float64       // max $/h for the 1x RTX 5090 examiner (default $1.50/h)
	Image    string        // pinned immutable vLLM Docker image digest
	Model    string        // pinned Hugging Face model ID
	Revision string        // pinned Hugging Face commit SHA
}

// ExaminerBudget passes the fleet's live machine, hourly, and daily budget limits into Reconcile
// so an examiner rental never exceeds --max, --max-hourly, or --daily-budget. Zero MaxWorkers or
// MaxHourly is honored as an explicit zero ceiling (no examiner rental permitted); AccountingErr
// defers new rentals while still allowing teardown and timeout cleanup.
type ExaminerBudget struct {
	LiveWorkers   int
	MaxWorkers    int
	HourlyNow     float64 // live worker $/h (excluding any already-active examiner)
	MaxHourly     float64
	SpentToday    float64
	DailyBudget   float64
	AccountingErr error
}

// ExaminerOffer is one eligible 1x RTX 5090 offer on Vast.ai.
type ExaminerOffer struct {
	ID       string
	Card     string
	Price    float64
	CUDA     float64
	Region   string
	InetDown float64
}

// ExaminerLaunch is what RentExaminer passes to Vast.ai.
type ExaminerLaunch struct {
	Label      string
	Image      string
	Model      string
	Revision   string
	APIKey     string
	CertPEM    []byte
	KeyPEM     []byte
	TTLSeconds int
	HFToken    string
	DiskGB     int
}

// ExaminerInstance is one live or stopped examiner instance returned by Vast.ai.
type ExaminerInstance struct {
	ID       string
	Label    string
	State    string // starting | running | stopped
	Status   string
	PublicIP string
	HostPort string
	Price    float64
	Created  time.Time
}

// ExaminerProvider abstracts Vast.ai examiner provisioning so tests can mock it cleanly.
type ExaminerProvider interface {
	ExaminerOffers(ctx context.Context, maxPrice float64) ([]ExaminerOffer, error)
	RentExaminer(ctx context.Context, offerID string, l ExaminerLaunch) (string, error)
	ListExaminers(ctx context.Context) ([]ExaminerInstance, error)
	EndExaminer(ctx context.Context, id string) error
}

// examinerState is persisted to <home>/examiner.json (0600) so restarts never lose the random
// API key, pinned peer certificate, unrecorded terminal spend charges, or leak an orphaned 5090 on Vast.ai.
type examinerState struct {
	InstanceID    string      `json:"instance_id,omitempty"`
	RoundID       string      `json:"round_id,omitempty"`
	APIKey        string      `json:"api_key,omitempty"`
	CertPEM       string      `json:"cert_pem,omitempty"`
	Upstream      string      `json:"upstream,omitempty"`
	Price         float64     `json:"price,omitempty"`
	Created       time.Time   `json:"created,omitempty"`
	PendingSpends []podRecord `json:"pending_spends,omitempty"`
}

func examinerStatePath(home string) string { return filepath.Join(home, "examiner.json") }
func examinerOwnerPath(home string) string { return filepath.Join(home, "examiner-owner") }

func readExaminerState(home string) *examinerState {
	if home == "" {
		return nil
	}
	raw, err := os.ReadFile(examinerStatePath(home))
	if err != nil {
		return nil
	}
	var st examinerState
	if json.Unmarshal(raw, &st) != nil || (st.InstanceID == "" && len(st.PendingSpends) == 0) {
		return nil
	}
	return &st
}

func writeAtomic0600(dir, pattern, dest string, data []byte) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	fh, err := os.CreateTemp(dir, pattern)
	if err != nil {
		return err
	}
	tmp := fh.Name()
	if err := fh.Chmod(0o600); err != nil {
		_ = fh.Close()
		_ = os.Remove(tmp)
		return err
	}
	if _, err := fh.Write(data); err != nil {
		_ = fh.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := fh.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func writeExaminerState(home string, st *examinerState) error {
	path := examinerStatePath(home)
	if st == nil || (st.InstanceID == "" && len(st.PendingSpends) == 0) {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	raw, err := json.MarshalIndent(st, "", " ")
	if err != nil {
		return err
	}
	return writeAtomic0600(home, "examiner.json.tmp-*", path, raw)
}

func randomHex(nBytes int) (string, error) {
	b := make([]byte, nBytes)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func loadOrCreateExaminerOwner(home string) string {
	if home != "" {
		if raw, err := os.ReadFile(examinerOwnerPath(home)); err == nil {
			if id := strings.TrimSpace(string(raw)); len(id) == 12 {
				if _, err := hex.DecodeString(id); err == nil {
					return id
				}
			}
		}
	}
	id, err := randomHex(6)
	if err != nil {
		id = fmt.Sprintf("%012x", time.Now().UnixNano()&0xffffffffffff)
	}
	if home != "" {
		_ = writeAtomic0600(home, "examiner-owner.tmp-*", examinerOwnerPath(home), []byte(id+"\n"))
	}
	return id
}

// examinerBelongsToOwner reports whether a provider instance label belongs strictly to this
// controller deployment so separate deployments sharing a Vast account never delete each other.
func examinerBelongsToOwner(label, ownerID string) bool {
	return ownerID != "" && strings.HasPrefix(label, examinerLabelPrefix+"-"+ownerID+"-")
}

func filterOwnedExaminers(insts []ExaminerInstance, ownerID, savedID string) []ExaminerInstance {
	var out []ExaminerInstance
	for _, inst := range insts {
		if examinerBelongsToOwner(inst.Label, ownerID) || (savedID != "" && inst.ID == savedID) {
			out = append(out, inst)
		}
	}
	return out
}

// examinerLabelRound extracts the round ID from labels formatted as
// "stet-examiner-<owner>-<round>-<8hex>" or legacy "stet-examiner-<round>[-<8hex>]".
func examinerLabelRound(label, ownerID string) string {
	rest := ""
	if ownerID != "" {
		if r, ok := strings.CutPrefix(label, examinerLabelPrefix+"-"+ownerID+"-"); ok {
			rest = r
		}
	}
	if rest == "" {
		var ok bool
		rest, ok = strings.CutPrefix(label, examinerLabelPrefix+"-")
		if !ok || rest == "" {
			return ""
		}
	}
	if i := strings.LastIndex(rest, "-"); i > 0 && len(rest)-i-1 == 8 {
		if _, err := hex.DecodeString(rest[i+1:]); err == nil {
			return rest[:i]
		}
	}
	return rest
}

// generateExaminerCert creates a fresh per-instance ECDSA P-256 X.509 certificate for peer-verified
// TLS between stet-cloud and the ephemeral Vast.ai vLLM instance.
func generateExaminerCert(now time.Time, ttl time.Duration) (certPEM, keyPEM []byte, err error) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	serialLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serial, err := rand.Int(rand.Reader, serialLimit)
	if err != nil {
		return nil, nil, err
	}
	validFor := max(ttl*2, 24*time.Hour)
	tmpl := &x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: examinerServerName},
		NotBefore:    now.Add(-5 * time.Minute),
		NotAfter:     now.Add(validFor),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames:     []string{examinerServerName},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, err
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, nil, err
	}
	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	return certPEM, keyPEM, nil
}

// examinerTLSConfig builds a strict TLS client configuration that verifies and pins the exact
// per-instance X.509 certificate before sending any credentials or exam material over the network.
func examinerTLSConfig(certPEM []byte, now func() time.Time) (*tls.Config, error) {
	block, _ := pem.Decode(certPEM)
	if block == nil || block.Type != "CERTIFICATE" {
		return nil, errors.New("missing or invalid examiner PEM certificate")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing examiner certificate: %w", err)
	}
	pool := x509.NewCertPool()
	pool.AddCert(cert)
	expectedDER := append([]byte(nil), block.Bytes...)
	return &tls.Config{
		MinVersion: tls.VersionTLS12,
		ServerName: examinerServerName,
		RootCAs:    pool,
		Time:       now,
		VerifyPeerCertificate: func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
			if len(rawCerts) == 0 || !bytes.Equal(rawCerts[0], expectedDER) {
				return errors.New("examiner peer certificate does not match pinned instance certificate")
			}
			return nil
		},
	}, nil
}

func activeExaminerFromDisk(home string, now time.Time, timeout time.Duration) (live int, hourly float64, accrued float64) {
	st := readExaminerState(home)
	if st == nil {
		return 0, 0, 0
	}
	dayStart := startOfDay(now)
	if st.InstanceID != "" {
		live = 1
		hourly = st.Price
		start := st.Created
		if start.IsZero() || start.After(now) {
			start = now
		}
		end := now
		if timeout <= 0 {
			timeout = 45 * time.Minute
		}
		if !st.Created.IsZero() {
			if capAt := st.Created.Add(timeout); end.After(capAt) {
				end = capAt
			}
		}
		rec := podRecord{Cost: st.Price, Start: start, End: end}
		accrued += rec.dollarsBetween(dayStart, end)
	}
	for _, p := range st.PendingSpends {
		if p.ID != "" && p.ID != st.InstanceID && !spentHasExaminer(home, p.ID) {
			end := p.End
			if end.IsZero() {
				end = now
			}
			accrued += p.dollarsBetween(dayStart, end)
		}
	}
	return live, hourly, accrued
}

// Examiner manages the lifecycle of an ephemeral 1x RTX 5090 vLLM server and serves an
// authenticated local loopback reverse proxy for stet-examine.
type Examiner struct {
	cfg     ExaminerConfig
	home    string
	prov    ExaminerProvider
	now     func() time.Time
	say     func(string, ...any)
	dryRun  bool
	hfToken string
	onSpent func(podRecord) error

	fileMu       sync.Mutex
	recorded     map[string]bool
	mu           sync.RWMutex
	loaded       bool
	owner        string
	state        examinerState
	ready        bool
	proxy        *httputil.ReverseProxy
	proxyTr      *http.Transport
	upstream     string
	lastProbeErr string
}

func newExaminer(cfg ExaminerConfig, home string, prov ExaminerProvider, now func() time.Time, say func(string, ...any), dryRun bool) *Examiner {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 45 * time.Minute
	}
	if cfg.MaxPrice <= 0 {
		cfg.MaxPrice = 1.50
	}
	if cfg.Image == "" {
		cfg.Image = defaultExaminerImage
	}
	if cfg.Model == "" {
		cfg.Model = defaultExaminerModel
	}
	if cfg.Revision == "" {
		cfg.Revision = defaultExaminerRevision
	}
	if now == nil {
		now = time.Now
	}
	if say == nil {
		say = func(string, ...any) {}
	}
	return &Examiner{
		cfg:     cfg,
		home:    home,
		prov:    prov,
		now:     now,
		say:     say,
		dryRun:  dryRun,
		hfToken: os.Getenv("HF_TOKEN"),
	}
}

// isLoopbackAddr requires --examiner-addr to bind to localhost/loopback only so the unauthenticated
// local proxy endpoint is never accidentally exposed on a public interface.
func isLoopbackAddr(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Start binds the local loopback HTTP reverse proxy on cfg.Addr and returns the bound address
// and a shutdown function.
func (e *Examiner) Start(ctx context.Context) (net.Addr, func() error, error) {
	if !isLoopbackAddr(e.cfg.Addr) {
		return nil, nil, fmt.Errorf("--examiner-addr %q must be a loopback address (e.g. 127.0.0.1:8000)", e.cfg.Addr)
	}
	ln, err := net.Listen("tcp", e.cfg.Addr)
	if err != nil {
		return nil, nil, fmt.Errorf("listening on --examiner-addr %s: %w", e.cfg.Addr, err)
	}
	srv := &http.Server{Handler: e, ReadHeaderTimeout: 30 * time.Second}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}()
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			e.say("examiner proxy stopped: %v", err)
		}
	}()
	closeFn := func() error {
		sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(sctx)
	}
	return ln.Addr(), closeFn, nil
}

// ServeHTTP proxies requests to the active ephemeral 5090 vLLM server once /v1/models is verified
// over peer-verified TLS, injecting Authorization: Bearer <apiKey>. While no examiner GPU is ready,
// it answers 503.
func (e *Examiner) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	e.mu.RLock()
	ready, proxy := e.ready, e.proxy
	e.mu.RUnlock()
	if !ready || proxy == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"ephemeral examiner GPU is starting or idle"}`))
		return
	}
	proxy.ServeHTTP(w, r)
}

func (e *Examiner) saveState(st *examinerState) error {
	if e.home == "" || e.dryRun {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	return writeExaminerState(e.home, st)
}

func (e *Examiner) persistState(st *examinerState) error {
	if e.home == "" || e.dryRun {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	if st != nil && st.InstanceID != "" {
		e.mu.RLock()
		match := e.state.InstanceID == st.InstanceID && e.state.APIKey == st.APIKey && e.state.CertPEM == st.CertPEM
		pending := append([]podRecord(nil), e.state.PendingSpends...)
		e.mu.RUnlock()
		if !match {
			return nil
		}
		cp := *st
		cp.PendingSpends = pending
		return writeExaminerState(e.home, &cp)
	}
	return writeExaminerState(e.home, st)
}

// setUpstream activates the local proxy ONLY if the active instance ID, API key, and pinned peer
// certificate still match what was probed, preventing a stale background ProbeIfBooting from
// reviving a destroyed VM or corrupting a replacement VM's state.
func (e *Examiner) setUpstream(expectedID, upstream, apiKey, certPEM string) (examinerState, bool, error) {
	u, err := url.Parse(upstream)
	if err != nil {
		return examinerState{}, false, err
	}
	if u.Scheme != "https" {
		return examinerState{}, false, fmt.Errorf("examiner upstream must use https, got %q", u.Scheme)
	}
	tlsCfg, err := examinerTLSConfig([]byte(certPEM), e.now)
	if err != nil {
		return examinerState{}, false, err
	}
	tr := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSClientConfig:       tlsCfg,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: 10 * time.Minute,
		IdleConnTimeout:       90 * time.Second,
	}
	rp := &httputil.ReverseProxy{
		Transport: tr,
		Director: func(req *http.Request) {
			req.URL.Scheme = "https"
			req.URL.Host = u.Host
			req.Host = u.Host
			req.Header.Set("Authorization", "Bearer "+apiKey)
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte(`{"error":"ephemeral examiner upstream unavailable"}`))
		},
	}
	e.mu.Lock()
	if expectedID == "" || e.state.InstanceID != expectedID || e.state.APIKey != apiKey || e.state.CertPEM != certPEM {
		e.mu.Unlock()
		tr.CloseIdleConnections()
		return examinerState{}, false, nil
	}
	oldTr := e.proxyTr
	e.upstream = upstream
	e.proxy = rp
	e.proxyTr = tr
	e.ready = true
	e.state.Upstream = upstream
	st := e.state
	e.mu.Unlock()
	if oldTr != nil {
		oldTr.CloseIdleConnections()
	}
	return st, true, nil
}

func (e *Examiner) proxyOffline() {
	e.mu.Lock()
	oldTr := e.proxyTr
	e.ready = false
	e.proxy = nil
	e.proxyTr = nil
	e.upstream = ""
	e.lastProbeErr = ""
	e.mu.Unlock()
	if oldTr != nil {
		oldTr.CloseIdleConnections()
	}
}

func (e *Examiner) clearState() {
	e.mu.Lock()
	oldTr := e.proxyTr
	pending := append([]podRecord(nil), e.state.PendingSpends...)
	e.state = examinerState{PendingSpends: pending}
	e.ready = false
	e.proxy = nil
	e.proxyTr = nil
	e.upstream = ""
	e.lastProbeErr = ""
	st := e.state
	e.mu.Unlock()
	if oldTr != nil {
		oldTr.CloseIdleConnections()
	}
	_ = e.saveState(&st)
}

func (e *Examiner) ensureLoaded() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.loaded {
		return
	}
	e.loaded = true
	e.owner = loadOrCreateExaminerOwner(e.home)
	if e.home != "" {
		if st := readExaminerState(e.home); st != nil {
			e.state = *st
		}
	}
}

func (e *Examiner) ownerID() string {
	e.ensureLoaded()
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.owner
}

// ActiveSnapshot reports the live count (0 or 1), hourly price, and accrued spend today (UTC)
// of any active examiner managed by this controller (plus any terminal charges pending durable
// flush to spent.jsonl).
func (e *Examiner) ActiveSnapshot(now time.Time) (live int, hourly float64, accrued float64) {
	if e == nil {
		return 0, 0, 0
	}
	e.ensureLoaded()
	e.mu.RLock()
	st := e.state
	pending := append([]podRecord(nil), e.state.PendingSpends...)
	e.mu.RUnlock()
	dayStart := startOfDay(now)
	if st.InstanceID != "" {
		live = 1
		hourly = st.Price
		start := st.Created
		if start.IsZero() || start.After(now) {
			start = now
		}
		end := now
		if !st.Created.IsZero() && e.cfg.Timeout > 0 {
			if capAt := st.Created.Add(e.cfg.Timeout); end.After(capAt) {
				end = capAt
			}
		}
		rec := podRecord{Cost: st.Price, Start: start, End: end}
		accrued += rec.dollarsBetween(dayStart, end)
	}
	for _, p := range pending {
		if p.ID != "" && p.ID != st.InstanceID && !spentHasExaminer(e.home, p.ID) {
			end := p.End
			if end.IsZero() {
				end = now
			}
			accrued += p.dollarsBetween(dayStart, end)
		}
	}
	return live, hourly, accrued
}

func (e *Examiner) isBooting() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.state.InstanceID != "" && !e.ready
}

// roundNeedingExam returns the ID of the first open round whose exam has not been posted yet.
func roundNeedingExam(w *Work) string {
	if w == nil {
		return ""
	}
	for _, r := range w.Rounds {
		if r.ExamAt == 0 {
			return r.ID
		}
	}
	return ""
}

func randomAPIKey() (string, error) { return randomHex(32) }

func (e *Examiner) noteProbeIssue(msg string) {
	if msg == "" {
		return
	}
	e.mu.Lock()
	changed := e.lastProbeErr != msg
	if changed {
		e.lastProbeErr = msg
	}
	e.mu.Unlock()
	if changed {
		e.say("examiner HTTPS probe issue: %s", msg)
	}
}

// probeModels verifies over peer-verified pinned TLS that the remote vLLM server responds 200 to
// GET /v1/models with Bearer auth and is serving the exact pinned model@revision.
func (e *Examiner) probeModels(ctx context.Context, upstream, apiKey, certPEM string) bool {
	u, err := url.Parse(upstream)
	if err != nil || u.Scheme != "https" {
		return false
	}
	tlsCfg, err := examinerTLSConfig([]byte(certPEM), e.now)
	if err != nil {
		e.noteProbeIssue(err.Error())
		return false
	}
	tr := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		DialContext:         (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
		TLSClientConfig:     tlsCfg,
		TLSHandshakeTimeout: 10 * time.Second,
	}
	defer tr.CloseIdleConnections()
	cl := &http.Client{Timeout: 10 * time.Second, Transport: tr}
	req, err := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(upstream, "/")+"/v1/models", nil)
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := cl.Do(req)
	if err != nil {
		es := err.Error()
		if strings.Contains(es, "x509:") || strings.Contains(es, "tls:") || strings.Contains(es, "pinned instance certificate") {
			e.noteProbeIssue(es)
		}
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		e.noteProbeIssue(fmt.Sprintf("%s/v1/models returned HTTP %d", upstream, resp.StatusCode))
		return false
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return false
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.Unmarshal(raw, &out) != nil {
		e.noteProbeIssue(fmt.Sprintf("%s/v1/models returned invalid JSON", upstream))
		return false
	}
	wantID := e.cfg.Model
	if e.cfg.Revision != "" {
		wantID = e.cfg.Model + "@" + e.cfg.Revision
	}
	for _, m := range out.Data {
		if m.ID == wantID {
			e.mu.Lock()
			e.lastProbeErr = ""
			e.mu.Unlock()
			return true
		}
	}
	e.noteProbeIssue(fmt.Sprintf("%s/v1/models missing expected model %q", upstream, wantID))
	return false
}

func spentHasExaminer(home, id string) bool {
	if home == "" || id == "" {
		return false
	}
	fh, err := os.Open(filepath.Join(home, "spent.jsonl"))
	if err != nil {
		return false
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		var r podRecord
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.Market == "vast" && r.For == "examiner" && r.ID == id {
			return true
		}
	}
	return false
}

func (e *Examiner) recordSpend(rec podRecord) error {
	if e.onSpent == nil || rec.ID == "" {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	if e.recorded == nil {
		e.recorded = map[string]bool{}
	}
	if e.recorded[rec.ID] || spentHasExaminer(e.home, rec.ID) {
		e.recorded[rec.ID] = true
		return nil
	}
	if err := e.onSpent(rec); err != nil {
		return err
	}
	e.recorded[rec.ID] = true
	return nil
}

func (e *Examiner) queuePendingSpend(rec podRecord) {
	if rec.ID == "" {
		return
	}
	e.mu.Lock()
	for i := range e.state.PendingSpends {
		if e.state.PendingSpends[i].ID == rec.ID {
			// Preserve the first recorded terminal End timestamp instead of inflating it on retries.
			e.mu.Unlock()
			return
		}
	}
	e.state.PendingSpends = append(e.state.PendingSpends, rec)
	st := e.state
	st.PendingSpends = append([]podRecord(nil), e.state.PendingSpends...)
	e.mu.Unlock()
	if err := e.saveState(&st); err != nil {
		e.say("warning: could not persist pending examiner spend in examiner.json: %v", err)
	}
}

func (e *Examiner) flushPendingSpends() error {
	if e.onSpent == nil || e.dryRun {
		return nil
	}
	e.mu.RLock()
	pending := append([]podRecord(nil), e.state.PendingSpends...)
	e.mu.RUnlock()
	if len(pending) == 0 {
		return nil
	}
	var keep []podRecord
	var firstErr error
	for _, rec := range pending {
		if err := e.recordSpend(rec); err != nil {
			keep = append(keep, rec)
			if firstErr == nil {
				firstErr = fmt.Errorf("recording examiner spend for %s: %w", rec.ID, err)
			}
		}
	}
	if len(keep) == len(pending) {
		return firstErr
	}
	flushed := map[string]bool{}
	for _, rec := range pending {
		flushed[rec.ID] = true
	}
	for _, rec := range keep {
		delete(flushed, rec.ID)
	}
	e.mu.Lock()
	var next []podRecord
	for _, rec := range e.state.PendingSpends {
		if !flushed[rec.ID] {
			next = append(next, rec)
		}
	}
	e.state.PendingSpends = next
	st := e.state
	st.PendingSpends = append([]podRecord(nil), e.state.PendingSpends...)
	e.mu.Unlock()
	if err := e.saveState(&st); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

func (e *Examiner) recordEndedAt(id, label string, price float64, start, end time.Time, roundID, why string) error {
	if e.onSpent == nil || id == "" {
		return nil
	}
	if start.IsZero() || start.After(end) {
		start = end
	}
	if roundID == "" {
		roundID = examinerLabelRound(label, e.ownerID())
	}
	rec := podRecord{
		ID: id, Market: "vast", Handle: examinerLabelPrefix, GPU: "RTX 5090",
		Cloud: "verified", Cost: price, Start: start, Alive: end, End: end,
		For: "examiner", Round: roundID, Why: why,
	}
	if err := e.recordSpend(rec); err != nil {
		e.queuePendingSpend(rec)
		return fmt.Errorf("recording examiner spend for %s: %w", id, err)
	}
	return nil
}

func (e *Examiner) recordEnded(inst ExaminerInstance, fallbackStart time.Time, roundID, why string) error {
	now := e.now()
	start := inst.Created
	if start.IsZero() {
		start = fallbackStart
	}
	return e.recordEndedAt(inst.ID, inst.Label, inst.Price, start, now, roundID, why)
}

// recordMissingSaved records a bounded, idempotent terminal spend charge for a saved examiner
// instance that vanished from the provider (e.g. terminated by its container-side TTL watchdog
// or ended during a controller outage) so accrued spend never vanishes from daily budget totals.
func (e *Examiner) recordMissingSaved(saved examinerState, why string) error {
	if saved.InstanceID == "" || saved.Price <= 0 || e.dryRun {
		return nil
	}
	now := e.now()
	start := saved.Created
	if start.IsZero() || start.After(now) {
		start = now
	}
	end := now
	if !saved.Created.IsZero() && e.cfg.Timeout > 0 {
		if capAt := saved.Created.Add(e.cfg.Timeout); end.After(capAt) {
			end = capAt
		}
	}
	return e.recordEndedAt(saved.InstanceID, "", saved.Price, start, end, saved.RoundID, why)
}

// EnforceTimeout checks owned examiner instances against the hard --examiner-timeout TTL and
// destroys any expired instance WITHOUT querying the Stet server, so Stet outages can never
// disable timeout enforcement.
func (e *Examiner) EnforceTimeout(ctx context.Context) error {
	if e == nil || e.prov == nil {
		return nil
	}
	e.ensureLoaded()
	firstErr := e.flushPendingSpends()
	owner := e.ownerID()
	insts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		return fmt.Errorf("listing examiner instances for timeout check: %w", err)
	}
	e.mu.RLock()
	saved := e.state
	e.mu.RUnlock()
	owned := filterOwnedExaminers(insts, owner, saved.InstanceID)
	now := e.now()

	foundSaved := false
	for _, inst := range owned {
		if inst.ID == saved.InstanceID {
			foundSaved = true
		}
		var created time.Time
		rid := ""
		if inst.ID == saved.InstanceID {
			created = saved.Created
			rid = saved.RoundID
		}
		if created.IsZero() {
			created = inst.Created
		}
		if created.IsZero() || now.Sub(created) <= e.cfg.Timeout {
			continue
		}
		if e.dryRun {
			e.say("would end timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
			continue
		}
		if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
			e.proxyOffline()
			e.say("could not end timed-out examiner GPU (machine %s): %v", inst.ID, err)
			if firstErr == nil {
				firstErr = fmt.Errorf("ending timed-out examiner GPU %s: %w", inst.ID, err)
			}
			continue
		}
		e.say("ended examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
		if rerr := e.recordEnded(inst, created, rid, "exceeded examiner timeout"); rerr != nil && firstErr == nil {
			firstErr = rerr
		}
		if inst.ID == saved.InstanceID {
			e.clearState()
			saved = examinerState{}
		}
	}
	if !foundSaved && saved.InstanceID != "" && !e.dryRun {
		if rerr := e.recordMissingSaved(saved, "examiner disappeared before timeout check (self-terminated or ended)"); rerr != nil && firstErr == nil {
			firstErr = rerr
		}
		e.clearState()
	}
	return firstErr
}

// Reconcile brings the ephemeral 1x RTX 5090 examiner in line with /admin/work:
//   - Scopes all discovery, adoption, and teardown strictly to this deployment's owner ID.
//   - When no open round needs an exam, destroys every owned examiner instance at once.
//   - When an open round has ExamAt == 0, ensures at most one owned live RTX 5090 is running with
//     a known per-instance API key and pinned TLS certificate, enforces hard --examiner-timeout,
//     refuses to rent a replacement until any live owned orphan is confirmed destroyed, respects
//     fleet --max / --max-hourly / --daily-budget, and activates the local reverse proxy only once
//     https://<ip>:<port>/v1/models passes peer-verified TLS.
func (e *Examiner) Reconcile(ctx context.Context, w *Work, budget ExaminerBudget) error {
	if e == nil || e.prov == nil {
		return nil
	}
	e.ensureLoaded()
	spendErr := e.flushPendingSpends()
	owner := e.ownerID()
	needRound := roundNeedingExam(w)
	rawInsts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		return fmt.Errorf("listing examiner instances: %w", err)
	}
	e.mu.RLock()
	saved := e.state
	wasReady := e.ready
	wasUpstream := e.upstream
	e.mu.RUnlock()
	insts := filterOwnedExaminers(rawInsts, owner, saved.InstanceID)
	now := e.now()

	// Case 1: no open round needs an exam -> tear down every owned examiner instance immediately.
	if needRound == "" {
		var firstEndErr error
		endedSaved := false
		for _, inst := range insts {
			if e.dryRun {
				e.say("would end examiner GPU (machine %s): no open round needs an exam", inst.ID)
				continue
			}
			if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
				e.say("could not end examiner GPU (machine %s): %v", inst.ID, err)
				if firstEndErr == nil {
					firstEndErr = fmt.Errorf("ending examiner GPU %s: %w", inst.ID, err)
				}
				continue
			}
			var fallbackStart time.Time
			rid := ""
			if inst.ID == saved.InstanceID {
				fallbackStart = saved.Created
				rid = saved.RoundID
				endedSaved = true
			}
			start := inst.Created
			if start.IsZero() {
				start = fallbackStart
			}
			dur := now.Sub(start).Round(time.Second)
			if start.IsZero() || dur < 0 {
				dur = 0
			}
			e.say("ended examiner GPU (machine %s, up %s): exam posted", inst.ID, dur)
			if rerr := e.recordEnded(inst, fallbackStart, rid, "exam posted"); rerr != nil && spendErr == nil {
				spendErr = rerr
			}
			if inst.ID == saved.InstanceID {
				e.clearState()
				saved, wasReady, wasUpstream = examinerState{}, false, ""
			}
		}
		if e.dryRun {
			return nil
		}
		if firstEndErr != nil {
			e.proxyOffline()
			return firstEndErr
		}
		if saved.InstanceID != "" && !endedSaved {
			if rerr := e.recordMissingSaved(saved, "examiner disappeared before teardown (self-terminated or ended)"); rerr != nil && spendErr == nil {
				spendErr = rerr
			}
		}
		if saved.InstanceID != "" || wasReady {
			e.clearState()
		}
		return spendErr
	}

	// Case 2: needRound needs an exam. Find if our known owned instance is alive, has a pinned
	// cert + key, and is within TTL; destroy any stopped, expired, duplicate, or orphaned owned
	// instances. Never rent a replacement while any live owned orphan fails deletion.
	var active *ExaminerInstance
	var strayEndErr error
	endedSaved := false
	for i := range insts {
		inst := insts[i]
		keep := inst.State != "stopped" &&
			saved.InstanceID != "" &&
			inst.ID == saved.InstanceID &&
			saved.APIKey != "" &&
			saved.CertPEM != "" &&
			active == nil
		if keep {
			created := saved.Created
			if created.IsZero() {
				created = inst.Created
			}
			if !created.IsZero() && now.Sub(created) > e.cfg.Timeout {
				if e.dryRun {
					e.say("would end timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
					return nil
				}
				if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
					e.proxyOffline()
					e.say("could not end timed-out examiner GPU (machine %s): %v", inst.ID, err)
					return fmt.Errorf("ending timed-out examiner GPU %s: %w", inst.ID, err)
				}
				e.say("ended examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
				if rerr := e.recordEnded(inst, saved.Created, saved.RoundID, "exceeded examiner timeout"); rerr != nil && spendErr == nil {
					spendErr = rerr
				}
				endedSaved = true
				e.clearState()
				saved, wasReady, wasUpstream = examinerState{}, false, ""
				continue
			}
		}
		if keep {
			active = &insts[i]
			continue
		}
		if e.dryRun {
			e.say("would end stray examiner GPU (machine %s, state %s)", inst.ID, inst.State)
			continue
		}
		if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
			e.say("could not end stray examiner GPU (machine %s): %v", inst.ID, err)
			if strayEndErr == nil {
				strayEndErr = fmt.Errorf("ending stray examiner GPU %s: %w", inst.ID, err)
			}
			continue
		}
		var fallbackStart time.Time
		rid := ""
		if inst.ID == saved.InstanceID {
			fallbackStart = saved.Created
			rid = saved.RoundID
			endedSaved = true
		}
		e.say("ended stray examiner GPU (machine %s, state %s)", inst.ID, inst.State)
		if rerr := e.recordEnded(inst, fallbackStart, rid, "stray or orphaned examiner instance"); rerr != nil && spendErr == nil {
			spendErr = rerr
		}
		if inst.ID == saved.InstanceID {
			e.clearState()
			saved, wasReady, wasUpstream = examinerState{}, false, ""
		}
	}

	if active != nil {
		if active.State != "stopped" && active.PublicIP != "" && active.HostPort != "" {
			upstream := "https://" + net.JoinHostPort(active.PublicIP, active.HostPort)
			if !wasReady || wasUpstream != upstream {
				if e.probeModels(ctx, upstream, saved.APIKey, saved.CertPEM) {
					if st, ok, err := e.setUpstream(saved.InstanceID, upstream, saved.APIKey, saved.CertPEM); err == nil && ok {
						_ = e.persistState(&st)
						boot := now.Sub(saved.Created).Round(time.Second)
						if saved.Created.IsZero() || boot < 0 {
							boot = 0
						}
						e.say("examiner GPU ready in %s (%s -> %s, %s, TLS pinned)", boot, e.cfg.Addr, upstream, e.cfg.Model)
					}
				}
			}
		}
		return spendErr
	}

	// Do NOT rent a replacement while an owned orphan could not be confirmed destroyed.
	if strayEndErr != nil {
		e.proxyOffline()
		return strayEndErr
	}

	// Saved instance is no longer listed -> record its bounded spend before clearing stale state.
	if saved.InstanceID != "" && !endedSaved {
		if rerr := e.recordMissingSaved(saved, "examiner disappeared (self-terminated or ended)"); rerr != nil && spendErr == nil {
			spendErr = rerr
		}
	}
	if saved.InstanceID != "" || wasReady {
		e.clearState()
	}

	if spendErr != nil {
		e.say("not renting examiner GPU for %s: spend log unavailable (%v)", needRound, spendErr)
		return spendErr
	}

	if budget.AccountingErr != nil {
		e.say("not renting examiner GPU for %s: fleet accounting unavailable (%v)", needRound, budget.AccountingErr)
		return budget.AccountingErr
	}
	if budget.MaxWorkers <= 0 || budget.LiveWorkers >= budget.MaxWorkers {
		e.say("not renting examiner GPU for %s: %d machines already live and --max is %d; waiting for a slot", needRound, budget.LiveWorkers, budget.MaxWorkers)
		return nil
	}
	if budget.DailyBudget > 0 && budget.SpentToday >= budget.DailyBudget {
		e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, budget.SpentToday, budget.DailyBudget)
		return nil
	}
	headroom := budget.MaxHourly - budget.HourlyNow
	if budget.MaxHourly <= 0 || headroom <= 1e-9 {
		e.say("not renting examiner GPU for %s: live hourly spend $%.2f/h leaves no room under --max-hourly $%.2f/h", needRound, budget.HourlyNow, budget.MaxHourly)
		return nil
	}
	effMaxPrice := e.cfg.MaxPrice
	if effMaxPrice <= 0 || headroom < effMaxPrice {
		effMaxPrice = headroom
	}

	offers, err := e.prov.ExaminerOffers(ctx, effMaxPrice)
	if err != nil {
		return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", err)
	}
	if len(offers) == 0 {
		e.say("Vast.ai has no 1x RTX 5090 free for %s examiner under $%.2f/h (headroom $%.2f/h); retrying on next tick", needRound, effMaxPrice, headroom)
		return nil
	}
	var fitting []ExaminerOffer
	for _, off := range offers {
		if budget.HourlyNow+off.Price <= budget.MaxHourly+1e-9 && (e.cfg.MaxPrice <= 0 || off.Price <= e.cfg.MaxPrice+1e-9) {
			fitting = append(fitting, off)
		}
	}
	if len(fitting) == 0 {
		e.say("not renting examiner GPU for %s: cheapest 1x RTX 5090 ($%.2f/h) plus $%.2f/h live exceeds --max-hourly $%.2f/h",
			needRound, offers[0].Price, budget.HourlyNow, budget.MaxHourly)
		return nil
	}
	if e.dryRun {
		e.say("would rent examiner GPU on Vast.ai for %s: %s at $%.2f/h (offer %s)", needRound, fitting[0].Card, fitting[0].Price, fitting[0].ID)
		return nil
	}
	apiKey, err := randomAPIKey()
	if err != nil {
		return err
	}
	certPEM, keyPEM, err := generateExaminerCert(now, e.cfg.Timeout)
	if err != nil {
		return fmt.Errorf("generating examiner TLS certificate: %w", err)
	}
	label := examinerLabelPrefix + "-" + owner + "-" + needRound + "-" + apiKey[:8]
	ttlSeconds := max(60, int(e.cfg.Timeout.Seconds()))
	for _, off := range fitting {
		id, rerr := e.prov.RentExaminer(ctx, off.ID, ExaminerLaunch{
			Label:      label,
			Image:      e.cfg.Image,
			Model:      e.cfg.Model,
			Revision:   e.cfg.Revision,
			APIKey:     apiKey,
			CertPEM:    certPEM,
			KeyPEM:     keyPEM,
			TTLSeconds: ttlSeconds,
			HFToken:    e.hfToken,
			DiskGB:     examinerDiskGB,
		})
		if errors.Is(rerr, errNoRoom) {
			continue
		}
		if rerr != nil {
			return fmt.Errorf("renting examiner GPU on Vast.ai: %w", rerr)
		}
		st := examinerState{
			InstanceID: id,
			RoundID:    needRound,
			APIKey:     apiKey,
			CertPEM:    string(certPEM),
			Price:      off.Price,
			Created:    now,
		}
		e.mu.Lock()
		oldTr := e.proxyTr
		st.PendingSpends = append([]podRecord(nil), e.state.PendingSpends...)
		e.state = st
		e.ready = false
		e.proxy = nil
		e.proxyTr = nil
		e.upstream = ""
		e.mu.Unlock()
		if oldTr != nil {
			oldTr.CloseIdleConnections()
		}
		if err := e.persistState(&st); err != nil {
			e.say("warning: could not save examiner.json: %v", err)
		}
		e.say("rented examiner GPU on Vast.ai for %s: %s, $%.2f/h (machine %s); warming %s (TLS pinned, TTL %s)", needRound, off.Card, off.Price, id, e.cfg.Model, e.cfg.Timeout)
		return nil
	}
	e.say("none of the %d RTX 5090 offers tried for %s examiner were available; retrying on next tick", len(fitting), needRound)
	return nil
}

// ProbeIfBooting polls Vast + https://<ip>:<port>/v1/models (over pinned TLS) once when an
// examiner GPU is currently booting so the local proxy turns ready seconds after vLLM loads.
func (e *Examiner) ProbeIfBooting(ctx context.Context) {
	if e == nil || e.prov == nil || !e.isBooting() {
		return
	}
	owner := e.ownerID()
	e.mu.RLock()
	saved := e.state
	ready := e.ready
	e.mu.RUnlock()
	if ready || saved.InstanceID == "" || saved.APIKey == "" || saved.CertPEM == "" {
		return
	}
	rawInsts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		return
	}
	for _, inst := range filterOwnedExaminers(rawInsts, owner, saved.InstanceID) {
		if inst.ID == saved.InstanceID && inst.State != "stopped" && inst.PublicIP != "" && inst.HostPort != "" {
			upstream := "https://" + net.JoinHostPort(inst.PublicIP, inst.HostPort)
			if e.probeModels(ctx, upstream, saved.APIKey, saved.CertPEM) {
				if st, ok, err := e.setUpstream(saved.InstanceID, upstream, saved.APIKey, saved.CertPEM); err == nil && ok {
					_ = e.persistState(&st)
					boot := e.now().Sub(saved.Created).Round(time.Second)
					if saved.Created.IsZero() || boot < 0 {
						boot = 0
					}
					e.say("examiner GPU ready in %s (%s -> %s, %s, TLS pinned)", boot, e.cfg.Addr, upstream, e.cfg.Model)
				}
			}
			return
		}
	}
}

func appendExaminerSpend(path string, rec podRecord) error {
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := fh.Write(append(line, '\n')); err != nil {
		_ = fh.Close()
		return err
	}
	if err := fh.Sync(); err != nil {
		_ = fh.Close()
		return err
	}
	return fh.Close()
}

// startExaminer initializes and starts the Fleet's automated ephemeral examiner if --examiner-addr
// is configured and a Vast.ai key is present.
func (f *Fleet) startExaminer(ctx context.Context) (func() error, error) {
	if f.o.Examiner.Addr == "" {
		return func() error { return nil }, nil
	}
	if err := f.refresh(); err != nil {
		return nil, err
	}
	vm, ok := f.market("vast").(*vastMarket)
	if !ok || vm == nil {
		return nil, fmt.Errorf("--examiner-addr requires VAST_API_KEY in %s (or environment)", f.o.Credentials)
	}
	ex := newExaminer(f.o.Examiner, f.o.Home, vm, f.now, f.say, f.o.DryRun)
	if creds, err := readCredentials(f.o.Credentials); err == nil && creds["HF_TOKEN"] != "" && ex.hfToken == "" {
		ex.hfToken = creds["HF_TOKEN"]
	}
	ex.onSpent = func(rec podRecord) error {
		return appendExaminerSpend(f.spentLog(), rec)
	}
	addr, stopProxy, err := ex.Start(ctx)
	if err != nil {
		return nil, err
	}
	f.examiner = ex
	f.say("ephemeral examiner proxy listening on http://%s (1x RTX 5090 up to $%.2f/h, timeout %s, pinned TLS upstream)", addr.String(), ex.cfg.MaxPrice, ex.cfg.Timeout)
	go func() {
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				ex.ProbeIfBooting(ctx)
			}
		}
	}()
	return stopProxy, nil
}
