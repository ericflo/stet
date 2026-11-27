package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
//   2. Rents one verified 1x RTX 5090 on Vast.ai running the pinned immutable vLLM Docker digest
//      and pinned Inkling-2-Mini revision with a fresh random 256-bit --api-key.
//   3. Probes /v1/models on the remote instance, then enables a local loopback reverse proxy on
//      --examiner-addr that injects Authorization: Bearer <api_key> so `stet-examine.service`
//      points at http://127.0.0.1:8000/v1 with zero manual provisioning or shared static keys.
//   4. Destroys the Vast.ai instance immediately once every open round has ExamAt > 0 (or on
//      hard --examiner-timeout TTL, and cleans up any orphaned stet-examiner instances on restart).

const (
	defaultExaminerImage    = "vllm/vllm-openai@sha256:d24e5fc73f0b859bd76c9d2f6f6fa4da3001afd7637f2610876420f3027ea6c7"
	defaultExaminerModel    = "thinkingmachines/Inkling-2-Mini"
	defaultExaminerRevision = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
	examinerLabelPrefix     = "stet-examiner"
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
	Label    string
	Image    string
	Model    string
	Revision string
	APIKey   string
	HFToken  string
	DiskGB   int
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
// API key or leak an orphaned 5090 on Vast.ai.
type examinerState struct {
	InstanceID string    `json:"instance_id"`
	RoundID    string    `json:"round_id"`
	APIKey     string    `json:"api_key"`
	Upstream   string    `json:"upstream,omitempty"`
	Price      float64   `json:"price"`
	Created    time.Time `json:"created"`
}

func examinerStatePath(home string) string { return filepath.Join(home, "examiner.json") }

func readExaminerState(home string) *examinerState {
	raw, err := os.ReadFile(examinerStatePath(home))
	if err != nil {
		return nil
	}
	var st examinerState
	if json.Unmarshal(raw, &st) != nil || st.InstanceID == "" {
		return nil
	}
	return &st
}

func writeExaminerState(home string, st *examinerState) error {
	path := examinerStatePath(home)
	if st == nil || st.InstanceID == "" {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	if err := os.MkdirAll(home, 0o700); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(st, "", " ")
	if err != nil {
		return err
	}
	fh, err := os.CreateTemp(home, "examiner.json.tmp-*")
	if err != nil {
		return err
	}
	tmp := fh.Name()
	if err := fh.Chmod(0o600); err != nil {
		_ = fh.Close()
		_ = os.Remove(tmp)
		return err
	}
	if _, err := fh.Write(raw); err != nil {
		_ = fh.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := fh.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// examinerLabelRound extracts the round ID from labels formatted as
// "stet-examiner-<round>" or "stet-examiner-<round>-<8hex>".
func examinerLabelRound(label string) string {
	rest, ok := strings.CutPrefix(label, examinerLabelPrefix+"-")
	if !ok || rest == "" {
		return ""
	}
	if i := strings.LastIndex(rest, "-"); i > 0 && len(rest)-i-1 == 8 {
		if _, err := hex.DecodeString(rest[i+1:]); err == nil {
			return rest[:i]
		}
	}
	return rest
}

// Examiner manages the lifecycle of an ephemeral 1x RTX 5090 vLLM server and serves an
// authenticated local loopback reverse proxy for stet-examine.
type Examiner struct {
	cfg        ExaminerConfig
	home       string
	prov       ExaminerProvider
	probeHTTP  *http.Client
	proxyTrans http.RoundTripper
	now        func() time.Time
	say        func(string, ...any)
	dryRun     bool
	hfToken    string
	onSpent    func(podRecord)

	fileMu   sync.Mutex
	mu       sync.RWMutex
	loaded   bool
	state    examinerState
	ready    bool
	proxy    *httputil.ReverseProxy
	upstream string
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
		cfg:       cfg,
		home:      home,
		prov:      prov,
		probeHTTP: &http.Client{Timeout: 10 * time.Second},
		proxyTrans: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			DialContext:           (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			ResponseHeaderTimeout: 10 * time.Minute,
			IdleConnTimeout:       90 * time.Second,
		},
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

// ServeHTTP proxies requests to the active ephemeral 5090 vLLM server once /v1/models is verified,
// injecting Authorization: Bearer <apiKey>. While no examiner GPU is ready, it answers 503.
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

func (e *Examiner) persistState(st *examinerState) error {
	if e.home == "" || e.dryRun {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	if st != nil && st.InstanceID != "" {
		e.mu.RLock()
		match := e.state.InstanceID == st.InstanceID && e.state.APIKey == st.APIKey
		e.mu.RUnlock()
		if !match {
			return nil
		}
	}
	return writeExaminerState(e.home, st)
}

// setUpstream activates the local proxy ONLY if the active instance ID and API key still match
// what was probed, preventing a stale background ProbeIfBooting from reviving a destroyed VM or
// corrupting a replacement VM's state.
func (e *Examiner) setUpstream(expectedID, upstream, apiKey string) (examinerState, bool, error) {
	u, err := url.Parse(upstream)
	if err != nil {
		return examinerState{}, false, err
	}
	rp := &httputil.ReverseProxy{
		Transport: e.proxyTrans,
		Director: func(req *http.Request) {
			req.URL.Scheme = u.Scheme
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
	if expectedID == "" || e.state.InstanceID != expectedID || e.state.APIKey != apiKey {
		e.mu.Unlock()
		return examinerState{}, false, nil
	}
	e.upstream = upstream
	e.proxy = rp
	e.ready = true
	e.state.Upstream = upstream
	st := e.state
	e.mu.Unlock()
	return st, true, nil
}

func (e *Examiner) clearState() {
	e.mu.Lock()
	e.state = examinerState{}
	e.ready = false
	e.proxy = nil
	e.upstream = ""
	e.mu.Unlock()
	_ = e.persistState(nil)
}

func (e *Examiner) ensureLoaded() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.loaded {
		return
	}
	e.loaded = true
	if e.home != "" {
		if st := readExaminerState(e.home); st != nil {
			e.state = *st
		}
	}
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

func randomAPIKey() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

// probeModels verifies that the remote vLLM server responds 200 to GET /v1/models with Bearer
// auth and is serving the pinned model ID.
func (e *Examiner) probeModels(ctx context.Context, upstream, apiKey string) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(upstream, "/")+"/v1/models", nil)
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := e.probeHTTP.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
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
		return false
	}
	wantID := e.cfg.Model
	if e.cfg.Revision != "" {
		wantID = e.cfg.Model + "@" + e.cfg.Revision
	}
	for _, m := range out.Data {
		if m.ID == wantID {
			return true
		}
	}
	return false
}

func (e *Examiner) recordEnded(inst ExaminerInstance, fallbackStart time.Time, roundID, why string) {
	if e.onSpent == nil {
		return
	}
	now := e.now()
	start := inst.Created
	if start.IsZero() {
		start = fallbackStart
	}
	if start.IsZero() || start.After(now) {
		start = now
	}
	if roundID == "" {
		roundID = examinerLabelRound(inst.Label)
	}
	e.onSpent(podRecord{
		ID: inst.ID, Market: "vast", Handle: examinerLabelPrefix, GPU: "RTX 5090",
		Cloud: "verified", Cost: inst.Price, Start: start, Alive: now, End: now,
		For: "examiner", Round: roundID, Why: why,
	})
}

// Reconcile brings the ephemeral 1x RTX 5090 examiner in line with /admin/work:
//   - When no open round needs an exam, destroys any live/orphaned examiner instances at once.
//   - When an open round has ExamAt == 0, ensures exactly one live RTX 5090 is running with a
//     known per-instance API key, enforces hard --examiner-timeout TTL, probes /v1/models, and
//     activates the local reverse proxy once verified.
func (e *Examiner) Reconcile(ctx context.Context, w *Work, spentToday, dailyBudget float64) error {
	if e == nil || e.prov == nil {
		return nil
	}
	e.ensureLoaded()
	needRound := roundNeedingExam(w)
	insts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		return fmt.Errorf("listing examiner instances: %w", err)
	}
	now := e.now()

	e.mu.RLock()
	saved := e.state
	wasReady := e.ready
	wasUpstream := e.upstream
	e.mu.RUnlock()

	// Case 1: no open round needs an exam -> tear down every examiner instance immediately.
	if needRound == "" {
		allEnded := true
		for _, inst := range insts {
			if e.dryRun {
				e.say("would end examiner GPU (machine %s): no open round needs an exam", inst.ID)
				continue
			}
			if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
				e.say("could not end examiner GPU (machine %s): %v", inst.ID, err)
				allEnded = false
				continue
			}
			var fallbackStart time.Time
			rid := ""
			if inst.ID == saved.InstanceID {
				fallbackStart = saved.Created
				rid = saved.RoundID
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
			e.recordEnded(inst, fallbackStart, rid, "exam posted")
		}
		if e.dryRun {
			return nil
		}
		if !allEnded {
			// Keep InstanceID persisted so the next tick retries teardown, while taking proxy offline.
			e.mu.Lock()
			e.ready = false
			e.proxy = nil
			e.upstream = ""
			e.mu.Unlock()
			return nil
		}
		if saved.InstanceID != "" || wasReady {
			e.clearState()
		}
		return nil
	}

	// Case 2: needRound needs an exam. Find if our known instance is alive & within TTL; destroy
	// any stopped, expired, duplicate, or orphaned instances (orphans have unknown random keys).
	var active *ExaminerInstance
	for i := range insts {
		inst := insts[i]
		keep := inst.State != "stopped" &&
			saved.InstanceID != "" &&
			inst.ID == saved.InstanceID &&
			saved.APIKey != "" &&
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
					// Take proxy offline but keep saved.InstanceID so we retry destroying this VM
					// on the next tick instead of leaking it and renting a duplicate 5090.
					e.mu.Lock()
					e.ready = false
					e.proxy = nil
					e.upstream = ""
					e.mu.Unlock()
					e.say("could not end timed-out examiner GPU (machine %s): %v", inst.ID, err)
					return fmt.Errorf("ending timed-out examiner GPU %s: %w", inst.ID, err)
				}
				e.say("ended examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
				e.recordEnded(inst, saved.Created, saved.RoundID, "exceeded examiner timeout")
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
			continue
		}
		var fallbackStart time.Time
		rid := ""
		if inst.ID == saved.InstanceID {
			fallbackStart = saved.Created
			rid = saved.RoundID
		}
		e.say("ended stray examiner GPU (machine %s, state %s)", inst.ID, inst.State)
		e.recordEnded(inst, fallbackStart, rid, "stray or orphaned examiner instance")
	}

	if active != nil {
		if active.State == "running" && active.PublicIP != "" && active.HostPort != "" {
			upstream := "http://" + net.JoinHostPort(active.PublicIP, active.HostPort)
			if !wasReady || wasUpstream != upstream {
				if e.probeModels(ctx, upstream, saved.APIKey) {
					if st, ok, err := e.setUpstream(saved.InstanceID, upstream, saved.APIKey); err == nil && ok {
						_ = e.persistState(&st)
						boot := now.Sub(saved.Created).Round(time.Second)
						if saved.Created.IsZero() || boot < 0 {
							boot = 0
						}
						e.say("examiner GPU ready in %s (%s -> %s, %s)", boot, e.cfg.Addr, upstream, e.cfg.Model)
					}
				}
			}
		}
		return nil
	}

	// Saved instance is no longer listed -> clear stale state before renting a new one.
	if saved.InstanceID != "" || wasReady {
		e.clearState()
	}

	if dailyBudget > 0 && spentToday >= dailyBudget {
		e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, spentToday, dailyBudget)
		return nil
	}

	offers, err := e.prov.ExaminerOffers(ctx, e.cfg.MaxPrice)
	if err != nil {
		return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", err)
	}
	if len(offers) == 0 {
		e.say("Vast.ai has no 1x RTX 5090 free for %s examiner under $%.2f/h; retrying on next tick", needRound, e.cfg.MaxPrice)
		return nil
	}
	if e.dryRun {
		e.say("would rent examiner GPU on Vast.ai for %s: %s at $%.2f/h (offer %s)", needRound, offers[0].Card, offers[0].Price, offers[0].ID)
		return nil
	}
	apiKey, err := randomAPIKey()
	if err != nil {
		return err
	}
	label := examinerLabelPrefix + "-" + needRound + "-" + apiKey[:8]
	for _, off := range offers {
		id, rerr := e.prov.RentExaminer(ctx, off.ID, ExaminerLaunch{
			Label:    label,
			Image:    e.cfg.Image,
			Model:    e.cfg.Model,
			Revision: e.cfg.Revision,
			APIKey:   apiKey,
			HFToken:  e.hfToken,
			DiskGB:   examinerDiskGB,
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
			Price:      off.Price,
			Created:    now,
		}
		e.mu.Lock()
		e.state = st
		e.ready = false
		e.proxy = nil
		e.upstream = ""
		e.mu.Unlock()
		if err := e.persistState(&st); err != nil {
			e.say("warning: could not save examiner.json: %v", err)
		}
		e.say("rented examiner GPU on Vast.ai for %s: %s, $%.2f/h (machine %s); warming %s", needRound, off.Card, off.Price, id, e.cfg.Model)
		return nil
	}
	e.say("none of the %d RTX 5090 offers tried for %s examiner were available; retrying on next tick", len(offers), needRound)
	return nil
}

// ProbeIfBooting polls Vast + /v1/models once when an examiner GPU is currently booting so the
// local proxy turns ready seconds after vLLM loads rather than waiting a full scale interval.
func (e *Examiner) ProbeIfBooting(ctx context.Context) {
	if e == nil || e.prov == nil || !e.isBooting() {
		return
	}
	e.mu.RLock()
	saved := e.state
	ready := e.ready
	e.mu.RUnlock()
	if ready || saved.InstanceID == "" || saved.APIKey == "" {
		return
	}
	insts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		return
	}
	for _, inst := range insts {
		if inst.ID == saved.InstanceID && inst.State == "running" && inst.PublicIP != "" && inst.HostPort != "" {
			upstream := "http://" + net.JoinHostPort(inst.PublicIP, inst.HostPort)
			if e.probeModels(ctx, upstream, saved.APIKey) {
				if st, ok, err := e.setUpstream(saved.InstanceID, upstream, saved.APIKey); err == nil && ok {
					_ = e.persistState(&st)
					boot := e.now().Sub(saved.Created).Round(time.Second)
					if saved.Created.IsZero() || boot < 0 {
						boot = 0
					}
					e.say("examiner GPU ready in %s (%s -> %s, %s)", boot, e.cfg.Addr, upstream, e.cfg.Model)
				}
			}
			return
		}
	}
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
	ex.onSpent = func(rec podRecord) {
		line, err := json.Marshal(rec)
		if err != nil {
			return
		}
		if fh, err := os.OpenFile(f.spentLog(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600); err == nil {
			_, _ = fh.Write(append(line, '\n'))
			_ = fh.Close()
		}
	}
	addr, stopProxy, err := ex.Start(ctx)
	if err != nil {
		return nil, err
	}
	f.examiner = ex
	f.say("ephemeral examiner proxy listening on http://%s (1x RTX 5090 up to $%.2f/h, timeout %s)", addr.String(), ex.cfg.MaxPrice, ex.cfg.Timeout)
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
