package main

import (
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
	"math"
	"math/big"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"sort"
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
	examinerDownloadGB      = 18.0 // pinned Inkling-2-Mini model weight pull billed on metered-ingress hosts
)

// errRentRejected marks a provider response that definitively refused to create an instance
// (HTTP 4xx or 2xx with explicit success=false), as opposed to ambiguous timeouts / 5xx or
// incomplete 2xx responses where Vast may still have created the contract and a stale listing
// may briefly omit it.
var errRentRejected = errors.New("examiner rental rejected by provider")

// errRentNotSent marks a transport failure proven to have occurred before any connection was
// established or request bytes were sent (such as dial ECONNREFUSED or non-timeout DNS failure).
var errRentNotSent = errors.New("examiner rental request was not sent")

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
	LiveWorkers       int
	MaxWorkers        int
	HourlyNow         float64 // live worker $/h (excluding any already-active examiner)
	MaxHourly         float64
	SpentToday        float64
	DailyBudget       float64
	AccountingErr     error
	CurrentSpentToday func() (float64, error) // optional fresh today-spend reader recomputed after cleanup, before rent
}

// ExaminerOffer is one eligible 1x RTX 5090 offer on Vast.ai.
type ExaminerOffer struct {
	ID              string
	Card            string
	Price           float64
	DiskCost        float64
	TransferDollars float64 // one-time cold-start download charge for the 18 GB model pull on metered-ingress hosts
	CUDA            float64
	Region          string
	InetDown        float64
}

// effectiveHourly returns the conservative first-hour outlay for an ephemeral examiner offer
// (hourly rental + disk rate plus any one-time cold-start model download charge).
func (o ExaminerOffer) effectiveHourly() float64 {
	eff := o.Price
	if nonNegFinite(o.TransferDollars) && o.TransferDollars > 0 {
		eff += o.TransferDollars
	}
	return eff
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
	ExpireAt   int64 // absolute UTC Unix deadline for container-side self-destruct watchdog
	HFToken    string
	DiskGB     int
}

// ExaminerInstance is one live or stopped examiner instance returned by Vast.ai.
type ExaminerInstance struct {
	ID              string
	Label           string
	State           string // starting | running | stopped
	Status          string
	PublicIP        string
	HostPort        string
	Price           float64
	BasePrice       float64 `json:"base_price,omitempty"`
	DiskCost        float64 `json:"disk_cost,omitempty"`
	TransferDollars float64 `json:"transfer_dollars,omitempty"`
	PartialRate     bool    `json:"partial_rate,omitempty"`
	Created         time.Time
	ObservedLiveAt  time.Time `json:"observed_live_at,omitempty"`
}

// ExaminerProvider abstracts Vast.ai examiner provisioning so tests can mock it cleanly.
type ExaminerProvider interface {
	ExaminerOffers(ctx context.Context, maxPrice float64) ([]ExaminerOffer, error)
	RentExaminer(ctx context.Context, offerID string, l ExaminerLaunch) (string, error)
	ListExaminers(ctx context.Context) ([]ExaminerInstance, error)
	EndExaminer(ctx context.Context, id string) error
}

// examinerOfferSearcher is an optional ExaminerProvider extension that accepts an upper bound on
// one-time cold-start transfer charges (maxTransfer < 0 means unlimited) so multi-tier searches
// fall through to slower affordable/free-ingress tiers when faster tiers exceed remaining daily budget.
type examinerOfferSearcher interface {
	ExaminerOffersWithTransferCap(ctx context.Context, maxPrice, maxTransfer float64) ([]ExaminerOffer, error)
}

type examinerTieredOfferSearcher interface {
	ExaminerOffersByTier(ctx context.Context, maxPrice, maxTransfer float64, startTier int, excludeIDs map[string]bool) ([]ExaminerOffer, int, error)
}

// examinerState is persisted to <home>/examiner.json (0600) so restarts never lose the random
// API key, pinned peer certificate, pre-rental reservation, unrecorded terminal spend charges,
// or leak an orphaned 5090 on Vast.ai.
type examinerState struct {
	InstanceID      string             `json:"instance_id,omitempty"`
	PendingRent     bool               `json:"pending_rent,omitempty"`
	ReservedLabel   string             `json:"reserved_label,omitempty"`
	RoundID         string             `json:"round_id,omitempty"`
	APIKey          string             `json:"api_key,omitempty"`
	CertPEM         string             `json:"cert_pem,omitempty"`
	Upstream        string             `json:"upstream,omitempty"`
	Price           float64            `json:"price,omitempty"`
	DiskCost        float64            `json:"disk_cost,omitempty"`
	TransferDollars float64            `json:"transfer_dollars,omitempty"`
	Created         time.Time          `json:"created,omitempty"`
	ObservedLiveAt  time.Time          `json:"observed_live_at,omitempty"`
	PendingSpends   []podRecord        `json:"pending_spends,omitempty"`
	Unresolved      []ExaminerInstance `json:"unresolved,omitempty"`
}

func validRate(p float64) bool { return p > 0 && !math.IsNaN(p) && !math.IsInf(p, 0) }

func examinerStatePath(home string) string { return filepath.Join(home, "examiner.json") }
func examinerOwnerPath(home string) string { return filepath.Join(home, "examiner-owner") }

func examinerStateEmpty(st *examinerState) bool {
	return st == nil || (st.InstanceID == "" && !st.PendingRent && len(st.PendingSpends) == 0 && len(st.Unresolved) == 0)
}

func readExaminerStateErr(home string) (*examinerState, error) {
	if home == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(examinerStatePath(home))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading examiner.json: %w", err)
	}
	var st examinerState
	if err := json.Unmarshal(raw, &st); err != nil {
		return nil, fmt.Errorf("decoding examiner.json: %w", err)
	}
	if examinerStateEmpty(&st) {
		return nil, nil
	}
	return &st, nil
}

func readExaminerState(home string) *examinerState {
	st, _ := readExaminerStateErr(home)
	return st
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
	if examinerStateEmpty(st) {
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

func ensureExaminerOwner(home string, create bool) (string, error) {
	if home != "" {
		raw, err := os.ReadFile(examinerOwnerPath(home))
		if err == nil {
			if id := strings.TrimSpace(string(raw)); len(id) == 12 {
				if _, decErr := hex.DecodeString(id); decErr == nil {
					return id, nil
				}
			}
			return "", errors.New("decoding examiner-owner: invalid owner id")
		}
		if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("reading examiner-owner: %w", err)
		}
		if !create {
			return "", nil
		}
	}
	id, err := randomHex(6)
	if err != nil {
		return "", fmt.Errorf("generating examiner owner id: %w", err)
	}
	if home != "" && create {
		if err := writeAtomic0600(home, "examiner-owner.tmp-*", examinerOwnerPath(home), []byte(id+"\n")); err != nil {
			return "", fmt.Errorf("saving examiner-owner: %w", err)
		}
	}
	return id, nil
}

func loadOrCreateExaminerOwner(home string) string {
	id, _ := ensureExaminerOwner(home, true)
	return id
}

// examinerBelongsToOwner reports whether a provider instance label belongs strictly to this
// controller deployment so separate deployments sharing a Vast account never delete each other.
func examinerBelongsToOwner(label, ownerID string) bool {
	return ownerID != "" && strings.HasPrefix(label, examinerLabelPrefix+"-"+ownerID+"-")
}

func filterOwnedExaminers(insts []ExaminerInstance, ownerID, savedID, reservedLabel string) []ExaminerInstance {
	var out []ExaminerInstance
	for _, inst := range insts {
		if examinerBelongsToOwner(inst.Label, ownerID) ||
			(savedID != "" && inst.ID == savedID) ||
			(reservedLabel != "" && inst.Label == reservedLabel) {
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

// examinerAccrualEnd returns the conservative accrual end time for an examiner instance whose
// termination has not been independently timestamped by the provider. Because a slow image pull,
// broken startup, or controller outage can let a contract outlive created+timeout, we do not
// optimistically cap accrual at created+timeout before provider confirmation; accrual runs
// through max(now, observedLive) until termination is confirmed.
func examinerAccrualEnd(created, observedLive, now time.Time, _ time.Duration) time.Time {
	end := now
	if !observedLive.IsZero() && observedLive.After(end) {
		end = observedLive
	}
	if !created.IsZero() && end.Before(created) {
		end = created
	}
	return end
}

func boundedExaminerAccrual(price float64, created, observedLive, now, dayStart time.Time, timeout time.Duration) float64 {
	start := created
	if start.IsZero() || start.After(now) {
		start = now
	}
	end := examinerAccrualEnd(created, observedLive, now, timeout)
	rec := podRecord{Cost: price, Start: start, End: end}
	return rec.dollarsBetween(dayStart, end)
}

func higherValidRate(a, b float64) float64 {
	switch {
	case validRate(a) && validRate(b):
		return max(a, b)
	case validRate(a):
		return a
	case validRate(b):
		return b
	default:
		return 0
	}
}

func nonNegMax(a, b float64) float64 {
	best := 0.0
	if nonNegFinite(a) && a > best {
		best = a
	}
	if nonNegFinite(b) && b > best {
		best = b
	}
	return best
}

func boundedExaminerTransfer(transfer float64, created, now, dayStart time.Time) float64 {
	if !validRate(transfer) {
		return 0
	}
	if created.IsZero() {
		return transfer
	}
	if !created.Before(dayStart) && !created.After(now) {
		return transfer
	}
	return 0
}

func snapshotFromState(st examinerState, home string, now time.Time, timeout time.Duration, maxPrice float64) (live int, hourly float64, accrued float64) {
	fallbackRate := maxPrice
	if !validRate(fallbackRate) {
		fallbackRate = 1.50
	}
	dayStart := startOfDay(now)
	seenLive := map[string]bool{}
	if st.InstanceID != "" || st.PendingRent {
		if st.InstanceID != "" && spentHasExaminer(home, st.InstanceID) {
			seenLive[st.InstanceID] = true
		} else {
			live++
			obs := st.ObservedLiveAt
			created := st.Created
			rate := st.Price
			transfer := nonNegMax(st.TransferDollars, 0)
			if st.InstanceID != "" {
				seenLive[st.InstanceID] = true
				if u, ok := findUnresolved(st.Unresolved, st.InstanceID); ok {
					if u.ObservedLiveAt.After(obs) {
						obs = u.ObservedLiveAt
					}
					if created.IsZero() && !u.Created.IsZero() {
						created = u.Created
					}
					rate = higherValidRate(rate, u.Price)
					transfer = nonNegMax(transfer, u.TransferDollars)
				}
			}
			if !validRate(rate) {
				rate = fallbackRate
			}
			hourly += rate + transfer
			accrued += boundedExaminerAccrual(rate, created, obs, now, dayStart, timeout) + boundedExaminerTransfer(transfer, created, now, dayStart)
		}
	}
	for _, u := range st.Unresolved {
		if u.ID == "" || seenLive[u.ID] {
			continue
		}
		seenLive[u.ID] = true
		if spentHasExaminer(home, u.ID) {
			continue
		}
		live++
		rate := u.Price
		if !validRate(rate) {
			rate = fallbackRate
		}
		transfer := nonNegMax(u.TransferDollars, 0)
		hourly += rate + transfer
		accrued += boundedExaminerAccrual(rate, u.Created, u.ObservedLiveAt, now, dayStart, timeout) + boundedExaminerTransfer(transfer, u.Created, now, dayStart)
	}
	for _, p := range st.PendingSpends {
		if p.ID != "" && !seenLive[p.ID] && !spentHasExaminer(home, p.ID) {
			cp := p
			if !validRate(cp.Cost) {
				cp.Cost = fallbackRate
			}
			end := cp.End
			if end.IsZero() {
				end = now
			}
			accrued += cp.dollarsBetween(dayStart, end)
		}
	}
	return live, hourly, accrued
}

func activeExaminerFromDisk(home string, now time.Time, timeout time.Duration, maxPrice float64) (live int, hourly float64, accrued float64, err error) {
	st, err := readExaminerStateErr(home)
	if err != nil {
		return 0, 0, 0, err
	}
	if st == nil {
		return 0, 0, 0, nil
	}
	live, hourly, accrued = snapshotFromState(*st, home, now, timeout, maxPrice)
	return live, hourly, accrued, nil
}

// examinerDiskID identifies the active instance / pre-rental reservation last observed or written
// cleanly to examiner.json so dirty recovery can distinguish a peer controller's newer disk write
// from this controller's own stale pre-failure file.
type examinerDiskID struct {
	InstanceID    string
	PendingRent   bool
	ReservedLabel string
	APIKey        string
	CertPEM       string
}

func diskIDOf(st *examinerState) examinerDiskID {
	if st == nil {
		return examinerDiskID{}
	}
	return examinerDiskID{
		InstanceID:    st.InstanceID,
		PendingRent:   st.PendingRent,
		ReservedLabel: st.ReservedLabel,
		APIKey:        st.APIKey,
		CertPEM:       st.CertPEM,
	}
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

	fileMu            sync.Mutex
	recorded          map[string]bool
	mu                sync.RWMutex
	loaded            bool
	dirty             bool           // true when last saveState/persistState failed so in-memory state leads disk
	diskUnreadable    bool           // true when last readExaminerStateErr failed; blocks overwriting/removing examiner.json
	lastDisk          examinerDiskID // active identity on disk at last clean read or write
	owner             string
	state             examinerState
	ready             bool
	proxy             *httputil.ReverseProxy
	proxyTr           *http.Transport
	upstream          string
	lastProbeErr      string
	blockedHourlyNeed float64 // cheapest valid examiner offer price blocked only by --max-hourly on last Reconcile
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

func (e *Examiner) writeStateFile(st *examinerState) error {
	if e.home == "" || e.dryRun {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	e.mu.RLock()
	unreadable := e.diskUnreadable
	e.mu.RUnlock()
	if unreadable {
		return errors.New("examiner.json unreadable; refusing to overwrite disk state")
	}
	err := writeExaminerState(e.home, st)
	if err == nil {
		e.mu.Lock()
		e.dirty = false
		e.lastDisk = diskIDOf(st)
		e.mu.Unlock()
	}
	return err
}

func (e *Examiner) saveState(st *examinerState) error {
	if e.home == "" || e.dryRun {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	e.mu.RLock()
	unreadable := e.diskUnreadable
	e.mu.RUnlock()
	if unreadable {
		return errors.New("examiner.json unreadable; refusing to overwrite disk state")
	}
	err := writeExaminerState(e.home, st)
	e.mu.Lock()
	e.dirty = err != nil
	if err == nil {
		e.lastDisk = diskIDOf(st)
	}
	e.mu.Unlock()
	return err
}

func (e *Examiner) persistState(st *examinerState) error {
	if e.home == "" || e.dryRun {
		return nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	e.mu.RLock()
	unreadable := e.diskUnreadable
	e.mu.RUnlock()
	if unreadable {
		return errors.New("examiner.json unreadable; refusing to overwrite disk state")
	}
	var err error
	written := st
	if st != nil && st.InstanceID != "" {
		e.mu.RLock()
		match := e.state.InstanceID == st.InstanceID && e.state.APIKey == st.APIKey && e.state.CertPEM == st.CertPEM
		pending := append([]podRecord(nil), e.state.PendingSpends...)
		unresolved := append([]ExaminerInstance(nil), e.state.Unresolved...)
		obs := e.state.ObservedLiveAt
		memPrice := e.state.Price
		memDisk := e.state.DiskCost
		memTransfer := e.state.TransferDollars
		e.mu.RUnlock()
		if !match {
			return nil
		}
		cp := *st
		cp.PendingSpends = pending
		cp.Unresolved = unresolved
		cp.Price = higherValidRate(cp.Price, memPrice)
		cp.DiskCost = nonNegMax(cp.DiskCost, memDisk)
		cp.TransferDollars = nonNegMax(cp.TransferDollars, memTransfer)
		if obs.After(cp.ObservedLiveAt) {
			cp.ObservedLiveAt = obs
		}
		written = &cp
		err = writeExaminerState(e.home, &cp)
	} else {
		err = writeExaminerState(e.home, st)
	}
	e.mu.Lock()
	e.dirty = err != nil
	if err == nil {
		e.lastDisk = diskIDOf(written)
	}
	e.mu.Unlock()
	return err
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
	unresolved := append([]ExaminerInstance(nil), e.state.Unresolved...)
	e.state = examinerState{PendingSpends: pending, Unresolved: unresolved}
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

func (e *Examiner) fallbackHourlyRate() float64 {
	if validRate(e.cfg.MaxPrice) {
		return e.cfg.MaxPrice
	}
	return 1.50
}

// pendingRentActive reports whether an unresolved pre-rental reservation must remain held.
// Because Vast.ai create calls are not idempotent, an ambiguous create whose response was lost
// could still have produced a live paid contract that appears late; keep holding the slot and
// hourly charge until the provider confirms its outcome (instance appears and is adopted/ended,
// or create returned an explicit errRentRejected / errNoRoom).
func (e *Examiner) pendingRentActive(saved examinerState, _ time.Time) bool {
	return saved.PendingRent && saved.InstanceID == ""
}

func mergePendingSpends(a, b []podRecord) []podRecord {
	seen := map[string]bool{}
	var out []podRecord
	for _, list := range [][]podRecord{a, b} {
		for _, r := range list {
			if r.ID == "" || seen[r.ID] {
				continue
			}
			seen[r.ID] = true
			out = append(out, r)
		}
	}
	return out
}

func mergeUnresolvedLists(a, b []ExaminerInstance) []ExaminerInstance {
	byID := map[string]ExaminerInstance{}
	var order []string
	for _, list := range [][]ExaminerInstance{a, b} {
		for _, u := range list {
			if u.ID == "" {
				continue
			}
			if prev, ok := byID[u.ID]; ok {
				byID[u.ID] = mergeExaminerInstance(prev, u)
			} else {
				byID[u.ID] = u
				order = append(order, u.ID)
			}
		}
	}
	out := make([]ExaminerInstance, 0, len(order))
	for _, id := range order {
		out = append(out, byID[id])
	}
	return out
}

// reloadState reloads durable examiner.json under fileMu so multiple controllers sharing a fleet
// home (e.g. standalone `stet-cloud examiner` and `stet-cloud scale --follow` serialized by the
// fleet lock) always observe each other's rentals and teardowns instead of killing active VMs as
// strays. When a prior write failed (e.dirty), disk is inspected first so recovery NEVER clobbers
// or deletes an active examiner persisted by another controller.
func (e *Examiner) reloadState(tryFlush bool) error {
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	e.mu.Lock()
	var ownerErr error
	if e.owner == "" {
		if id, err := ensureExaminerOwner(e.home, !e.dryRun); err != nil {
			ownerErr = err
		} else {
			e.owner = id
		}
	}
	if e.home == "" {
		e.loaded = true
		e.mu.Unlock()
		return ownerErr
	}
	dirty := e.dirty
	wasUnreadable := e.diskUnreadable
	lastDisk := e.lastDisk
	memSt := e.cloneStateLocked()
	e.mu.Unlock()

	diskSt, stateErr := readExaminerStateErr(e.home)
	if stateErr != nil {
		e.mu.Lock()
		e.diskUnreadable = true
		e.mu.Unlock()
		if dirty && !tryFlush {
			return ownerErr
		}
		return stateErr
	}
	if dirty {
		curDiskID := diskIDOf(diskSt)
		if curDiskID != lastDisk {
			// A peer controller genuinely moved examiner.json since our last clean sync.
			// Adopt disk's active state and merge any unsaved PendingSpends/Unresolved obligations
			// rather than clobbering the peer's state.
			var merged examinerState
			if diskSt != nil {
				merged = *diskSt
			}
			merged.PendingSpends = mergePendingSpends(merged.PendingSpends, memSt.PendingSpends)
			merged.Unresolved = mergeUnresolvedLists(merged.Unresolved, memSt.Unresolved)
			var basePending []podRecord
			var baseUnresolved []ExaminerInstance
			if diskSt != nil {
				basePending = diskSt.PendingSpends
				baseUnresolved = diskSt.Unresolved
			}
			needWrite := len(merged.PendingSpends) != len(basePending) ||
				len(merged.Unresolved) != len(baseUnresolved)
			var werr error
			if tryFlush && needWrite && !e.dryRun {
				werr = writeExaminerState(e.home, &merged)
			}
			e.mu.Lock()
			e.loaded = true
			e.diskUnreadable = false
			e.lastDisk = curDiskID
			if !needWrite || (tryFlush && werr == nil) {
				e.dirty = false
				if tryFlush && werr == nil {
					e.lastDisk = diskIDOf(&merged)
				}
			}
			changed := e.state.InstanceID != merged.InstanceID ||
				e.state.APIKey != merged.APIKey ||
				e.state.CertPEM != merged.CertPEM
			e.state = merged
			var oldTr *http.Transport
			if changed {
				oldTr = e.proxyTr
				e.ready = false
				e.proxy = nil
				e.proxyTr = nil
				e.upstream = ""
				e.lastProbeErr = ""
			}
			e.mu.Unlock()
			if oldTr != nil {
				oldTr.CloseIdleConnections()
			}
			if werr != nil {
				return werr
			}
			return ownerErr
		}
		// Disk still holds our own stale pre-failure state (no peer moved it): NEVER resurrect
		// stale disk state into memory; keep memSt for ActiveSnapshot, and flush memSt on tryFlush.
		if !tryFlush {
			e.mu.Lock()
			e.diskUnreadable = false
			e.mu.Unlock()
			return ownerErr
		}
		merged := memSt
		if diskSt != nil {
			merged.PendingSpends = mergePendingSpends(memSt.PendingSpends, diskSt.PendingSpends)
			merged.Unresolved = mergeUnresolvedLists(memSt.Unresolved, diskSt.Unresolved)
			if memSt.InstanceID != "" &&
				memSt.InstanceID == diskSt.InstanceID &&
				memSt.APIKey == diskSt.APIKey &&
				memSt.CertPEM == diskSt.CertPEM {
				merged.Price = higherValidRate(merged.Price, diskSt.Price)
				merged.DiskCost = nonNegMax(merged.DiskCost, diskSt.DiskCost)
				merged.TransferDollars = nonNegMax(merged.TransferDollars, diskSt.TransferDollars)
				if diskSt.ObservedLiveAt.After(merged.ObservedLiveAt) {
					merged.ObservedLiveAt = diskSt.ObservedLiveAt
				}
			}
		}
		if err := writeExaminerState(e.home, &merged); err != nil {
			e.mu.Lock()
			e.diskUnreadable = false
			e.mu.Unlock()
			return err
		}
		e.mu.Lock()
		if e.state.InstanceID != "" &&
			e.state.InstanceID == merged.InstanceID &&
			e.state.APIKey == merged.APIKey &&
			e.state.CertPEM == merged.CertPEM {
			merged.Price = higherValidRate(merged.Price, e.state.Price)
			merged.DiskCost = nonNegMax(merged.DiskCost, e.state.DiskCost)
			merged.TransferDollars = nonNegMax(merged.TransferDollars, e.state.TransferDollars)
			if e.state.ObservedLiveAt.After(merged.ObservedLiveAt) {
				merged.ObservedLiveAt = e.state.ObservedLiveAt
			}
		}
		e.state = merged
		e.lastDisk = diskIDOf(&merged)
		e.dirty = false
		e.diskUnreadable = false
		e.loaded = true
		e.mu.Unlock()
		return ownerErr
	}
	var next examinerState
	if diskSt != nil {
		next = *diskSt
	}
	var werr error
	needFlushRecovered := false
	needRecoverWrite := false
	needRateWrite := false
	if wasUnreadable && (len(memSt.PendingSpends) > 0 || len(memSt.Unresolved) > 0) {
		var basePending []podRecord
		var baseUnresolved []ExaminerInstance
		if diskSt != nil {
			basePending = diskSt.PendingSpends
			baseUnresolved = diskSt.Unresolved
		}
		next.PendingSpends = mergePendingSpends(next.PendingSpends, memSt.PendingSpends)
		next.Unresolved = mergeUnresolvedLists(next.Unresolved, memSt.Unresolved)
		if len(next.PendingSpends) != len(basePending) || len(next.Unresolved) != len(baseUnresolved) {
			needRecoverWrite = true
		}
	}
	if diskSt != nil && memSt.InstanceID != "" &&
		memSt.InstanceID == diskSt.InstanceID &&
		memSt.APIKey == diskSt.APIKey &&
		memSt.CertPEM == diskSt.CertPEM {
		next.Price = higherValidRate(next.Price, memSt.Price)
		next.DiskCost = nonNegMax(next.DiskCost, memSt.DiskCost)
		next.TransferDollars = nonNegMax(next.TransferDollars, memSt.TransferDollars)
		if memSt.ObservedLiveAt.After(next.ObservedLiveAt) {
			next.ObservedLiveAt = memSt.ObservedLiveAt
		}
		if next.Price > diskSt.Price+1e-9 || next.DiskCost > diskSt.DiskCost+1e-9 || next.TransferDollars > diskSt.TransferDollars+1e-9 {
			needRateWrite = true
		}
	}
	if (needRecoverWrite || needRateWrite) && !e.dryRun {
		if tryFlush {
			werr = writeExaminerState(e.home, &next)
			needFlushRecovered = werr != nil
		} else if needRecoverWrite {
			needFlushRecovered = true
		}
	}
	e.mu.Lock()
	e.loaded = true
	e.diskUnreadable = false
	if needFlushRecovered {
		e.dirty = true
		e.lastDisk = diskIDOf(diskSt)
	} else {
		e.lastDisk = diskIDOf(&next)
	}
	changed := e.state.InstanceID != next.InstanceID ||
		e.state.APIKey != next.APIKey ||
		e.state.CertPEM != next.CertPEM
	if !changed && e.state.InstanceID != "" && e.state.InstanceID == next.InstanceID {
		next.Price = higherValidRate(next.Price, e.state.Price)
		next.DiskCost = nonNegMax(next.DiskCost, e.state.DiskCost)
		next.TransferDollars = nonNegMax(next.TransferDollars, e.state.TransferDollars)
		if e.state.ObservedLiveAt.After(next.ObservedLiveAt) {
			next.ObservedLiveAt = e.state.ObservedLiveAt
		}
	}
	e.state = next
	var oldTr *http.Transport
	if changed {
		oldTr = e.proxyTr
		e.ready = false
		e.proxy = nil
		e.proxyTr = nil
		e.upstream = ""
		e.lastProbeErr = ""
	}
	e.mu.Unlock()
	if oldTr != nil {
		oldTr.CloseIdleConnections()
	}
	if werr != nil {
		return werr
	}
	return ownerErr
}

func (e *Examiner) ensureLoaded() error {
	return e.reloadState(false)
}

func (e *Examiner) ownerID() string {
	e.mu.RLock()
	if e.owner != "" {
		id := e.owner
		e.mu.RUnlock()
		return id
	}
	e.mu.RUnlock()
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.owner == "" {
		if id, err := ensureExaminerOwner(e.home, !e.dryRun); err == nil {
			e.owner = id
		}
	}
	return e.owner
}

func (e *Examiner) cloneStateLocked() examinerState {
	st := e.state
	st.PendingSpends = append([]podRecord(nil), e.state.PendingSpends...)
	st.Unresolved = append([]ExaminerInstance(nil), e.state.Unresolved...)
	return st
}

func (e *Examiner) setUnresolved(insts []ExaminerInstance) {
	e.mu.Lock()
	had := len(e.state.Unresolved) > 0
	prevByID := map[string]ExaminerInstance{}
	for _, p := range e.state.Unresolved {
		if p.ID != "" {
			prevByID[p.ID] = p
		}
	}
	merged := make([]ExaminerInstance, 0, len(insts))
	for _, cur := range insts {
		if prev, ok := prevByID[cur.ID]; ok {
			cur = mergeExaminerInstance(prev, cur)
		}
		merged = append(merged, cur)
	}
	e.state.Unresolved = merged
	shouldSave := e.loaded && (had || len(merged) > 0)
	st := e.cloneStateLocked()
	e.mu.Unlock()
	if shouldSave {
		if err := e.saveState(&st); err != nil {
			e.say("warning: could not persist unresolved examiner state in examiner.json: %v", err)
		}
	}
}

func examinerObservedRate(savedPrice, savedDisk float64, inst ExaminerInstance) (price, disk float64) {
	disk = max(0, savedDisk)
	origDisk := max(0, inst.DiskCost)
	if nonNegFinite(inst.DiskCost) && inst.DiskCost > disk {
		disk = inst.DiskCost
	}
	price = higherValidRate(savedPrice, inst.Price)
	if validRate(inst.BasePrice) {
		price = higherValidRate(price, inst.BasePrice+disk)
	} else if inst.PartialRate && validRate(inst.Price) {
		price = higherValidRate(price, inst.Price+max(0, disk-origDisk))
	}
	return price, disk
}

func mergeExaminerInstance(prev, cur ExaminerInstance) ExaminerInstance {
	if cur.Label == "" {
		cur.Label = prev.Label
	}
	if cur.Created.IsZero() {
		cur.Created = prev.Created
	}
	cur.Price, cur.DiskCost = examinerObservedRate(prev.Price, prev.DiskCost, cur)
	cur.TransferDollars = nonNegMax(prev.TransferDollars, cur.TransferDollars)
	if prev.ObservedLiveAt.After(cur.ObservedLiveAt) {
		cur.ObservedLiveAt = prev.ObservedLiveAt
	}
	return cur
}

// updateUnresolved refreshes persisted Unresolved after a partial pass (like EnforceTimeout) that
// only attempts teardown on expired instances: removes instances no longer listed or confirmed
// ended, and upserts any owned instances whose EndExaminer just failed.
func (e *Examiner) updateUnresolved(owned []ExaminerInstance, ended map[string]bool, failed []ExaminerInstance) {
	listed := map[string]ExaminerInstance{}
	for _, inst := range owned {
		listed[inst.ID] = inst
	}
	e.mu.Lock()
	had := len(e.state.Unresolved) > 0
	byID := map[string]ExaminerInstance{}
	for _, prev := range e.state.Unresolved {
		if ended[prev.ID] {
			continue
		}
		if cur, ok := listed[prev.ID]; ok {
			byID[prev.ID] = mergeExaminerInstance(prev, cur)
		}
	}
	for _, inst := range failed {
		if prev, ok := byID[inst.ID]; ok {
			byID[inst.ID] = mergeExaminerInstance(prev, inst)
		} else {
			byID[inst.ID] = inst
		}
	}
	next := make([]ExaminerInstance, 0, len(byID))
	for _, inst := range byID {
		next = append(next, inst)
	}
	e.state.Unresolved = next
	shouldSave := e.loaded && (had || len(next) > 0)
	st := e.cloneStateLocked()
	e.mu.Unlock()
	if shouldSave {
		if err := e.saveState(&st); err != nil {
			e.say("warning: could not persist unresolved examiner state in examiner.json: %v", err)
		}
	}
}

// ActiveSnapshot reports the live count, hourly price, and accrued spend today (UTC) of any
// active examiner, pre-rental reservation, or unresolved owned orphan managed by this controller
// (plus any terminal charges pending durable flush to spent.jsonl). Returns a non-nil error if
// examiner.json / examiner-owner cannot be read cleanly.
func (e *Examiner) ActiveSnapshot(now time.Time) (live int, hourly float64, accrued float64, err error) {
	if e == nil {
		return 0, 0, 0, nil
	}
	if err := e.ensureLoaded(); err != nil {
		return 0, 0, 0, err
	}
	e.mu.RLock()
	st := e.cloneStateLocked()
	e.mu.RUnlock()
	live, hourly, accrued = snapshotFromState(st, e.home, now, e.cfg.Timeout, e.cfg.MaxPrice)
	return live, hourly, accrued, nil
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

func (e *Examiner) setBlockedHourly(price float64) {
	if e == nil {
		return
	}
	if !validRate(price) {
		price = 0
	}
	e.mu.Lock()
	e.blockedHourlyNeed = price
	e.mu.Unlock()
}

func (e *Examiner) blockedHourly() float64 {
	if e == nil {
		return 0
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.blockedHourlyNeed
}

func cheapestValidExaminerOffer(offers []ExaminerOffer, maxPrice, maxTransfer float64) float64 {
	best := 0.0
	for _, off := range offers {
		if !validRate(off.Price) {
			continue
		}
		if maxTransfer >= 0 && nonNegMax(off.TransferDollars, 0) > maxTransfer+1e-9 {
			continue
		}
		eff := off.effectiveHourly()
		if maxPrice > 0 && eff > maxPrice+1e-9 {
			continue
		}
		if best == 0 || eff < best {
			best = eff
		}
	}
	return best
}

func spentHasExaminerErr(home, id string) (bool, error) {
	if home == "" || id == "" {
		return false, nil
	}
	sum, err := readSpentFile(filepath.Join(home, "spent.jsonl"))
	if err != nil {
		return false, err
	}
	return sum.byExaminerID[id], nil
}

func spentHasExaminer(home, id string) bool {
	has, err := spentHasExaminerErr(home, id)
	return err == nil && has
}

func (e *Examiner) recordSpend(rec podRecord) (bool, error) {
	if e.onSpent == nil || rec.ID == "" {
		return false, nil
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	if e.recorded == nil {
		e.recorded = map[string]bool{}
	}
	if e.recorded[rec.ID] {
		return false, nil
	}
	has, err := spentHasExaminerErr(e.home, rec.ID)
	if err != nil {
		return false, err
	}
	if has {
		e.recorded[rec.ID] = true
		return false, nil
	}
	if err := e.onSpent(rec); err != nil {
		return false, err
	}
	e.recorded[rec.ID] = true
	return true, nil
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
	st := e.cloneStateLocked()
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
		if spentHasExaminer(e.home, rec.ID) {
			continue
		}
		if !validRate(rec.Cost) {
			rec.Cost = e.fallbackHourlyRate()
		}
		if _, err := e.recordSpend(rec); err != nil {
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
	st := e.cloneStateLocked()
	e.mu.Unlock()
	if err := e.saveState(&st); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

func (e *Examiner) recordEndedAt(id, label string, price float64, start, end time.Time, roundID, why string) error {
	_, err := e.recordEndedAtFull(id, label, price, 0, start, end, roundID, why)
	return err
}

func (e *Examiner) recordEndedAtDollars(id, label string, price float64, start, end time.Time, roundID, why string) (float64, error) {
	return e.recordEndedAtFull(id, label, price, 0, start, end, roundID, why)
}

func (e *Examiner) recordEndedAtFull(id, label string, price, transferDollars float64, start, end time.Time, roundID, why string) (float64, error) {
	if id == "" {
		return 0, nil
	}
	if start.IsZero() || start.After(end) {
		start = end
	}
	if roundID == "" {
		roundID = examinerLabelRound(label, e.ownerID())
	}
	if !validRate(price) {
		if e.onSpent != nil && spentHasExaminer(e.home, id) {
			return 0, nil
		}
		// Never write a $0.00 spend record or deadlock future autoscaling when the provider omits
		// a stray instance's hourly rate: record conservatively at the configured max-price cap.
		price = e.fallbackHourlyRate()
		if why != "" {
			why += " (provider rate unavailable; estimated at max price)"
		} else {
			why = "provider rate unavailable; estimated at max price"
		}
	}
	fixed := nonNegMax(transferDollars, 0)
	rec := podRecord{
		ID: id, Market: "vast", Handle: examinerLabelPrefix, GPU: "RTX 5090",
		Cloud: "verified", Cost: price, FixedCost: fixed, Start: start, Alive: end, End: end,
		For: "examiner", Round: roundID, Why: why,
	}
	now := e.now()
	todayDollars := rec.dollarsBetween(startOfDay(now), end)
	if e.onSpent == nil {
		return todayDollars, nil
	}
	wrote, err := e.recordSpend(rec)
	if err != nil {
		e.queuePendingSpend(rec)
		return todayDollars, fmt.Errorf("recording examiner spend for %s: %w", id, err)
	}
	if !wrote {
		return 0, nil
	}
	return todayDollars, nil
}

func (e *Examiner) recordEnded(inst ExaminerInstance, fallbackStart time.Time, fallbackPrice float64, roundID, why string) error {
	_, err := e.recordEndedDollars(inst, fallbackStart, fallbackPrice, roundID, why)
	return err
}

func (e *Examiner) recordEndedDollars(inst ExaminerInstance, fallbackStart time.Time, fallbackPrice float64, roundID, why string) (float64, error) {
	now := e.now()
	start := inst.Created
	if !fallbackStart.IsZero() && (start.IsZero() || start.After(fallbackStart)) {
		start = fallbackStart
	}
	price := higherValidRate(inst.Price, fallbackPrice)
	return e.recordEndedAtFull(inst.ID, inst.Label, price, inst.TransferDollars, start, now, roundID, why)
}

// recordMissingSaved records a bounded, idempotent terminal spend charge for a saved examiner
// instance that vanished from the provider (e.g. terminated by its container-side TTL watchdog
// or ended during a controller outage) so accrued spend never vanishes from daily budget totals.
func (e *Examiner) recordMissingSaved(saved examinerState, why string) error {
	_, err := e.recordMissingSavedDollars(saved, why)
	return err
}

func (e *Examiner) recordMissingSavedDollars(saved examinerState, why string) (float64, error) {
	if saved.InstanceID == "" || e.dryRun {
		return 0, nil
	}
	now := e.now()
	created := saved.Created
	obs := saved.ObservedLiveAt
	price := saved.Price
	transfer := nonNegMax(saved.TransferDollars, 0)
	if u, ok := findUnresolved(saved.Unresolved, saved.InstanceID); ok {
		if u.ObservedLiveAt.After(obs) {
			obs = u.ObservedLiveAt
		}
		if created.IsZero() && !u.Created.IsZero() {
			created = u.Created
		}
		price = higherValidRate(price, u.Price)
		transfer = nonNegMax(transfer, u.TransferDollars)
	}
	start := created
	if start.IsZero() || start.After(now) {
		start = now
	}
	end := examinerAccrualEnd(created, obs, now, e.cfg.Timeout)
	return e.recordEndedAtFull(saved.InstanceID, saved.ReservedLabel, price, transfer, start, end, saved.RoundID, why)
}

// recordMissingInstance records a bounded, idempotent terminal spend charge for a tracked
// examiner instance (saved or previously unresolved orphan) that vanished from the provider so
// its accrued daily spend never disappears from budget totals.
func (e *Examiner) recordMissingInstance(inst ExaminerInstance, why string) error {
	_, err := e.recordMissingInstanceDollars(inst, why)
	return err
}

func (e *Examiner) recordMissingInstanceDollars(inst ExaminerInstance, why string) (float64, error) {
	if inst.ID == "" || e.dryRun {
		return 0, nil
	}
	now := e.now()
	start := inst.Created
	if start.IsZero() || start.After(now) {
		start = now
	}
	end := examinerAccrualEnd(inst.Created, inst.ObservedLiveAt, now, e.cfg.Timeout)
	return e.recordEndedAtFull(inst.ID, inst.Label, inst.Price, inst.TransferDollars, start, end, "", why)
}

func findUnresolved(prev []ExaminerInstance, id string) (ExaminerInstance, bool) {
	for _, u := range prev {
		if u.ID != "" && u.ID == id {
			return u, true
		}
	}
	return ExaminerInstance{}, false
}

func savedOrphanInstance(saved examinerState, prev []ExaminerInstance) ExaminerInstance {
	obs := saved.ObservedLiveAt
	if u, ok := findUnresolved(prev, saved.InstanceID); ok {
		if obs.After(u.ObservedLiveAt) {
			u.ObservedLiveAt = obs
		}
		if u.Created.IsZero() {
			u.Created = saved.Created
		}
		u.Price, u.DiskCost = examinerObservedRate(saved.Price, saved.DiskCost, u)
		u.TransferDollars = nonNegMax(u.TransferDollars, saved.TransferDollars)
		return u
	}
	return ExaminerInstance{
		ID:              saved.InstanceID,
		Label:           saved.ReservedLabel,
		State:           "unknown",
		Price:           saved.Price,
		DiskCost:        saved.DiskCost,
		TransferDollars: saved.TransferDollars,
		Created:         saved.Created,
		ObservedLiveAt:  obs,
	}
}

func (e *Examiner) noteSavedObservedLive(saved *examinerState, now time.Time) {
	if now.After(saved.ObservedLiveAt) {
		saved.ObservedLiveAt = now
	}
	e.mu.Lock()
	if e.state.InstanceID == saved.InstanceID && now.After(e.state.ObservedLiveAt) {
		e.state.ObservedLiveAt = now
	}
	e.mu.Unlock()
}

// noteSavedPrice raises saved.Price and e.state.Price whenever the provider reports a higher
// valid hourly rate (or higher base compute rate plus known disk storage cost) for the same
// instance, never lowering a conservative reserved/known rate. Reports whether persisted fields
// changed.
func (e *Examiner) noteSavedPrice(saved *examinerState, inst ExaminerInstance) bool {
	price, disk := examinerObservedRate(saved.Price, saved.DiskCost, inst)
	transfer := nonNegMax(saved.TransferDollars, inst.TransferDollars)
	changed := false
	if validRate(price) && (!validRate(saved.Price) || price > saved.Price+1e-9) {
		saved.Price = price
		changed = true
	}
	if nonNegFinite(disk) && disk > saved.DiskCost+1e-9 {
		saved.DiskCost = disk
		changed = true
	}
	if nonNegFinite(transfer) && transfer > saved.TransferDollars+1e-9 {
		saved.TransferDollars = transfer
		changed = true
	}
	e.mu.Lock()
	if e.state.InstanceID == saved.InstanceID {
		ePrice, eDisk := examinerObservedRate(e.state.Price, e.state.DiskCost, inst)
		ePrice = higherValidRate(ePrice, saved.Price)
		if saved.DiskCost > eDisk {
			eDisk = saved.DiskCost
		}
		eTransfer := nonNegMax(e.state.TransferDollars, saved.TransferDollars)
		if validRate(ePrice) && (!validRate(e.state.Price) || ePrice > e.state.Price+1e-9) {
			e.state.Price = ePrice
			changed = true
		}
		if nonNegFinite(eDisk) && eDisk > e.state.DiskCost+1e-9 {
			e.state.DiskCost = eDisk
			changed = true
		}
		if nonNegFinite(eTransfer) && eTransfer > e.state.TransferDollars+1e-9 {
			e.state.TransferDollars = eTransfer
			changed = true
		}
	}
	e.mu.Unlock()
	return changed
}

// teardownKnownDueOnListErr attempts direct EndExaminer teardown on known due instance IDs
// (tracked saved.InstanceID and Unresolved orphans) even while provider listing is failing, so a
// listing outage never disables timeout or idle cleanup (including containers still pulling their
// image before their in-container watchdog starts).
func (e *Examiner) teardownKnownDueOnListErr(ctx context.Context, saved examinerState, allDue bool, now time.Time) {
	_, savedWasUnresolved := findUnresolved(saved.Unresolved, saved.InstanceID)
	savedDue := saved.InstanceID != "" &&
		(allDue || savedWasUnresolved || saved.Created.IsZero() || now.Sub(saved.Created) > e.cfg.Timeout || saved.APIKey == "" || saved.CertPEM == "")
	if e.dryRun {
		if savedDue {
			e.say("would end known due examiner GPU (machine %s) during listing outage", saved.InstanceID)
		}
		for _, u := range saved.Unresolved {
			if u.ID != "" && u.ID != saved.InstanceID {
				e.say("would end known unresolved examiner GPU (machine %s) during listing outage", u.ID)
			}
		}
		return
	}
	endedIDs := map[string]bool{}
	var keepUnresolved []ExaminerInstance
	changedUnresolved := false
	for _, u := range saved.Unresolved {
		if u.ID == "" || u.ID == saved.InstanceID {
			continue
		}
		if err := e.prov.EndExaminer(ctx, u.ID); err != nil {
			keepUnresolved = append(keepUnresolved, u)
			e.say("could not end known unresolved examiner GPU (machine %s) during listing outage: %v", u.ID, err)
			continue
		}
		endedIDs[u.ID] = true
		changedUnresolved = true
		e.say("ended known unresolved examiner GPU (machine %s) during listing outage", u.ID)
		_ = e.recordMissingInstance(u, "unresolved examiner ended during provider listing outage")
	}
	if savedDue {
		if err := e.prov.EndExaminer(ctx, saved.InstanceID); err != nil {
			e.proxyOffline()
			keepUnresolved = append(keepUnresolved, savedOrphanInstance(saved, saved.Unresolved))
			changedUnresolved = true
			e.say("could not end known due examiner GPU (machine %s) during listing outage: %v", saved.InstanceID, err)
		} else {
			endedIDs[saved.InstanceID] = true
			changedUnresolved = true
			why := "examiner ended during provider listing outage"
			if !allDue && !saved.Created.IsZero() && now.Sub(saved.Created) > e.cfg.Timeout {
				why = "timed-out examiner ended during provider listing outage"
			}
			e.say("ended known examiner GPU (machine %s) during listing outage", saved.InstanceID)
			_ = e.recordMissingSaved(saved, why)
			e.clearState()
		}
	}
	if changedUnresolved || len(endedIDs) > 0 {
		e.setUnresolved(keepUnresolved)
	}
}

// confirmMissingUnresolved checks previously unresolved orphans that are absent from the latest
// provider listing. Never drops an orphan or bills terminal spend on omission alone: calls
// EndExaminer so termination is confirmed (200/204/404 -> nil) before recording bounded spend,
// and retains the orphan in Unresolved if EndExaminer fails.
func (e *Examiner) confirmMissingUnresolved(ctx context.Context, prevUnresolved, listed []ExaminerInstance, skipID, why string) (still []ExaminerInstance, ended map[string]bool, endedToday float64, endErr, spendErr error) {
	if len(prevUnresolved) == 0 || e.dryRun {
		return nil, nil, 0, nil, nil
	}
	listedIDs := make(map[string]bool, len(listed))
	for _, inst := range listed {
		listedIDs[inst.ID] = true
	}
	ended = map[string]bool{}
	for _, prev := range prevUnresolved {
		if prev.ID == "" || prev.ID == skipID || listedIDs[prev.ID] {
			continue
		}
		if err := e.prov.EndExaminer(ctx, prev.ID); err != nil {
			still = append(still, prev)
			e.say("could not confirm termination of missing examiner GPU (machine %s): %v", prev.ID, err)
			if endErr == nil {
				endErr = fmt.Errorf("confirming termination of missing examiner GPU %s: %w", prev.ID, err)
			}
			continue
		}
		ended[prev.ID] = true
		d, rerr := e.recordMissingInstanceDollars(prev, why)
		endedToday += d
		if rerr != nil && spendErr == nil {
			spendErr = rerr
		}
	}
	return still, ended, endedToday, endErr, spendErr
}

// EnforceTimeout checks owned examiner instances against the hard --examiner-timeout TTL and
// destroys any expired instance WITHOUT querying the Stet server, so Stet outages can never
// disable timeout enforcement.
func (e *Examiner) EnforceTimeout(ctx context.Context) error {
	if e == nil || e.prov == nil {
		return nil
	}
	loadErr := e.reloadState(true)
	firstErr := e.flushPendingSpends()
	if firstErr == nil && loadErr != nil {
		firstErr = loadErr
	}
	e.mu.RLock()
	owner := e.owner
	saved := e.cloneStateLocked()
	e.mu.RUnlock()
	if owner == "" && saved.InstanceID == "" && saved.ReservedLabel == "" && len(saved.Unresolved) == 0 {
		return firstErr
	}
	insts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		e.teardownKnownDueOnListErr(ctx, saved, false, e.now())
		return fmt.Errorf("listing examiner instances for timeout check: %w", err)
	}
	owned := filterOwnedExaminers(insts, owner, saved.InstanceID, saved.ReservedLabel)
	now := e.now()
	for i := range owned {
		if prev, ok := findUnresolved(saved.Unresolved, owned[i].ID); ok {
			owned[i] = mergeExaminerInstance(prev, owned[i])
		}
		owned[i].ObservedLiveAt = now
	}
	if err := e.enforceOwnedTimeouts(ctx, owned, saved, now, false); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

// enforceOwnedTimeouts ends only owned examiner instances that have independently exceeded
// --examiner-timeout (plus missing unresolved orphans and missing timed-out saved contracts),
// leaving unexpired live instances untouched. Shared by EnforceTimeout and Reconcile's degraded
// state-unavailable path so both enforce identical proxy teardown, missing-orphan confirmation,
// and spend recording without ever clobbering an unreadable examiner.json on disk.
func (e *Examiner) enforceOwnedTimeouts(ctx context.Context, owned []ExaminerInstance, saved examinerState, now time.Time, logDeferred bool) error {
	var firstErr error
	foundSaved := false
	endedIDs := map[string]bool{}
	var failedEnds []ExaminerInstance
	for _, inst := range owned {
		matchesSaved := (saved.InstanceID != "" && inst.ID == saved.InstanceID) ||
			(saved.ReservedLabel != "" && inst.Label == saved.ReservedLabel)
		if inst.ID == saved.InstanceID {
			foundSaved = true
			e.noteSavedObservedLive(&saved, now)
			if e.noteSavedPrice(&saved, inst) && !logDeferred && !e.dryRun {
				if perr := e.persistState(&saved); perr != nil && firstErr == nil {
					firstErr = fmt.Errorf("saving updated examiner.json price: %w", perr)
				}
			}
		}
		var created time.Time
		fallbackPrice := 0.0
		rid := ""
		if matchesSaved {
			created = saved.Created
			fallbackPrice, _ = examinerObservedRate(saved.Price, saved.DiskCost, inst)
			rid = saved.RoundID
		}
		if created.IsZero() {
			created = inst.Created
		}
		if created.IsZero() || now.Sub(created) <= e.cfg.Timeout {
			if logDeferred {
				e.say("examiner state unavailable; deferring stray check for live examiner GPU (machine %s) until examiner.json is readable", inst.ID)
			}
			continue
		}
		if e.dryRun {
			e.say("would end timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
			continue
		}
		if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
			e.promoteReservedInstance(&saved, inst)
			if inst.ID == saved.InstanceID {
				foundSaved = true
				e.noteSavedObservedLive(&saved, now)
			}
			if inst.Created.IsZero() && !created.IsZero() {
				inst.Created = created
			}
			fallbackDisk := 0.0
			if matchesSaved {
				fallbackDisk = saved.DiskCost
				inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
			}
			inst.Price, inst.DiskCost = examinerObservedRate(fallbackPrice, fallbackDisk, inst)
			inst.ObservedLiveAt = now
			e.proxyOffline()
			failedEnds = append(failedEnds, inst)
			e.say("could not end timed-out examiner GPU (machine %s): %v", inst.ID, err)
			if firstErr == nil {
				firstErr = fmt.Errorf("ending timed-out examiner GPU %s: %w", inst.ID, err)
			}
			continue
		}
		endedIDs[inst.ID] = true
		if matchesSaved {
			inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
		}
		e.say("ended examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
		if rerr := e.recordEnded(inst, created, fallbackPrice, rid, "exceeded examiner timeout"); rerr != nil && firstErr == nil {
			firstErr = rerr
		}
		if matchesSaved {
			e.clearState()
			saved.InstanceID, saved.PendingRent, saved.ReservedLabel = "", false, ""
		}
	}
	_, savedWasUnresolved := findUnresolved(saved.Unresolved, saved.InstanceID)
	savedNeedsEnd := !foundSaved && saved.InstanceID != "" &&
		(savedWasUnresolved || saved.Created.IsZero() || now.Sub(saved.Created) > e.cfg.Timeout)
	if !e.dryRun {
		stillUnresolved, confirmedEnded, _, uEndErr, uSpendErr := e.confirmMissingUnresolved(ctx, saved.Unresolved, owned, saved.InstanceID, "unresolved examiner orphan confirmed ended during timeout check")
		for id := range confirmedEnded {
			endedIDs[id] = true
		}
		failedEnds = append(failedEnds, stillUnresolved...)
		if uEndErr != nil && firstErr == nil {
			firstErr = uEndErr
		}
		if uSpendErr != nil && firstErr == nil {
			firstErr = uSpendErr
		}
		if savedNeedsEnd {
			if err := e.prov.EndExaminer(ctx, saved.InstanceID); err != nil {
				e.proxyOffline()
				failedEnds = append(failedEnds, savedOrphanInstance(saved, saved.Unresolved))
				e.say("could not end missing timed-out examiner GPU (machine %s): %v", saved.InstanceID, err)
				if firstErr == nil {
					firstErr = fmt.Errorf("ending missing timed-out examiner GPU %s: %w", saved.InstanceID, err)
				}
			} else {
				endedIDs[saved.InstanceID] = true
				e.say("ended missing timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", saved.InstanceID, e.cfg.Timeout)
				if rerr := e.recordMissingSaved(saved, "timed-out examiner confirmed ended after omission from provider listing"); rerr != nil && firstErr == nil {
					firstErr = rerr
				}
				e.clearState()
			}
		}
		e.updateUnresolved(owned, endedIDs, failedEnds)
	} else if savedNeedsEnd {
		e.say("would end missing timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", saved.InstanceID, e.cfg.Timeout)
	}
	return firstErr
}

// promoteReservedInstance binds a materialized PendingRent instance to saved.InstanceID and
// e.state when it is adopted or when teardown fails on it, so snapshotFromState deduplicates it
// against Unresolved (exact 1x slot/hourly/accrued count) and later disappearance bills it once
// and clears state cleanly without leaving PendingRent set.
func (e *Examiner) promoteReservedInstance(saved *examinerState, inst ExaminerInstance) {
	if saved.InstanceID != "" || !saved.PendingRent || saved.ReservedLabel == "" || inst.Label != saved.ReservedLabel {
		return
	}
	saved.InstanceID = inst.ID
	saved.PendingRent = false
	saved.ReservedLabel = ""
	saved.Price, saved.DiskCost = examinerObservedRate(saved.Price, saved.DiskCost, inst)
	saved.TransferDollars = nonNegMax(saved.TransferDollars, inst.TransferDollars)
	e.mu.Lock()
	e.state.InstanceID = saved.InstanceID
	e.state.PendingRent = false
	e.state.ReservedLabel = ""
	e.state.Price = higherValidRate(e.state.Price, saved.Price)
	e.state.DiskCost = nonNegMax(e.state.DiskCost, saved.DiskCost)
	e.state.TransferDollars = nonNegMax(e.state.TransferDollars, saved.TransferDollars)
	e.mu.Unlock()
}

// Reconcile brings the ephemeral 1x RTX 5090 examiner in line with /admin/work:
//   - Scopes all discovery, adoption, and teardown strictly to this deployment's owner ID.
//   - When no open round needs an exam, destroys every owned examiner instance at once.
//   - When an open round has ExamAt == 0, ensures at most one owned live RTX 5090 is running with
//     a known per-instance API key and pinned TLS certificate, enforces hard --examiner-timeout,
//     refuses to rent a replacement until any live owned orphan is confirmed destroyed, persists a
//     durable pre-rental reservation in examiner.json before renting, respects fleet --max /
//     --max-hourly / --daily-budget, and activates the local reverse proxy only once
//     https://<ip>:<port>/v1/models passes peer-verified TLS.
func (e *Examiner) Reconcile(ctx context.Context, w *Work, budget ExaminerBudget) error {
	if e == nil || e.prov == nil {
		return nil
	}
	e.setBlockedHourly(0)
	loadErr := e.reloadState(true)
	spendErr := e.flushPendingSpends()
	needRound := roundNeedingExam(w)
	e.mu.RLock()
	owner := e.owner
	saved := e.cloneStateLocked()
	wasReady := e.ready
	wasUpstream := e.upstream
	e.mu.RUnlock()
	preNow := e.now()
	_, _, preAccrued := snapshotFromState(saved, e.home, preNow, e.cfg.Timeout, e.cfg.MaxPrice)
	prevUnresolved := append([]ExaminerInstance(nil), saved.Unresolved...)
	rawInsts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		e.teardownKnownDueOnListErr(ctx, saved, needRound == "", e.now())
		return fmt.Errorf("listing examiner instances: %w", err)
	}
	insts := filterOwnedExaminers(rawInsts, owner, saved.InstanceID, saved.ReservedLabel)
	now := e.now()
	listedIDs := make(map[string]bool, len(insts))
	for i := range insts {
		if prev, ok := findUnresolved(prevUnresolved, insts[i].ID); ok {
			insts[i] = mergeExaminerInstance(prev, insts[i])
		}
		insts[i].ObservedLiveAt = now
		listedIDs[insts[i].ID] = true
	}

	// Case 1: no open round needs an exam -> tear down every owned examiner instance immediately.
	if needRound == "" {
		var firstEndErr error
		var failedEnds []ExaminerInstance
		endedSaved := false
		for _, inst := range insts {
			matchesSaved := (saved.InstanceID != "" && inst.ID == saved.InstanceID) ||
				(saved.ReservedLabel != "" && inst.Label == saved.ReservedLabel)
			if e.dryRun {
				e.say("would end examiner GPU (machine %s): no open round needs an exam", inst.ID)
				continue
			}
			if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
				e.promoteReservedInstance(&saved, inst)
				if inst.ID == saved.InstanceID {
					e.noteSavedObservedLive(&saved, now)
					e.noteSavedPrice(&saved, inst)
					if inst.Created.IsZero() && !saved.Created.IsZero() {
						inst.Created = saved.Created
					}
					inst.Price, inst.DiskCost = examinerObservedRate(saved.Price, saved.DiskCost, inst)
					inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
				}
				inst.ObservedLiveAt = now
				failedEnds = append(failedEnds, inst)
				e.say("could not end examiner GPU (machine %s): %v", inst.ID, err)
				if firstEndErr == nil {
					firstEndErr = fmt.Errorf("ending examiner GPU %s: %w", inst.ID, err)
				}
				continue
			}
			var fallbackStart time.Time
			fallbackPrice := 0.0
			rid := ""
			if matchesSaved {
				fallbackStart = saved.Created
				fallbackPrice, _ = examinerObservedRate(saved.Price, saved.DiskCost, inst)
				inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
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
			if rerr := e.recordEnded(inst, fallbackStart, fallbackPrice, rid, "exam posted"); rerr != nil && spendErr == nil {
				spendErr = rerr
			}
			if matchesSaved {
				e.clearState()
				saved, wasReady, wasUpstream = examinerState{}, false, ""
			}
		}
		if e.dryRun {
			if saved.InstanceID != "" && !listedIDs[saved.InstanceID] {
				e.say("would end missing examiner GPU (machine %s): no open round needs an exam", saved.InstanceID)
			}
			return loadErr
		}
		stillUnresolved, _, _, uEndErr, uSpendErr := e.confirmMissingUnresolved(ctx, prevUnresolved, insts, saved.InstanceID, "unresolved examiner orphan confirmed ended during idle cleanup")
		failedEnds = append(failedEnds, stillUnresolved...)
		if uEndErr != nil && firstEndErr == nil {
			firstEndErr = uEndErr
		}
		if uSpendErr != nil && spendErr == nil {
			spendErr = uSpendErr
		}
		if saved.InstanceID != "" && !endedSaved && !listedIDs[saved.InstanceID] {
			if err := e.prov.EndExaminer(ctx, saved.InstanceID); err != nil {
				failedEnds = append(failedEnds, savedOrphanInstance(saved, prevUnresolved))
				e.say("could not end missing examiner GPU (machine %s): %v", saved.InstanceID, err)
				if firstEndErr == nil {
					firstEndErr = fmt.Errorf("ending missing examiner GPU %s: %w", saved.InstanceID, err)
				}
			} else {
				if rerr := e.recordMissingSaved(saved, "examiner confirmed ended during idle cleanup"); rerr != nil && spendErr == nil {
					spendErr = rerr
				}
				e.clearState()
				saved, wasReady, wasUpstream = examinerState{}, false, ""
				endedSaved = true
			}
		}
		e.setUnresolved(failedEnds)
		if firstEndErr != nil {
			e.proxyOffline()
			return firstEndErr
		}
		if saved.InstanceID == "" && !endedSaved && e.pendingRentActive(saved, now) {
			e.proxyOffline()
			if spendErr != nil {
				return spendErr
			}
			return loadErr
		}
		if saved.InstanceID != "" || saved.PendingRent || wasReady {
			e.clearState()
		}
		if spendErr != nil {
			return spendErr
		}
		return loadErr
	}

	// Case 2: needRound needs an exam.
	// If examiner.json could not be read or flushed cleanly (loadErr != nil), defer any
	// identity-dependent stray deletion until durable state is readable so a restarted or
	// degraded controller never destroys the active round's examiner; independently established
	// timeout expiry, missing-orphan confirmation, and proxy teardown still proceed.
	if loadErr != nil {
		if timeoutErr := e.enforceOwnedTimeouts(ctx, insts, saved, now, true); timeoutErr != nil {
			return timeoutErr
		}
		e.say("not renting examiner GPU for %s: examiner state unavailable (%v)", needRound, loadErr)
		return loadErr
	}

	// Find if our known owned instance is alive, has a pinned cert + key, and is within TTL;
	// destroy any stopped, expired, duplicate, or orphaned owned instances. Never rent a
	// replacement while any live owned orphan fails deletion.
	var active *ExaminerInstance
	var strayEndErr error
	var failedEnds []ExaminerInstance
	endedSaved := false
	endedToday := 0.0
	dryRunTimedOut := false
	for i := range insts {
		inst := insts[i]
		if inst.State != "stopped" &&
			saved.InstanceID == "" &&
			saved.PendingRent &&
			saved.ReservedLabel != "" &&
			inst.Label == saved.ReservedLabel &&
			saved.APIKey != "" &&
			saved.CertPEM != "" &&
			active == nil {
			e.promoteReservedInstance(&saved, inst)
			e.noteSavedObservedLive(&saved, now)
			if err := e.persistState(&saved); err != nil && spendErr == nil {
				spendErr = fmt.Errorf("saving adopted examiner.json: %w", err)
			}
		}
		keep := inst.State != "stopped" &&
			saved.InstanceID != "" &&
			inst.ID == saved.InstanceID &&
			saved.APIKey != "" &&
			saved.CertPEM != "" &&
			active == nil
		if keep {
			e.noteSavedObservedLive(&saved, now)
			if e.noteSavedPrice(&saved, inst) && !e.dryRun {
				if err := e.persistState(&saved); err != nil && spendErr == nil {
					spendErr = fmt.Errorf("saving updated examiner.json price: %w", err)
				}
			}
			created := saved.Created
			if created.IsZero() {
				created = inst.Created
			}
			if !created.IsZero() && now.Sub(created) > e.cfg.Timeout {
				if e.dryRun {
					dryRunTimedOut = true
					e.say("would end timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
					continue
				}
				if err := e.prov.EndExaminer(ctx, inst.ID); err != nil {
					if inst.Created.IsZero() {
						inst.Created = created
					}
					inst.Price, inst.DiskCost = examinerObservedRate(saved.Price, saved.DiskCost, inst)
					inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
					inst.ObservedLiveAt = now
					e.proxyOffline()
					failedEnds = append(failedEnds, inst)
					e.say("could not end timed-out examiner GPU (machine %s): %v", inst.ID, err)
					if strayEndErr == nil {
						strayEndErr = fmt.Errorf("ending timed-out examiner GPU %s: %w", inst.ID, err)
					}
					continue
				}
				inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
				e.say("ended examiner GPU (machine %s): exceeded --examiner-timeout (%s)", inst.ID, e.cfg.Timeout)
				d, rerr := e.recordEndedDollars(inst, saved.Created, saved.Price, saved.RoundID, "exceeded examiner timeout")
				endedToday += d
				if rerr != nil && spendErr == nil {
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
			e.promoteReservedInstance(&saved, inst)
			if inst.ID == saved.InstanceID {
				e.noteSavedObservedLive(&saved, now)
				e.noteSavedPrice(&saved, inst)
				if inst.Created.IsZero() && !saved.Created.IsZero() {
					inst.Created = saved.Created
				}
				inst.Price, inst.DiskCost = examinerObservedRate(saved.Price, saved.DiskCost, inst)
				inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
			}
			inst.ObservedLiveAt = now
			failedEnds = append(failedEnds, inst)
			e.say("could not end stray examiner GPU (machine %s): %v", inst.ID, err)
			if strayEndErr == nil {
				strayEndErr = fmt.Errorf("ending stray examiner GPU %s: %w", inst.ID, err)
			}
			continue
		}
		matchesSaved := (saved.InstanceID != "" && inst.ID == saved.InstanceID) ||
			(saved.ReservedLabel != "" && inst.Label == saved.ReservedLabel)
		var fallbackStart time.Time
		fallbackPrice := 0.0
		rid := ""
		if matchesSaved {
			fallbackStart = saved.Created
			fallbackPrice, _ = examinerObservedRate(saved.Price, saved.DiskCost, inst)
			inst.TransferDollars = nonNegMax(inst.TransferDollars, saved.TransferDollars)
			rid = saved.RoundID
			endedSaved = true
		}
		e.say("ended stray examiner GPU (machine %s, state %s)", inst.ID, inst.State)
		d, rerr := e.recordEndedDollars(inst, fallbackStart, fallbackPrice, rid, "stray or orphaned examiner instance")
		endedToday += d
		if rerr != nil && spendErr == nil {
			spendErr = rerr
		}
		if matchesSaved {
			e.clearState()
			saved, wasReady, wasUpstream = examinerState{}, false, ""
		}
	}
	missingSavedHeld := false
	if !e.dryRun {
		stillUnresolved, _, uToday, uEndErr, uSpendErr := e.confirmMissingUnresolved(ctx, prevUnresolved, insts, saved.InstanceID, "unresolved examiner orphan confirmed ended during reconciliation")
		endedToday += uToday
		failedEnds = append(failedEnds, stillUnresolved...)
		if uEndErr != nil && strayEndErr == nil {
			strayEndErr = uEndErr
		}
		if uSpendErr != nil && spendErr == nil {
			spendErr = uSpendErr
		}
		if saved.InstanceID != "" && !endedSaved && !listedIDs[saved.InstanceID] {
			// Never clear a healthy acknowledged contract or record terminal spend on a single
			// missing provider listing (eventual consistency or delayed label visibility).
			// However, if the instance was already condemned (Unresolved from a prior failed
			// teardown), lacks credentials to adopt, has a zero creation time, or has exceeded TTL,
			// confirm termination via EndExaminer now instead of stalling synthesis.
			_, wasUnresolved := findUnresolved(prevUnresolved, saved.InstanceID)
			timedOut := saved.Created.IsZero() || now.Sub(saved.Created) > e.cfg.Timeout
			unadoptable := saved.APIKey == "" || saved.CertPEM == ""
			if timedOut || wasUnresolved || unadoptable {
				if err := e.prov.EndExaminer(ctx, saved.InstanceID); err != nil {
					failedEnds = append(failedEnds, savedOrphanInstance(saved, prevUnresolved))
					e.say("could not end missing examiner GPU (machine %s): %v", saved.InstanceID, err)
					if strayEndErr == nil {
						strayEndErr = fmt.Errorf("ending missing examiner GPU %s: %w", saved.InstanceID, err)
					}
				} else {
					why := "examiner confirmed ended after omission from provider listing"
					if timedOut {
						e.say("ended missing timed-out examiner GPU (machine %s): exceeded --examiner-timeout (%s)", saved.InstanceID, e.cfg.Timeout)
						why = "timed-out examiner confirmed ended after omission from provider listing"
					} else {
						e.say("ended missing examiner GPU (machine %s): confirmed terminated", saved.InstanceID)
					}
					d, rerr := e.recordMissingSavedDollars(saved, why)
					endedToday += d
					if rerr != nil && spendErr == nil {
						spendErr = rerr
					}
					e.clearState()
					saved, wasReady, wasUpstream = examinerState{}, false, ""
					endedSaved = true
				}
			} else {
				missingSavedHeld = true
				e.say("examiner GPU %s omitted from provider listing; keeping reservation until it reappears or terminates", saved.InstanceID)
			}
		}
		e.setUnresolved(failedEnds)
	} else if dryRunTimedOut || (saved.InstanceID != "" && !listedIDs[saved.InstanceID]) {
		return nil
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
		if strayEndErr != nil {
			return strayEndErr
		}
		if spendErr != nil {
			return spendErr
		}
		return nil
	}

	// Do NOT rent a replacement while an owned orphan could not be confirmed destroyed.
	if strayEndErr != nil {
		e.proxyOffline()
		return strayEndErr
	}

	// Keep an acknowledged contract omitted from a stale listing intact (credentials, slot,
	// hourly charge) without renting a replacement or recording terminal spend.
	if missingSavedHeld {
		if spendErr != nil {
			return spendErr
		}
		return nil
	}

	// Preserve an unresolved pre-rental reservation across stale provider listings until the
	// provider confirms its outcome (the delayed instance appears and is adopted or ended), so a
	// lost create confirmation can never trigger a second paid rental exceeding --max/--max-hourly.
	if saved.InstanceID == "" && saved.PendingRent && !endedSaved {
		elapsed := time.Duration(0)
		if !saved.Created.IsZero() && now.After(saved.Created) {
			elapsed = now.Sub(saved.Created).Round(time.Second)
		}
		if e.cfg.Timeout > 0 && elapsed > e.cfg.Timeout {
			e.say("examiner GPU reservation %s remains unresolved after %s (past --examiner-timeout %s); holding slot and budget until provider confirms outcome (run `stet-cloud examiner resolve` after verifying on Vast.ai)", saved.ReservedLabel, elapsed, e.cfg.Timeout)
		} else {
			e.say("waiting for pending examiner GPU reservation %s to appear (%s elapsed)", saved.ReservedLabel, elapsed)
		}
		if spendErr != nil {
			return spendErr
		}
		return nil
	} else if saved.InstanceID != "" || wasReady {
		e.clearState()
	}

	if !e.dryRun && owner == "" {
		return errors.New("examiner owner id unavailable")
	}

	if spendErr != nil {
		e.say("not renting examiner GPU for %s: spend log unavailable (%v)", needRound, spendErr)
		return spendErr
	}

	if budget.AccountingErr != nil {
		e.say("not renting examiner GPU for %s: fleet accounting unavailable (%v)", needRound, budget.AccountingErr)
		return budget.AccountingErr
	}
	if budget.MaxWorkers <= 0 {
		e.say("not renting examiner GPU for %s: %d machines already live and --max is %d; waiting for a slot", needRound, budget.LiveWorkers, budget.MaxWorkers)
		return nil
	}
	currentSpentToday := func(at time.Time) (float64, error) {
		if budget.CurrentSpentToday != nil {
			freshSpent, err := budget.CurrentSpentToday()
			if err != nil {
				e.say("not renting examiner GPU for %s: fleet accounting unavailable (%v)", needRound, err)
				return 0, err
			}
			return freshSpent, nil
		}
		e.mu.RLock()
		curSt := e.cloneStateLocked()
		e.mu.RUnlock()
		_, _, postAccrued := snapshotFromState(curSt, e.home, at, e.cfg.Timeout, e.cfg.MaxPrice)
		return max(0, budget.SpentToday-preAccrued) + endedToday + postAccrued, nil
	}
	spentToday, err := currentSpentToday(now)
	if err != nil {
		return err
	}
	if budget.DailyBudget > 0 && spentToday >= budget.DailyBudget {
		e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, spentToday, budget.DailyBudget)
		return nil
	}
	transferCapFor := func(spent float64) float64 {
		if budget.DailyBudget <= 0 {
			return -1
		}
		return max(0, budget.DailyBudget-spent)
	}
	ts, canTierSearch := e.prov.(examinerTieredOfferSearcher)
	_, canCapTransfer := e.prov.(examinerOfferSearcher)
	searchOffers := func(maxPrice, maxTransfer float64) ([]ExaminerOffer, error) {
		if canTierSearch {
			offs, _, err := ts.ExaminerOffersByTier(ctx, maxPrice, maxTransfer, 0, nil)
			return offs, err
		}
		if s, ok := e.prov.(examinerOfferSearcher); ok {
			return s.ExaminerOffersWithTransferCap(ctx, maxPrice, maxTransfer)
		}
		return e.prov.ExaminerOffers(ctx, maxPrice)
	}
	minFeasibleHourly := func(maxPrice, maxTransfer float64, excludeIDs map[string]bool) float64 {
		if canTierSearch {
			best := 0.0
			start := 0
			for {
				offs, tier, perr := ts.ExaminerOffersByTier(ctx, maxPrice, maxTransfer, start, excludeIDs)
				if perr != nil {
					return 0
				}
				if len(offs) == 0 {
					break
				}
				if cand := cheapestValidExaminerOffer(offs, maxPrice, maxTransfer); cand > 0 && (best == 0 || cand < best) {
					best = cand
				}
				if tier < start {
					break
				}
				start = tier + 1
			}
			return best
		}
		probe, perr := searchOffers(maxPrice, maxTransfer)
		if perr != nil {
			return 0
		}
		if len(excludeIDs) > 0 {
			filtered := make([]ExaminerOffer, 0, len(probe))
			for _, off := range probe {
				if !excludeIDs[off.ID] {
					filtered = append(filtered, off)
				}
			}
			probe = filtered
		}
		return cheapestValidExaminerOffer(probe, maxPrice, maxTransfer)
	}
	maxTransfer := transferCapFor(spentToday)
	maxBudget := budget.MaxHourly
	if e.cfg.MaxPrice > 0 && e.cfg.MaxPrice < maxBudget {
		maxBudget = e.cfg.MaxPrice
	}
	if budget.LiveWorkers >= budget.MaxWorkers {
		e.say("not renting examiner GPU for %s: %d machines already live and --max is %d; waiting for a slot", needRound, budget.LiveWorkers, budget.MaxWorkers)
		if budget.MaxHourly > 0 && maxBudget > 0 {
			e.setBlockedHourly(minFeasibleHourly(maxBudget, maxTransfer, nil))
		}
		return nil
	}
	headroom := budget.MaxHourly - budget.HourlyNow
	if budget.MaxHourly <= 0 || headroom <= 1e-9 {
		e.say("not renting examiner GPU for %s: live hourly spend $%.2f/h leaves no room under --max-hourly $%.2f/h", needRound, budget.HourlyNow, budget.MaxHourly)
		if budget.MaxHourly > 0 && maxBudget > 0 {
			e.setBlockedHourly(minFeasibleHourly(maxBudget, maxTransfer, nil))
		}
		return nil
	}
	effMaxPrice := e.cfg.MaxPrice
	if effMaxPrice <= 0 || headroom < effMaxPrice {
		effMaxPrice = headroom
	}

	filterFitting := func(raw []ExaminerOffer) []ExaminerOffer {
		var out []ExaminerOffer
		for _, off := range raw {
			if !validRate(off.Price) {
				continue
			}
			eff := off.effectiveHourly()
			if budget.HourlyNow+eff <= budget.MaxHourly+1e-9 && (e.cfg.MaxPrice <= 0 || eff <= e.cfg.MaxPrice+1e-9) {
				out = append(out, off)
			}
		}
		sort.SliceStable(out, func(i, j int) bool {
			ei, ej := out[i].effectiveHourly(), out[j].effectiveHourly()
			if ei != ej {
				return ei < ej
			}
			return out[i].Price < out[j].Price
		})
		return out
	}

	var offers []ExaminerOffer
	curTier := 0
	if canTierSearch {
		offers, curTier, err = ts.ExaminerOffersByTier(ctx, effMaxPrice, maxTransfer, 0, nil)
	} else {
		offers, err = searchOffers(effMaxPrice, maxTransfer)
	}
	if err != nil {
		return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", err)
	}
	if len(offers) == 0 {
		if effMaxPrice < maxBudget-1e-9 {
			if need := minFeasibleHourly(maxBudget, maxTransfer, nil); need > 0 {
				e.setBlockedHourly(need)
				e.say("not renting examiner GPU for %s: cheapest 1x RTX 5090 ($%.2f/h) plus $%.2f/h live exceeds --max-hourly $%.2f/h",
					needRound, need, budget.HourlyNow, budget.MaxHourly)
				return nil
			}
		}
		e.say("Vast.ai has no 1x RTX 5090 free for %s examiner under $%.2f/h (headroom $%.2f/h); retrying on next tick", needRound, effMaxPrice, headroom)
		return nil
	}
	fitting := filterFitting(offers)
	allUnfit := offers
	if len(fitting) == 0 && canTierSearch {
		for {
			more, nextTier, merr := ts.ExaminerOffersByTier(ctx, effMaxPrice, maxTransfer, curTier+1, nil)
			if merr != nil {
				return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", merr)
			}
			if len(more) == 0 || nextTier <= curTier {
				break
			}
			curTier = nextTier
			allUnfit = append(allUnfit, more...)
			if nextFit := filterFitting(more); len(nextFit) > 0 {
				fitting = nextFit
				break
			}
		}
	}
	if len(fitting) == 0 {
		need := cheapestValidExaminerOffer(allUnfit, maxBudget, maxTransfer)
		if effMaxPrice < maxBudget-1e-9 {
			if cand := minFeasibleHourly(maxBudget, maxTransfer, nil); cand > 0 && (need == 0 || cand < need) {
				need = cand
			}
		}
		e.setBlockedHourly(need)
		if need > 0 {
			e.say("not renting examiner GPU for %s: cheapest 1x RTX 5090 ($%.2f/h) plus $%.2f/h live exceeds --max-hourly $%.2f/h",
				needRound, need, budget.HourlyNow, budget.MaxHourly)
		} else if validRate(allUnfit[0].Price) {
			e.say("not renting examiner GPU for %s: cheapest 1x RTX 5090 ($%.2f/h) plus $%.2f/h live exceeds --max-hourly $%.2f/h",
				needRound, allUnfit[0].effectiveHourly(), budget.HourlyNow, budget.MaxHourly)
		} else {
			e.say("not renting examiner GPU for %s: no valid priced 1x RTX 5090 offers fit under --max-hourly $%.2f/h",
				needRound, budget.MaxHourly)
		}
		return nil
	}
	if e.dryRun {
		spentNow, serr := currentSpentToday(e.now())
		if serr != nil {
			return serr
		}
		if budget.DailyBudget > 0 && spentNow >= budget.DailyBudget {
			e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, spentNow, budget.DailyBudget)
			return nil
		}
		findDryRun := func(list []ExaminerOffer) (ExaminerOffer, bool) {
			for _, off := range list {
				transfer := nonNegMax(off.TransferDollars, 0)
				if budget.DailyBudget > 0 && transfer > 0 && spentNow+transfer > budget.DailyBudget+1e-9 {
					continue
				}
				return off, true
			}
			return ExaminerOffer{}, false
		}
		chosen, ok := findDryRun(fitting)
		if !ok && canTierSearch && budget.DailyBudget > 0 {
			for !ok {
				tighter := transferCapFor(spentNow)
				more, nextTier, merr := ts.ExaminerOffersByTier(ctx, effMaxPrice, tighter, curTier+1, nil)
				if merr != nil {
					return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", merr)
				}
				if len(more) == 0 || nextTier <= curTier {
					break
				}
				curTier = nextTier
				spentNow, serr = currentSpentToday(e.now())
				if serr != nil {
					return serr
				}
				if spentNow >= budget.DailyBudget {
					e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, spentNow, budget.DailyBudget)
					return nil
				}
				chosen, ok = findDryRun(filterFitting(more))
			}
		} else if !ok && canCapTransfer && budget.DailyBudget > 0 {
			if tighter := transferCapFor(spentNow); tighter < maxTransfer-1e-9 {
				more, merr := searchOffers(effMaxPrice, tighter)
				if merr != nil {
					return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", merr)
				}
				spentNow, serr = currentSpentToday(e.now())
				if serr != nil {
					return serr
				}
				if spentNow >= budget.DailyBudget {
					e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, spentNow, budget.DailyBudget)
					return nil
				}
				chosen, ok = findDryRun(filterFitting(more))
			}
		}
		if ok {
			transfer := nonNegMax(chosen.TransferDollars, 0)
			if transfer > 0 {
				e.say("would rent examiner GPU on Vast.ai for %s: %s at $%.2f/h + $%.2f transfer (offer %s)", needRound, chosen.Card, chosen.Price, transfer, chosen.ID)
			} else {
				e.say("would rent examiner GPU on Vast.ai for %s: %s at $%.2f/h (offer %s)", needRound, chosen.Card, chosen.Price, chosen.ID)
			}
			return nil
		}
		e.say("not renting examiner GPU for %s: cold-start download charge would pass --daily-budget $%.2f (about $%.2f spent today)", needRound, budget.DailyBudget, spentNow)
		return nil
	}
	apiKey, err := randomAPIKey()
	if err != nil {
		return err
	}
	label := examinerLabelPrefix + "-" + owner + "-" + needRound + "-" + apiKey[:8]
	ttlSeconds := max(60, int(e.cfg.Timeout.Seconds()))
	tried := 0
	skippedTransferDaily := 0
	triedIDs := map[string]bool{}
	transferRequeried := false
	for {
		noRoomThisPass := 0
		skippedThisPass := 0
		for _, off := range fitting {
			if triedIDs[off.ID] {
				continue
			}
			attemptNow := e.now()
			spentNow, serr := currentSpentToday(attemptNow)
			if serr != nil {
				return serr
			}
			if budget.DailyBudget > 0 && spentNow >= budget.DailyBudget {
				e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, spentNow, budget.DailyBudget)
				return nil
			}
			transfer := nonNegMax(off.TransferDollars, 0)
			if budget.DailyBudget > 0 && transfer > 0 && spentNow+transfer > budget.DailyBudget+1e-9 {
				skippedTransferDaily++
				skippedThisPass++
				continue
			}
			certPEM, keyPEM, err := generateExaminerCert(attemptNow, e.cfg.Timeout)
			if err != nil {
				return fmt.Errorf("generating examiner TLS certificate: %w", err)
			}
			expireAt := attemptNow.Add(time.Duration(ttlSeconds) * time.Second).Unix()
			triedIDs[off.ID] = true
			tried++
			e.mu.RLock()
			pendingCopy := append([]podRecord(nil), e.state.PendingSpends...)
			e.mu.RUnlock()
			reserved := examinerState{
				PendingRent:     true,
				ReservedLabel:   label,
				RoundID:         needRound,
				APIKey:          apiKey,
				CertPEM:         string(certPEM),
				Price:           off.Price,
				DiskCost:        off.DiskCost,
				TransferDollars: transfer,
				Created:         attemptNow,
				PendingSpends:   pendingCopy,
			}
			if err := e.writeStateFile(&reserved); err != nil {
				return fmt.Errorf("saving pre-rental examiner.json: %w", err)
			}
			e.mu.Lock()
			oldTr := e.proxyTr
			reserved.PendingSpends = append([]podRecord(nil), e.state.PendingSpends...)
			e.state = reserved
			e.ready = false
			e.proxy = nil
			e.proxyTr = nil
			e.upstream = ""
			e.mu.Unlock()
			if oldTr != nil {
				oldTr.CloseIdleConnections()
			}
			id, rerr := e.prov.RentExaminer(ctx, off.ID, ExaminerLaunch{
				Label:      label,
				Image:      e.cfg.Image,
				Model:      e.cfg.Model,
				Revision:   e.cfg.Revision,
				APIKey:     apiKey,
				CertPEM:    certPEM,
				KeyPEM:     keyPEM,
				TTLSeconds: ttlSeconds,
				ExpireAt:   expireAt,
				HFToken:    e.hfToken,
				DiskGB:     examinerDiskGB,
			})
			if errors.Is(rerr, errNoRoom) {
				// Offer was definitively unavailable; clear the failed reservation immediately so
				// subsequent offer attempts do not count a phantom PendingRent against daily spend.
				e.clearState()
				noRoomThisPass++
				continue
			}
			if errors.Is(rerr, errRentRejected) || errors.Is(rerr, errRentNotSent) {
				// Provider explicitly refused to create an instance (4xx / explicit success=false) OR
				// transport proved the create request never reached the provider (e.g. dial
				// ECONNREFUSED); clear the reservation immediately so future examiners are not blocked.
				e.clearState()
				return fmt.Errorf("renting examiner GPU on Vast.ai: %w", rerr)
			}
			if rerr != nil {
				// Keep PendingRent intact on ambiguous/transient errors (e.g. network timeout or 5xx
				// where Vast may still have created the VM): separate CLI processes continue seeing
				// the reserved slot, and subsequent Reconcile polls hold the reservation across stale
				// listings until the provider confirms its outcome (the VM appears and is adopted/ended).
				return fmt.Errorf("renting examiner GPU on Vast.ai: %w", rerr)
			}
			st := examinerState{
				InstanceID:      id,
				RoundID:         needRound,
				APIKey:          apiKey,
				CertPEM:         string(certPEM),
				Price:           off.Price,
				DiskCost:        off.DiskCost,
				TransferDollars: transfer,
				Created:         attemptNow,
			}
			e.mu.Lock()
			st.PendingSpends = append([]podRecord(nil), e.state.PendingSpends...)
			st.Unresolved = append([]ExaminerInstance(nil), e.state.Unresolved...)
			e.state = st
			e.mu.Unlock()
			if err := e.persistState(&st); err != nil {
				e.say("could not save examiner.json after renting %s (%v); unwinding rental", id, err)
				if endErr := e.prov.EndExaminer(ctx, id); endErr == nil {
					_ = e.recordEnded(ExaminerInstance{ID: id, Label: label, Price: off.Price, TransferDollars: transfer, Created: attemptNow}, attemptNow, off.Price, needRound, "unwound after examiner.json save failure")
					e.clearState()
					return fmt.Errorf("saving examiner.json after renting %s (unwound rental): %w", id, err)
				} else {
					e.setUnresolved([]ExaminerInstance{{ID: id, Label: label, State: "starting", Price: off.Price, DiskCost: off.DiskCost, TransferDollars: transfer, Created: attemptNow}})
					return fmt.Errorf("saving examiner.json after renting %s (unwind failed: %v): %w", id, endErr, err)
				}
			}
			if transfer > 0 {
				e.say("rented examiner GPU on Vast.ai for %s: %s, $%.2f/h + $%.2f transfer (machine %s); warming %s (TLS pinned, TTL %s)", needRound, off.Card, off.Price, transfer, id, e.cfg.Model, e.cfg.Timeout)
			} else {
				e.say("rented examiner GPU on Vast.ai for %s: %s, $%.2f/h (machine %s); warming %s (TLS pinned, TTL %s)", needRound, off.Card, off.Price, id, e.cfg.Model, e.cfg.Timeout)
			}
			return nil
		}
		if canTierSearch && (noRoomThisPass > 0 || skippedThisPass > 0) {
			// Every offer in the current bandwidth tier was definitively unavailable (errNoRoom)
			// or exceeded remaining daily transfer budget: advance to slower bandwidth tiers
			// (500 -> 200 Mbps) while skipping offer IDs already tried in this pass.
			freshSpent, serr := currentSpentToday(e.now())
			if serr != nil {
				return serr
			}
			if budget.DailyBudget > 0 && freshSpent >= budget.DailyBudget {
				e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, freshSpent, budget.DailyBudget)
				return nil
			}
			maxTransfer = transferCapFor(freshSpent)
			advanced := false
			for {
				more, nextTier, merr := ts.ExaminerOffersByTier(ctx, effMaxPrice, maxTransfer, curTier+1, triedIDs)
				if merr != nil {
					return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", merr)
				}
				if len(more) == 0 || nextTier <= curTier {
					break
				}
				curTier = nextTier
				if nextFitting := filterFitting(more); len(nextFitting) > 0 {
					fitting = nextFitting
					advanced = true
					break
				}
			}
			if advanced {
				continue
			}
		} else if !transferRequeried && canCapTransfer && skippedThisPass > 0 && budget.DailyBudget > 0 {
			// Non-tiered provider fallback: if spend accrued during the search/retries shrank the
			// remaining daily budget and eliminated offers on transfer cost, re-query once tighter.
			transferRequeried = true
			freshSpent, serr := currentSpentToday(e.now())
			if serr != nil {
				return serr
			}
			if freshSpent < budget.DailyBudget {
				if tighter := transferCapFor(freshSpent); tighter < maxTransfer-1e-9 {
					maxTransfer = tighter
					more, merr := searchOffers(effMaxPrice, maxTransfer)
					if merr != nil {
						return fmt.Errorf("searching Vast.ai RTX 5090 offers for examiner: %w", merr)
					}
					if nextFitting := filterFitting(more); len(nextFitting) > 0 {
						fitting = nextFitting
						continue
					}
				}
			}
		}
		break
	}
	e.clearState()
	if tried == 0 && skippedTransferDaily > 0 {
		e.say("not renting examiner GPU for %s: %d available RTX 5090 offer(s) had cold-start download charges exceeding remaining --daily-budget $%.2f", needRound, skippedTransferDaily, budget.DailyBudget)
		return nil
	}
	if tried > 0 && effMaxPrice < maxBudget-1e-9 && maxBudget > 0 {
		// Stale cheap offer(s) under current headroom stocked out (errNoRoom), and no other offer
		// fit under effMaxPrice. Probe maxBudget excluding triedIDs so tick() can reclaim idle
		// workers when a live offer exists under --max-hourly.
		freshSpent, serr := currentSpentToday(e.now())
		if serr != nil {
			return serr
		}
		if budget.DailyBudget > 0 && freshSpent >= budget.DailyBudget {
			e.say("not renting examiner GPU for %s: about $%.2f spent today, and the daily budget is $%.2f", needRound, freshSpent, budget.DailyBudget)
			return nil
		}
		probeTransfer := transferCapFor(freshSpent)
		if need := minFeasibleHourly(maxBudget, probeTransfer, triedIDs); need > 0 {
			e.setBlockedHourly(need)
			e.say("not renting examiner GPU for %s: %d offer(s) under headroom were unavailable and cheapest remaining 1x RTX 5090 ($%.2f/h) plus $%.2f/h live exceeds --max-hourly $%.2f/h",
				needRound, tried, need, budget.HourlyNow, budget.MaxHourly)
			return nil
		}
	}
	e.say("none of the %d RTX 5090 offers tried for %s examiner were available; retrying on next tick", tried, needRound)
	return nil
}

func (e *Examiner) diskMatchesSaved(saved examinerState) (ok bool, changed bool) {
	if e.home == "" || e.dryRun {
		return true, false
	}
	e.fileMu.Lock()
	defer e.fileMu.Unlock()
	e.mu.RLock()
	dirty := e.dirty
	e.mu.RUnlock()
	if dirty {
		return true, false
	}
	diskSt, err := readExaminerStateErr(e.home)
	if err != nil {
		return false, false
	}
	if diskSt == nil || diskSt.InstanceID != saved.InstanceID || diskSt.APIKey != saved.APIKey || diskSt.CertPEM != saved.CertPEM {
		return false, true
	}
	return true, false
}

// ProbeIfBooting polls Vast + https://<ip>:<port>/v1/models (over pinned TLS) once when an
// examiner GPU is currently booting so the local proxy turns ready seconds after vLLM loads.
// Strictly read-only on examiner.json so a stale background probe can never clobber another
// controller's durable state.
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
	if ok, changed := e.diskMatchesSaved(saved); !ok {
		if changed {
			e.proxyOffline()
		}
		return
	}
	rawInsts, err := e.prov.ListExaminers(ctx)
	if err != nil {
		return
	}
	for _, inst := range filterOwnedExaminers(rawInsts, owner, saved.InstanceID, saved.ReservedLabel) {
		if inst.ID == saved.InstanceID && inst.State != "stopped" {
			e.noteSavedPrice(&saved, inst)
		}
		if inst.ID == saved.InstanceID && inst.State != "stopped" && inst.PublicIP != "" && inst.HostPort != "" {
			upstream := "https://" + net.JoinHostPort(inst.PublicIP, inst.HostPort)
			if e.probeModels(ctx, upstream, saved.APIKey, saved.CertPEM) {
				if ok, changed := e.diskMatchesSaved(saved); !ok {
					if changed {
						e.proxyOffline()
					}
					return
				}
				if _, ok, err := e.setUpstream(saved.InstanceID, upstream, saved.APIKey, saved.CertPEM); err == nil && ok {
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
	_, err := appendSpendRecord(path, rec, false)
	return err
}

// resolvePendingExaminer verifies an unresolved PendingRent reservation in examiner.json
// against the provider: if a matching instance is listed it is adopted into InstanceID (or
// terminated and billed when end is true). If the provider lists zero matching instances, the
// reservation is kept intact unless the operator explicitly verified absence and passed
// clearLabel matching st.ReservedLabel (preserving PendingSpends and Unresolved).
func resolvePendingExaminer(
	ctx context.Context,
	home string,
	prov ExaminerProvider,
	nowFn func() time.Time,
	say func(string, ...any),
	dryRun bool,
	end bool,
	clearLabel string,
	onSpent func(podRecord) error,
) error {
	if nowFn == nil {
		nowFn = time.Now
	}
	if say == nil {
		say = func(string, ...any) {}
	}
	clearLabel = strings.TrimSpace(clearLabel)
	st, err := readExaminerStateErr(home)
	if err != nil {
		return err
	}
	hasPending := st != nil && st.PendingRent && st.InstanceID == ""
	hasUnresolved := st != nil && len(st.Unresolved) > 0
	if !hasPending && !hasUnresolved {
		if clearLabel != "" {
			return fmt.Errorf("no pending examiner reservation matching --clear %q", clearLabel)
		}
		say("no pending examiner rental reservation to resolve")
		return nil
	}
	now := nowFn()
	clearActiveInstance := func(target *examinerState) {
		target.InstanceID = ""
		target.PendingRent = false
		target.ReservedLabel = ""
		target.RoundID = ""
		target.APIKey = ""
		target.CertPEM = ""
		target.Upstream = ""
		target.Price = 0
		target.DiskCost = 0
		target.TransferDollars = 0
		target.Created = time.Time{}
		target.ObservedLiveAt = time.Time{}
	}
	endUnresolved := func(u *ExaminerInstance, roundID, why string, next *examinerState) (ended bool, endErr, spendErr error) {
		if err := prov.EndExaminer(ctx, u.ID); err != nil {
			u.ObservedLiveAt = now
			return false, fmt.Errorf("ending unresolved examiner GPU %s: %w", u.ID, err), nil
		}
		price := u.Price
		transfer := nonNegMax(u.TransferDollars, 0)
		if u.ID == st.InstanceID || st.PendingRent || !validRate(price) {
			price, _ = examinerObservedRate(st.Price, st.DiskCost, *u)
		}
		if u.ID == st.InstanceID || st.PendingRent {
			transfer = nonNegMax(transfer, st.TransferDollars)
		}
		if !validRate(price) {
			price = 1.50
		}
		start := u.Created
		if start.IsZero() {
			start = st.Created
		}
		if start.IsZero() || start.After(now) {
			start = now
		}
		rec := podRecord{
			ID: u.ID, Market: "vast", Handle: "examiner", GPU: "RTX 5090",
			Cost: price, FixedCost: transfer, Start: start, Alive: now, End: now,
			For: "examiner", Round: roundID, Why: why,
		}
		if onSpent != nil && !spentHasExaminer(home, u.ID) {
			if serr := onSpent(rec); serr != nil {
				next.PendingSpends = mergePendingSpends(next.PendingSpends, []podRecord{rec})
				spendErr = fmt.Errorf("recording ended examiner spend: %w", serr)
			}
		}
		say("ended unresolved examiner GPU (machine %s)", u.ID)
		return true, nil, spendErr
	}
	if !hasPending {
		if clearLabel != "" {
			return fmt.Errorf("no pending examiner reservation matching --clear %q", clearLabel)
		}
		if !end {
			e := fmt.Errorf("examiner.json has %d unresolved examiner instance(s) awaiting termination; run `stet-cloud examiner resolve --end` to retry", len(st.Unresolved))
			say("%v", e)
			return e
		}
		if dryRun {
			for _, u := range st.Unresolved {
				say("would retry ending unresolved examiner GPU (machine %s)", u.ID)
			}
			return nil
		}
		next := *st
		next.Unresolved = nil
		var firstEndErr, spendErr error
		for i := range st.Unresolved {
			u := st.Unresolved[i]
			if u.ID == "" {
				continue
			}
			ok, eErr, sErr := endUnresolved(&u, st.RoundID, "resolved unresolved examiner teardown (--end)", &next)
			if !ok {
				next.Unresolved = append(next.Unresolved, u)
				if firstEndErr == nil {
					firstEndErr = eErr
				}
			} else if next.InstanceID != "" && u.ID == next.InstanceID {
				clearActiveInstance(&next)
			}
			if sErr != nil && spendErr == nil {
				spendErr = sErr
			}
		}
		if werr := writeExaminerState(home, &next); werr != nil {
			return fmt.Errorf("updating examiner.json after resolving unresolved instances: %w", werr)
		}
		if firstEndErr != nil {
			return firstEndErr
		}
		return spendErr
	}
	rawInsts, err := prov.ListExaminers(ctx)
	if err != nil {
		return fmt.Errorf("listing examiner instances on provider: %w", err)
	}
	var matches []ExaminerInstance
	for i := range rawInsts {
		if st.ReservedLabel != "" && rawInsts[i].Label == st.ReservedLabel {
			matches = append(matches, rawInsts[i])
		}
	}
	if len(matches) > 0 {
		if clearLabel != "" {
			return fmt.Errorf("refusing --clear %q: Vast.ai lists %d matching examiner instance(s) (first: %s); run `stet-cloud examiner resolve` to adopt or `--end` to terminate", clearLabel, len(matches), matches[0].ID)
		}
		if end {
			if dryRun {
				for _, m := range matches {
					say("would end examiner GPU (machine %s) for reservation %s", m.ID, st.ReservedLabel)
				}
				for _, u := range st.Unresolved {
					say("would retry ending unresolved examiner GPU (machine %s)", u.ID)
				}
				return nil
			}
			next := examinerState{
				PendingSpends: append([]podRecord(nil), st.PendingSpends...),
			}
			var firstEndErr, spendErr error
			var failedEnds []ExaminerInstance
			handledIDs := map[string]bool{}
			for _, match := range matches {
				handledIDs[match.ID] = true
				if err := prov.EndExaminer(ctx, match.ID); err != nil {
					u := match
					if u.Created.IsZero() {
						u.Created = st.Created
					}
					u.Price, u.DiskCost = examinerObservedRate(st.Price, st.DiskCost, u)
					u.TransferDollars = nonNegMax(u.TransferDollars, st.TransferDollars)
					u.ObservedLiveAt = now
					failedEnds = append(failedEnds, u)
					if firstEndErr == nil {
						firstEndErr = fmt.Errorf("ending examiner GPU %s for reservation: %w", match.ID, err)
					}
					continue
				}
				price, _ := examinerObservedRate(st.Price, st.DiskCost, match)
				transfer := nonNegMax(match.TransferDollars, st.TransferDollars)
				start := match.Created
				if start.IsZero() {
					start = st.Created
				}
				if start.IsZero() || start.After(now) {
					start = now
				}
				rec := podRecord{
					ID: match.ID, Market: "vast", Handle: "examiner", GPU: "RTX 5090",
					Cost: price, FixedCost: transfer, Start: start, Alive: now, End: now,
					For: "examiner", Round: st.RoundID, Why: "resolved pending examiner reservation (--end)",
				}
				if onSpent != nil && !spentHasExaminer(home, match.ID) {
					if serr := onSpent(rec); serr != nil {
						next.PendingSpends = mergePendingSpends(next.PendingSpends, []podRecord{rec})
						if spendErr == nil {
							spendErr = fmt.Errorf("recording ended examiner spend: %w", serr)
						}
					}
				}
				say("ended examiner GPU (machine %s) for reservation %s", match.ID, st.ReservedLabel)
			}
			for i := range st.Unresolved {
				u := st.Unresolved[i]
				if u.ID == "" || handledIDs[u.ID] {
					continue
				}
				ok, eErr, sErr := endUnresolved(&u, st.RoundID, "resolved unresolved examiner teardown (--end)", &next)
				if !ok {
					failedEnds = append(failedEnds, u)
					if firstEndErr == nil {
						firstEndErr = eErr
					}
				}
				if sErr != nil && spendErr == nil {
					spendErr = sErr
				}
			}
			if len(failedEnds) > 0 {
				// Persist failed teardowns strictly in Unresolved (with InstanceID/APIKey/CertPEM
				// cleared) so snapshotFromState continues accounting for slot/rate/spend AND both
				// `examiner resolve --end` retries and Reconcile unconditionally retry EndExaminer.
				next.Unresolved = mergeUnresolvedLists(next.Unresolved, failedEnds)
			}
			if werr := writeExaminerState(home, &next); werr != nil {
				return fmt.Errorf("updating examiner.json after resolving %s: %w", st.ReservedLabel, werr)
			}
			if firstEndErr != nil {
				return firstEndErr
			}
			return spendErr
		}
		// Adopt mode: prefer a live (non-stopped) match over any stopped duplicate; any remaining
		// duplicates will be terminated as strays on the next Reconcile tick.
		match := matches[0]
		for _, cand := range matches {
			if cand.State != "stopped" {
				match = cand
				break
			}
		}
		if dryRun {
			say("would adopt examiner GPU (machine %s, state %s) for reservation %s", match.ID, match.State, st.ReservedLabel)
			return nil
		}
		label := st.ReservedLabel
		st.InstanceID = match.ID
		st.PendingRent = false
		st.ReservedLabel = ""
		st.Price, st.DiskCost = examinerObservedRate(st.Price, st.DiskCost, match)
		st.TransferDollars = nonNegMax(st.TransferDollars, match.TransferDollars)
		st.ObservedLiveAt = now
		if err := writeExaminerState(home, st); err != nil {
			return fmt.Errorf("saving adopted examiner.json: %w", err)
		}
		say("confirmed examiner GPU (machine %s, state %s) for reservation %s: adopted into examiner.json", match.ID, match.State, label)
		return nil
	}
	if clearLabel == "" {
		next := *st
		var firstEndErr, spendErr error
		if end && len(st.Unresolved) > 0 {
			if dryRun {
				for _, u := range st.Unresolved {
					say("would retry ending unresolved examiner GPU (machine %s)", u.ID)
				}
			} else {
				next.Unresolved = nil
				for i := range st.Unresolved {
					u := st.Unresolved[i]
					if u.ID == "" {
						continue
					}
					ok, eErr, sErr := endUnresolved(&u, st.RoundID, "resolved unresolved examiner teardown (--end)", &next)
					if !ok {
						next.Unresolved = append(next.Unresolved, u)
						if firstEndErr == nil {
							firstEndErr = eErr
						}
					}
					if sErr != nil && spendErr == nil {
						spendErr = sErr
					}
				}
				if werr := writeExaminerState(home, &next); werr != nil {
					return fmt.Errorf("updating examiner.json after retrying unresolved teardowns: %w", werr)
				}
			}
		}
		e := fmt.Errorf("pending examiner reservation %s is not listed on Vast.ai yet; keeping reservation (verify on Vast.ai and re-run `stet-cloud examiner resolve --clear %s` only if confirmed absent)", st.ReservedLabel, st.ReservedLabel)
		say("%v", e)
		if firstEndErr != nil {
			return firstEndErr
		}
		if spendErr != nil {
			return spendErr
		}
		return e
	}
	if clearLabel != st.ReservedLabel {
		return fmt.Errorf("--clear %q does not match pending examiner reservation %q; keeping reservation", clearLabel, st.ReservedLabel)
	}
	if dryRun {
		say("would clear verified pending examiner reservation %s (not listed on Vast.ai)", st.ReservedLabel)
		if end {
			for _, u := range st.Unresolved {
				say("would retry ending unresolved examiner GPU (machine %s)", u.ID)
			}
		}
		return nil
	}
	label := st.ReservedLabel
	next := examinerState{
		PendingSpends: append([]podRecord(nil), st.PendingSpends...),
	}
	var firstEndErr, spendErr error
	if end {
		for i := range st.Unresolved {
			u := st.Unresolved[i]
			if u.ID == "" {
				continue
			}
			ok, eErr, sErr := endUnresolved(&u, st.RoundID, "resolved unresolved examiner teardown (--end)", &next)
			if !ok {
				next.Unresolved = append(next.Unresolved, u)
				if firstEndErr == nil {
					firstEndErr = eErr
				}
			}
			if sErr != nil && spendErr == nil {
				spendErr = sErr
			}
		}
	} else {
		next.Unresolved = append([]ExaminerInstance(nil), st.Unresolved...)
	}
	if err := writeExaminerState(home, &next); err != nil {
		return fmt.Errorf("clearing pending examiner reservation %s: %w", label, err)
	}
	say("cleared pending examiner reservation %s via --clear (not listed on Vast.ai; no spend recorded)", label)
	if firstEndErr != nil {
		return firstEndErr
	}
	return spendErr
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
