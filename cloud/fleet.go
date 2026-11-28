package main

import (
	"bytes"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed worker.sh
var workerScript string

type offerKey struct {
	long, curators bool
}

// Fleet is the pool of keys, the machines rented with them, and what has been spent.
type Fleet struct {
	o *Options
	// The markets a key is found for, and why the others are off: read again at each look, so
	// that a key added later counts without a restart (market.go). fixed: given, as in the tests.
	markets  []Market
	off      map[string]string
	keysSeen map[string]string // each market's settings when its client was made
	fixed    bool
	arming   *Arming
	// Why each market's offers could not be had, the last time every market was asked.
	offerErrs map[string]error
	// The offers asked for in this tick, kept for it alone: a tick asks once per role.
	tickOffers map[offerKey][]Offer
	// What was last said of each market whose offers could not be had, said again only
	// once it changes.
	offerSaid map[string]string
	// What was last seen of each machine rented, to note what changes (seen.go).
	seen map[string]seenRental
	// The tail of the log of each machine near the end of its grace, to see whether it is still
	// getting there (seen.go), and those said to be.
	lastLog  map[string]string
	slowSaid map[string]bool
	logGrew  map[string]time.Duration
	// The record of the look under way, and when the markets' accounts and offers were last
	// kept, and the rounds open at the last look (history.go).
	cur        *lookRecord
	balancesAt time.Time
	offersAt   time.Time
	openRounds map[string]bool
	// The day whose files before it were last compressed.
	compressedDay string
	// probe: tools/cage_check.py, gzipped and base64'd, for a machine rented by stet-cloud
	// probe: it takes no work, and says whether the sandbox holds (probe.go).
	probe string
	st    *Stet
	op    *Key // the organisers' own key, for asking what is waiting
	out   io.Writer
	now   func() time.Time
	keys  []*Key // the pool, by handle
	// What was last said of graders waiting for their time, said again only once it changes.
	waitSaid string
	// What has been seen machines take (calibrate.go): read once, kept when it changes, and
	// the grades of settled rounds counted at most every ten minutes.
	cal      *Calibration
	calDirty bool
	calAt    time.Time
	calErr   string
	// Pods seen before they first asked for work, so that how long they took is seen too.
	starting map[string]bool
	// Whether a machine rented now trains once for its first grade of a round, no round open at
	// the last look wanting its baseline measured (baselinesWanted): it then trains only the
	// student.
	onePass bool
	// lease is the server's job lease from /admin/work (defaults to defaultLease when absent).
	lease time.Duration
	// examiner manages the optional ephemeral 1x RTX 5090 vLLM server and local auth proxy
	// when --examiner-addr is configured (examiner.go).
	examiner *Examiner
	// datedCharges caches per-market, per-contract, per-UTC-day bandwidth charges reconciled from
	// provider billing endpoints (e.g. Vast /api/v0/charges) so both active and recently retired
	// owned contracts reflect delayed bandwidth billing without rewriting spent.jsonl.
	datedCharges map[string]map[string]map[string]dayTransferCharge
	// sayMu serializes writes to f.out and concurrent appends/transitions on f.cur.Said from
	// background examiner probe/proxy goroutines.
	sayMu sync.Mutex
}

// --- the pool of keys ---------------------------------------------------------------------------

func (f *Fleet) keyDir() string { return filepath.Join(f.o.Home, "workers") }

func (f *Fleet) loadKeys() error {
	paths, err := filepath.Glob(filepath.Join(f.keyDir(), "*.json"))
	if err != nil {
		return err
	}
	var keys []*Key
	for _, p := range paths {
		k, err := loadKey(p)
		if err != nil {
			return err
		}
		if k.Handle == "" {
			k.Handle = strings.TrimSuffix(filepath.Base(p), ".json")
		}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].Handle < keys[j].Handle })
	f.keys = keys
	return nil
}

var handleRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,22}[a-z0-9]$`)

// addKeys makes n more keys, named prefix-01, prefix-02 and on from the last one there is.
func (f *Fleet) addKeys(n int, prefix string) ([]*Key, error) {
	have := map[string]bool{}
	for _, k := range f.keys {
		have[k.Handle] = true
	}
	var made []*Key
	for i := 1; len(made) < n; i++ {
		handle := fmt.Sprintf("%s-%02d", prefix, i)
		if have[handle] {
			continue
		}
		if !handleRe.MatchString(handle) || strings.Contains(handle, "--") {
			return made, fmt.Errorf("%q is not a handle the server takes: 3-24 lowercase letters, digits and single hyphens", handle)
		}
		k, err := newKey(handle)
		if err != nil {
			return made, err
		}
		if err := k.save(filepath.Join(f.keyDir(), handle+".json")); err != nil {
			return made, err
		}
		made = append(made, k)
		f.keys = append(f.keys, k)
	}
	sort.Slice(f.keys, func(i, j int) bool { return f.keys[i].Handle < f.keys[j].Handle })
	return made, nil
}

// rename gives a key a new handle, the same prefix and a random end, before it has an
// account or a pod by the old one.
func (f *Fleet) rename(k *Key) error {
	prefix := k.Handle
	if i := strings.LastIndex(prefix, "-"); i > 0 {
		prefix = prefix[:i]
	}
	have := map[string]bool{}
	for _, other := range f.keys {
		have[other.Handle] = true
	}
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	for {
		b := make([]byte, 4)
		if _, err := rand.Read(b); err != nil {
			return err
		}
		for i := range b {
			b[i] = letters[int(b[i])%len(letters)]
		}
		handle := prefix + "-" + string(b)
		if have[handle] || !handleRe.MatchString(handle) {
			continue
		}
		old := filepath.Join(f.keyDir(), k.Handle+".json")
		k.Handle = handle
		if err := k.save(filepath.Join(f.keyDir(), handle+".json")); err != nil {
			return err
		}
		return os.Remove(old)
	}
}

// lock keeps two of these from renting at once on one pool: two would each see the same
// free keys, and rent two pods for one key. It is held for one look and what follows it,
// so `down` can be run while `scale --follow` is, and waits for its turn.
func (f *Fleet) lock(wait time.Duration) (func(), error) {
	if err := os.MkdirAll(f.o.Home, 0o700); err != nil {
		return nil, err
	}
	fh, err := os.OpenFile(filepath.Join(f.o.Home, "lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	deadline := time.Now().Add(wait)
	for syscall.Flock(int(fh.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) != nil {
		if time.Now().After(deadline) {
			fh.Close()
			return nil, errors.New("another stet-cloud (or stet-runpod) is at work on this pool (" + f.o.Home + ") and has not finished; try again")
		}
		time.Sleep(250 * time.Millisecond)
	}
	return func() { fh.Close() }, nil
}

// --- what is rented, and what it costs ----------------------------------------------------------
//
// The markets forget a machine once it is ended, and say little of what it cost. So what was
// rented is written down here: each machine while it lives (fleet.json), and each one once it
// is gone (spent.jsonl), with the hours it ran and what it cost an hour. A machine that ended
// itself while nothing was watching is taken to have run until a little after it was last
// seen. These are estimates; each market's billing page is the account.

type podRecord struct {
	ID     string `json:"id"`
	Market string `json:"market,omitempty"` // "": runpod, as stet-runpod kept them
	Grades int    `json:"grades,omitempty"` // how many grades it was rented to take before an exam (0: one)
	// Tracked: whether it asked for work is kept (Asked), for how often its kind starts.
	Tracked bool `json:"tracked,omitempty"`
	Asked   bool `json:"asked,omitempty"`
	// Gone: the market stopped listing it, or stopped it, without the fleet ending it.
	Gone bool `json:"gone,omitempty"`
	// Stopped: confirmed stopped by provider by-ID lookup while omitted from listing, kept in
	// fleet.json until m.End(ctx, id) confirms destruction (e.g. to stop Vast disk billing).
	Stopped bool `json:"stopped,omitempty"`
	// Offline: confirmed offline by provider (e.g. Vast actual_status=offline), exempt from both
	// compute and storage charges while confirmed offline; unobserved gaps still accrue conservatively.
	Offline bool `json:"offline,omitempty"`
	// Terminating: operator or resolve --end requested teardown of this exact contract ID, but
	// provider deletion failed transiently; persisted until deletion or confirmed absence so
	// retrying `workers resolve --end` and `scale` ticks keep terminating it even while the
	// fleet wants workers.
	Terminating bool `json:"terminating,omitempty"`
	// Why it ended, as the fleet said it, and the cause when it failed (seen.go).
	Why   string `json:"why,omitempty"`
	Cause string `json:"cause,omitempty"`
	// The host it is on and its CPU, where its market says; whether it was rented to run
	// curators; and, if the server would not hand it any, why (curators.go).
	Host     string `json:"host,omitempty"`
	CPU      string `json:"cpu,omitempty"`
	Curators bool   `json:"curators,omitempty"`
	Refused  string `json:"refused,omitempty"`
	// What it was rented for (curators, early grades, grades, bigger students), the round, and
	// when it was expected to report its last grade (history.go).
	For                   string                       `json:"for,omitempty"`
	Round                 string                       `json:"round,omitempty"`
	Report                time.Time                    `json:"report,omitzero"`
	Handle                string                       `json:"handle"`
	GPU                   string                       `json:"gpu"`
	Cloud                 string                       `json:"cloud"`
	PendingID             string                       `json:"pending_id,omitempty"` // original pendingRent.ID when adopted/ended from l.Pending
	Cost                  float64                      `json:"cost"`
	DiskCost              float64                      `json:"disk_cost,omitempty"`
	DiskCostUnknown       bool                         `json:"disk_cost_unknown,omitempty"`
	TransferCost          float64                      `json:"transfer_cost,omitempty"`
	FixedCost             float64                      `json:"fixed_cost,omitempty"`              // exact fixed order fee (Clore) or role-based cold-start transfer estimate (Vast) incurred at Start
	FixedCostEstimated    bool                         `json:"fixed_cost_estimated,omitempty"`    // true when FixedCost is a pre-launch transfer estimate rather than an exact provider fee
	ObservedTransferCost  float64                      `json:"observed_transfer_cost,omitempty"`  // effective cumulative provider-reported observed transfer spend
	TransferObserved      bool                         `json:"transfer_observed,omitempty"`       // true when the provider reported observed transfer usage for this contract
	ObservedTransferDown  float64                      `json:"observed_transfer_down,omitempty"`  // cumulative observed download transfer spend when reported
	TransferDownObserved  bool                         `json:"transfer_down_observed,omitempty"`  // true when the provider reported download transfer usage
	TransferDownUnpriced  bool                         `json:"transfer_down_unpriced,omitempty"`  // true when the provider reported positive/invalid download usage without a usable rate
	ObservedTransferUp    float64                      `json:"observed_transfer_up,omitempty"`    // cumulative observed upload transfer spend when reported
	TransferUpObserved    bool                         `json:"transfer_up_observed,omitempty"`    // true when the provider reported upload transfer usage
	TransferUpUnpriced    bool                         `json:"transfer_up_unpriced,omitempty"`    // true when the provider reported positive/invalid upload usage without a usable rate
	ObservedTransferTotal float64                      `json:"observed_transfer_total,omitempty"` // explicit combined provider transfer_cost when reported
	TransferTotalObserved bool                         `json:"transfer_total_observed,omitempty"` // true when the provider reported explicit combined transfer_cost
	TransferTotalUnpriced bool                         `json:"transfer_total_unpriced,omitempty"` // true when the provider reported a malformed/non-finite transfer_cost or sum overflow
	TransferCharges       []transferCharge             `json:"transfer_charges,omitempty"`        // incremental observed transfer charges attributed to the UTC day they were accounted for
	DatedTransfer         map[string]dayTransferCharge `json:"dated_transfer,omitempty"`          // per-UTC-day ("2006-01-02") provider-billed transfer reconciled from billing endpoints
	PartialCost           bool                         `json:"partial_cost,omitempty"`
	Start                 time.Time                    `json:"start"`
	Alive                 time.Time                    `json:"alive"`                 // last seen running
	StoppedAt             time.Time                    `json:"stopped_at,omitzero"`   // when current stopped interval began (disk billing continues until ResumeAt/End)
	StoppedSeen           time.Time                    `json:"stopped_seen,omitzero"` // last confirmed stopped observation (bounds confirmed storage across downtime)
	ResumeAt              time.Time                    `json:"resume_at,omitzero"`    // when current running interval began after prior Segments
	End                   time.Time                    `json:"end,omitzero"`
	Segments              []billingSegment             `json:"segments,omitempty"` // completed prior billing intervals preserved across stop/resume
}

// transferCharge records one incremental increase in cumulative observed transfer spend against the
// timestamp at which it was accounted for so multi-day workers attribute new-day transfer to the
// UTC day that incurred it rather than backdating it onto Start. Dir optionally tags directional
// charges ("down" or "up") so per-day billing reconciliation never drops or double-counts across
// mixed directional/listing polls.
type transferCharge struct {
	At     time.Time `json:"at"`
	Amount float64   `json:"amount"`
	Dir    string    `json:"dir,omitempty"`
}

// transferReading is one provider snapshot of cumulative observed transfer usage (per-direction
// and/or explicit combined total).
type transferReading struct {
	Effective     float64
	Observed      bool
	Down          float64
	DownObserved  bool
	DownUnpriced  bool
	Up            float64
	UpObserved    bool
	UpUnpriced    bool
	Total         float64
	TotalObserved bool
	TotalUnpriced bool
}

// transferState is the shared cumulative + timestamped observed-transfer accounting state used by
// both worker podRecords and ephemeral Examiner instances.
type transferState struct {
	Observed      float64
	HasObserved   bool
	Down          float64
	DownObserved  bool
	DownUnpriced  bool
	Up            float64
	UpObserved    bool
	UpUnpriced    bool
	Total         float64
	TotalObserved bool
	TotalUnpriced bool
	Charges       []transferCharge
	Dated         map[string]dayTransferCharge
}

func (s transferState) incomplete() bool {
	return s.DownUnpriced || s.UpUnpriced || s.TotalUnpriced
}

func cloneTransferCharges(src []transferCharge) []transferCharge {
	if len(src) == 0 {
		return nil
	}
	dst := make([]transferCharge, len(src))
	copy(dst, src)
	return dst
}

func cloneDatedTransfer(src map[string]dayTransferCharge) map[string]dayTransferCharge {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]dayTransferCharge, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneBillingSegments(src []billingSegment) []billingSegment {
	if len(src) == 0 {
		return nil
	}
	dst := make([]billingSegment, len(src))
	copy(dst, src)
	return dst
}

func (s transferState) clone() transferState {
	s.Charges = cloneTransferCharges(s.Charges)
	s.Dated = cloneDatedTransfer(s.Dated)
	return s
}

func (s transferState) effectiveObserved() float64 {
	eff := 0.0
	if nonNegFinite(s.Observed) {
		eff = s.Observed
	}
	dir := 0.0
	if s.DownObserved && nonNegFinite(s.Down) {
		dir += s.Down
	}
	if s.UpObserved && nonNegFinite(s.Up) {
		dir += s.Up
	}
	if (s.DownObserved || s.UpObserved) && dir > eff {
		eff = dir
	}
	if s.TotalObserved && nonNegFinite(s.Total) && s.Total > eff {
		eff = s.Total
	}
	return eff
}

func (s transferState) chargesSum() float64 {
	sum := 0.0
	for _, tc := range s.Charges {
		if validRate(tc.Amount) {
			sum += tc.Amount
		}
	}
	return sum
}

func transferChargeTime(start, now time.Time) time.Time {
	if now.IsZero() {
		return start
	}
	if !start.IsZero() && start.Before(now) && now.Equal(startOfDay(now)) {
		// A reading polled at exact 00:00:00 UTC for a contract started on an earlier UTC day
		// reflects usage accumulated before midnight (0s of the new day have elapsed).
		return now.Add(-time.Nanosecond)
	}
	if !start.IsZero() && now.Before(start) {
		return start
	}
	return now
}

func (s *transferState) appendCharge(at time.Time, delta float64) {
	s.appendDirCharge(at, delta, "")
}

func (s *transferState) appendDirCharge(at time.Time, delta float64, dir string) {
	if !validRate(delta) || delta <= 1e-9 {
		return
	}
	for i := len(s.Charges) - 1; i >= 0; i-- {
		if s.Charges[i].At.Equal(at) && s.Charges[i].Dir == dir {
			s.Charges[i].Amount += delta
			return
		}
	}
	s.Charges = append(s.Charges, transferCharge{At: at, Amount: delta, Dir: dir})
}

func (s transferState) chargesOnDayByDir(day time.Time) (down, up, untyped float64) {
	for _, tc := range s.Charges {
		if !validRate(tc.Amount) || tc.At.IsZero() || !startOfDay(tc.At).Equal(day) {
			continue
		}
		switch tc.Dir {
		case "down":
			down += tc.Amount
		case "up":
			up += tc.Amount
		default:
			untyped += tc.Amount
		}
	}
	return down, up, untyped
}

// mergeDatedCharges reconciles per-UTC-day provider bandwidth charges (e.g. Vast GET /api/v0/charges)
// idempotently into transferState. Zero/empty inputs are ignored so pending actuals never erase a
// pre-launch FixedCost estimate; per-day, per-direction reconciliation against existing timestamped
// charges guarantees Day 2 / mixed-direction charges are never dropped or double-counted across
// overlapping instance-listing counters and repeated billing polls.
func (s *transferState) mergeDatedCharges(days map[string]dayTransferCharge, start, now time.Time) bool {
	if len(days) == 0 {
		return false
	}
	keys := make([]string, 0, len(days))
	for k := range days {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	changed := false
	for _, dayKey := range keys {
		d := days[dayKey]
		inDown, inUp := 0.0, 0.0
		if nonNegFinite(d.Down) {
			inDown = d.Down
		}
		if nonNegFinite(d.Up) {
			inUp = d.Up
		}
		if inDown <= 1e-9 && inUp <= 1e-9 {
			continue
		}
		day, err := time.Parse("2006-01-02", dayKey)
		if err != nil {
			continue
		}
		day = day.UTC()
		prevDay := s.Dated[dayKey]
		newDay := dayTransferCharge{
			Down: max(prevDay.Down, inDown),
			Up:   max(prevDay.Up, inUp),
		}
		if newDay.Down <= prevDay.Down+1e-9 && newDay.Up <= prevDay.Up+1e-9 {
			continue
		}
		dayDeltaDown := max(0, newDay.Down-prevDay.Down)
		dayDeltaUp := max(0, newDay.Up-prevDay.Up)
		prevEff := s.effectiveObserved()
		prevLegacy := max(0, prevEff-s.chargesSum())

		if s.Dated == nil {
			s.Dated = map[string]dayTransferCharge{}
		}
		s.Dated[dayKey] = newDay
		changed = true

		existDown, existUp, existUntyped := s.chargesOnDayByDir(day)
		if !start.IsZero() && startOfDay(start).Equal(day) {
			existUntyped += prevLegacy
		}
		needDown := max(0, newDay.Down-existDown)
		needUp := max(0, newDay.Up-existUp)
		if existUntyped > 0 {
			if sub := min(existUntyped, needDown); sub > 0 {
				needDown -= sub
				existUntyped -= sub
			}
			if sub := min(existUntyped, needUp); sub > 0 {
				needUp -= sub
			}
		}

		datedDown, datedUp := 0.0, 0.0
		for _, dv := range s.Dated {
			if nonNegFinite(dv.Down) {
				datedDown += dv.Down
			}
			if nonNegFinite(dv.Up) {
				datedUp += dv.Up
			}
		}
		s.HasObserved = true
		if newDay.Down > 0 {
			nextDown := max(s.Down+min(needDown, dayDeltaDown), datedDown)
			if !s.DownObserved || nextDown > s.Down+1e-9 {
				s.DownObserved = true
				s.Down = nextDown
			}
			s.DownUnpriced = false
		}
		if newDay.Up > 0 {
			nextUp := max(s.Up+min(needUp, dayDeltaUp), datedUp)
			if !s.UpObserved || nextUp > s.Up+1e-9 {
				s.UpObserved = true
				s.Up = nextUp
			}
			s.UpUnpriced = false
		}

		chargeAt := day
		if !start.IsZero() && startOfDay(start).Equal(day) && start.After(chargeAt) {
			chargeAt = start.UTC()
		}
		if !now.IsZero() && startOfDay(now).Equal(day) && chargeAt.After(now) {
			chargeAt = now.UTC()
		}
		if needDown > 1e-9 {
			s.appendDirCharge(chargeAt, needDown, "down")
		}
		if needUp > 1e-9 {
			s.appendDirCharge(chargeAt, needUp, "up")
		}
		newEff := max(s.effectiveObserved(), s.chargesSum())
		if newEff > s.Observed+1e-9 {
			s.Observed = newEff
		}
		if s.TotalUnpriced && s.DownObserved && s.UpObserved && !s.DownUnpriced && !s.UpUnpriced && s.Down+s.Up >= newEff-1e-9 {
			s.TotalUnpriced = false
		}
	}
	return changed
}

func (s *transferState) mergeReading(tr transferReading, fixedEstimated bool, state string, start, now time.Time) bool {
	if !tr.Observed && !tr.DownUnpriced && !tr.UpUnpriced && !tr.TotalUnpriced {
		return false
	}
	prevEff := s.effectiveObserved()
	changed := false
	if tr.TotalObserved && nonNegFinite(tr.Total) {
		if tr.Total >= prevEff-1e-9 && (s.DownUnpriced || s.UpUnpriced || s.TotalUnpriced) {
			s.DownUnpriced = false
			s.UpUnpriced = false
			s.TotalUnpriced = false
			changed = true
		}
	} else {
		if tr.TotalUnpriced && !s.TotalUnpriced {
			s.TotalUnpriced = true
			changed = true
		}
		if tr.DownObserved && nonNegFinite(tr.Down) {
			if s.DownUnpriced && tr.Down >= s.Down-1e-9 {
				s.DownUnpriced = false
				changed = true
			}
		} else if tr.DownUnpriced && !s.DownUnpriced {
			s.DownUnpriced = true
			changed = true
		}
		if tr.UpObserved && nonNegFinite(tr.Up) {
			if s.UpUnpriced && tr.Up >= s.Up-1e-9 {
				s.UpUnpriced = false
				changed = true
			}
		} else if tr.UpUnpriced && !s.UpUnpriced {
			s.UpUnpriced = true
			changed = true
		}
		if s.TotalUnpriced && !tr.TotalUnpriced &&
			tr.DownObserved && nonNegFinite(tr.Down) && tr.Down >= s.Down-1e-9 &&
			tr.UpObserved && nonNegFinite(tr.Up) && tr.Up >= s.Up-1e-9 &&
			tr.Down+tr.Up >= prevEff-1e-9 {
			s.TotalUnpriced = false
			changed = true
		}
	}
	if !tr.Observed {
		return changed
	}
	effIn := tr.Effective
	if !nonNegFinite(effIn) {
		return changed
	}
	if fixedEstimated && effIn == 0 && state == "starting" {
		return changed
	}
	prevDown, prevUp := s.Down, s.Up
	if !s.HasObserved {
		s.HasObserved = true
		changed = true
	}
	if tr.DownObserved && nonNegFinite(tr.Down) {
		if !s.DownObserved || tr.Down > s.Down+1e-9 {
			s.DownObserved = true
			if tr.Down > s.Down {
				s.Down = tr.Down
			}
			changed = true
		}
	}
	if tr.UpObserved && nonNegFinite(tr.Up) {
		if !s.UpObserved || tr.Up > s.Up+1e-9 {
			s.UpObserved = true
			if tr.Up > s.Up {
				s.Up = tr.Up
			}
			changed = true
		}
	}
	if tr.TotalObserved && nonNegFinite(tr.Total) {
		if !s.TotalObserved || tr.Total > s.Total+1e-9 {
			s.TotalObserved = true
			if tr.Total > s.Total {
				s.Total = tr.Total
			}
			changed = true
		}
	}
	if !tr.DownObserved && !tr.UpObserved && !tr.TotalObserved && effIn > s.Observed+1e-9 {
		s.Observed = effIn
		changed = true
	}
	newEff := s.effectiveObserved()
	if newEff > s.Observed+1e-9 {
		s.Observed = newEff
		changed = true
	}
	if delta := newEff - prevEff; delta > 1e-9 {
		at := transferChargeTime(start, now)
		dDown := max(0, s.Down-prevDown)
		dUp := max(0, s.Up-prevUp)
		switch {
		case dDown > 1e-9 && dUp <= 1e-9 && math.Abs(dDown-delta) <= 1e-9:
			s.appendDirCharge(at, delta, "down")
		case dUp > 1e-9 && dDown <= 1e-9 && math.Abs(dUp-delta) <= 1e-9:
			s.appendDirCharge(at, delta, "up")
		case dDown > 1e-9 && dUp > 1e-9 && math.Abs((dDown+dUp)-delta) <= 1e-9:
			s.appendDirCharge(at, dDown, "down")
			s.appendDirCharge(at, dUp, "up")
		default:
			s.appendDirCharge(at, delta, "")
		}
		changed = true
	}
	return changed
}

func combineTransferCharges(a, b []transferCharge) []transferCharge {
	out := make([]transferCharge, 0, len(a)+len(b))
	for _, tc := range a {
		if validRate(tc.Amount) && tc.Amount > 1e-9 {
			out = append(out, tc)
		}
	}
	for _, tc := range b {
		if validRate(tc.Amount) && tc.Amount > 1e-9 {
			out = append(out, tc)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if !out[i].At.Equal(out[j].At) {
			return out[i].At.Before(out[j].At)
		}
		return out[i].Dir < out[j].Dir
	})
	var merged []transferCharge
	for _, tc := range out {
		if n := len(merged); n > 0 && merged[n-1].At.Equal(tc.At) && merged[n-1].Dir == tc.Dir {
			merged[n-1].Amount += tc.Amount
		} else {
			merged = append(merged, tc)
		}
	}
	return merged
}

func (s *transferState) mergeState(other transferState, fixedEstimated bool, state string, start, now time.Time) bool {
	otherEff := other.effectiveObserved()
	hasValue := other.HasObserved || len(other.Charges) > 0 || len(other.Dated) > 0 || other.DownObserved || other.UpObserved || other.TotalObserved
	if !hasValue && !other.DownUnpriced && !other.UpUnpriced && !other.TotalUnpriced {
		return false
	}
	prevEff := s.effectiveObserved()
	changed := false
	sCovered := s.TotalObserved && !s.incomplete() && s.Total >= otherEff-1e-9 && (!other.TotalObserved || s.Total > other.Total+1e-9)
	otherCovered := other.TotalObserved && nonNegFinite(other.Total) && !other.incomplete() && other.Total >= prevEff-1e-9 && (!s.TotalObserved || other.Total > s.Total+1e-9)
	if otherCovered {
		if s.DownUnpriced || s.UpUnpriced || s.TotalUnpriced {
			s.DownUnpriced = false
			s.UpUnpriced = false
			s.TotalUnpriced = false
			changed = true
		}
	} else if !sCovered {
		if other.TotalUnpriced {
			sBothDirs := s.DownObserved && s.UpObserved && !s.incomplete() &&
				s.Down >= other.Down-1e-9 && s.Up >= other.Up-1e-9 && prevEff >= otherEff-1e-9 &&
				(!other.DownObserved || !other.UpObserved || s.Down > other.Down+1e-9 || s.Up > other.Up+1e-9)
			if !s.TotalUnpriced && !sBothDirs {
				s.TotalUnpriced = true
				changed = true
			}
		} else if s.TotalUnpriced && !other.incomplete() &&
			other.DownObserved && nonNegFinite(other.Down) && other.Down >= s.Down-1e-9 &&
			other.UpObserved && nonNegFinite(other.Up) && other.Up >= s.Up-1e-9 &&
			other.Down+other.Up >= prevEff-1e-9 &&
			(!s.DownObserved || !s.UpObserved || other.Down > s.Down+1e-9 || other.Up > s.Up+1e-9) {
			s.TotalUnpriced = false
			changed = true
		}
		if other.DownUnpriced {
			if !s.DownUnpriced && !(s.DownObserved && !s.DownUnpriced && (!other.DownObserved || s.Down > other.Down+1e-9)) {
				s.DownUnpriced = true
				changed = true
			}
		} else if other.DownObserved && nonNegFinite(other.Down) && (!s.DownObserved || other.Down > s.Down+1e-9) {
			if s.DownUnpriced {
				s.DownUnpriced = false
				changed = true
			}
		}
		if other.UpUnpriced {
			if !s.UpUnpriced && !(s.UpObserved && !s.UpUnpriced && (!other.UpObserved || s.Up > other.Up+1e-9)) {
				s.UpUnpriced = true
				changed = true
			}
		} else if other.UpObserved && nonNegFinite(other.Up) && (!s.UpObserved || other.Up > s.Up+1e-9) {
			if s.UpUnpriced {
				s.UpUnpriced = false
				changed = true
			}
		}
	}
	if !hasValue {
		return changed
	}
	if fixedEstimated && otherEff == 0 && len(other.Charges) == 0 && state == "starting" {
		return changed
	}
	prevChargesSum := s.chargesSum()
	prevDownOnly := s.DownObserved && !s.UpObserved && !s.TotalObserved
	prevUpOnly := s.UpObserved && !s.DownObserved && !s.TotalObserved
	otherSum := other.chargesSum()
	otherDownOnly := other.DownObserved && !other.UpObserved && !other.TotalObserved
	otherUpOnly := other.UpObserved && !other.DownObserved && !other.TotalObserved
	if !s.HasObserved {
		s.HasObserved = true
		changed = true
	}
	if other.DownObserved && nonNegFinite(other.Down) {
		if !s.DownObserved || other.Down > s.Down+1e-9 {
			s.DownObserved = true
			if other.Down > s.Down {
				s.Down = other.Down
			}
			changed = true
		}
	}
	if other.UpObserved && nonNegFinite(other.Up) {
		if !s.UpObserved || other.Up > s.Up+1e-9 {
			s.UpObserved = true
			if other.Up > s.Up {
				s.Up = other.Up
			}
			changed = true
		}
	}
	if other.TotalObserved && nonNegFinite(other.Total) {
		if !s.TotalObserved || other.Total > s.Total+1e-9 {
			s.TotalObserved = true
			if other.Total > s.Total {
				s.Total = other.Total
			}
			changed = true
		}
	}
	if nonNegFinite(other.Observed) && other.Observed > s.Observed+1e-9 {
		s.Observed = other.Observed
		changed = true
	}
	newEff := s.effectiveObserved()
	if newEff > s.Observed+1e-9 {
		s.Observed = newEff
		changed = true
	}
	disjointDir := (prevDownOnly && otherUpOnly) || (prevUpOnly && otherDownOnly)
	prevLegacy := max(0, prevEff-prevChargesSum)
	otherLegacy := max(0, otherEff-otherSum)
	covered := max(prevEff, prevChargesSum)
	prevDated := cloneDatedTransfer(s.Dated)
	replacedCharges := false
	switch {
	case disjointDir && len(other.Charges) > 0 && prevChargesSum+otherSum <= newEff+1e-9:
		s.Charges = combineTransferCharges(s.Charges, other.Charges)
		covered = max(max(prevEff, otherEff), min(newEff, prevChargesSum+otherSum+prevLegacy+otherLegacy))
		changed = true
	case len(s.Charges) > 0 && len(other.Charges) > 0 && otherSum > 0 &&
		(otherEff <= prevLegacy+1e-9 || prevEff <= otherLegacy+1e-9) &&
		prevChargesSum+otherSum <= newEff+1e-9:
		s.Charges = combineTransferCharges(s.Charges, other.Charges)
		covered = max(max(prevEff, otherEff), s.chargesSum())
		changed = true
	case (otherSum > prevChargesSum+1e-9 && otherEff >= prevEff-1e-9) ||
		(len(s.Charges) == 0 && len(other.Charges) > 0 && otherEff >= prevEff-1e-9):
		s.Charges = cloneTransferCharges(other.Charges)
		s.Dated = cloneDatedTransfer(other.Dated)
		replacedCharges = true
		covered = max(max(prevEff, otherEff), otherSum)
		changed = true
	}
	// Only append a new timestamped delta for genuine cumulative increases not already covered
	// by either side's persisted Charges or pre-existing legacy Observed baseline.
	if delta := newEff - covered; delta > 1e-9 {
		s.appendCharge(transferChargeTime(start, now), delta)
		changed = true
	}
	if replacedCharges {
		if s.mergeDatedCharges(prevDated, start, now) {
			changed = true
		}
	} else if len(other.Dated) > 0 {
		if s.mergeDatedCharges(other.Dated, start, now) {
			changed = true
		}
	}
	return changed
}

func (s *transferState) clearUnconfirmedZero(fixedEstimated bool) {
	if fixedEstimated && s.HasObserved && s.effectiveObserved() == 0 && len(s.Charges) == 0 {
		s.HasObserved = false
		s.DownObserved = false
		s.UpObserved = false
		s.TotalObserved = false
	}
}

func (s transferState) dollarsBetween(fixed float64, fixedEstimated bool, start, from, to time.Time, retired bool) float64 {
	fixedVal := 0.0
	if validRate(fixed) {
		fixedVal = fixed
	}
	startInWindow := !start.IsZero() && !start.Before(from) && !start.After(to)
	effObs := s.effectiveObserved()
	sumCharges := 0.0
	windowCharges := 0.0
	for _, tc := range s.Charges {
		if !validRate(tc.Amount) {
			continue
		}
		sumCharges += tc.Amount
		if !tc.At.IsZero() && !tc.At.Before(from) && !tc.At.After(to) {
			windowCharges += tc.Amount
		}
	}
	legacyDelta := 0.0
	if s.HasObserved && effObs > sumCharges+1e-9 {
		legacyDelta = effObs - sumCharges
	}
	totalObs := sumCharges + legacyDelta

	if !fixedEstimated {
		total := 0.0
		if startInWindow {
			total += fixedVal
		}
		if s.HasObserved || len(s.Charges) > 0 {
			total += windowCharges
			if startInWindow {
				total += legacyDelta
			}
		}
		return total
	}
	if !s.HasObserved && len(s.Charges) == 0 {
		if startInWindow {
			return fixedVal
		}
		return 0
	}
	total := windowCharges
	if startInWindow {
		total += legacyDelta
		if (!retired || s.incomplete()) && fixedVal > totalObs {
			total += fixedVal - totalObs
		}
	}
	return total
}

func rentalTransferReading(p *Rental) transferReading {
	if p == nil {
		return transferReading{}
	}
	tr := transferReading{
		Effective:     max(0, p.ObservedTransferCost),
		Observed:      p.TransferObserved,
		Down:          max(0, p.ObservedTransferDown),
		DownObserved:  p.TransferDownObserved,
		DownUnpriced:  p.TransferDownUnpriced,
		Up:            max(0, p.ObservedTransferUp),
		UpObserved:    p.TransferUpObserved,
		UpUnpriced:    p.TransferUpUnpriced,
		Total:         max(0, p.ObservedTransferTotal),
		TotalObserved: p.TransferTotalObserved,
		TotalUnpriced: p.TransferTotalUnpriced,
	}
	if tr.DownObserved || tr.UpObserved || tr.TotalObserved {
		tr.Observed = true
		dir := 0.0
		if tr.DownObserved {
			dir += tr.Down
		}
		if tr.UpObserved {
			dir += tr.Up
		}
		if dir > tr.Effective {
			tr.Effective = dir
		}
		if tr.TotalObserved && tr.Total > tr.Effective {
			tr.Effective = tr.Total
		}
	}
	return tr
}

func (r podRecord) transferState() transferState {
	return transferState{
		Observed:      r.ObservedTransferCost,
		HasObserved:   r.TransferObserved,
		Down:          r.ObservedTransferDown,
		DownObserved:  r.TransferDownObserved,
		DownUnpriced:  r.TransferDownUnpriced,
		Up:            r.ObservedTransferUp,
		UpObserved:    r.TransferUpObserved,
		UpUnpriced:    r.TransferUpUnpriced,
		Total:         r.ObservedTransferTotal,
		TotalObserved: r.TransferTotalObserved,
		TotalUnpriced: r.TransferTotalUnpriced,
		Charges:       r.TransferCharges,
		Dated:         r.DatedTransfer,
	}
}

func (r *podRecord) setTransferState(ts transferState) {
	r.ObservedTransferCost = ts.Observed
	r.TransferObserved = ts.HasObserved
	r.ObservedTransferDown = ts.Down
	r.TransferDownObserved = ts.DownObserved
	r.TransferDownUnpriced = ts.DownUnpriced
	r.ObservedTransferUp = ts.Up
	r.TransferUpObserved = ts.UpObserved
	r.TransferUpUnpriced = ts.UpUnpriced
	r.ObservedTransferTotal = ts.Total
	r.TransferTotalObserved = ts.TotalObserved
	r.TransferTotalUnpriced = ts.TotalUnpriced
	r.TransferCharges = ts.Charges
	r.DatedTransfer = ts.Dated
}

// billingSegment is one completed, immutable billing interval for a worker contract so stop/resume
// transitions never reprice earlier hours across controller restarts or UTC day boundaries.
type billingSegment struct {
	Start       time.Time `json:"start"`
	End         time.Time `json:"end"`
	Rate        float64   `json:"rate"`
	Stopped     bool      `json:"stopped,omitempty"`
	Offline     bool      `json:"offline,omitempty"`
	RateUnknown bool      `json:"rate_unknown,omitempty"`
}

func (r *podRecord) market() string {
	if r.Market == "" {
		return "runpod"
	}
	return r.Market
}

// unknownStoppedFallbackRate is the conservative hourly storage rate applied in daily spend when a
// stopped contract with an attached disk never reported a storage rate (DiskCostUnknown=true) and
// also has no known positive running rate.
const unknownStoppedFallbackRate = 0.50

// runningSpendRate returns the ongoing hourly compute+disk rate used for daily spend accrual.
// When FixedCost or ObservedTransferCost already accounts for a one-time startup charge (e.g. Vast
// cold-start download or Clore order creation fee) whose amortized equivalent TransferCost is
// folded into Cost for hourly cap comparisons, subtract TransferCost so daily spend bills one-time
// startup/transfer charges once at Start plus actual ongoing hourly compute+disk without double-counting.
func (r podRecord) runningSpendRate() float64 {
	if !validRate(r.Cost) {
		return 0
	}
	if (validRate(r.FixedCost) || r.TransferObserved || len(r.TransferCharges) > 0) && validRate(r.TransferCost) {
		return max(0, r.Cost-r.TransferCost)
	}
	return r.Cost
}

// stoppedHourlyRate is the reported ongoing hourly rate billed while the worker is stopped with an
// attached disk (e.g. Vast.ai storage billing before the instance is destroyed). Returns 0 when
// storage is free or unknown (callers check DiskCostUnknown separately for live unknown-rate gates).
func (r podRecord) stoppedHourlyRate() float64 {
	if validRate(r.DiskCost) {
		return r.DiskCost
	}
	return 0
}

// stoppedSpendRate returns the hourly rate used for daily spend over a stopped interval: the
// reported DiskCost when known, 0 when storage is confirmed free (!DiskCostUnknown), or a
// conservative estimate when DiskCostUnknown is true (the worker's known running rate, or
// unknownStoppedFallbackRate when neither storage nor compute rate was ever reported) so deleting
// an unknown-rate stopped worker stops future billing without erasing already-incurred hours.
func (r podRecord) stoppedSpendRate() float64 {
	if validRate(r.DiskCost) {
		return r.DiskCost
	}
	if !r.DiskCostUnknown {
		return 0
	}
	if rate := r.runningSpendRate(); validRate(rate) {
		return rate
	}
	return unknownStoppedFallbackRate
}

func (r podRecord) currentRunStart() time.Time {
	if !r.ResumeAt.IsZero() {
		return r.ResumeAt
	}
	if n := len(r.Segments); n > 0 {
		return r.Segments[n-1].End
	}
	return r.Start
}

func (r podRecord) currentStopStart(now time.Time) time.Time {
	if !r.StoppedAt.IsZero() {
		return r.StoppedAt
	}
	if !r.Alive.IsZero() {
		return r.Alive
	}
	if rs := r.currentRunStart(); !rs.IsZero() {
		return rs
	}
	return now
}

// resolveStopTime picks when a running -> stopped or running -> ended transition takes effect: a
// trustworthy provider timestamp bounded within [minAt, now] when reported, or else now so a
// delayed observation never retroactively refunds unconfirmed compute time.
func resolveStopTime(minAt, now, providerAt time.Time) time.Time {
	if !providerAt.IsZero() && !providerAt.After(now) && (minAt.IsZero() || !providerAt.Before(minAt)) {
		return providerAt
	}
	return now
}

// markStopped records that GPU compute has stopped without a provider stop timestamp, preserving
// the current offline status when advancing an already-stopped/offline interval.
func (r *podRecord) markStopped(now time.Time) {
	r.markStoppedStateAt(now, time.Time{}, r.Offline)
}

// markStoppedAt records that GPU compute has stopped (billing attached storage), closing any
// completed running interval into Segments and setting StoppedAt to where storage billing takes over.
func (r *podRecord) markStoppedAt(now, providerStopAt time.Time) {
	r.markStoppedStateAt(now, providerStopAt, false)
}

// markStoppedStateAt records that GPU compute has stopped, distinguishing confirmed offline
// intervals (offline=true, e.g. Vast actual_status=offline, exempt from compute and storage charges)
// from ordinary stopped intervals (offline=false, billing attached storage). Repeated observations
// in the same state advance StoppedSeen so confirmed offline/storage intervals survive restarts,
// while unobserved transitions to/from unreachable or resumed states bill conservatively.
func (r *podRecord) markStoppedStateAt(now, providerStopAt time.Time, offline bool) {
	if r.Stopped {
		if r.Offline == offline {
			if r.StoppedAt.IsZero() {
				r.StoppedAt = r.currentStopStart(now)
			}
			if r.StoppedSeen.IsZero() || now.After(r.StoppedSeen) {
				r.StoppedSeen = now
			}
			if !r.StoppedAt.IsZero() && r.StoppedSeen.Before(r.StoppedAt) {
				r.StoppedSeen = r.StoppedAt
			}
			return
		}
		stopStart := r.currentStopStart(now)
		if stopStart.After(now) {
			stopStart = now
		}
		if len(r.Segments) == 0 && !r.Start.IsZero() && stopStart.After(r.Start) {
			r.Segments = append(r.Segments, billingSegment{Start: r.Start, End: stopStart, Rate: r.runningSpendRate(), Stopped: false})
		}
		if !r.Offline && offline {
			// Stopped -> Offline: bill attached storage conservatively through the first confirmed
			// offline observation (now), then start a zero-cost offline interval at now.
			if !stopStart.IsZero() && now.After(stopStart) {
				r.Segments = append(r.Segments, billingSegment{
					Start: stopStart, End: now, Rate: r.stoppedHourlyRate(), Stopped: true, Offline: false,
					RateUnknown: r.DiskCostUnknown && !validRate(r.DiskCost),
				})
			}
			r.Offline = true
			r.StoppedAt = now
			r.StoppedSeen = now
			return
		}
		// Offline -> Stopped: preserve the confirmed offline interval [stopStart, lastOffline] at
		// $0/h and conservatively begin stopped storage billing at lastOffline.
		lastOffline := r.StoppedSeen
		if lastOffline.IsZero() || lastOffline.Before(stopStart) {
			lastOffline = stopStart
		}
		if lastOffline.After(now) {
			lastOffline = now
		}
		if !stopStart.IsZero() && lastOffline.After(stopStart) {
			r.Segments = append(r.Segments, billingSegment{
				Start: stopStart, End: lastOffline, Rate: 0, Stopped: true, Offline: true,
			})
		}
		r.Offline = false
		r.StoppedAt = lastOffline
		r.StoppedSeen = now
		if r.StoppedSeen.Before(r.StoppedAt) {
			r.StoppedSeen = r.StoppedAt
		}
		return
	}
	rs := r.currentRunStart()
	minStop := rs
	if !r.Alive.IsZero() && (minStop.IsZero() || r.Alive.After(minStop)) {
		minStop = r.Alive
	}
	stopAt := resolveStopTime(minStop, now, providerStopAt)
	if stopAt.After(now) {
		stopAt = now
	}
	if !rs.IsZero() && stopAt.Before(rs) {
		stopAt = rs
	}
	if !rs.IsZero() && stopAt.After(rs) {
		r.Segments = append(r.Segments, billingSegment{Start: rs, End: stopAt, Rate: r.runningSpendRate(), Stopped: false})
	}
	r.Stopped = true
	r.Offline = offline
	r.StoppedAt = stopAt
	r.StoppedSeen = now
	if r.StoppedSeen.Before(stopAt) {
		r.StoppedSeen = stopAt
	}
	r.ResumeAt = time.Time{}
}

// markLive records that the worker is currently running without a provider resume timestamp.
func (r *podRecord) markLive(now time.Time) {
	r.markLiveAt(now, time.Time{})
}

// markLiveAt records that the worker is currently running, closing any prior stopped/offline
// interval into Segments so resuming a stopped/offline contract preserves confirmed offline/storage
// time while conservatively budgeting any unobserved downtime after the last confirmed observation
// (StoppedSeen) at the compute rate unless a trustworthy providerResumedAt in [StoppedSeen, now] is supplied.
func (r *podRecord) markLiveAt(now, providerResumedAt time.Time) {
	if r.Stopped {
		stopStart := r.currentStopStart(now)
		if stopStart.After(now) {
			stopStart = now
		}
		// Backfill the initial pre-stop compute interval for legacy records persisted before
		// Segments existed (or newly adopted stopped records that later resume).
		if len(r.Segments) == 0 && !r.Start.IsZero() && stopStart.After(r.Start) {
			r.Segments = append(r.Segments, billingSegment{Start: r.Start, End: stopStart, Rate: r.runningSpendRate(), Stopped: false})
		}
		lastStopped := r.StoppedSeen
		if lastStopped.IsZero() || lastStopped.Before(stopStart) {
			lastStopped = stopStart
		}
		if lastStopped.After(now) {
			lastStopped = now
		}
		resumeAt := lastStopped
		if !providerResumedAt.IsZero() && !providerResumedAt.Before(lastStopped) && !providerResumedAt.After(now) {
			resumeAt = providerResumedAt
		}
		if !stopStart.IsZero() && resumeAt.After(stopStart) {
			seg := billingSegment{Start: stopStart, End: resumeAt, Stopped: true, Offline: r.Offline}
			if !r.Offline {
				seg.Rate = r.stoppedHourlyRate()
				seg.RateUnknown = r.DiskCostUnknown && !validRate(r.DiskCost)
			}
			r.Segments = append(r.Segments, seg)
		}
		r.Stopped = false
		r.Offline = false
		r.StoppedAt = time.Time{}
		r.StoppedSeen = time.Time{}
		r.ResumeAt = resumeAt
	}
	if now.After(r.Alive) || r.Alive.IsZero() {
		r.Alive = now
	}
}

func (r podRecord) dollarsBetween(from, to time.Time) float64 {
	return r.dollarsBetweenMode(from, to, !r.End.IsZero())
}

func (r podRecord) dollarsBetweenMode(from, to time.Time, retired bool) float64 {
	end := r.End
	if end.IsZero() {
		end = r.Alive
	}
	total := r.transferState().dollarsBetween(r.FixedCost, r.FixedCostEstimated, r.Start, from, to, retired)
	runRate := r.runningSpendRate()
	stopRate := r.stoppedSpendRate()
	tailStopRate := stopRate
	if r.Offline {
		tailStopRate = 0
	}
	effRunRate := runRate
	if !validRate(effRunRate) && stopRate > 0 {
		effRunRate = stopRate
	}
	if len(r.Segments) > 0 {
		for _, seg := range r.Segments {
			if seg.Offline {
				continue
			}
			segEnd := seg.End
			if !end.IsZero() && segEnd.After(end) {
				segEnd = end
			}
			rate := seg.Rate
			if !validRate(rate) {
				if seg.Stopped {
					if seg.RateUnknown {
						if validRate(r.DiskCost) {
							rate = r.DiskCost
						} else if validRate(runRate) {
							rate = runRate
						} else {
							rate = unknownStoppedFallbackRate
						}
					}
				} else if validRate(effRunRate) {
					rate = effRunRate
				}
			}
			if validRate(rate) {
				if a, b := maxTime(seg.Start, from), minTime(segEnd, to); b.After(a) {
					total += b.Sub(a).Hours() * rate
				}
			}
		}
		if r.Stopped {
			tailStart := r.StoppedAt
			if tailStart.IsZero() {
				tailStart = r.Segments[len(r.Segments)-1].End
			}
			if tailStopRate > 0 && !tailStart.IsZero() && end.After(tailStart) {
				if a, b := maxTime(tailStart, from), minTime(end, to); b.After(a) {
					total += b.Sub(a).Hours() * tailStopRate
				}
			}
		} else {
			tailStart := r.currentRunStart()
			if validRate(effRunRate) && !tailStart.IsZero() && end.After(tailStart) {
				if a, b := maxTime(tailStart, from), minTime(end, to); b.After(a) {
					total += b.Sub(a).Hours() * effRunRate
				}
			}
		}
		return total
	}
	if r.Stopped {
		gpuEnd := r.StoppedAt
		if gpuEnd.IsZero() {
			gpuEnd = r.Alive
			if gpuEnd.IsZero() {
				gpuEnd = r.Start
			}
		}
		if !end.IsZero() && gpuEnd.After(end) {
			gpuEnd = end
		}
		if validRate(effRunRate) {
			if a, b := maxTime(r.Start, from), minTime(gpuEnd, to); b.After(a) {
				total += b.Sub(a).Hours() * effRunRate
			}
		}
		if tailStopRate > 0 && !gpuEnd.IsZero() && end.After(gpuEnd) {
			if a, b := maxTime(gpuEnd, from), minTime(end, to); b.After(a) {
				total += b.Sub(a).Hours() * tailStopRate
			}
		}
		return total
	}
	a, b := maxTime(r.Start, from), minTime(end, to)
	if !b.After(a) || !validRate(effRunRate) {
		return total
	}
	return total + b.Sub(a).Hours()*effRunRate
}

func maxTime(a, b time.Time) time.Time {
	if a.After(b) {
		return a
	}
	return b
}

func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}

// pendingRent records a worker creation attempt persisted BEFORE calling m.Rent so that an
// ambiguous outcome (incomplete 2xx without contract ID, HTTP 5xx, timeout, or crash) reserves
// its slot, handle, and hourly rate durably until the contract is adopted from provider listings
// or explicitly resolved via `stet-cloud workers resolve`.
type pendingRent struct {
	ID                 string    `json:"id"`
	Market             string    `json:"market"`
	Handle             string    `json:"handle"`
	OfferID            string    `json:"offer_id,omitempty"`
	GPU                string    `json:"gpu"`
	Cloud              string    `json:"cloud,omitempty"`
	Cost               float64   `json:"cost"`
	DiskCost           float64   `json:"disk_cost,omitempty"`
	TransferCost       float64   `json:"transfer_cost,omitempty"`
	FixedCost          float64   `json:"fixed_cost,omitempty"`
	FixedCostEstimated bool      `json:"fixed_cost_estimated,omitempty"`
	Created            time.Time `json:"created"`
	Grades             int       `json:"grades,omitempty"`
	Host               string    `json:"host,omitempty"`
	CPU                string    `json:"cpu,omitempty"`
	Curators           bool      `json:"curators,omitempty"`
	For                string    `json:"for,omitempty"`
	Round              string    `json:"round,omitempty"`
	Report             time.Time `json:"report,omitzero"`
	Error              string    `json:"error,omitempty"`
	KnownIDs           []string  `json:"known_ids,omitempty"`
}

func (pr *pendingRent) dollarsBetween(from, to time.Time) float64 {
	if pr == nil || (!validRate(pr.Cost) && !validRate(pr.FixedCost)) {
		return 0
	}
	r := podRecord{Cost: pr.Cost, TransferCost: pr.TransferCost, FixedCost: pr.FixedCost, FixedCostEstimated: pr.FixedCostEstimated, Start: pr.Created, End: to}
	return r.dollarsBetween(from, to)
}

type ledger struct {
	Pods    map[string]*podRecord   `json:"pods"`
	Pending map[string]*pendingRent `json:"pending,omitempty"`
}

func (f *Fleet) readLedger() (*ledger, error) {
	l := &ledger{Pods: map[string]*podRecord{}, Pending: map[string]*pendingRent{}}
	raw, err := os.ReadFile(filepath.Join(f.o.Home, "fleet.json"))
	if errors.Is(err, os.ErrNotExist) {
		return l, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, l); err != nil {
		return nil, fmt.Errorf("fleet.json: %w", err)
	}
	if l.Pods == nil {
		l.Pods = map[string]*podRecord{}
	}
	if l.Pending == nil {
		l.Pending = map[string]*pendingRent{}
	}
	return l, nil
}

func (f *Fleet) writeLedger(l *ledger) error {
	raw, err := json.MarshalIndent(l, "", " ")
	if err != nil {
		return err
	}
	path := filepath.Join(f.o.Home, "fleet.json")
	if err := os.WriteFile(path+".tmp", raw, 0o600); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

func (f *Fleet) spentLog() string { return filepath.Join(f.o.Home, "spent.jsonl") }

// owner is the handle of the worker key a machine was rented with: the name the market keeps
// for it, or where it keeps none (Clore), the record's.
func owner(l *ledger, r *Rental) string {
	if r.Name != "" {
		return r.Name
	}
	if rec := l.Pods[ledgerKey(r.Market, r.ID)]; rec != nil {
		return rec.Handle
	}
	return ""
}

// matchPendingRentals returns pointers into rentals that match an unresolved pendingRent
// (same market, not already tracked in l.Pods or already retired in spentSum, not in pr.KnownIDs,
// and matching provider identity/handle), ordered live-first, then stopped, then gone/expired,
// and newest-first within each state.
// For Clore (which carries no worker handle on orders), an explicit server ID match (si ==
// pr.OfferID/pr.Host) is required so recovery on a shared account never adopts or cancels
// somebody else's order on a different server.
func matchPendingRentals(pr *pendingRent, l *ledger, spentSum *spentSummary, rentals []Rental) []*Rental {
	known := make(map[string]bool, len(pr.KnownIDs))
	for _, id := range pr.KnownIDs {
		known[id] = true
	}
	var matches []*Rental
	for i := range rentals {
		p := &rentals[i]
		if p.Market != pr.Market || p.ID == "" || known[p.ID] {
			continue
		}
		if !p.live() && p.State != "stopped" && p.State != "gone" {
			continue
		}
		if l.Pods[ledgerKey(p.Market, p.ID)] != nil {
			continue
		}
		if spentSum != nil && spentSum.hasContract(p.Market, p.ID) {
			continue
		}
		if pr.Market == "clore" {
			wantServer := pr.OfferID
			if wantServer == "" {
				wantServer = pr.Host
			}
			if wantServer == "" || p.Host == "" || p.Host != wantServer {
				continue
			}
			if p.State == "gone" && (p.Created.IsZero() || pr.Created.IsZero()) {
				continue
			}
			if !p.Created.IsZero() && !pr.Created.IsZero() && p.Created.Before(pr.Created.Add(-2*time.Minute)) {
				continue
			}
			matches = append(matches, p)
			continue
		}
		if p.Name == pr.Handle {
			if p.State == "gone" {
				// Require a credible post-reservation creation timestamp before attributing an
				// already-gone non-Clore contract (e.g. a pre-existing DELETING pod).
				if p.Created.IsZero() || pr.Created.IsZero() || p.Created.Before(pr.Created.Add(-2*time.Minute)) {
					continue
				}
			}
			matches = append(matches, p)
		}
	}
	stateRank := func(r *Rental) int {
		if r.State == "running" || r.State == "starting" {
			return 3
		}
		if r.State == "unknown" {
			return 2
		}
		if r.State == "stopped" {
			return 1
		}
		return 0
	}
	sort.SliceStable(matches, func(i, j int) bool {
		ri, rj := stateRank(matches[i]), stateRank(matches[j])
		if ri != rj {
			return ri > rj
		}
		if !matches[i].Created.Equal(matches[j].Created) {
			return matches[i].Created.After(matches[j].Created)
		}
		return matches[i].ID > matches[j].ID
	})
	return matches
}

// resolvePendingMatches computes candidate matches for every pending reservation in pkeys plus
// the distinct worker handles claiming each candidate contract, so callers leave genuinely
// ambiguous matches (multiple contracts for one reservation, or multiple handles competing for
// the same contract) pending rather than guessing or canceling unrelated contracts.
func resolvePendingMatches(l *ledger, spentSum *spentSummary, rentalsByMarket map[string][]Rental, pkeys []string) (map[string][]*Rental, map[string]map[string]bool) {
	matchesByID := make(map[string][]*Rental, len(pkeys))
	claimants := map[string]map[string]bool{}
	for _, id := range pkeys {
		pr := l.Pending[id]
		if pr == nil {
			continue
		}
		rs, ok := rentalsByMarket[pr.Market]
		if !ok {
			continue
		}
		ms := matchPendingRentals(pr, l, spentSum, rs)
		matchesByID[id] = ms
		for _, p := range ms {
			key := ledgerKey(p.Market, p.ID)
			if claimants[key] == nil {
				claimants[key] = map[string]bool{}
			}
			claimants[key][pr.Handle] = true
		}
	}
	return matchesByID, claimants
}

func rentalIDList(rs []*Rental) string {
	ids := make([]string, 0, len(rs))
	for _, r := range rs {
		if r != nil && r.ID != "" {
			ids = append(ids, r.ID)
		}
	}
	return strings.Join(ids, ", ")
}

// applyOfferRateToRental merges a newly created Rental's provider rate with the Offer it was
// rented from, ensuring a base-only/partial, lower, or zero initial listing never erases known
// storage, transfer, or order-fee components.
func applyOfferRateToRental(o Offer, r *Rental) {
	if r == nil {
		return
	}
	origDisk := max(0, r.DiskCost)
	disk := origDisk
	if nonNegFinite(o.DiskCost) && o.DiskCost > disk {
		disk = o.DiskCost
	}
	r.DiskCost = disk
	if validRate(r.DiskCost) {
		r.DiskCostUnknown = false
	}
	origTransfer := max(0, r.TransferCost)
	transfer := origTransfer
	if nonNegFinite(o.TransferCost) && o.TransferCost > transfer {
		transfer = o.TransferCost
	}
	r.TransferCost = transfer
	if nonNegFinite(o.FixedCost) && o.FixedCost > r.FixedCost {
		r.FixedCost = o.FixedCost
		r.FixedCostEstimated = o.FixedCostEstimated
	} else if o.FixedCostEstimated && r.FixedCost > 0 {
		r.FixedCostEstimated = true
	}
	missingDisk := max(0, disk-origDisk)
	missingTransfer := max(0, transfer-origTransfer)
	switch {
	case validRate(r.Price) && !r.PartialRate:
		eff := r.Price + missingTransfer
		if validRate(r.BasePrice) && r.BasePrice+disk+transfer > eff {
			eff = r.BasePrice + disk + transfer
		}
		if validRate(o.Price) && o.Price > eff {
			eff = o.Price
		}
		r.Price = eff
	case validRate(r.Price) && r.PartialRate:
		reconstructed := r.Price + missingDisk + missingTransfer
		if validRate(o.Price) {
			r.Price = max(o.Price, reconstructed)
			r.PartialRate = false
		} else {
			r.Price = reconstructed
		}
	default:
		if validRate(o.Price) {
			r.Price = o.Price
			r.PartialRate = false
		} else {
			r.Price = 0
		}
	}
}

// reconcileRentalRate merges a provider-reported Rental rate with a persisted podRecord,
// preserving known storage/transfer/all-in cost components whenever the provider returns a
// partial (base-only), lower (e.g. stopped disk-only or compute-only), or zero rate.
func reconcileRentalRate(r *podRecord, p *Rental) {
	reconcileRentalRateAt(r, p, time.Time{})
}

func reconcileRentalRateAt(r *podRecord, p *Rental, now time.Time) {
	if p == nil {
		return
	}
	if r == nil {
		if !validRate(p.Price) {
			p.Price = 0
		}
		return
	}
	origDisk := max(0, p.DiskCost)
	if nonNegFinite(p.DiskCost) && p.DiskCost > r.DiskCost {
		r.DiskCost = p.DiskCost
	}
	p.DiskCost = r.DiskCost
	if validRate(r.DiskCost) {
		r.DiskCostUnknown = false
		p.DiskCostUnknown = false
	} else if p.State == "stopped" && !p.Offline {
		r.DiskCostUnknown = p.DiskCostUnknown
	} else if r.DiskCostUnknown && !p.Offline {
		p.DiskCostUnknown = true
	}
	missingDisk := max(0, r.DiskCost-origDisk)
	origTransfer := max(0, p.TransferCost)
	if nonNegFinite(p.TransferCost) && p.TransferCost > r.TransferCost {
		r.TransferCost = p.TransferCost
	}
	p.TransferCost = r.TransferCost
	if nonNegFinite(p.FixedCost) && p.FixedCost > r.FixedCost {
		r.FixedCost = p.FixedCost
		r.FixedCostEstimated = p.FixedCostEstimated
	}
	p.FixedCost = r.FixedCost
	if r.FixedCostEstimated {
		p.FixedCostEstimated = true
	}
	ts := r.transferState()
	ts.mergeReading(rentalTransferReading(p), r.FixedCostEstimated, p.State, r.Start, now)
	r.setTransferState(ts)
	p.TransferDownUnpriced = r.TransferDownUnpriced
	p.TransferUpUnpriced = r.TransferUpUnpriced
	p.TransferTotalUnpriced = r.TransferTotalUnpriced
	if r.TransferObserved {
		p.TransferObserved = true
		p.ObservedTransferCost = r.ObservedTransferCost
		p.ObservedTransferDown = r.ObservedTransferDown
		p.TransferDownObserved = r.TransferDownObserved
		p.ObservedTransferUp = r.ObservedTransferUp
		p.TransferUpObserved = r.TransferUpObserved
		p.ObservedTransferTotal = r.ObservedTransferTotal
		p.TransferTotalObserved = r.TransferTotalObserved
	}
	missingTransfer := max(0, r.TransferCost-origTransfer)
	switch {
	case validRate(p.Price) && !p.PartialRate:
		eff := p.Price + missingTransfer
		if validRate(p.BasePrice) && p.BasePrice+r.DiskCost+r.TransferCost > eff {
			eff = p.BasePrice + r.DiskCost + r.TransferCost
		}
		if validRate(r.Cost) && r.Cost > eff {
			eff = r.Cost
		}
		r.Cost = eff
		if p.State != "stopped" {
			r.PartialCost = false
		}
		p.Price = eff
		p.PartialRate = r.PartialCost
	case validRate(p.Price) && p.PartialRate:
		reconstructed := p.Price + missingDisk + missingTransfer
		if !r.PartialCost && validRate(r.Cost) {
			eff := max(r.Cost, reconstructed)
			r.Cost = eff
			p.Price = eff
			p.PartialRate = false
		} else {
			eff := reconstructed
			if validRate(r.Cost) && r.Cost > eff {
				eff = r.Cost
			}
			r.Cost = eff
			r.PartialCost = true
			p.Price = eff
			p.PartialRate = true
		}
	default:
		if validRate(r.Cost) {
			p.Price = r.Cost
			p.PartialRate = r.PartialCost
		} else {
			p.Price = 0
			p.PartialRate = false
		}
	}
}

func rentalStatePriority(r *Rental) int {
	if r == nil {
		return -1
	}
	switch r.State {
	case "running", "starting":
		return 2
	case "unknown":
		return 1
	default:
		return 0
	}
}

func betterRental(prev, cur *Rental) bool {
	if prev == nil {
		return true
	}
	if prev.Unconfirmed != cur.Unconfirmed {
		return prev.Unconfirmed
	}
	if pi, ci := rentalStatePriority(prev), rentalStatePriority(cur); pi != ci {
		return ci > pi
	}
	if !prev.live() {
		if prev.DiskCostUnknown != cur.DiskCostUnknown {
			return prev.DiskCostUnknown
		}
		if validRate(prev.DiskCost) != validRate(cur.DiskCost) {
			return validRate(cur.DiskCost)
		}
	}
	prevFull := validRate(prev.Price) && !prev.PartialRate
	curFull := validRate(cur.Price) && !cur.PartialRate
	if !prevFull && curFull {
		return true
	}
	return !validRate(prev.Price) && validRate(cur.Price)
}

// note brings the record up to date with the machines the markets list now:
//   - unresolved pending worker creates are adopted automatically when the provider lists one
//     unambiguous matching contract, and otherwise remain reserved in l.Pending;
//   - live rentals retain their last known positive hourly rate if a listing omits or zeroes Price;
//   - stopped rentals (whether listed directly or returned by Lookup) stay in fleet.json with
//     Stopped=true until m.End(ctx, id) confirms deletion or Lookup confirms absence (404);
//   - recorded machines omitted from an answered listing are verified via rentalLookup before
//     retiring, preserving their slot, state, and positive rate while status is uncertain.
func (f *Fleet) note(ctx context.Context, l *ledger, rentals []Rental, answered map[string]bool) ([]Rental, error) {
	now := f.now()
	if l.Pending == nil {
		l.Pending = map[string]*pendingRent{}
	}
	spentSum, err := f.readSpentSummary()
	if err != nil {
		return rentals, err
	}
	// 1. Adopt any unresolved pending worker creates that now have a single unambiguous match
	//    in an answered listing; leave genuinely ambiguous matches pending. Drop any pending
	//    create already retired to spent.jsonl before a prior failed ledger write.
	if len(l.Pending) > 0 {
		for id := range l.Pending {
			if spentSum.byPendingID[id] {
				delete(l.Pending, id)
			}
		}
		pkeys := make([]string, 0, len(l.Pending))
		for id := range l.Pending {
			pkeys = append(pkeys, id)
		}
		sort.Strings(pkeys)
		rentalsByMarket := map[string][]Rental{}
		for mName, ok := range answered {
			if ok {
				rentalsByMarket[mName] = rentals
			}
		}
		matchesByID, claimants := resolvePendingMatches(l, spentSum, rentalsByMarket, pkeys)
		adopted := map[string]bool{}
		for _, id := range pkeys {
			pr := l.Pending[id]
			if pr == nil || !answered[pr.Market] {
				continue
			}
			matches := matchesByID[id]
			if len(matches) == 0 {
				continue
			}
			if len(matches) > 1 {
				f.say("pending worker %s on %s matches %d untracked contracts (%s); leaving pending (verify on %s before resolving)",
					pr.Handle, marketTitle(pr.Market), len(matches), rentalIDList(matches), marketTitle(pr.Market))
				continue
			}
			p := matches[0]
			key := ledgerKey(p.Market, p.ID)
			if len(claimants[key]) > 1 {
				f.say("pending worker %s on %s shares candidate contract %s with another pending handle; leaving pending",
					pr.Handle, marketTitle(pr.Market), p.ID)
				continue
			}
			if adopted[key] {
				f.say("pending worker reservation %s (%s on %s) matches already-adopted contract %s; keeping reservation pending until verified",
					id, pr.Handle, marketTitle(pr.Market), p.ID)
				continue
			}
			adopted[key] = true
			if p.Name == "" {
				p.Name = pr.Handle
			}
			gpu := p.Card
			if gpu == "" {
				gpu = pr.GPU
			}
			host := p.Host
			if host == "" {
				host = pr.Host
			}
			start := p.Created
			if !pr.Created.IsZero() && (start.IsZero() || start.After(pr.Created)) {
				start = pr.Created
			}
			if start.IsZero() || start.After(now) {
				start = now
			}
			if p.State == "gone" {
				// Uniquely attributable terminal/expired provider contract (e.g. a Clore order
				// that already expired before first reconciliation): retire it once through the
				// supported or conservatively observed end time instead of leaving the pending
				// reservation accruing indefinitely.
				statusWord := strings.ToLower(p.Status)
				if statusWord == "" {
					statusWord = "gone"
				}
				rec := &podRecord{
					ID: p.ID, Market: p.Market, Handle: pr.Handle, GPU: gpu, Cloud: pr.Cloud, PendingID: pr.ID,
					Cost: pr.Cost, DiskCost: pr.DiskCost, TransferCost: pr.TransferCost,
					FixedCost: pr.FixedCost, FixedCostEstimated: pr.FixedCostEstimated,
					PartialCost: !validRate(pr.Cost), Start: start, Alive: now, Tracked: true, Gone: true,
					Why:    "its market already has it " + statusWord,
					Grades: pr.Grades, Host: host, CPU: pr.CPU, Curators: pr.Curators,
					For: pr.For, Round: pr.Round, Report: pr.Report,
				}
				reconcileRentalRateAt(rec, p, now)
				endAt := resolveStopTime(start, now, p.EndedAt)
				l.Pods[key] = rec
				if rerr := f.retire(l, key, endAt); rerr != nil {
					delete(l.Pods, key)
					return rentals, rerr
				}
				delete(l.Pending, id)
				f.event(p, pr.Handle, "gone", rec.Why)
				f.say("retired already-%s pending worker %s on %s (machine %s)", statusWord, pr.Handle, marketTitle(p.Market), p.ID)
				continue
			}
			alive := now
			var stoppedAt, stoppedSeen time.Time
			if p.State == "stopped" {
				// Use a trustworthy provider stop timestamp when available; otherwise preserve
				// the conservative compute-rate accrual reserved while pending up through this
				// first confirmed stop, then accrue storage from now forward.
				stoppedAt = resolveStopTime(start, now, p.StoppedAt)
				stoppedSeen = now
			} else if !p.live() {
				alive = start
			}
			rec := &podRecord{
				ID: p.ID, Market: p.Market, Handle: pr.Handle, GPU: gpu, Cloud: pr.Cloud, PendingID: pr.ID,
				Cost: pr.Cost, DiskCost: pr.DiskCost, TransferCost: pr.TransferCost,
				FixedCost: pr.FixedCost, FixedCostEstimated: pr.FixedCostEstimated,
				PartialCost: !validRate(pr.Cost),
				Start:       start, Alive: alive, StoppedAt: stoppedAt, StoppedSeen: stoppedSeen, Tracked: true, Stopped: p.State == "stopped", Offline: p.State == "stopped" && p.Offline,
				Grades: pr.Grades, Host: host, CPU: pr.CPU, Curators: pr.Curators,
				For: pr.For, Round: pr.Round, Report: pr.Report,
			}
			reconcileRentalRateAt(rec, p, now)
			l.Pods[key] = rec
			delete(l.Pending, id)
			f.event(p, pr.Handle, "rented", "adopted after uncertain create")
			f.say("adopted pending worker %s on %s (machine %s)", pr.Handle, marketTitle(p.Market), p.ID)
		}
	}

	listed := map[string]*Rental{}
	for i := range rentals {
		key := ledgerKey(rentals[i].Market, rentals[i].ID)
		if betterRental(listed[key], &rentals[i]) {
			listed[key] = &rentals[i]
		}
	}
	knownHandles := map[string]bool{}
	for _, k := range f.keys {
		knownHandles[k.Handle] = true
	}
	for i := range rentals {
		p := &rentals[i]
		h := owner(l, p)
		if !knownHandles[h] {
			continue
		}
		if !p.live() && p.State != "stopped" {
			continue
		}
		key := ledgerKey(p.Market, p.ID)
		r := l.Pods[key]
		if r == nil { // rented by another copy of this tool, or by hand
			cost := 0.0
			if validRate(p.Price) {
				cost = p.Price
			}
			start := p.Created
			if start.IsZero() {
				start = now
			}
			alive := now
			var stoppedAt, stoppedSeen time.Time
			if p.State == "stopped" {
				stoppedAt = resolveStopTime(start, now, p.StoppedAt)
				stoppedSeen = now
			}
			r = &podRecord{ID: p.ID, Market: p.Market, Handle: h, Host: p.Host, GPU: p.Card,
				Cost: cost, DiskCost: max(0, p.DiskCost), DiskCostUnknown: p.DiskCostUnknown,
				FixedCost: max(0, p.FixedCost), FixedCostEstimated: p.FixedCostEstimated,
				PartialCost: p.PartialRate || p.State == "stopped",
				Start:       start, Alive: alive, StoppedAt: stoppedAt, StoppedSeen: stoppedSeen, Tracked: true, Stopped: p.State == "stopped", Offline: p.State == "stopped" && p.Offline}
			l.Pods[key] = r
		}
		if r.Host == "" && p.Host != "" {
			r.Host = p.Host
		}
		if r.Terminating {
			p.Terminating = true
		}
		reconcileRentalRateAt(r, p, now)
		if p.live() {
			r.markLiveAt(now, p.ResumedAt)
		}
	}
	retireGone := func(key string, r *podRecord, p *Rental) error {
		if p == nil {
			// If an instance vanished from provider listings without a terminal billing payload,
			// do not let an earlier zero-transfer snapshot erase the pre-launch estimate.
			ts := r.transferState()
			ts.clearUnconfirmedZero(r.FixedCostEstimated)
			r.setTransferState(ts)
		}
		r.Gone = true
		gone := Rental{Market: r.market(), ID: r.ID, Card: r.GPU}
		why := "its market no longer lists it"
		var providerEndAt time.Time
		if p != nil {
			gone = *p
			providerEndAt = p.EndedAt
			why = "its market has it " + strings.ToLower(p.Status)
			if p.Detail != "" && p.Detail != strings.ToLower(p.Status) {
				why += ": " + p.Detail
			}
		}
		if r.Asked {
			why = "ended by itself, or by its market: " + why
		} else {
			r.Cause = causeGone
			why = "taken away before it asked for work: " + why
		}
		r.Why = why
		f.event(&gone, r.Handle, "gone", why)
		minEnd := r.currentRunStart()
		if !r.Alive.IsZero() && (minEnd.IsZero() || r.Alive.After(minEnd)) {
			minEnd = r.Alive
		}
		if r.Stopped {
			stoppedAt := r.StoppedAt
			if stoppedAt.IsZero() {
				stoppedAt = r.currentStopStart(now)
			}
			if !stoppedAt.IsZero() && (minEnd.IsZero() || stoppedAt.After(minEnd)) {
				minEnd = stoppedAt
			}
			if !r.StoppedSeen.IsZero() && (minEnd.IsZero() || r.StoppedSeen.After(minEnd)) {
				minEnd = r.StoppedSeen
			}
		}
		// Without a trustworthy provider end timestamp, keep the budget charged through the first
		// confirmed absence (now) rather than retroactively refunding compute back to Alive, and
		// conservatively budget any unobserved [StoppedSeen, endAt] gap on a stopped worker at
		// the compute rate.
		endAt := resolveStopTime(minEnd, now, providerEndAt)
		trustedEnd := !providerEndAt.IsZero() && endAt.Equal(providerEndAt)
		if r.Stopped && !trustedEnd {
			r.markLiveAt(endAt, time.Time{})
		}
		return f.retire(l, key, endAt)
	}
	keys := make([]string, 0, len(l.Pods))
	for key := range l.Pods {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var extra []Rental
	for _, key := range keys {
		r := l.Pods[key]
		if r == nil {
			continue
		}
		p := listed[key]
		// If a prior confirmed retirement already wrote this contract's terminal row to
		// spent.jsonl before a failed fleet.json write, clear the stale ledger entry immediately
		// whenever the contract is not actively listed (avoiding Clore zombies, bogus "gone"
		// events, and transient outage lockouts).
		if r.ID != "" && spentSum.hasContract(r.market(), r.ID) && (p == nil || (!p.live() && p.State != "stopped")) {
			delete(l.Pods, key)
			continue
		}
		if !answered[r.market()] {
			continue
		}
		if p != nil && p.live() {
			reconcileRentalRateAt(r, p, now)
			r.markLiveAt(now, p.ResumedAt)
			continue
		}
		if p != nil {
			if p.Name == "" {
				p.Name = r.Handle
			}
			reconcileRentalRateAt(r, p, now)
			if p.State == "stopped" {
				// Keep stopped instance in fleet.json (with Stopped=true) until m.End(ctx, id)
				// confirms provider deletion or Lookup confirms absence (e.g. to stop Vast disk
				// billing even if a delete fails and a later listing omits the stopped worker).
				wasStopped := r.Stopped
				r.markStoppedStateAt(now, p.StoppedAt, p.Offline)
				if !wasStopped {
					r.Gone = true
					why := "its market has it " + strings.ToLower(p.Status)
					if p.Detail != "" && p.Detail != strings.ToLower(p.Status) {
						why += ": " + p.Detail
					}
					r.Why = why
					f.event(p, r.Handle, "stopped", why)
				}
				continue
			}
			if err := retireGone(key, r, p); err != nil {
				return rentals, err
			}
			continue
		}
		// Known worker omitted from a successful listing: verify by ID before retiring so a
		// truncated/eventually-consistent listing can never free a live slot, drop a stopped
		// worker awaiting deletion, or erase its rate.
		if lu, ok := f.market(r.market()).(rentalLookup); ok {
			fresh, exists, lerr := lu.Lookup(ctx, r.ID)
			switch {
			case lerr == nil && !exists:
				if err := retireGone(key, r, nil); err != nil {
					return rentals, err
				}
			case lerr == nil && exists && fresh != nil && fresh.live():
				if fresh.Name == "" {
					fresh.Name = r.Handle
				}
				if fresh.Host == "" {
					fresh.Host = r.Host
				}
				fresh.Terminating = r.Terminating
				reconcileRentalRateAt(r, fresh, now)
				r.markLiveAt(now, fresh.ResumedAt)
				extra = append(extra, *fresh)
			case lerr == nil && exists && fresh != nil:
				if fresh.Name == "" {
					fresh.Name = r.Handle
				}
				if fresh.Host == "" {
					fresh.Host = r.Host
				}
				fresh.Terminating = r.Terminating
				reconcileRentalRateAt(r, fresh, now)
				if fresh.State == "stopped" {
					r.markStoppedStateAt(now, fresh.StoppedAt, fresh.Offline)
					extra = append(extra, *fresh)
				} else if err := retireGone(key, r, fresh); err != nil {
					return rentals, err
				}
			default:
				extra = append(extra, recordedRental(r, "unconfirmed (omitted from listing)"))
			}
			continue
		}
		extra = append(extra, recordedRental(r, "unconfirmed (omitted from listing)"))
	}
	if len(extra) > 0 {
		rentals = append(rentals, extra...)
	}
	f.syncDatedCharges(ctx, l, answered, now)
	return rentals, f.writeLedger(l)
}

// syncDatedCharges polls provider billing endpoints (e.g. Vast.ai GET /api/v0/charges) for
// answered markets and reconciles dated bandwidth charges exclusively onto owned contracts
// (tracked in l.Pods, plus cached on f.datedCharges for owned contracts retired in spent.jsonl).
// Unrelated contracts on shared provider accounts are strictly ignored.
func (f *Fleet) syncDatedCharges(ctx context.Context, l *ledger, answered map[string]bool, now time.Time) {
	todayStart := startOfDay(now)
	minLookback := todayStart.AddDate(0, 0, -7)
	for _, m := range f.markets {
		if !answered[m.Name()] {
			continue
		}
		tcLookup, ok := m.(transferChargesLookup)
		if !ok {
			continue
		}
		from := todayStart
		for _, r := range l.Pods {
			if r != nil && r.market() == m.Name() && !r.Start.IsZero() {
				if sd := startOfDay(r.Start); sd.Before(from) {
					from = sd
				}
			}
		}
		if from.Before(minLookback) {
			from = minLookback
		}
		byContract, err := tcLookup.TransferCharges(ctx, from, now)
		if err != nil {
			continue
		}
		if len(byContract) > 0 {
			if f.datedCharges == nil {
				f.datedCharges = map[string]map[string]map[string]dayTransferCharge{}
			}
			mCache := f.datedCharges[m.Name()]
			if mCache == nil {
				mCache = map[string]map[string]dayTransferCharge{}
				f.datedCharges[m.Name()] = mCache
			}
			for cid, days := range byContract {
				if cid == "" || len(days) == 0 {
					continue
				}
				cCache := mCache[cid]
				if cCache == nil {
					cCache = map[string]dayTransferCharge{}
					mCache[cid] = cCache
				}
				for dk, dv := range days {
					prev := cCache[dk]
					cCache[dk] = dayTransferCharge{
						Down: max(prev.Down, max(0, dv.Down)),
						Up:   max(prev.Up, max(0, dv.Up)),
					}
				}
			}
		}
		mCache := f.datedCharges[m.Name()]
		if len(mCache) == 0 {
			continue
		}
		for _, r := range l.Pods {
			if r == nil || r.market() != m.Name() || r.ID == "" {
				continue
			}
			days := mCache[r.ID]
			if len(days) == 0 {
				continue
			}
			ts := r.transferState().clone()
			if ts.mergeDatedCharges(days, r.Start, now) {
				r.setTransferState(ts)
			}
		}
	}
}

type spentSummary struct {
	records            []podRecord
	byContract         map[string]podRecord // first terminal record per durable contract key (market:id)
	byPendingID        map[string]bool      // pending create IDs already durably retired in spent.jsonl
	byExaminerID       map[string]bool      // examiner instance IDs already durably retired in spent.jsonl
	incompleteTail     bool                 // true when a trailing partial JSON object (no '\n') needs truncation
	tailOffset         int64                // byte offset right after the last '\n' when incompleteTail is true
	needLeadingNewline bool                 // true when the file ends with a valid JSON object lacking '\n'
}

func (s *spentSummary) hasContract(market, id string) bool {
	if s == nil || id == "" {
		return false
	}
	if market == "" {
		market = "runpod"
	}
	_, ok := s.byContract[ledgerKey(market, id)]
	return ok
}

func (s *spentSummary) addRecord(r podRecord) {
	if r.Market == "vast" && r.For == "examiner" && r.ID != "" {
		s.byExaminerID[r.ID] = true
	}
	if r.ID != "" {
		ck := ledgerKey(r.market(), r.ID)
		if _, seen := s.byContract[ck]; seen {
			return
		}
		s.byContract[ck] = r
	}
	if r.PendingID != "" {
		s.byPendingID[r.PendingID] = true
	}
	s.records = append(s.records, r)
}

func parseSpentRecord(line []byte) (podRecord, error) {
	var r podRecord
	if len(line) == 0 || line[0] != '{' {
		return r, errors.New("expected JSON object")
	}
	if err := json.Unmarshal(line, &r); err != nil {
		return r, err
	}
	return r, nil
}

// isIncompleteJSONObjectPrefix reports whether tail (already TrimSpace'd) failed json.Unmarshal
// solely because it was truncated mid-object at EOF, rather than containing glued records,
// mid-line syntax errors, or type-corrupted JSON fields.
func isIncompleteJSONObjectPrefix(tail []byte, err error) bool {
	if len(tail) == 0 || tail[0] != '{' {
		return false
	}
	var se *json.SyntaxError
	if !errors.As(err, &se) {
		return false
	}
	msg := se.Error()
	return msg == "unexpected end of JSON input" ||
		(se.Offset == int64(len(tail)) && strings.HasPrefix(msg, "invalid character ' ' "))
}

// parseSpentBytes parses spent.jsonl strictly:
//   - Every complete newline-terminated non-empty line MUST parse as a JSON object; malformed
//     history surfaces an explicit error instead of silently dropping spend.
//   - An unterminated final fragment that is a strict incomplete JSON object prefix is classified
//     as an uncommitted partial write (incompleteTail=true, tailOffset=offset after last '\n') so
//     writers under lock truncate it back to tailOffset before appending the retried record.
//     Glued records, type-corrupted objects, or mid-line syntax errors without '\n' fail closed.
//   - A valid JSON object lacking a trailing '\n' is accepted and marked needLeadingNewline=true so
//     the next append separates lines cleanly.
func parseSpentBytes(path string, raw []byte) (*spentSummary, error) {
	sum := &spentSummary{
		byContract:   map[string]podRecord{},
		byPendingID:  map[string]bool{},
		byExaminerID: map[string]bool{},
	}
	pos := 0
	lineNum := 1
	for pos < len(raw) {
		idx := bytes.IndexByte(raw[pos:], '\n')
		if idx >= 0 {
			line := bytes.TrimSpace(raw[pos : pos+idx])
			pos += idx + 1
			if len(line) > 0 {
				r, err := parseSpentRecord(line)
				if err != nil {
					return nil, fmt.Errorf("corrupt spend log %s line %d: %w", path, lineNum, err)
				}
				sum.addRecord(r)
			}
			lineNum++
			continue
		}
		tailStart := int64(pos)
		tail := bytes.TrimSpace(raw[pos:])
		if len(tail) == 0 {
			sum.incompleteTail = true
			sum.tailOffset = tailStart
			break
		}
		r, err := parseSpentRecord(tail)
		if err == nil {
			sum.addRecord(r)
			sum.needLeadingNewline = true
			break
		}
		if !isIncompleteJSONObjectPrefix(tail, err) {
			return nil, fmt.Errorf("corrupt spend log %s line %d: %w", path, lineNum, err)
		}
		sum.incompleteTail = true
		sum.tailOffset = tailStart
		break
	}
	return sum, nil
}

// readSpentFile reads and strictly validates a spent.jsonl file under a shared flock.
func readSpentFile(path string) (*spentSummary, error) {
	fh, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return &spentSummary{
			byContract:   map[string]podRecord{},
			byPendingID:  map[string]bool{},
			byExaminerID: map[string]bool{},
		}, nil
	}
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	if err := syscall.Flock(int(fh.Fd()), syscall.LOCK_SH); err != nil {
		return nil, err
	}
	defer syscall.Flock(int(fh.Fd()), syscall.LOCK_UN)
	raw, err := io.ReadAll(fh)
	if err != nil {
		return nil, err
	}
	return parseSpentBytes(path, raw)
}

// appendSpendRecord durably appends a terminal podRecord to spent.jsonl under an exclusive flock:
//   - Fails closed on any unrecoverable malformed newline-terminated line in spent.jsonl.
//   - Recovers an unterminated partial trailing fragment by truncating back to the last valid
//     newline and syncing before appending.
//   - Separates lines if the previous valid record lacked a trailing newline.
//   - Deduplicates idempotently by durable contract identity (market:id) or examiner instance ID.
//   - Syncs and closes the file before returning nil so callers only clear live state once durable.
func appendSpendRecord(path string, rec podRecord, dedupByContract bool) (bool, error) {
	line, err := json.Marshal(rec)
	if err != nil {
		return false, err
	}
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return false, err
	}
	fd := int(fh.Fd())
	if err := syscall.Flock(fd, syscall.LOCK_EX); err != nil {
		_ = fh.Close()
		return false, err
	}
	closeFile := func() error {
		if fh == nil {
			return nil
		}
		_ = syscall.Flock(fd, syscall.LOCK_UN)
		cerr := fh.Close()
		fh = nil
		return cerr
	}
	defer func() { _ = closeFile() }()

	raw, err := io.ReadAll(fh)
	if err != nil {
		return false, err
	}
	sum, err := parseSpentBytes(path, raw)
	if err != nil {
		return false, err
	}
	writeOffset := int64(len(raw))
	if sum.incompleteTail {
		if err := fh.Truncate(sum.tailOffset); err != nil {
			return false, err
		}
		if err := fh.Sync(); err != nil {
			return false, err
		}
		writeOffset = sum.tailOffset
	}
	if dedupByContract && rec.ID != "" && sum.hasContract(rec.market(), rec.ID) {
		if !sum.incompleteTail {
			if err := fh.Sync(); err != nil {
				return false, err
			}
		}
		return false, closeFile()
	}
	if !dedupByContract && rec.Market == "vast" && rec.For == "examiner" && rec.ID != "" && sum.byExaminerID[rec.ID] {
		if !sum.incompleteTail {
			if err := fh.Sync(); err != nil {
				return false, err
			}
		}
		return false, closeFile()
	}
	if _, err := fh.Seek(writeOffset, io.SeekStart); err != nil {
		return false, err
	}
	var payload []byte
	if !sum.incompleteTail && sum.needLeadingNewline {
		payload = append(payload, '\n')
	}
	payload = append(payload, line...)
	payload = append(payload, '\n')
	n, err := fh.Write(payload)
	if err != nil || n != len(payload) {
		if err == nil {
			err = io.ErrShortWrite
		}
		_ = fh.Truncate(writeOffset)
		_ = fh.Sync()
		return false, err
	}
	if err := fh.Sync(); err != nil {
		return false, err
	}
	if err := closeFile(); err != nil {
		return false, err
	}
	return true, nil
}

// readSpentSummary parses spent.jsonl once and deduplicates terminal records by durable contract
// identity (market:id), preserving the first confirmed terminal record and end time.
func (f *Fleet) readSpentSummary() (*spentSummary, error) {
	return readSpentFile(f.spentLog())
}

// retire moves a machine that has ended from the record of live ones to the log of what was
// spent, which is only ever added to. Retirement is idempotent per durable contract identity
// (market:id): if a prior attempt already appended the terminal row before a failed fleet.json
// write, retrying preserves the first confirmed end time and clears l.Pods without appending a
// duplicate row. An incomplete trailing fragment from an interrupted write is truncated under lock
// and the complete terminal row is fsynced before l.Pods[key] is cleared.
func (f *Fleet) retire(l *ledger, key string, end time.Time) error {
	r := l.Pods[key]
	if r == nil {
		return nil
	}
	recCopy := *r
	recCopy.TransferCharges = cloneTransferCharges(r.TransferCharges)
	recCopy.DatedTransfer = cloneDatedTransfer(r.DatedTransfer)
	recCopy.Segments = cloneBillingSegments(r.Segments)
	if recCopy.Stopped {
		recCopy.markStopped(end)
	}
	recCopy.End = end
	if _, err := appendSpendRecord(f.spentLog(), recCopy, true); err != nil {
		return err
	}
	r.End = end
	delete(l.Pods, key)
	return nil
}

// spent is about what was spent since a moment: deduplicated ended pods from spent.jsonl, live
// pods not yet retired to spent.jsonl, and unresolved pending worker reservations not already
// retired to spent.jsonl.
func (f *Fleet) spent(l *ledger, since time.Time) (float64, error) {
	return f.spentFor(l, since, nil)
}

// spentFor computes UTC spend since `since`, conservatively charging any stopped ledger contract
// whose current provider status could not be confirmed (`unconfirmed[key]`) at its compute rate
// over the unobserved [StoppedSeen, now] interval without mutating fleet.json on disk.
func (f *Fleet) spentFor(l *ledger, since time.Time, unconfirmed map[string]bool) (float64, error) {
	now := f.now()
	sum, err := f.readSpentSummary()
	if err != nil {
		return 0, err
	}
	total := 0.0
	for _, r := range sum.records {
		rc := r
		if rc.ID != "" && len(f.datedCharges[rc.market()][rc.ID]) > 0 {
			ts := rc.transferState().clone()
			if ts.mergeDatedCharges(f.datedCharges[rc.market()][rc.ID], rc.Start, now) {
				rc.setTransferState(ts)
			}
		}
		total += rc.dollarsBetween(since, now)
	}
	for key, r := range l.Pods {
		if r == nil {
			continue
		}
		if r.ID != "" {
			if _, retired := sum.byContract[ledgerKey(r.market(), r.ID)]; retired {
				continue
			}
		}
		rc := *r
		rc.Segments = cloneBillingSegments(r.Segments)
		if rc.ID != "" && len(f.datedCharges[rc.market()][rc.ID]) > 0 {
			ts := rc.transferState().clone()
			if ts.mergeDatedCharges(f.datedCharges[rc.market()][rc.ID], rc.Start, now) {
				rc.setTransferState(ts)
			}
		}
		if unconfirmed[key] && rc.Stopped {
			rc.markLiveAt(now, time.Time{})
		}
		rc.End = now
		total += rc.dollarsBetweenMode(since, now, false)
	}
	for _, pr := range l.Pending {
		if pr == nil || sum.byPendingID[pr.ID] {
			continue
		}
		total += pr.dollarsBetween(since, now)
	}
	return total, nil
}

func startOfDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

// --- looking -------------------------------------------------------------------------------------

// View is one look at everything: the work, the pool and its machines, and the money.
type View struct {
	Work                   *Work
	Workers                []*Worker
	Ledger                 *ledger
	LiveRentals            []*Rental           // deduplicated live fleet worker contracts (one per unique market/id)
	PendingRents           []*pendingRent      // unresolved pending worker create reservations
	OrphanRentals          []*Rental           // fleet ledger contracts whose key is no longer in f.keys
	UnknownRateKeys        map[string]string   // live contracts / pending reservations with no known positive hourly rate
	UnconfirmedKeys        map[string]bool     // ledger contracts whose current provider status could not be confirmed (outage/lookup error)
	MarketKnownIDs         map[string][]string // all non-gone contract IDs listed per market (including unowned on shared accounts)
	Hourly                 float64             // dollars an hour, every live contract + pending create + active examiner together
	Today                  float64             // about what has been spent today (UTC), including any active examiner
	ExaminerLive           int                 // 1 when an ephemeral examiner GPU is active, else 0
	ExaminerHourly         float64             // active examiner's hourly rate (included in Hourly)
	ExaminerAccrued        float64             // active examiner's accrued spend today (included in Today)
	ExaminerReservedHourly float64             // hourly headroom held back for a required examiner not yet rented
}

func unknownRateLabel(market, id, handle string) string {
	name := handle
	if name == "" {
		name = id
	}
	if market == "" {
		return name
	}
	if name != id && id != "" {
		return fmt.Sprintf("%s (%s on %s)", name, id, marketTitle(market))
	}
	return fmt.Sprintf("%s on %s", name, marketTitle(market))
}

func (v *View) noteUnknownRate(key, label string) {
	if v == nil || key == "" {
		return
	}
	if v.UnknownRateKeys == nil {
		v.UnknownRateKeys = map[string]string{}
	}
	v.UnknownRateKeys[key] = label
}

func (v *View) hourlyUnknown() bool {
	return v != nil && len(v.UnknownRateKeys) > 0
}

func (v *View) unknownRateSummary() string {
	if v == nil || len(v.UnknownRateKeys) == 0 {
		return ""
	}
	labels := make([]string, 0, len(v.UnknownRateKeys))
	for _, lbl := range v.UnknownRateKeys {
		labels = append(labels, lbl)
	}
	sort.Strings(labels)
	return strings.Join(labels, ", ")
}

func (v *View) liveWorkerCount() int {
	if v == nil {
		return 0
	}
	return len(v.LiveRentals) + len(v.PendingRents)
}

func (v *View) totalLiveCount() int {
	if v == nil {
		return 0
	}
	return v.liveWorkerCount() + v.ExaminerLive
}

func (v *View) liveWorkerHourly() float64 {
	if v == nil {
		return 0
	}
	return max(0, v.Hourly-v.ExaminerHourly)
}

// extraWorkerLiveCount is the number of live worker contracts or pending reservations beyond
// each Worker's single primary slot (e.g. duplicate live contracts sharing one handle, multiple
// pending creates on one handle, or live contracts whose key was revoked).
func (v *View) extraWorkerLiveCount() int {
	if v == nil {
		return 0
	}
	primary := 0
	for _, w := range v.Workers {
		if (w.Rental != nil && (w.Rental.live() || w.Rental.Unconfirmed)) || (w.Rental == nil && w.Pending != nil) {
			primary++
		}
	}
	return max(0, v.liveWorkerCount()-primary)
}

func (v *View) addLiveRental(r *Rental) {
	if v == nil || r == nil || !r.live() {
		return
	}
	if r.ID != "dry-run" {
		key := ledgerKey(r.Market, r.ID)
		for _, existing := range v.LiveRentals {
			if existing.ID != "dry-run" && ledgerKey(existing.Market, existing.ID) == key {
				return
			}
		}
	}
	v.LiveRentals = append(v.LiveRentals, r)
}

func (v *View) removeLiveRental(market, id string) {
	if v == nil {
		return
	}
	key := ledgerKey(market, id)
	out := v.LiveRentals[:0]
	for _, existing := range v.LiveRentals {
		if ledgerKey(existing.Market, existing.ID) != key {
			out = append(out, existing)
		}
	}
	v.LiveRentals = out
	delete(v.UnknownRateKeys, key)
	delete(v.UnconfirmedKeys, key)
}

func (f *Fleet) examinerSnapshot() (live int, hourly float64, accrued float64, err error) {
	now := f.now()
	if f.examiner != nil {
		return f.examiner.ActiveSnapshot(now)
	}
	return activeExaminerFromDisk(f.o.Home, now, f.o.Examiner.Timeout, f.o.Examiner.MaxPrice)
}

// seeStart notes how long a machine took to first ask for work, if it was seen before it had.
func (f *Fleet) seeStart(w *Worker) {
	if f.starting == nil {
		f.starting = map[string]bool{}
	}
	id := ledgerKey(w.Rental.Market, w.Rental.ID)
	if w.Machine == nil || w.Machine.LastSeen < w.Rental.Created.Unix() {
		f.starting[id] = true
		return
	}
	if f.starting[id] {
		delete(f.starting, id)
		f.noteBoot(id, time.Unix(w.Machine.LastSeen, 0).Sub(w.Rental.Created))
	}
}

// refresh reads the markets' keys, and which markets are armed, again: a key added or a
// market armed since the last look counts now. A market whose settings are as they were keeps
// its client, and what the client has learned.
func (f *Fleet) refresh() error {
	a, err := readArming(f.o.Home)
	if err != nil {
		return err
	}
	f.arming = a
	if f.fixed {
		return nil
	}
	creds, err := readCredentials(f.o.Credentials)
	if err != nil {
		return err
	}
	had := map[string]Market{}
	for _, m := range f.markets {
		had[m.Name()] = m
	}
	seen := map[string]string{}
	f.markets, f.off = nil, map[string]string{}
	for _, name := range marketOrder {
		sig := name + "|" + f.o.Cloud
		for _, k := range credentialNames {
			if strings.HasPrefix(k, strings.ToUpper(name)+"_") {
				sig += "|" + creds[k]
			}
		}
		m := had[name]
		if m == nil || f.keysSeen[name] != sig {
			var why string
			if m, why = openMarket(f.o, name, creds); m == nil {
				f.off[name] = why
				continue
			}
		}
		f.markets, seen[name] = append(f.markets, m), sig
	}
	f.keysSeen = seen
	return nil
}

// market is the client of a market a key is found for; nil when there is none.
func (f *Fleet) market(name string) Market {
	for _, m := range f.markets {
		if m.Name() == name {
			return m
		}
	}
	return nil
}

// runpod is RunPod's own client, for what only RunPod does here (logs, ssh).
func (f *Fleet) runpod() *RunPod {
	if m, ok := f.market("runpod").(*runpodMarket); ok {
		return m.rp
	}
	return nil
}

// armed says whether machines may be rented on a market.
func (f *Fleet) armed(name string) bool {
	if f.arming == nil {
		return name == "runpod"
	}
	_, ok := f.arming.Armed[name]
	return ok
}

// armedHave is the markets machines may be rented on, as the start of a sentence: "RunPod
// has", "Vast.ai and RunPod have".
func (f *Fleet) armedHave() string {
	var names []string
	for _, m := range f.markets {
		if f.armed(m.Name()) {
			names = append(names, marketTitle(m.Name()))
		}
	}
	switch len(names) {
	case 0:
		return "No market armed has"
	case 1:
		return names[0] + " has"
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1] + " have"
}

// rentals are the machines on every market a key is found for, and which markets answered. A
// market that does not answer is passed over only when nothing may be rented on it and the
// record has nothing there: otherwise its machines could not be told from free keys, and a
// key could be rented twice.
func (f *Fleet) rentals(ctx context.Context, l *ledger) ([]Rental, map[string]bool, error) {
	spentSum, err := f.readSpentSummary()
	if err != nil {
		return nil, nil, err
	}
	recorded := map[string]int{}
	for _, r := range l.Pods {
		if r == nil || spentSum.hasContract(r.market(), r.ID) {
			continue
		}
		recorded[r.market()]++
	}
	for _, pr := range l.Pending {
		if pr != nil && pr.Market != "" && !spentSum.byPendingID[pr.ID] {
			recorded[pr.Market]++
		}
	}
	for _, name := range marketOrder {
		if n := recorded[name]; n > 0 && f.market(name) == nil {
			return nil, nil, fmt.Errorf("%d machines on %s are in the record (%s), but there is no key for it: %s",
				n, marketTitle(name), filepath.Join(f.o.Home, "fleet.json"), f.off[name])
		}
	}
	answered := map[string]bool{}
	var all []Rental
	for _, m := range f.markets {
		rs, err := m.Rentals(ctx)
		if err != nil {
			// Its machines are taken to be as the record has them, so that their keys are not
			// rented again; nothing of it is retired until it answers.
			if recorded[m.Name()] > 0 {
				f.say("%s did not say what it has rented (%v): its %d machines are taken to be as they were", marketTitle(m.Name()), err, recorded[m.Name()])
				all = append(all, recordedRentals(l, m.Name(), spentSum)...)
			}
			continue
		}
		answered[m.Name()] = true
		all = append(all, rs...)
	}
	return all, answered, nil
}

func recordedRental(r *podRecord, status string) Rental {
	state := "running"
	if r.Stopped {
		state = "stopped"
	}
	cost := r.Cost
	if !validRate(cost) {
		cost = 0
	}
	return Rental{
		Market: r.market(), ID: r.ID, Name: r.Handle, Host: r.Host, Card: r.GPU,
		Price: cost, DiskCost: r.DiskCost, DiskCostUnknown: r.DiskCostUnknown, TransferCost: r.TransferCost,
		FixedCost: r.FixedCost, FixedCostEstimated: r.FixedCostEstimated,
		ObservedTransferCost: r.ObservedTransferCost, TransferObserved: r.TransferObserved,
		ObservedTransferDown: r.ObservedTransferDown, TransferDownObserved: r.TransferDownObserved, TransferDownUnpriced: r.TransferDownUnpriced,
		ObservedTransferUp: r.ObservedTransferUp, TransferUpObserved: r.TransferUpObserved, TransferUpUnpriced: r.TransferUpUnpriced,
		ObservedTransferTotal: r.ObservedTransferTotal, TransferTotalObserved: r.TransferTotalObserved, TransferTotalUnpriced: r.TransferTotalUnpriced,
		PartialRate: r.PartialCost, State: state, Offline: r.Stopped && r.Offline, Status: status, Created: r.Start, Terminating: r.Terminating, Unconfirmed: true,
	}
}

// recordedRentals are a market's machines as the record has them, live, excluding contracts
// already durably retired in spent.jsonl.
func recordedRentals(l *ledger, market string, spentSum *spentSummary) []Rental {
	var out []Rental
	for _, r := range l.Pods {
		if r != nil && r.market() == market && !spentSum.hasContract(r.market(), r.ID) {
			out = append(out, recordedRental(r, "as recorded"))
		}
	}
	return out
}

func (f *Fleet) look(ctx context.Context) (*View, error) {
	work, err := f.st.work(ctx, f.op)
	if err != nil {
		return nil, err
	}
	f.onePass = !baselinesWanted(work)
	if work.LeaseSeconds > 0 {
		f.lease = time.Duration(work.LeaseSeconds) * time.Second
	}
	if err := f.refresh(); err != nil {
		return &View{Work: work}, err
	}
	l, err := f.readLedger()
	if err != nil {
		return &View{Work: work}, err
	}
	rentals, answered, err := f.rentals(ctx, l)
	if err != nil {
		return &View{Work: work, Ledger: l}, err
	}
	if rentals, err = f.note(ctx, l, rentals, answered); err != nil {
		return &View{Work: work, Ledger: l}, err
	}
	v := &View{Work: work, Ledger: l, MarketKnownIDs: map[string][]string{}}
	asked := false
	live := map[string]bool{}
	knownHandles := map[string]bool{}
	for _, k := range f.keys {
		knownHandles[k.Handle] = true
	}
	// Record every live, stopped, or gone rental ID seen on each market (including unowned or
	// already-expired contracts on shared accounts) so subsequent worker creates include them in
	// Launch.KnownIDs / pr.KnownIDs.
	marketSeen := map[string]map[string]bool{}
	for i := range rentals {
		p := &rentals[i]
		if p.Market == "" || p.ID == "" || (!p.live() && p.State != "stopped" && p.State != "gone") {
			continue
		}
		if marketSeen[p.Market] == nil {
			marketSeen[p.Market] = map[string]bool{}
		}
		if !marketSeen[p.Market][p.ID] {
			marketSeen[p.Market][p.ID] = true
			v.MarketKnownIDs[p.Market] = append(v.MarketKnownIDs[p.Market], p.ID)
		}
	}
	for mName := range v.MarketKnownIDs {
		sort.Strings(v.MarketKnownIDs[mName])
	}
	// Deduplicate all rentals by exact (market, contract ID) so capacity and hourly spend are
	// counted once per live contract ID even when multiple contracts share a single handle.
	byKey := map[string]*Rental{}
	var orderKeys []string
	for i := range rentals {
		p := &rentals[i]
		key := ledgerKey(p.Market, p.ID)
		prev := byKey[key]
		if prev == nil {
			byKey[key] = p
			orderKeys = append(orderKeys, key)
		} else if betterRental(prev, p) {
			byKey[key] = p
		}
	}
	// Defense-in-depth: ensure any tracked ledger pod absent from rentals is still accounted for.
	podKeys := make([]string, 0, len(l.Pods))
	for key := range l.Pods {
		podKeys = append(podKeys, key)
	}
	sort.Strings(podKeys)
	for _, key := range podKeys {
		if _, ok := byKey[key]; !ok && l.Pods[key] != nil {
			rr := recordedRental(l.Pods[key], "as recorded")
			byKey[key] = &rr
			orderKeys = append(orderKeys, key)
		}
	}
	byHandle := map[string][]*Rental{}
	for _, key := range orderKeys {
		p := byKey[key]
		if !p.live() && p.State != "stopped" {
			continue
		}
		rec := l.Pods[key]
		h := owner(l, p)
		if !knownHandles[h] && rec == nil {
			continue
		}
		if rec != nil {
			reconcileRentalRateAt(rec, p, f.now())
			if rec.Terminating {
				p.Terminating = true
			}
			if p.Host == "" && rec.Host != "" {
				p.Host = rec.Host
			}
		}
		if p.Unconfirmed {
			if v.UnconfirmedKeys == nil {
				v.UnconfirmedKeys = map[string]bool{}
			}
			v.UnconfirmedKeys[key] = true
		}
		if p.live() || p.Unconfirmed {
			// When provider listing/lookup could not confirm a stopped contract's current status,
			// conservatively reserve its slot and running compute rate until status is confirmed.
			v.LiveRentals = append(v.LiveRentals, p)
			if validRate(p.Price) {
				v.Hourly += p.Price
			}
			if !validRate(p.Price) || p.PartialRate {
				v.noteUnknownRate(key, unknownRateLabel(p.Market, p.ID, h))
			}
		} else if p.State == "stopped" {
			if !p.Offline {
				diskRate := p.DiskCost
				if rec != nil && rec.stoppedHourlyRate() > diskRate {
					diskRate = rec.stoppedHourlyRate()
				}
				if validRate(diskRate) {
					v.Hourly += diskRate
				} else if p.DiskCostUnknown || (rec != nil && rec.DiskCostUnknown) {
					v.noteUnknownRate(key, unknownRateLabel(p.Market, p.ID, h))
				}
			}
		}
		if knownHandles[h] {
			byHandle[h] = append(byHandle[h], p)
		} else {
			v.OrphanRentals = append(v.OrphanRentals, p)
		}
	}
	pendingByHandle := map[string][]*pendingRent{}
	if len(l.Pending) > 0 {
		pkeys := make([]string, 0, len(l.Pending))
		for id := range l.Pending {
			pkeys = append(pkeys, id)
		}
		sort.Strings(pkeys)
		for _, id := range pkeys {
			pr := l.Pending[id]
			if pr == nil {
				continue
			}
			v.PendingRents = append(v.PendingRents, pr)
			if validRate(pr.Cost) {
				v.Hourly += pr.Cost
			} else {
				v.noteUnknownRate("pending:"+pr.ID, unknownRateLabel(pr.Market, pr.ID, pr.Handle))
			}
			pendingByHandle[pr.Handle] = append(pendingByHandle[pr.Handle], pr)
		}
	}
	for _, k := range f.keys {
		w := &Worker{Key: k, Machine: work.machine(k.Handle)}
		// Prefer the newest live contract for dispatch (or newest stopped contract when none is
		// live), keeping every additional contract sharing this handle in ExtraRentals so all
		// live contracts count toward capacity/spend and stopped duplicates are cleaned up.
		list := byHandle[k.Handle]
		rentalRank := func(r *Rental) int {
			if (r.State == "running" || r.State == "starting") && !r.Unconfirmed {
				return 3
			}
			if r.Unconfirmed {
				return 2
			}
			if r.State == "unknown" {
				return 1
			}
			return 0
		}
		sort.SliceStable(list, func(i, j int) bool {
			ri, rj := rentalRank(list[i]), rentalRank(list[j])
			if ri != rj {
				return ri > rj
			}
			if list[i].Terminating != list[j].Terminating {
				return !list[i].Terminating
			}
			if !list[i].Created.Equal(list[j].Created) {
				return list[i].Created.After(list[j].Created)
			}
			return list[i].ID > list[j].ID
		})
		if len(list) > 0 {
			w.Rental = list[0]
			if len(list) > 1 {
				w.ExtraRentals = list[1:]
			}
		}
		if ps := pendingByHandle[k.Handle]; len(ps) > 0 {
			w.Pending = ps[0]
		}
		if w.Rental != nil {
			if rec := l.Pods[ledgerKey(w.Rental.Market, w.Rental.ID)]; rec != nil {
				if !validRate(w.Rental.Price) && validRate(rec.Cost) {
					w.Rental.Price = rec.Cost
				}
				w.Plan = rec.Grades
				if !rec.Asked && w.Machine != nil && w.Machine.LastSeen >= w.Rental.Created.Unix() {
					rec.Asked, asked = true, true
					detail := fmt.Sprintf("first asked for work %s after it was rented", time.Unix(w.Machine.LastSeen, 0).Sub(w.Rental.Created).Round(time.Second))
					if w.Machine.Harness != "" {
						detail += ", running stet " + w.Machine.Harness
					}
					f.event(w.Rental, k.Handle, "asked", detail)
				}
				if why, no := w.Machine.refusedCurators(w.Rental.Created); no && rec.Curators {
					w.NoCurators = why
					if rec.Refused == "" {
						rec.Refused, asked = why, true
						f.noCurators(w.Rental, rec, why)
					}
				}
				if !rec.Curators {
					w.GradeOnly = true
				}
			}
			if w.NoCurators != "" || (w.Machine != nil && w.Machine.Curators != nil && !*w.Machine.Curators && w.Machine.LastSeen >= w.Rental.Created.Unix()) {
				w.GradeOnly = true
			}
			w.Grace, w.Speed = f.grace(w.Rental.Market), f.speedOf(w.Rental.Card)
			if m := f.market(w.Rental.Market); m != nil && !m.Caps().SelfEnd {
				w.Silence = quietWithoutSelfEnd
			}
			if f.stillInstalling(ctx, w, f.now()) {
				w.Grace *= 2
			}
		}
		if w.Rental != nil && w.Rental.live() && w.Rental.State != "unknown" {
			f.seeStart(w)
		}
		if window := earlyWindow(work); w.Rental != nil && window > 0 {
			// The latest a machine of its kind could be rented and still start its grades as
			// the server's window opens for it: the most it may hold, trained one after another,
			// ending within the window of the first exam. One waiting for work before then waits
			// only to spend.
			if due := f.firstExamDue(work); !due.IsZero() {
				o := Offer{Market: w.Rental.Market, Card: w.Rental.Card}
				p, _ := f.pass(o.Card)
				chain := f.lead(o) - sitting + time.Duration(max(work.EarlyGradesPerMachine-1, 0))*p
				w.Window = due.Add(-window - chain)
			}
		}
		if w.Rental != nil {
			f.watch(w, f.now())
		}
		for _, p := range list {
			live[ledgerKey(p.Market, p.ID)] = true
		}
		v.Workers = append(v.Workers, w)
	}
	f.forget(live)
	if asked {
		if err := f.writeLedger(l); err != nil {
			return &View{Work: work, Ledger: l}, err
		}
	}
	if v.Today, err = f.spentFor(l, startOfDay(f.now()), v.UnconfirmedKeys); err != nil {
		return &View{Work: work, Ledger: l}, err
	}
	exLive, exHourly, exAccrued, err := f.examinerSnapshot()
	if err != nil {
		return &View{Work: work, Ledger: l}, err
	}
	v.ExaminerLive, v.ExaminerHourly, v.ExaminerAccrued = exLive, exHourly, exAccrued
	v.Hourly += exHourly
	v.Today += exAccrued
	return v, nil
}

// --- renting and ending --------------------------------------------------------------------------

// Not the kinds a grade can be trusted to: those without bf16 (before Ampere), slices of
// one GPU shared with others, a laptop's, and makers other than NVIDIA (grades are taken on
// CUDA).
var unfitGPU = regexp.MustCompile(`(?i)V100|\bT4\b|P100|P40|\bP4\b|K80|M60|Quadro|RTX\s*20[5-9]\d|TITAN|GTX|MIG|Laptop`)

// quietWithoutSelfEnd is how long a machine on a market where it cannot end itself may say
// nothing, holding no job, before it is ended: a stet waiting for work asks far more often.
const quietWithoutSelfEnd = 5 * time.Minute

// need is what every machine rented must have.
func (f *Fleet) need() Need {
	return Need{MinGPUMemGB: f.o.MinGPUMemory, MinCPUs: f.o.MinCPU, MinRAMGB: f.o.MinRAM, DiskGB: f.o.Disk, MinCUDA: f.o.MinCUDA}
}

// graderDisk is the disk a machine that only grades needs, GB: stet and its libraries (~5.4 GB),
// student weights (~1.2 GB), and a round's corpus (~4.8 GB), with headroom for uv cache and
// multiple cached rounds. --disk (default 120 GB) is for machines that run curators too, which
// keep 80 GB free.
const graderDisk = 35

// defaultLease is the assumed server job lease (STET_LEASE) when /admin/work omits lease_seconds,
// and leaseMargin is the headroom kept for corpus download, SFT probe, and exam sitting on top of
// two training passes (baseline + candidate).
const (
	defaultLease = 5 * time.Hour
	leaseMargin  = 30 * time.Minute
)

// curators says whether machines on a market may run curators: on RunPod, and where a probe
// showed stet's container sandbox to hold and curators were let run there (markets.json).
func (f *Fleet) curators(market string) bool {
	if m := f.market(market); m != nil && m.Caps().Curators {
		return true
	}
	if f.arming != nil {
		_, ok := f.arming.Curators[market]
		return ok
	}
	return false
}

// needFor is what a machine on a market must have for its role: grade-only rentals (or markets
// where stet's container sandbox is not known to hold) need only a grader's disk (35 GB) and
// grading VRAM floor; curator rentals need the full curator disk and curator VRAM floor.
func (f *Fleet) needFor(m Market, curators bool) Need {
	n := f.need()
	if !curators || !f.curators(m.Name()) {
		n.DiskGB = min(n.DiskGB, graderDisk)
		n.Curators = false
	} else {
		n.Curators = true
		if f.o.MinCuratorGPUMemory > 0 {
			n.MinGPUMemGB = max(n.MinGPUMemGB, f.o.MinCuratorGPUMemory)
		}
	}
	return n
}

// needOn is the curator-capable market requirement (or grader disk where curators cannot run).
func (f *Fleet) needOn(m Market) Need {
	return f.needFor(m, true)
}

// unfitReason explains why an offer's GPU cannot be trusted with a grade, hold one, or finish a
// two-pass baseline grade inside the server's job lease, or returns "" when it is eligible.
func (f *Fleet) unfitReason(c Offer) string {
	if c.Maker != "NVIDIA" {
		return fmt.Sprintf("non-NVIDIA maker %q", c.Maker)
	}
	if c.MemoryGB < f.o.MinGPUMemory {
		return fmt.Sprintf("%d GB VRAM below --min-gpu-memory %d GB", c.MemoryGB, f.o.MinGPUMemory)
	}
	if unfitGPU.MatchString(c.Model) || unfitGPU.MatchString(c.Card) || unfitGPU.MatchString(c.Name) {
		return "unsupported GPU architecture/class (no native bf16 or unfit for grading)"
	}
	lease := f.lease
	if lease <= 0 {
		lease = defaultLease
	}
	p, _ := f.pass(c.Card)
	if 2*p+leaseMargin > lease {
		return fmt.Sprintf("too slow for %s job lease (estimated %s for two passes + %s margin)",
			lease.Round(time.Minute), (2*p + leaseMargin).Round(time.Minute), leaseMargin.Round(time.Minute))
	}
	return ""
}

// fits says whether an offer's GPU can be trusted with a grade, holds one, and is fast enough to
// finish a two-pass baseline grade inside the server's job lease.
func (f *Fleet) fits(c Offer) bool {
	return f.unfitReason(c) == ""
}

// offerName is an offer's card, with its market where that is not RunPod.
func offerName(o Offer) string {
	if o.Market == "runpod" || o.Market == "" {
		return o.Card
	}
	return o.Card + " on " + marketTitle(o.Market)
}

// choices are what to try, looked up afresh each time machines are rented: every fit kind of
// machine the armed markets have free now (with all, every market a key is found for), the
// fastest first, or with --cheapest the cheapest a grade first (the more trusted first where
// that is the same); or the fit ones among those named with --gpu, in that order, each cheapest
// first. For the organisers' bigger students (long), it is what a student costs on each kind that
// counts, under --long-max-price. With all, a market whose offers cannot be had is passed over and
// why is kept in offerErrs; otherwise that is an error.
func (f *Fleet) choices(ctx context.Context, long, all bool) ([]Offer, error) {
	return f.choicesFor(ctx, long, false, all)
}

func (f *Fleet) choicesFor(ctx context.Context, long, curators, all bool) ([]Offer, error) {
	key := offerKey{long: long, curators: curators}
	if got, ok := f.tickOffers[key]; ok && !all {
		return slices.Clone(got), nil
	}
	out, err := f.choicesNowFor(ctx, long, curators, all)
	if err == nil && !all && f.tickOffers != nil {
		f.tickOffers[key] = slices.Clone(out)
	}
	return out, err
}

// offersWithin is the most a market is waited for to say what it offers, its tries again
// included. Offers are asked before the machines that failed are ended, and what they held
// handed back: in round 10 RunPod's catalog took 4 minutes, four tries of a minute, while a
// machine its host had taken offline held 2 grades.
var offersWithin = 30 * time.Second

func (f *Fleet) choicesNow(ctx context.Context, long, all bool) ([]Offer, error) {
	return f.choicesNowFor(ctx, long, false, all)
}

func (f *Fleet) choicesNowFor(ctx context.Context, long, curators, all bool) ([]Offer, error) {
	if err := f.refresh(); err != nil {
		return nil, err
	}
	ceiling := f.maxPrice(long)
	var offers []Offer
	f.offerErrs = map[string]error{}
	asked := 0
	if f.offerSaid == nil {
		f.offerSaid = map[string]string{}
	}
	defer func() {
		// A market whose offers could not be had is passed over this time, and said once.
		if !all {
			for name, err := range f.offerErrs {
				if msg := fmt.Sprintf("%s's offers could not be had: %v", marketTitle(name), err); msg != f.offerSaid[name] {
					f.say("%s; renting on the other markets", msg)
					f.offerSaid[name] = msg
				}
			}
			for name := range f.offerSaid {
				if f.offerErrs[name] == nil {
					delete(f.offerSaid, name)
				}
			}
		}
	}()
	for _, m := range f.markets {
		if f.o.Market != "" && m.Name() != f.o.Market {
			continue
		}
		if !all && !f.armed(m.Name()) && !(m.Name() == f.o.Market && m.Caps().Rents) {
			continue
		}
		octx, cancel := context.WithTimeout(ctx, offersWithin)
		began := f.now()
		got, err := m.Offers(octx, f.needFor(m, curators || f.probe != ""))
		cancel()
		f.offersSeen(m.Name(), f.now().Sub(began), len(got), err)
		if err != nil {
			f.offerErrs[m.Name()] = err
			continue
		}
		asked++
		for _, o := range got {
			if !validRate(o.Price) {
				continue
			}
			if ceiling <= 0 || o.Price <= ceiling+1e-9 {
				offers = append(offers, o)
			}
		}
	}
	if asked == 0 && len(f.offerErrs) > 0 && !all {
		for _, err := range f.offerErrs {
			return nil, err // no market could say what it has
		}
	}
	rates := f.startRates()
	for i := range offers {
		offers[i].Rate = rates.of(offers[i])
	}
	sortOffers(offers)
	var out []Offer
	if len(f.o.GPUs) > 0 {
		seen := make([]bool, len(offers))
		var rejected []string
		seenRej := map[string]bool{}
		for _, want := range f.o.GPUs {
			for idx, c := range offers {
				if seen[idx] {
					continue
				}
				if strings.EqualFold(c.Model, want) || strings.EqualFold(c.Name, want) || strings.EqualFold(c.Card, want) {
					if reason := f.unfitReason(c); reason != "" {
						rkey := c.Card + ":" + reason
						if !seenRej[rkey] {
							seenRej[rkey] = true
							rejected = append(rejected, fmt.Sprintf("%s (%s)", offerName(c), reason))
						}
						continue
					}
					seen[idx] = true
					out = append(out, c)
				}
			}
		}
		if len(out) == 0 && len(rejected) > 0 && !all {
			f.say("ignoring ineligible --gpu %s: %s", strings.Join(f.o.GPUs, ","), strings.Join(rejected, "; "))
		}
		return out, nil
	}
	for _, c := range offers {
		if f.fits(c) {
			out = append(out, c)
		}
	}
	switch {
	case long:
		byStudent(out, f.speedOf)
	case f.o.Cheapest:
		byGrade(out, f.speedOf)
	default:
		byPace(out, f.speedOf)
	}
	return out, nil
}

// firstExamDue is when the first exam whose grades are wanted before it is due.
func (f *Fleet) firstExamDue(w *Work) time.Time {
	var due time.Time
	for _, r := range w.Rounds {
		if r.GradesEarly > 0 {
			if d := f.o.Policy.examDue(r, f.now()); !d.IsZero() && (due.IsZero() || d.Before(due)) {
				due = d
			}
		}
	}
	return due
}

// examPlan is what forExam knows of each kind: its lead and its pass as learned, and the most
// grades the server lets one machine hold before an exam.
func (f *Fleet) examPlan(w *Work) examPlan {
	pl := examPlan{lead: f.lead, pass: func(c Offer) time.Duration { p, _ := f.pass(c.Card); return p }}
	if w != nil {
		pl.most = w.EarlyGradesPerMachine
		pl.window = earlyWindow(w)
	}
	return pl
}

// earlyWindow is the server's window for early grades: what it says, or, from a server that
// lets a machine hold several (with the window), its 10 minutes when it does not say.
func earlyWindow(w *Work) time.Duration {
	switch {
	case w.EarlyWindowSeconds > 0:
		return time.Duration(w.EarlyWindowSeconds) * time.Second
	case w.EarlyGradesPerMachine > 1:
		return 10 * time.Minute
	}
	return 0
}

// nextGrades is how many grades before an exam a machine rented now for them is taken to take:
// that of the kind forExam would rent now, or, while it waits for a cheaper kind's time, the
// most one may hold. One when the server lets a machine hold only one, or nothing is wanted.
func (f *Fleet) nextGrades(ctx context.Context, w *Work) int {
	pl := f.examPlan(w)
	now := f.now()
	if pl.most <= 1 || f.o.Policy.earlyWanted(w, now) == 0 {
		return 1
	}
	var due time.Time
	for _, r := range w.Rounds {
		if f.o.Policy.early(r, now) > 0 {
			if d := f.o.Policy.examDue(r, now); due.IsZero() || d.Before(due) {
				due = d
			}
		}
	}
	gpus, err := f.choices(ctx, false, false)
	if err != nil || len(gpus) == 0 || due.IsZero() {
		return 1
	}
	left := due.Sub(now)
	if now := forExam(gpus, left, f.o.Settle, f.o.Every, pl); len(now) > 0 {
		return pl.grades(now[0], left)
	}
	return pl.most
}

// launchFor is what a machine is rented with for a worker key.
func (f *Fleet) launchFor(k *Key, site string) Launch {
	env := map[string]string{
		"STET_WORKER_KEY":   base64.StdEncoding.EncodeToString(k.file()),
		"STET_API":          f.st.api,
		"STET_INSTALL":      site + "/install.sh",
		"STET_IDLE_MINUTES": fmt.Sprint(f.o.Idle),
		"STET_WORKER_EDITS": map[bool]string{true: "1", false: "0"}[f.o.Edits],
	}
	if f.o.LongMinutes > 0 {
		env["STET_FARM_MAX_MINUTES"] = fmt.Sprint(f.o.LongMinutes)
	}
	return Launch{Name: k.Handle, Image: f.o.Image, Env: env, Script: workerScript, Need: f.need(), SSHKey: f.o.SSHKey}
}

// launch rents up to n machines on keys that have no pod, in the order chosen, within what
// may be spent an hour. It says what it did as it goes.
func (f *Fleet) launch(ctx context.Context, v *View, n, long, early, runs int, due time.Time) (int, error) {
	return f.launchExcluding(ctx, v, n, long, early, runs, due, nil)
}

func (f *Fleet) launchExcluding(ctx context.Context, v *View, n, long, early, runs int, due time.Time, excludeKeys map[string]bool) (int, error) {
	rented, err := f.launchExcludingRentals(ctx, v, n, long, early, runs, due, excludeKeys)
	return len(rented), err
}

func (f *Fleet) launchExcludingRentals(ctx context.Context, v *View, n, long, early, runs int, due time.Time, excludeKeys map[string]bool) ([]*Rental, error) {
	if n <= 0 {
		return nil, nil
	}
	if v.hourlyUnknown() {
		f.say("not renting another machine: live hourly spend is incomplete (%s has no reported hourly rate yet); waiting for provider rate", v.unknownRateSummary())
		return nil, nil
	}
	site, err := f.st.site(ctx)
	if err != nil {
		return nil, err
	}
	vouched, err := f.st.vouched(ctx, f.op)
	if err != nil {
		return nil, err
	}
	// The machines for the bigger students first, each chosen for what a student costs on
	// it; then those for grades taken before an exam, each kind just in time; then those to
	// run curators, only where they may run; then the rest. Each is the fastest first (or with
	// --cheapest, the cheapest a grade): with the exam posted at the draw (Season 2), a curator
	// machine's grades are wanted as soon as its runs are done, and the first four to take one
	// also measure the round's baseline, so its speed is the round's. In Season 1 it held a
	// grade until the exam and was the cheapest a grade; in Season 2's first round two RTX
	// 5070s rented so set its last 35 minutes, for no less a pass than a 5090's.
	long = min(long, n)
	early = min(early, n-long)
	runs = min(runs, n-long-early)
	grades := n - long - early - runs
	var rented []*Rental
	for _, batch := range []struct {
		n              int
		long, curators bool
		due            time.Time
		what           string
	}{{long, true, false, time.Time{}, "bigger students"}, {early, false, false, due, "early grades"}, {runs, false, true, time.Time{}, "curators"}} {
		got, err := f.launchSomeRentals(ctx, v, batch.n, batch.long, batch.curators, batch.due, site, vouched, batch.what, excludeKeys)
		rented = append(rented, got...)
		if err != nil {
			return rented, err
		}
		if batch.curators && len(got) < batch.n && v.Work != nil {
			// When 24 GB curator GPUs are scarce, pass unfilled curator slots to 12 GB+ graders
			// if grades or farm tasks are actually waiting, rather than leaving those slots idle.
			extraWant := 0
			per := max(f.o.PerWorker, 1)
			gradesWaiting := 0
			for _, r := range v.Work.Rounds {
				gradesWaiting += r.GradesWanted
			}
			extraWant += (gradesWaiting + per - 1) / per
			if f.o.PerFarm > 0 {
				extraWant += (v.Work.FarmWaiting + f.o.PerFarm - 1) / f.o.PerFarm
			}
			if spare := min(batch.n-len(got), max(0, extraWant-grades)); spare > 0 {
				grades += spare
			}
		}
	}
	got, err := f.launchSomeRentals(ctx, v, grades, false, false, time.Time{}, site, vouched, "grades", excludeKeys)
	rented = append(rented, got...)
	return rented, err
}

func (f *Fleet) launchSomeRentals(ctx context.Context, v *View, n int, long, curators bool, due time.Time, site string, vouched map[string]Vouched, what string, excludeKeys map[string]bool) ([]*Rental, error) {
	if n <= 0 {
		return nil, nil
	}
	wantCurators := curators && f.o.Edits && f.probe == ""
	gpus, err := f.choicesFor(ctx, long, wantCurators, false)
	if err != nil {
		return nil, err
	}
	if wantCurators {
		// Machines to run curators, only where stet's container sandbox is known to hold, and
		// not on a host or a CPU that was under the season's floor this week.
		refused, err := f.readRefusals()
		if err != nil {
			f.say("where curators may not run is not known (%v): every host is tried", err)
		}
		var held []Offer
		for _, c := range gpus {
			if f.o.MinCuratorGPUMemory > 0 && c.MemoryGB < f.o.MinCuratorGPUMemory {
				continue
			}
			if _, no := refused.refused(c, f.now()); f.curators(c.Market) && !no {
				held = append(held, c)
			}
		}
		gpus = held
	}
	var grades func(Offer) int // how many grades each machine rented is to take before the exam
	// When each machine rented is expected to report its last grade: its lead, or, rented for
	// grades before an exam, once it has held them until the exam and sat it.
	plan := rentPlan{what: what, round: roundFor(v.Work, what, due), report: func(o Offer) time.Time { return f.now().Add(f.lead(o)) }}
	if !due.IsZero() && len(gpus) > 0 {
		left := due.Sub(f.now())
		pl := f.examPlan(v.Work)
		now := forExam(gpus, left, f.o.Settle, f.o.Every, pl)
		if len(now) == 0 {
			// The cheapest a round is still to come: it is rented in its time.
			msg := fmt.Sprintf("%d graders wanted for the exam due about %s UTC", n, due.UTC().Format("15:04"))
			if next, ok := waitingFor(gpus, left, f.o.Every, pl); ok {
				msg += fmt.Sprintf(": waiting for the cheapest a round, %s at $%.2f an hour (about $%.2f a grade), from %s UTC",
					offerName(next), next.Price, pl.cost(next, pl.full(next)), due.Add(-pl.full(next)).UTC().Format("15:04"))
				if g := pl.grades(next, pl.full(next)); g > 1 {
					msg += fmt.Sprintf(", to take %d grades each", g)
				}
			}
			if msg != f.waitSaid {
				f.say("%s", msg)
				f.waitSaid = msg
			}
			if f.cur != nil {
				f.cur.Plan = msg
			}
			return nil, nil
		}
		gpus = now
		grades = func(o Offer) int { return pl.grades(o, left) }
		plan.report = func(o Offer) time.Time { return f.now().Add(max(left+sitting, pl.lead(o))) }
	}
	if len(gpus) == 0 {
		f.say("%s no GPU free now that fits (%s cloud%s%s); trying again later", f.armedHave(), strings.ToLower(f.o.Cloud), f.ceiling(long), map[bool]string{true: ", --gpu " + strings.Join(f.o.GPUs, ","), false: ""}[len(f.o.GPUs) > 0])
		return nil, nil
	}
	var rented []*Rental
	for _, w := range quietestExcluding(v.Workers, excludeKeys) {
		if len(rented) == n {
			break
		}
		if f.o.DailyBudget > 0 {
			today, err := f.currentSpentToday(v)
			if err != nil {
				return rented, err
			}
			if today >= f.o.DailyBudget {
				f.say("not renting another machine: about $%.2f spent today, and the daily budget is $%.2f", today, f.o.DailyBudget)
				break
			}
		}
		if err := f.ready(ctx, w.Key, vouched); err != nil {
			return rented, err
		}
		pod, err := f.rent(ctx, v, w.Key, gpus, site, grades, plan, wantCurators)
		if pod != nil {
			// Preserve the provider-acknowledged rental ID even if local ledger persistence
			// failed so callers (such as cmdProbe) can still tear it down immediately.
			w.Rental = pod
			if rec := v.Ledger.Pods[ledgerKey(pod.Market, pod.ID)]; rec != nil {
				w.Plan = rec.Grades
			}
			rented = append(rented, pod)
		}
		if v.Ledger != nil {
			for _, pr := range v.Ledger.Pending {
				if pr != nil && pr.Handle == w.Key.Handle {
					w.Pending = pr
					break
				}
			}
		}
		if err != nil {
			return rented, err
		}
		if pod == nil {
			break // nothing fits: no point trying the next key
		}
	}
	return rented, nil
}

// quietest lists the keys no machine is rented with, the one the server has heard from
// longest ago first, and one it has never heard from before any. The server gives each
// account one run of an entry: a machine rented again with the key whose last machine ran
// the round's curators would find nothing it may run, wait out its idle minutes, end, and
// be rented again with the same key, while the curators' second runs waited.
func quietest(ws []*Worker) []*Worker {
	return quietestExcluding(ws, nil)
}

func quietestExcluding(ws []*Worker, exclude map[string]bool) []*Worker {
	var free []*Worker
	for _, w := range ws {
		if w.Rental == nil && w.Pending == nil && !exclude[w.Key.Handle] {
			free = append(free, w)
		}
	}
	seen := func(w *Worker) int64 {
		if w.Machine == nil {
			return 0
		}
		return w.Machine.LastSeen
	}
	sort.SliceStable(free, func(i, j int) bool { return seen(free[i]) < seen(free[j]) })
	return free
}

// ready makes sure the server takes a key as one of the organisers' rented machines: the
// organisers' key vouches for it if it does not count yet, and its account is made. A dry
// run says what it would do.
func (f *Fleet) ready(ctx context.Context, k *Key, vouched map[string]Vouched) error {
	if vouched[k.Public].Counts {
		if f.o.DryRun {
			return nil
		}
	} else if f.o.DryRun {
		f.say("would vouch for %s with %s's key", k.Handle, f.op.Handle)
		return nil
	} else {
		if err := f.st.vouch(ctx, f.op, k); err != nil {
			return fmt.Errorf("vouching for %s: %w", k.Handle, err)
		}
		f.say("vouched for %s with %s's key", k.Handle, f.op.Handle)
	}
	worker, err := f.st.account(ctx, k)
	for tries := 0; statusOf(err) == 409 && strings.Contains(err.Error(), "taken") && tries < 3; tries++ {
		// Somebody else has the handle, and handles are anybody's to take. The grant is
		// for the key, not the name, so the key takes another name.
		old := k.Handle
		if err := f.rename(k); err != nil {
			return err
		}
		f.say("%s is somebody else's handle on the server: this key is %s now", old, k.Handle)
		worker, err = f.st.account(ctx, k)
	}
	if err != nil {
		return fmt.Errorf("%s's account: %w", k.Handle, err)
	}
	if !worker {
		return fmt.Errorf("the Stet server does not take %s as one of the organisers' rented machines, though %s vouched for it", k.Handle, f.op.Handle)
	}
	return nil
}

// rent tries each kind of machine in turn, in the order chosen, until a market gives one.
// nil, nil: none could be had within what may be spent.
// grades, when not nil, is how many grades before an exam a machine of each kind is rented to
// take, kept in the record.
// rentPlan is what a machine is rented for: the work, the round, and when it is expected to
// report its last grade.
type rentPlan struct {
	what, round string
	report      func(Offer) time.Time
}

// roundFor is the round a machine rented for what is for: the one whose exam is due, the one
// with curators to run, or else the first open.
func roundFor(w *Work, what string, due time.Time) string {
	if w == nil || len(w.Rounds) == 0 {
		return ""
	}
	for _, r := range w.Rounds {
		switch {
		case what == "curators" && r.RunsWanted > 0:
			return r.ID
		case what == "early grades" && !due.IsZero() && r.ExamDue == due.Unix():
			return r.ID
		case what == "grades" && r.GradesWanted > 0:
			return r.ID
		}
	}
	return w.Rounds[0].ID
}

// currentSpentToday recomputes live UTC-day spend (spent.jsonl + live/stopped/pending workers in
// v.Ledger + active/pending examiner spend) at f.now() so batch worker creates re-check
// --daily-budget immediately before each paid provider call.
func (f *Fleet) currentSpentToday(v *View) (float64, error) {
	if v == nil || v.Ledger == nil {
		return 0, errors.New("fleet ledger unavailable")
	}
	now := f.now()
	base, err := f.spentFor(v.Ledger, startOfDay(now), v.UnconfirmedKeys)
	if err != nil {
		return 0, fmt.Errorf("reading daily spend: %w", err)
	}
	_, _, exAccrued, err := f.examinerSnapshot()
	if err != nil {
		return 0, fmt.Errorf("reading examiner spend: %w", err)
	}
	total := base + exAccrued
	if f.o.DryRun {
		for _, dr := range v.LiveRentals {
			if dr != nil && dr.ID == "dry-run" && validRate(dr.FixedCost) {
				total += dr.FixedCost
			}
		}
	}
	v.Today = total
	return total, nil
}

func (f *Fleet) rent(ctx context.Context, v *View, k *Key, gpus []Offer, site string, grades func(Offer) int, plan rentPlan, curators bool) (*Rental, error) {
	if v.hourlyUnknown() {
		f.say("not renting another machine: live hourly spend is incomplete (%s has no reported hourly rate yet); waiting for provider rate", v.unknownRateSummary())
		return nil, nil
	}
	if f.o.DailyBudget > 0 {
		today, err := f.currentSpentToday(v)
		if err != nil {
			return nil, err
		}
		if today >= f.o.DailyBudget {
			f.say("not renting another machine: about $%.2f spent today, and the daily budget is $%.2f", today, f.o.DailyBudget)
			return nil, nil
		}
	}
	tried, dearer, skippedDailyTransfer := 0, 0, 0
	failed := map[string]bool{} // markets that definitively rejected/could not be reached, passed over now
	for _, c := range gpus {
		if failed[c.Market] {
			continue
		}
		price := c.Price
		if !validRate(price) {
			continue
		}
		if v.Hourly+v.ExaminerReservedHourly+price > f.o.MaxHourly+1e-9 {
			dearer++
			continue
		}
		m := f.market(c.Market)
		if m == nil || (!f.armed(c.Market) && c.Market != f.o.Market) {
			continue // not to be rented on, whatever it offers (but by hand, with up --market)
		}
		fixed := 0.0
		if nonNegFinite(c.FixedCost) && c.FixedCost > 0 {
			fixed = c.FixedCost
		}
		if f.o.DailyBudget > 0 {
			today, err := f.currentSpentToday(v)
			if err != nil {
				return nil, err
			}
			if today >= f.o.DailyBudget {
				f.say("not renting another machine: about $%.2f spent today, and the daily budget is $%.2f", today, f.o.DailyBudget)
				return nil, nil
			}
			if fixed > 0 && today+fixed > f.o.DailyBudget+1e-9 {
				skippedDailyTransfer++
				continue
			}
		}
		tried++
		if f.o.DryRun {
			f.say("would rent %s on %s (%s cloud, about $%.2f an hour)", k.Handle, offerName(c), c.Trust, price)
			v.Hourly += price
			v.Today += fixed
			dr := &Rental{Market: c.Market, ID: "dry-run", Name: k.Handle, Card: c.Card, State: "starting", Status: "PROVISIONING", Price: price, FixedCost: fixed, Created: f.now()}
			v.addLiveRental(dr)
			return dr, nil
		}
		l := f.launchFor(k, site)
		knownIDs := map[string]bool{}
		var knownSlice []string
		addKnown := func(market, id string) {
			if market == c.Market && id != "" && !knownIDs[id] {
				knownIDs[id] = true
				knownSlice = append(knownSlice, id)
			}
		}
		if v.Ledger != nil {
			for _, rec := range v.Ledger.Pods {
				if rec != nil {
					addKnown(rec.market(), rec.ID)
				}
			}
		}
		for _, lr := range v.LiveRentals {
			if lr != nil {
				addKnown(lr.Market, lr.ID)
			}
		}
		for _, wk := range v.Workers {
			if wk.Rental != nil {
				addKnown(wk.Rental.Market, wk.Rental.ID)
			}
			for _, ex := range wk.ExtraRentals {
				if ex != nil {
					addKnown(ex.Market, ex.ID)
				}
			}
		}
		for _, id := range v.MarketKnownIDs[c.Market] {
			addKnown(c.Market, id)
		}
		sort.Strings(knownSlice)
		l.KnownIDs = knownIDs

		asCurator := curators && f.o.Edits && f.curators(c.Market) && (f.o.MinCuratorGPUMemory <= 0 || c.MemoryGB >= f.o.MinCuratorGPUMemory)
		if f.probe != "" {
			l.Need = f.needFor(m, true)
			l.Env["STET_PROBE"], l.Env["STET_WORKER_EDITS"] = f.probe, "1"
			if l.SSHKey == "" && c.Market != "runpod" {
				l.SSHKey = sshKey(l) // to read what it found, where the market lets a key in
			}
		} else {
			l.Need = f.needFor(m, asCurator)
			if asCurator {
				l.Env["STET_WORKER_EDITS"] = "1"
			} else {
				l.Env["STET_WORKER_EDITS"] = "0"
			}
		}
		cloud := c.Trust
		if c.Market == "runpod" {
			cloud = cloudOf(c)
		}
		plannedGrades := 0
		if grades != nil {
			if g := grades(c); g > 1 {
				plannedGrades = g
			}
		}
		var plannedReport time.Time
		if plan.report != nil {
			plannedReport = plan.report(c)
		}
		if f.o.DailyBudget > 0 {
			today, err := f.currentSpentToday(v)
			if err != nil {
				return nil, err
			}
			if today >= f.o.DailyBudget {
				f.say("not renting another machine: about $%.2f spent today, and the daily budget is $%.2f", today, f.o.DailyBudget)
				return nil, nil
			}
			if fixed > 0 && today+fixed > f.o.DailyBudget+1e-9 {
				tried--
				skippedDailyTransfer++
				continue
			}
		}
		if v.Ledger.Pending == nil {
			v.Ledger.Pending = map[string]*pendingRent{}
		}
		began := f.now()
		pr := &pendingRent{
			ID:                 newID()[:16],
			Market:             c.Market,
			Handle:             k.Handle,
			OfferID:            c.ID,
			GPU:                c.Card,
			Cloud:              cloud,
			Cost:               price,
			DiskCost:           c.DiskCost,
			TransferCost:       c.TransferCost,
			FixedCost:          fixed,
			FixedCostEstimated: c.FixedCostEstimated,
			Created:            began,
			Grades:             plannedGrades,
			Host:               c.Host,
			CPU:                c.CPU,
			Curators:           l.Env["STET_WORKER_EDITS"] == "1",
			For:                plan.what,
			Round:              plan.round,
			Report:             plannedReport,
			KnownIDs:           knownSlice,
		}
		v.Ledger.Pending[pr.ID] = pr
		if err := f.writeLedger(v.Ledger); err != nil {
			delete(v.Ledger.Pending, pr.ID)
			return nil, fmt.Errorf("persisting worker create reservation before rent: %w", err)
		}
		r, err := m.Rent(ctx, c, l)
		if err == nil && (r == nil || r.ID == "") {
			err = fmt.Errorf("%s answered without a rental ID", marketTitle(c.Market))
		}
		try := rentTry{Handle: k.Handle, For: plan.what, Market: c.Market, Offer: c.ID, Card: c.Card, Price: c.Price, MS: f.now().Sub(began).Milliseconds(), Outcome: "rented"}
		switch {
		case errors.Is(err, errNoRoom):
			try.Outcome = "no room: " + firstLine(err.Error())
		case err != nil:
			try.Outcome = firstLine(err.Error())
		default:
			try.ID = r.ID
		}
		f.rentSeen(try)
		if errors.Is(err, errNoRoom) || errors.Is(err, errRentRejected) || errors.Is(err, errRentNotSent) {
			// Definitively did NOT create a contract on the provider: safe to release the
			// reservation and try the next offer or market.
			delete(v.Ledger.Pending, pr.ID)
			if werr := f.writeLedger(v.Ledger); werr != nil {
				return nil, fmt.Errorf("clearing rejected worker create reservation: %w", werr)
			}
			if errors.Is(err, errNoRoom) {
				continue
			}
			f.say("renting %s on %s did not work: %v; trying the other markets", k.Handle, marketTitle(c.Market), err)
			failed[c.Market] = true
			continue
		}
		if err != nil {
			// Ambiguous / unknown create outcome (e.g. incomplete 2xx without contract ID, 5xx,
			// timeout, or lost response): keep the durable reservation in fleet.json, charge its
			// slot and hourly rate against this View, and NEVER fall through to another market.
			pr.Error = firstLine(err.Error())
			v.PendingRents = append(v.PendingRents, pr)
			v.Hourly += pr.Cost
			werr := f.writeLedger(v.Ledger)
			f.say("renting %s on %s had an uncertain outcome (%v); keeping slot and $%.2f/h reserved until reconciled (run `stet-cloud workers resolve` after checking %s)",
				k.Handle, marketTitle(c.Market), err, pr.Cost, marketTitle(c.Market))
			if werr != nil {
				return nil, fmt.Errorf("uncertain create for %s on %s (%v); also failed updating reservation: %w", k.Handle, marketTitle(c.Market), err, werr)
			}
			return nil, fmt.Errorf("uncertain create for %s on %s (slot and $%.2f/h reserved until reconciled): %w", k.Handle, marketTitle(c.Market), pr.Cost, err)
		}
		delete(v.Ledger.Pending, pr.ID)
		r.PendingID = pr.ID
		applyOfferRateToRental(c, r)
		if !validRate(r.Price) {
			r.Price = price
		}
		v.Hourly += r.Price
		v.addLiveRental(r)
		fixedEst := c.FixedCostEstimated || r.FixedCostEstimated
		rentNow := f.now()
		rec := &podRecord{ID: r.ID, Market: r.Market, Handle: k.Handle, GPU: c.Card, Cloud: cloud,
			Cost: r.Price, DiskCost: max(c.DiskCost, r.DiskCost), TransferCost: c.TransferCost,
			FixedCost: max(fixed, r.FixedCost), FixedCostEstimated: fixedEst,
			Start: began, Alive: rentNow, Tracked: true, PendingID: pr.ID,
			Host: c.Host, CPU: c.CPU, Curators: l.Env["STET_WORKER_EDITS"] == "1", For: plan.what, Round: plan.round, Report: plannedReport}
		ts := rec.transferState()
		ts.mergeReading(rentalTransferReading(r), rec.FixedCostEstimated, r.State, rec.Start, rentNow)
		rec.setTransferState(ts)
		// Seen from its start: a machine that asks for work before the next look still has its
		// boot counted (calibrate.go). Lium's first, at 41 s, never was.
		if f.starting == nil {
			f.starting = map[string]bool{}
		}
		f.starting[ledgerKey(r.Market, r.ID)] = true
		gradesSaid := ""
		if plannedGrades > 1 {
			rec.Grades, gradesSaid = plannedGrades, fmt.Sprintf(", to take %d grades before the exam", plannedGrades)
		}
		v.Ledger.Pods[ledgerKey(r.Market, r.ID)] = rec
		detail := fmt.Sprintf("%s, %s, $%.2f an hour, from offer %s%s; for %s", c.Card, c.Trust, r.Price, c.ID, gradesSaid, plan.what)
		if plan.round != "" {
			detail += " of " + plan.round
		}
		if !rec.Report.IsZero() {
			detail += fmt.Sprintf("; expected to report by %s UTC", rec.Report.UTC().Format("15:04"))
		}
		if c.Region != "" || c.CPU != "" {
			detail += fmt.Sprintf("; in %s on %s", c.Region, c.CPU)
		}
		f.event(r, k.Handle, "rented", detail)
		if err := f.writeLedger(v.Ledger); err != nil {
			return r, err
		}
		if c.Market == "runpod" {
			f.say("rented %s: %s, %s cloud, $%.2f an hour (pod %s)%s", k.Handle, c.Card, c.Trust, r.Price, r.ID, gradesSaid)
		} else {
			f.say("rented %s on %s: %s, %s, $%.2f an hour (machine %s)%s", k.Handle, marketTitle(c.Market), c.Card, c.Trust, r.Price, r.ID, gradesSaid)
		}
		return r, nil
	}
	switch {
	case tried == 0 && skippedDailyTransfer > 0:
		f.say("not renting another machine: %d available offer(s) had startup transfer/order charges exceeding remaining --daily-budget $%.2f (about $%.2f spent today)", skippedDailyTransfer, f.o.DailyBudget, v.Today)
	case tried == 0 && dearer > 0 && v.ExaminerReservedHourly > 0:
		f.say("not renting another machine: at $%.2f an hour or more (plus $%.2f/h reserved for the round's examiner) it would pass --max-hourly $%.2f", v.Hourly+cheapest(gpus), v.ExaminerReservedHourly, f.o.MaxHourly)
	case tried == 0 && dearer > 0:
		f.say("not renting another machine: at $%.2f an hour or more it would pass --max-hourly $%.2f", v.Hourly+cheapest(gpus), f.o.MaxHourly)
	case tried == 0:
		f.say("%s nothing that may be rented for %s just now; trying again later", f.armedHave(), k.Handle)
	case dearer > 0:
		f.say("%s none of the %d kinds of GPU tried free for %s just now, and the other %d would pass --max-hourly $%.2f; trying again later", f.armedHave(), tried, k.Handle, dearer, f.o.MaxHourly)
	default:
		f.say("%s none of the %d kinds of GPU tried free for %s just now; trying again later", f.armedHave(), tried, k.Handle)
	}
	return nil, nil
}

func cheapest(gpus []Offer) float64 {
	low := math.Inf(1)
	for _, c := range gpus {
		low = min(low, c.Price)
	}
	return low
}

// end ends machines and says why.
func (f *Fleet) end(ctx context.Context, v *View, endings []Ending) error {
	for _, e := range endings {
		r := e.Rental
		if r == nil && e.Worker != nil {
			r = e.Worker.Rental
		}
		if r == nil {
			continue
		}
		handle := r.Name
		if e.Worker != nil && e.Worker.Key != nil {
			handle = e.Worker.Key.Handle
		}
		if handle == "" {
			handle = r.ID
		}
		if f.o.DryRun {
			f.say("would end %s: %s", handle, e.Why)
			continue
		}
		m := f.market(r.Market)
		if m == nil {
			f.say("could not end %s: there is no key for %s now", handle, marketTitle(r.Market))
			continue
		}
		why, cause := e.Why, ""
		if e.Failed && e.Worker != nil && e.Worker.Rental == r {
			// What became of it is asked before it is gone: its market's word, and its log.
			var said string
			cause, said = f.diagnose(ctx, e.Worker)
			why += ": " + said
		}
		key := ledgerKey(r.Market, r.ID)
		rec := v.Ledger.Pods[key]
		preHourlyRate := 0.0
		switch {
		case r.live() || r.Unconfirmed:
			if validRate(r.Price) {
				preHourlyRate = r.Price
			}
		case r.State == "stopped" && !r.Offline:
			preHourlyRate = r.DiskCost
			if rec != nil && rec.stoppedHourlyRate() > preHourlyRate {
				preHourlyRate = rec.stoppedHourlyRate()
			}
		}
		unconfirmedStopped := rec != nil && rec.Stopped && (r.Unconfirmed || r.Status == "as recorded" || r.Status == "unconfirmed (omitted from listing)")
		if unconfirmedStopped {
			if lu, ok := m.(rentalLookup); ok {
				if fresh, exists, lerr := lu.Lookup(ctx, r.ID); lerr == nil && exists && fresh != nil {
					now := f.now()
					reconcileRentalRateAt(rec, fresh, now)
					if fresh.live() {
						rec.markLiveAt(now, fresh.ResumedAt)
						unconfirmedStopped = false
						if fresh.State != "unknown" && !e.Forced && !r.Terminating && !rec.Terminating {
							r.State = fresh.State
							r.Unconfirmed = false
							delete(v.UnconfirmedKeys, key)
							v.addLiveRental(r)
							if validRate(fresh.Price) {
								r.Price = fresh.Price
								r.PartialRate = fresh.PartialRate
								if !fresh.PartialRate {
									delete(v.UnknownRateKeys, key)
								}
								v.Hourly = max(0, v.Hourly-preHourlyRate) + fresh.Price
							}
							f.say("left %s on %s: provider confirmed it resumed running", handle, marketTitle(r.Market))
							continue
						}
					} else if fresh.State == "stopped" {
						rec.markStoppedStateAt(now, fresh.StoppedAt, fresh.Offline)
						unconfirmedStopped = false
					}
				}
			}
		}
		if err := m.End(ctx, r.ID); err != nil {
			// Said, and tried again at the next look; ledger entry is retained until deleted.
			f.say("could not end %s on %s: %v", handle, marketTitle(r.Market), err)
			continue
		}
		v.removeLiveRental(r.Market, r.ID)
		if validRate(preHourlyRate) {
			v.Hourly = max(0, v.Hourly-preHourlyRate)
		}
		endAt := f.now()
		if unconfirmedStopped && rec != nil && rec.Stopped {
			rec.markLiveAt(endAt, time.Time{})
		}
		if rec != nil {
			rec.Why, rec.Cause = why, cause
		}
		f.event(r, handle, "ended", why)
		if err := f.retire(v.Ledger, key, endAt); err != nil {
			return err
		}
		f.say("ended %s: %s", handle, why)
		if e.Worker != nil {
			if e.Worker.Rental == r {
				if len(e.Worker.ExtraRentals) > 0 {
					e.Worker.Rental = e.Worker.ExtraRentals[0]
					e.Worker.ExtraRentals = e.Worker.ExtraRentals[1:]
				} else {
					e.Worker.Rental = nil
				}
			} else {
				kept := e.Worker.ExtraRentals[:0]
				for _, ex := range e.Worker.ExtraRentals {
					if ex != r {
						kept = append(kept, ex)
					}
				}
				e.Worker.ExtraRentals = kept
			}
			// What it held is handed back only once no other contract remains on that key.
			if e.Worker.Rental == nil {
				f.release(ctx, []*Worker{e.Worker})
			}
		}
	}
	return f.writeLedger(v.Ledger)
}

// release hands back the jobs of machines that are gone. It is said, not stopped for, when it
// cannot be done: the job's lease runs out in the end anyway, and the next look tries again.
func (f *Fleet) release(ctx context.Context, gone []*Worker) {
	for _, wk := range gone {
		if f.o.DryRun {
			f.say("would hand back what %s holds: it is not rented any more", wk.Key.Handle)
			continue
		}
		jobs, students, err := f.st.release(ctx, f.op, wk.Key.Public)
		switch {
		case err != nil:
			f.say("could not hand back what %s holds, though it is not rented any more: %v", wk.Key.Handle, err)
		case jobs+students > 0:
			f.say("handed back what %s held when it was ended: %s", wk.Key.Handle, held(jobs, students))
		}
	}
}

func held(jobs, students int) string {
	var parts []string
	switch jobs {
	case 0:
	case 1:
		parts = append(parts, "1 job")
	default:
		parts = append(parts, fmt.Sprintf("%d jobs", jobs))
	}
	switch students {
	case 0:
	case 1:
		parts = append(parts, "1 of the organisers' students")
	default:
		parts = append(parts, fmt.Sprintf("%d of the organisers' students", students))
	}
	return strings.Join(parts, " and ")
}

func (f *Fleet) say(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	f.sayMu.Lock()
	defer f.sayMu.Unlock()
	fmt.Fprintf(f.out, "%s  %s\n", f.now().UTC().Format("15:04:05"), msg)
	if f.cur != nil {
		f.cur.Said = append(f.cur.Said, msg)
	}
}
