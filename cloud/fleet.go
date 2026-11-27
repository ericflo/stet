package main

import (
	"bufio"
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
	f.keys = nil
	for _, p := range paths {
		k, err := loadKey(p)
		if err != nil {
			return err
		}
		if k.Handle == "" {
			k.Handle = strings.TrimSuffix(filepath.Base(p), ".json")
		}
		f.keys = append(f.keys, k)
	}
	sort.Slice(f.keys, func(i, j int) bool { return f.keys[i].Handle < f.keys[j].Handle })
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
	For    string    `json:"for,omitempty"`
	Round  string    `json:"round,omitempty"`
	Report time.Time `json:"report,omitzero"`
	Handle string    `json:"handle"`
	GPU    string    `json:"gpu"`
	Cloud  string    `json:"cloud"`
	Cost   float64   `json:"cost"`
	Start  time.Time `json:"start"`
	Alive  time.Time `json:"alive"` // last seen running
	End    time.Time `json:"end,omitzero"`
}

func (r *podRecord) market() string {
	if r.Market == "" {
		return "runpod"
	}
	return r.Market
}

func (r podRecord) dollarsBetween(from, to time.Time) float64 {
	end := r.End
	if end.IsZero() {
		end = r.Alive
	}
	a, b := maxTime(r.Start, from), minTime(end, to)
	if !b.After(a) {
		return 0
	}
	return b.Sub(a).Hours() * r.Cost
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

type ledger struct {
	Pods map[string]*podRecord `json:"pods"`
}

func (f *Fleet) readLedger() (*ledger, error) {
	l := &ledger{Pods: map[string]*podRecord{}}
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

// note brings the record up to date with the machines the markets list now: the live ones
// were alive now, and those a market that answered no longer lists are gone.
func (f *Fleet) note(l *ledger, rentals []Rental, answered map[string]bool) error {
	now := f.now()
	listed := map[string]*Rental{}
	for i := range rentals {
		listed[ledgerKey(rentals[i].Market, rentals[i].ID)] = &rentals[i]
	}
	for _, k := range f.keys {
		for i := range rentals {
			p := &rentals[i]
			if owner(l, p) != k.Handle || !p.live() {
				continue
			}
			key := ledgerKey(p.Market, p.ID)
			r := l.Pods[key]
			if r == nil { // rented by another copy of this tool, or by hand
				r = &podRecord{ID: p.ID, Market: p.Market, Handle: k.Handle, GPU: p.Card, Cost: p.Price, Start: p.Created, Tracked: true}
				l.Pods[key] = r
			}
			r.Alive, r.Cost = now, p.Price
		}
	}
	for key, r := range l.Pods {
		if !answered[r.market()] {
			continue
		}
		if p := listed[key]; p == nil || !p.live() {
			r.Gone = true
			gone := Rental{Market: r.market(), ID: r.ID, Card: r.GPU}
			why := "its market no longer lists it"
			if p != nil {
				gone = *p
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
			if err := f.retire(l, key, minTime(now, r.Alive.Add(2*f.o.Every))); err != nil {
				return err
			}
		}
	}
	return f.writeLedger(l)
}

// retire moves a machine that has ended from the record of live ones to the log of what was
// spent, which is only ever added to.
func (f *Fleet) retire(l *ledger, key string, end time.Time) error {
	r := l.Pods[key]
	if r == nil {
		return nil
	}
	r.End = end
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	fh, err := os.OpenFile(f.spentLog(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer fh.Close()
	if _, err := fh.Write(append(line, '\n')); err != nil {
		return err
	}
	delete(l.Pods, key)
	return nil
}

// spent is about what was spent since a moment: ended pods from the log, and live ones.
func (f *Fleet) spent(l *ledger, since time.Time) (float64, error) {
	now := f.now()
	total := 0.0
	for _, r := range l.Pods {
		r := *r
		r.End = now
		total += r.dollarsBetween(since, now)
	}
	fh, err := os.Open(f.spentLog())
	if errors.Is(err, os.ErrNotExist) {
		return total, nil
	}
	if err != nil {
		return 0, err
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		var r podRecord
		if json.Unmarshal(sc.Bytes(), &r) == nil {
			total += r.dollarsBetween(since, now)
		}
	}
	return total, sc.Err()
}

func startOfDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

// --- looking -------------------------------------------------------------------------------------

// View is one look at everything: the work, the pool and its machines, and the money.
type View struct {
	Work            *Work
	Workers         []*Worker
	Ledger          *ledger
	Hourly          float64 // dollars an hour, every live machine (including any active examiner) together
	Today           float64 // about what has been spent today (UTC), including any active examiner
	ExaminerLive    int     // 1 when an ephemeral examiner GPU is active, else 0
	ExaminerHourly  float64 // active examiner's hourly rate (included in Hourly)
	ExaminerAccrued float64 // active examiner's accrued spend today (included in Today)
}

func (f *Fleet) examinerSnapshot() (live int, hourly float64, accrued float64) {
	now := f.now()
	if f.examiner != nil {
		return f.examiner.ActiveSnapshot(now)
	}
	return activeExaminerFromDisk(f.o.Home, now, f.o.Examiner.Timeout)
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
	recorded := map[string]int{}
	for _, r := range l.Pods {
		recorded[r.market()]++
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
				all = append(all, recordedRentals(l, m.Name())...)
			}
			continue
		}
		answered[m.Name()] = true
		all = append(all, rs...)
	}
	return all, answered, nil
}

// recordedRentals are a market's machines as the record has them, live.
func recordedRentals(l *ledger, market string) []Rental {
	var out []Rental
	for _, r := range l.Pods {
		if r.market() == market {
			out = append(out, Rental{Market: market, ID: r.ID, Name: r.Handle, Card: r.GPU, Price: r.Cost, State: "running", Status: "as recorded", Created: r.Start})
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
		return nil, err
	}
	l, err := f.readLedger()
	if err != nil {
		return nil, err
	}
	rentals, answered, err := f.rentals(ctx, l)
	if err != nil {
		return nil, err
	}
	if err := f.note(l, rentals, answered); err != nil {
		return nil, err
	}
	v := &View{Work: work, Ledger: l}
	asked := false
	live := map[string]bool{}
	for _, k := range f.keys {
		w := &Worker{Key: k, Machine: work.machine(k.Handle)}
		// The newest machine rented with that key. Ended ones are not listed at all; one that
		// stopped is, and is the one to end.
		for i := range rentals {
			p := &rentals[i]
			if owner(l, p) == k.Handle && (w.Rental == nil || p.Created.After(w.Rental.Created)) {
				w.Rental = p
			}
		}
		if w.Rental != nil {
			if rec := l.Pods[ledgerKey(w.Rental.Market, w.Rental.ID)]; rec != nil {
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
		if w.Rental != nil && w.Rental.live() {
			v.Hourly += w.Rental.Price
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
			live[ledgerKey(w.Rental.Market, w.Rental.ID)] = true
		}
		v.Workers = append(v.Workers, w)
	}
	f.forget(live)
	if asked {
		if err := f.writeLedger(l); err != nil {
			return nil, err
		}
	}
	if v.Today, err = f.spent(l, startOfDay(f.now())); err != nil {
		return nil, err
	}
	exLive, exHourly, exAccrued := f.examinerSnapshot()
	v.ExaminerLive, v.ExaminerHourly, v.ExaminerAccrued = exLive, exHourly, exAccrued
	v.Hourly += exHourly
	v.Today += exAccrued
	return v, nil
}

// --- renting and ending --------------------------------------------------------------------------

// Not the kinds a grade can be trusted to: those without bf16 (before Ampere), slices of
// one GPU shared with others, a laptop's, and makers other than NVIDIA (grades are taken on
// CUDA).
var unfitGPU = regexp.MustCompile(`(?i)V100|\bT4\b|P100|P40|\bP4\b|K80|M60|Quadro|RTX 20|TITAN|GTX|MIG|Laptop`)

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
	} else if f.o.MinCuratorGPUMemory > 0 {
		n.MinGPUMemGB = max(n.MinGPUMemGB, f.o.MinCuratorGPUMemory)
	}
	return n
}

// needOn is the curator-capable market requirement (or grader disk where curators cannot run).
func (f *Fleet) needOn(m Market) Need {
	return f.needFor(m, true)
}

// fits says whether an offer's GPU can be trusted with a grade, holds one, and is fast enough to
// finish a two-pass baseline grade inside the server's job lease.
func (f *Fleet) fits(c Offer) bool {
	if c.Maker != "NVIDIA" || c.MemoryGB < f.o.MinGPUMemory || unfitGPU.MatchString(c.Model) {
		return false
	}
	lease := f.lease
	if lease <= 0 {
		lease = defaultLease
	}
	p, _ := f.pass(c.Card)
	return 2*p+leaseMargin <= lease
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
// that is the same); or those named with --gpu, in that order, each cheapest first. For the
// organisers' bigger students (long), it is what a student costs on each kind that counts,
// under --long-max-price. With all, a market whose offers cannot be had is passed over and
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
		for _, want := range f.o.GPUs {
			for _, c := range offers {
				if strings.EqualFold(c.Model, want) || strings.EqualFold(c.Name, want) || strings.EqualFold(c.Card, want) {
					out = append(out, c)
				}
			}
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
	if n <= 0 {
		return 0, nil
	}
	site, err := f.st.site(ctx)
	if err != nil {
		return 0, err
	}
	vouched, err := f.st.vouched(ctx, f.op)
	if err != nil {
		return 0, err
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
	rented := 0
	for _, batch := range []struct {
		n              int
		long, curators bool
		due            time.Time
		what           string
	}{{long, true, false, time.Time{}, "bigger students"}, {early, false, false, due, "early grades"}, {runs, false, true, time.Time{}, "curators"}} {
		got, err := f.launchSome(ctx, v, batch.n, batch.long, batch.curators, batch.due, site, vouched, batch.what, excludeKeys)
		rented += got
		if err != nil {
			return rented, err
		}
		if batch.curators && got < batch.n && v.Work != nil {
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
			if spare := min(batch.n-got, max(0, extraWant-grades)); spare > 0 {
				grades += spare
			}
		}
	}
	got, err := f.launchSome(ctx, v, grades, false, false, time.Time{}, site, vouched, "grades", excludeKeys)
	rented += got
	return rented, err
}

func (f *Fleet) launchSome(ctx context.Context, v *View, n int, long, curators bool, due time.Time, site string, vouched map[string]Vouched, what string, excludeKeys map[string]bool) (int, error) {
	if n <= 0 {
		return 0, nil
	}
	wantCurators := curators && f.o.Edits && f.probe == ""
	gpus, err := f.choicesFor(ctx, long, wantCurators, false)
	if err != nil {
		return 0, err
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
			return 0, nil
		}
		gpus = now
		grades = func(o Offer) int { return pl.grades(o, left) }
		plan.report = func(o Offer) time.Time { return f.now().Add(max(left+sitting, pl.lead(o))) }
	}
	if len(gpus) == 0 {
		f.say("%s no GPU free now that fits (%s cloud%s%s); trying again later", f.armedHave(), strings.ToLower(f.o.Cloud), f.ceiling(long), map[bool]string{true: ", --gpu " + strings.Join(f.o.GPUs, ","), false: ""}[len(f.o.GPUs) > 0])
		return 0, nil
	}
	rented := 0
	for _, w := range quietestExcluding(v.Workers, excludeKeys) {
		if rented == n {
			break
		}
		if err := f.ready(ctx, w.Key, vouched); err != nil {
			return rented, err
		}
		pod, err := f.rent(ctx, v, w.Key, gpus, site, grades, plan, wantCurators)
		if err != nil {
			return rented, err
		}
		if pod == nil {
			break // nothing fits: no point trying the next key
		}
		w.Rental = pod
		if rec := v.Ledger.Pods[ledgerKey(pod.Market, pod.ID)]; rec != nil {
			w.Plan = rec.Grades
		}
		rented++
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
		if w.Rental == nil && !exclude[w.Key.Handle] {
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

func (f *Fleet) rent(ctx context.Context, v *View, k *Key, gpus []Offer, site string, grades func(Offer) int, plan rentPlan, curators bool) (*Rental, error) {
	tried, dearer := 0, 0
	failed := map[string]bool{} // markets that failed otherwise than with no room, passed over now
	for _, c := range gpus {
		if failed[c.Market] {
			continue
		}
		price := c.Price
		if v.Hourly+price > f.o.MaxHourly+1e-9 {
			dearer++
			continue
		}
		m := f.market(c.Market)
		if m == nil || (!f.armed(c.Market) && c.Market != f.o.Market) {
			continue // not to be rented on, whatever it offers (but by hand, with up --market)
		}
		tried++
		if f.o.DryRun {
			f.say("would rent %s on %s (%s cloud, about $%.2f an hour)", k.Handle, offerName(c), c.Trust, price)
			v.Hourly += price
			return &Rental{Market: c.Market, ID: "dry-run", Name: k.Handle, Card: c.Card, State: "starting", Status: "PROVISIONING", Price: price, Created: f.now()}, nil
		}
		l := f.launchFor(k, site)
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
		began := f.now()
		r, err := m.Rent(ctx, c, l)
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
		if errors.Is(err, errNoRoom) {
			continue
		}
		if err != nil {
			f.say("renting %s on %s did not work: %v; trying the other markets", k.Handle, marketTitle(c.Market), err)
			failed[c.Market] = true
			continue
		}
		v.Hourly += r.Price
		cloud := c.Trust
		if c.Market == "runpod" {
			cloud = cloudOf(c)
		}
		rec := &podRecord{ID: r.ID, Market: r.Market, Handle: k.Handle, GPU: c.Card, Cloud: cloud, Cost: r.Price, Start: f.now(), Alive: f.now(), Tracked: true,
			Host: c.Host, CPU: c.CPU, Curators: l.Env["STET_WORKER_EDITS"] == "1", For: plan.what, Round: plan.round}
		if plan.report != nil {
			rec.Report = plan.report(c)
		}
		// Seen from its start: a machine that asks for work before the next look still has its
		// boot counted (calibrate.go). Lium's first, at 41 s, never was.
		if f.starting == nil {
			f.starting = map[string]bool{}
		}
		f.starting[ledgerKey(r.Market, r.ID)] = true
		gradesSaid := ""
		if grades != nil {
			if g := grades(c); g > 1 {
				rec.Grades, gradesSaid = g, fmt.Sprintf(", to take %d grades before the exam", g)
			}
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
		r := e.Worker.Rental
		if r == nil {
			continue
		}
		if f.o.DryRun {
			f.say("would end %s: %s", e.Worker.Key.Handle, e.Why)
			continue
		}
		m := f.market(r.Market)
		if m == nil {
			f.say("could not end %s: there is no key for %s now", e.Worker.Key.Handle, marketTitle(r.Market))
			continue
		}
		why, cause := e.Why, ""
		if e.Failed {
			// What became of it is asked before it is gone: its market's word, and its log.
			var said string
			cause, said = f.diagnose(ctx, e.Worker)
			why += ": " + said
		}
		if err := m.End(ctx, r.ID); err != nil {
			// Said, and tried again at the next look; the others are ended meanwhile.
			f.say("could not end %s on %s: %v", e.Worker.Key.Handle, marketTitle(r.Market), err)
			continue
		}
		if r.live() {
			v.Hourly -= r.Price
		}
		key := ledgerKey(r.Market, r.ID)
		if rec := v.Ledger.Pods[key]; rec != nil {
			rec.Why, rec.Cause = why, cause
		}
		f.event(r, e.Worker.Key.Handle, "ended", why)
		if err := f.retire(v.Ledger, key, f.now()); err != nil {
			return err
		}
		f.say("ended %s: %s", e.Worker.Key.Handle, why)
		e.Worker.Rental = nil
		// What it held is handed back now: a grade held before an exam is kept for its
		// machine until then, and an ended machine would keep it from anybody else.
		f.release(ctx, []*Worker{e.Worker})
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
