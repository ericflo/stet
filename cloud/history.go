package main

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"
)

// --- what the fleet did, a look at a time ---------------------------------------------------------
//
// So that what was rented, what it cost, how long each market took to answer and where the work
// waited can be seen later, across every market, the fleet keeps:
//
//	looks.jsonl     one line a look: the work the server said was waiting, the machines in each
//	                state and on each market, what was spent an hour and today, how many machines
//	                the work wanted, were rented and were ended, everything the fleet said, how
//	                long each market took to say what it offers (and what went wrong), and each
//	                machine it tried to rent, how long that took, and what came of it.
//	balances.jsonl  each market's account, once an hour: what it really spent is what that went
//	                down by, less what was put in.
//
// Each machine's own story is in machines.jsonl (seen.go), with the round and the work it was
// rented for; what it cost, in spent.jsonl.

type lookRecord struct {
	T        time.Time      `json:"t"`
	TookMS   int64          `json:"took_ms"`
	StetMS   int64          `json:"stet_ms,omitempty"` // how long the server took to say what is waiting
	Rounds   []lookRound    `json:"rounds,omitempty"`
	Runners  int            `json:"runners"`
	Paused   bool           `json:"paused,omitempty"`
	NextDraw int64          `json:"next_draw,omitempty"`
	Farm     int            `json:"farm,omitempty"`
	States   map[string]int `json:"states,omitempty"`    // machines rented, by what they do
	Live     map[string]int `json:"by_market,omitempty"` // machines rented, by market
	Hourly   float64        `json:"hourly"`              // dollars an hour, every machine rented
	Today    float64        `json:"today"`               // about what was spent today (UTC)
	Target   int            `json:"target"`              // machines the work wanted
	Launch   int            `json:"launch"`              // of those, to rent now
	Rented   int            `json:"rented"`              // rented
	Ended    int            `json:"ended"`               // ended
	Note     string         `json:"note,omitempty"`      // why fewer were to be rented than wanted
	// Of Launch: for the bigger students, for grades before the exam due at ExamDue, and to
	// run curators; the rest for grades of a posted exam, or students.
	LaunchLong  int                    `json:"launch_long,omitempty"`
	LaunchEarly int                    `json:"launch_early,omitempty"`
	LaunchRuns  int                    `json:"launch_runs,omitempty"`
	ExamDue     int64                  `json:"exam_due,omitempty"`
	Next        int                    `json:"next_grades,omitempty"` // grades a machine rented now is to take before the exam
	Plan        string                 `json:"plan,omitempty"`        // what the graders still to come wait for
	Endings     []endingLook           `json:"endings,omitempty"`     // each machine to end, and why
	Released    int                    `json:"released,omitempty"`    // keys whose jobs were handed back
	Markets     map[string]*marketLook `json:"markets,omitempty"`     // each market asked what it offers
	Rents       []rentTry              `json:"rents,omitempty"`       // each machine tried
	Calls       []apiCall              `json:"calls,omitempty"`       // every request to a market or the server
	Said        []string               `json:"said,omitempty"`        // what the fleet said
	Error       string                 `json:"error,omitempty"`       // why the look went no further
	Fleet       string                 `json:"fleet,omitempty"`       // the commit this fleet was built from
	CLI         string                 `json:"cli,omitempty"`         // the stet the server hands out
	Server      string                 `json:"server,omitempty"`      // the commit the server runs
	// Work is all the server said is waiting, as it said it: every round, and every one of the
	// organisers' machines, what it holds, whether it has trained, and so on.
	Work json.RawMessage `json:"work,omitempty"`
}

// endingLook is one machine the fleet decided to end.
type endingLook struct {
	Handle string `json:"handle"`
	Market string `json:"market,omitempty"`
	ID     string `json:"id,omitempty"`
	Why    string `json:"why"`
	Failed bool   `json:"failed,omitempty"`
}

type lookRound struct {
	ID             string `json:"id"`
	Entries        int    `json:"entries"`
	RunsWanted     int    `json:"runs_wanted"`
	EditsUnderWay  int    `json:"runs_under_way"`
	GradesWanted   int    `json:"grades_wanted"`
	GradesUnderWay int    `json:"grades_under_way"`
	GradesEarly    int    `json:"grades_early"`
	DrawnAt        int64  `json:"drawn_at,omitempty"`
	ExamDue        int64  `json:"exam_due,omitempty"`
	ExamAt         int64  `json:"exam_at,omitempty"`
}

// marketLook is how one market answered what it offers.
type marketLook struct {
	MS     int64  `json:"ms"`
	Offers int    `json:"offers"`
	Error  string `json:"error,omitempty"`
}

// rentTry is one machine the fleet tried to rent.
type rentTry struct {
	Handle  string  `json:"handle"`
	For     string  `json:"for,omitempty"` // curators, early grades, grades, students, bigger students
	Market  string  `json:"market"`
	Offer   string  `json:"offer"`
	Card    string  `json:"card"`
	Price   float64 `json:"price"`
	MS      int64   `json:"ms"`
	Outcome string  `json:"outcome"` // rented, no room, or what went wrong
	ID      string  `json:"id,omitempty"`
}

func (f *Fleet) looksPath() string {
	return filepath.Join(f.historyDir(), "looks-"+f.now().UTC().Format("2006-01-02")+".jsonl")
}
func (f *Fleet) balancesPath() string { return filepath.Join(f.historyDir(), "balances.jsonl") }

// appendJSON adds one line to a file of them. Failing to is said, and nothing more: what is
// kept is for seeing later, and no reason to stop renting.
func (f *Fleet) appendJSON(path string, v any) {
	raw, err := json.Marshal(v)
	if err != nil {
		return
	}
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		f.sayMu.Lock()
		f.out.Write([]byte("keeping " + filepath.Base(path) + ": " + err.Error() + "\n"))
		f.sayMu.Unlock()
		return
	}
	defer fh.Close()
	fh.Write(append(raw, '\n'))
}

// beginLook starts the record of a look; endLook keeps it.
func (f *Fleet) beginLook() *lookRecord {
	rec := &lookRecord{T: f.now().UTC(), Fleet: fleetBuild()}
	f.sayMu.Lock()
	f.cur = rec
	f.sayMu.Unlock()
	collectCalls(rec)
	return rec
}

func (f *Fleet) endLook(start time.Time, err error) {
	stopCalls()
	f.sayMu.Lock()
	rec := f.cur
	f.cur = nil
	f.sayMu.Unlock()
	if rec == nil {
		return
	}
	rec.TookMS = f.now().Sub(start).Milliseconds()
	rec.CLI, rec.Server = f.st.cli, f.st.server
	if err != nil {
		rec.Error = err.Error()
	}
	f.appendJSON(f.looksPath(), rec)
	f.compressOld()
}

// decided notes what the look decided.
func (rec *lookRecord) decided(d Decision, next int) {
	if rec == nil {
		return
	}
	rec.Target, rec.Launch, rec.Note = d.Target, d.Launch, d.Note
	rec.LaunchLong, rec.LaunchEarly, rec.LaunchRuns, rec.Next = d.LaunchLong, d.LaunchEarly, d.LaunchRuns, next
	if !d.ExamDue.IsZero() {
		rec.ExamDue = d.ExamDue.Unix()
	}
	for _, e := range d.End {
		handle := ""
		if e.Worker != nil && e.Worker.Key != nil {
			handle = e.Worker.Key.Handle
		}
		r := e.Rental
		if r == nil && e.Worker != nil {
			r = e.Worker.Rental
		}
		if handle == "" && r != nil {
			handle = r.Name
		}
		el := endingLook{Handle: handle, Why: e.Why, Failed: e.Failed}
		if r != nil {
			el.Market, el.ID = r.Market, r.ID
		}
		rec.Endings = append(rec.Endings, el)
	}
	rec.Released = len(d.Release)
}

// seenWork notes what the server said is waiting, and the machines as they are.
func (rec *lookRecord) seenWork(v *View, p Policy, now time.Time) {
	if rec == nil || v == nil || v.Work == nil {
		return
	}
	w := v.Work
	rec.Work = w.Raw
	for _, r := range w.Rounds {
		rec.Rounds = append(rec.Rounds, lookRound{ID: r.ID, Entries: r.Entries, RunsWanted: r.RunsWanted, EditsUnderWay: r.EditsUnderWay,
			GradesWanted: r.GradesWanted, GradesUnderWay: r.GradesUnderWay, GradesEarly: r.GradesEarly, DrawnAt: r.DrawnAt, ExamDue: r.ExamDue, ExamAt: r.ExamAt})
	}
	rec.Runners, rec.Paused, rec.NextDraw, rec.Farm = w.Runners, w.Paused != 0, w.NextDrawAt, w.FarmWaiting
	rec.Hourly, rec.Today = v.Hourly, v.Today
	for _, wk := range v.Workers {
		st := wk.state(now, p)
		if st == "free" {
			continue
		}
		if rec.States == nil {
			rec.States, rec.Live = map[string]int{}, map[string]int{}
		}
		rec.States[st]++
		if wk.Rental != nil {
			rec.Live[wk.Rental.Market]++
		} else if wk.Pending != nil {
			rec.Live[wk.Pending.Market]++
		}
		for _, ex := range wk.ExtraRentals {
			if ex != nil && (ex.live() || ex.Unconfirmed) {
				rec.Live[ex.Market]++
			}
		}
	}
}

// offersSeen notes how one market answered what it offers.
func (f *Fleet) offersSeen(market string, took time.Duration, n int, err error) {
	if f.cur == nil {
		return
	}
	if f.cur.Markets == nil {
		f.cur.Markets = map[string]*marketLook{}
	}
	m := &marketLook{MS: took.Milliseconds(), Offers: n}
	if err != nil {
		m.Error = firstLine(err.Error())
	}
	f.cur.Markets[market] = m
}

// rentSeen notes one machine tried.
func (f *Fleet) rentSeen(t rentTry) {
	if f.cur != nil {
		f.cur.Rents = append(f.cur.Rents, t)
	}
}

// --- what each market's account says --------------------------------------------------------------

type balanceRecord struct {
	T       time.Time `json:"t"`
	Market  string    `json:"market"`
	Balance *float64  `json:"balance,omitempty"` // dollars left to spend, where the market says
	Note    string    `json:"note,omitempty"`
	Error   string    `json:"error,omitempty"`
}

// balancesEvery is how often each market's account is asked what is left in it.
const balancesEvery = time.Hour

// keepBalances asks each market what is left in its account, once an hour.
func (f *Fleet) keepBalances(ctx context.Context) {
	now := f.now()
	if !f.balancesAt.IsZero() && now.Sub(f.balancesAt) < balancesEvery {
		return
	}
	f.balancesAt = now
	names := make([]string, 0, len(f.markets))
	byName := map[string]Market{}
	for _, m := range f.markets {
		names = append(names, m.Name())
		byName[m.Name()] = m
	}
	sort.Strings(names)
	for _, name := range names {
		c, cancel := context.WithTimeout(ctx, 30*time.Second)
		a, err := byName[name].Check(c)
		cancel()
		rec := balanceRecord{T: now.UTC(), Market: name, Note: a.Note}
		if err != nil {
			rec.Error = firstLine(err.Error())
		} else if a.Known {
			b := a.Balance
			rec.Balance = &b
		}
		f.appendJSON(f.balancesPath(), rec)
	}
}

// --- every call to a market or the server, timed ------------------------------------------------

// apiCall is one request to a market or to the Stet server: where, what, how it went, and how
// long it took. Never its headers or its query, where the keys are.
type apiCall struct {
	At     time.Time `json:"at"`
	Who    string    `json:"who"` // runpod, vast, lium, salad, clore, stet, or the host
	Method string    `json:"method"`
	Path   string    `json:"path"`
	Status int       `json:"status,omitempty"`
	MS     int64     `json:"ms"`
	Error  string    `json:"error,omitempty"`
}

// callTimer is the transport every client of the fleet's goes through: it notes each call into
// the look under way, if any, as who's.
type callTimer struct {
	base http.RoundTripper
	who  string
}

// timedFor is the transport a client of who's is made with.
func timedFor(who string) http.RoundTripper { return &callTimer{base: http.DefaultTransport, who: who} }

var (
	callsMu sync.Mutex
	calls   *[]apiCall // the look's, while one is under way
)

func (t *callTimer) RoundTrip(req *http.Request) (*http.Response, error) {
	start := time.Now()
	resp, err := t.base.RoundTrip(req)
	who := t.who
	if who == "" || req.URL.Host != "" && whoIs(req.URL.Host) != req.URL.Host && whoIs(req.URL.Host) != who {
		who = whoIs(req.URL.Host) // another service's host, as a client's link may lead
	}
	c := apiCall{At: start.UTC(), Who: who, Method: req.Method, Path: req.URL.Path, MS: time.Since(start).Milliseconds()}
	if err != nil {
		c.Error = firstLine(err.Error())
	} else {
		c.Status = resp.StatusCode
	}
	callsMu.Lock()
	if calls != nil {
		*calls = append(*calls, c)
	}
	callsMu.Unlock()
	return resp, err
}

// whoIs is the market, or the server, a host is.
func whoIs(host string) string {
	h := strings.ToLower(host)
	for _, w := range []struct{ part, who string }{{"runpod", "runpod"}, {"vast.ai", "vast"}, {"lium", "lium"}, {"salad", "salad"}, {"clore", "clore"}, {"stetrun", "stet"}} {
		if strings.Contains(h, w.part) {
			return w.who
		}
	}
	return h
}

// collectCalls has the calls made from now on kept in rec, until stopCalls.
func collectCalls(rec *lookRecord) {
	callsMu.Lock()
	calls = &rec.Calls
	callsMu.Unlock()
}

func stopCalls() {
	callsMu.Lock()
	calls = nil
	callsMu.Unlock()
}

// --- what the markets offer, now and then --------------------------------------------------------

// offersEvery is how often what each market offers is kept, by kind.
const offersEvery = 10 * time.Minute

// offerKind is what a market offers of one card, of one kind of host, at one time.
type offerKind struct {
	T      time.Time `json:"t"`
	Market string    `json:"market"`
	Card   string    `json:"card"`
	Trust  string    `json:"trust"`
	N      int       `json:"n"`   // offers (Stock where the market says it, else 1 each)
	Min    float64   `json:"min"` // dollars an hour, all in
	Median float64   `json:"median"`
	Rate   float64   `json:"rate,omitempty"`  // how often such a machine was seen to start
	Speed  float64   `json:"speed,omitempty"` // against a 3090
	Grade  float64   `json:"grade,omitempty"` // dollars a grade, at the cheapest, on its pass and lead
}

// offersDue says whether what the markets offer is to be kept this look.
func (f *Fleet) offersDue() bool {
	return f.offersAt.IsZero() || f.now().Sub(f.offersAt) >= offersEvery
}

// keepOffers keeps what the markets offer, by kind, every offersEvery: what could have been
// rented, and at what, across every market.
func (f *Fleet) keepOffers(offers []Offer) {
	now := f.now()
	if len(offers) == 0 {
		return
	}
	f.offersAt = now
	type key struct{ market, card, trust string }
	prices := map[key][]float64{}
	first := map[key]Offer{}
	for _, o := range offers {
		k := key{o.Market, o.Card, o.Trust}
		if _, ok := first[k]; !ok {
			first[k] = o
		}
		n := max(o.Stock, 1)
		for range min(n, 50) {
			prices[k] = append(prices[k], o.Price)
		}
	}
	keys := make([]key, 0, len(prices))
	for k := range prices {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].market != keys[j].market {
			return keys[i].market < keys[j].market
		}
		return keys[i].card+keys[i].trust < keys[j].card+keys[j].trust
	})
	path := filepath.Join(f.historyDir(), "offers-"+now.UTC().Format("2006-01-02")+".jsonl")
	for _, k := range keys {
		ps := prices[k]
		sort.Float64s(ps)
		o := first[k]
		rec := offerKind{T: now.UTC(), Market: k.market, Card: k.card, Trust: k.trust, N: len(ps), Min: ps[0], Median: ps[len(ps)/2], Rate: o.Rate, Speed: f.speedOf(k.card)}
		o.Price = ps[0]
		if lead := f.lead(o); lead > 0 {
			rec.Grade = o.Price * lead.Hours()
		}
		f.appendJSON(path, rec)
	}
}

// --- the builds ------------------------------------------------------------------------------------

// fleetBuild is the commit this binary was built from, as Go stamps it ("+" if the tree had
// changes not committed).
func fleetBuild() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	rev, dirty := "", false
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			dirty = s.Value == "true"
		}
	}
	if len(rev) > 12 {
		rev = rev[:12]
	}
	if dirty {
		rev += "+"
	}
	return rev
}

// --- rounds, as they settle -----------------------------------------------------------------------

// roundsSettled keeps each round the work no longer has open, whole, as the server tells it
// (GET /api/v1/rounds/{id}), once it has settled: in history/rounds/, and a line in rounds.jsonl.
func (f *Fleet) roundsSettled(ctx context.Context, w *Work) {
	open := map[string]bool{}
	for _, r := range w.Rounds {
		open[r.ID] = true
	}
	if f.openRounds != nil {
		for id := range f.openRounds {
			if !open[id] {
				f.keepRound(ctx, id)
			}
		}
	}
	f.openRounds = open
}

func (f *Fleet) keepRound(ctx context.Context, id string) {
	f.st.site(ctx) // the builds, as they are now
	var raw json.RawMessage
	c, cancel := context.WithTimeout(ctx, 30*time.Second)
	err := f.st.do(c, "GET", "/api/v1/rounds/"+url.PathEscape(id), nil, nil, &raw)
	cancel()
	line := map[string]any{"t": f.now().UTC(), "round": id, "fleet": fleetBuild(), "cli": f.st.cli, "server": f.st.server}
	if err != nil {
		line["error"] = firstLine(err.Error())
	} else {
		dir := filepath.Join(f.historyDir(), "rounds")
		if os.MkdirAll(dir, 0o700) == nil {
			os.WriteFile(filepath.Join(dir, id+".json"), append([]byte(raw), '\n'), 0o600)
		}
		var r struct {
			Round struct {
				DrawnAt   int64 `json:"drawn_at"`
				ExamAt    int64 `json:"exam_at"`
				SettledAt int64 `json:"settled_at"`
			} `json:"round"`
		}
		if json.Unmarshal(raw, &r) == nil {
			line["drawn_at"], line["exam_at"], line["settled_at"] = r.Round.DrawnAt, r.Round.ExamAt, r.Round.SettledAt
		}
	}
	f.appendJSON(filepath.Join(f.historyDir(), "rounds.jsonl"), line)
}

// historyDir is where what the fleet did is kept, made if it is not there.
func (f *Fleet) historyDir() string {
	dir := filepath.Join(f.o.Home, "history")
	os.MkdirAll(dir, 0o700)
	return dir
}

// --- keeping it small ------------------------------------------------------------------------------

// compressOld gzips the days before today's of the files kept a day at a time, once a day: a
// look is a few kilobytes, the work said whole among them, and a day of them is some 10 MB.
func (f *Fleet) compressOld() {
	today := f.now().UTC().Format("2006-01-02")
	if f.compressedDay == today {
		return
	}
	f.compressedDay = today
	dir := f.historyDir()
	for _, pattern := range []string{"looks-*.jsonl", "offers-*.jsonl"} {
		paths, _ := filepath.Glob(filepath.Join(dir, pattern))
		for _, p := range paths {
			if strings.Contains(filepath.Base(p), today) {
				continue
			}
			if err := gzipFile(p); err != nil {
				f.say("compressing %s: %v", filepath.Base(p), err)
			}
		}
	}
}

// gzipFile writes path.gz and removes path.
func gzipFile(path string) error {
	in, err := os.Open(path)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := path + ".gz.tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	zw := gzip.NewWriter(out)
	if _, err := io.Copy(zw, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := zw.Close(); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path+".gz"); err != nil {
		return err
	}
	return os.Remove(path)
}
