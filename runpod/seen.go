package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// --- what is seen of each machine -----------------------------------------------------------------
//
// "Never asked for work" says that a machine failed, not why. So what is seen of each machine is
// kept, in machines.jsonl, one event a line: when it was rented and on what; each phase its
// market says it is in ("no container yet", "pulling image", "container up"), and its container
// started again; when it first asked for work; each change in what it does for the server; and
// how it ended, with the cause. A machine ended for not starting or for going quiet has what its
// market and its container last said fetched first, where the market says it (logs/), and the
// cause is in the line the fleet says, in its record in spent.jsonl, and in the event.

type event struct {
	T      time.Time `json:"t"`
	Key    string    `json:"key"`
	Market string    `json:"market"`
	Handle string    `json:"handle"`
	Card   string    `json:"card,omitempty"`
	Event  string    `json:"event"`
	Detail string    `json:"detail,omitempty"`
}

func (f *Fleet) eventsPath() string { return filepath.Join(f.o.Home, "machines.jsonl") }

// event notes one thing seen of a machine. Failing to is said, and nothing more.
func (f *Fleet) event(r *Rental, handle, what, detail string) {
	e := event{T: f.now().UTC(), Key: ledgerKey(r.Market, r.ID), Market: r.Market, Handle: handle, Card: r.Card, Event: what, Detail: detail}
	raw, err := json.Marshal(e)
	if err != nil {
		return
	}
	fh, err := os.OpenFile(f.eventsPath(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		f.say("keeping what was seen of %s: %v", handle, err)
		return
	}
	defer fh.Close()
	fh.Write(append(raw, '\n'))
}

// seenRental is what was last seen of a machine, to say only what changes.
type seenRental struct {
	phase, state, work, trouble string
	upFor                       time.Duration
}

// watch notes what changed of a machine since the last look.
func (f *Fleet) watch(w *Worker, now time.Time) {
	r := w.Rental
	key := ledgerKey(r.Market, r.ID)
	if f.seen == nil {
		f.seen = map[string]seenRental{}
	}
	prev, had := f.seen[key]
	cur := seenRental{phase: r.Phase, state: w.state(now, f.o.Policy), upFor: r.UpFor}
	if w.Machine != nil && w.Machine.LastSeen >= r.Created.Unix() {
		cur.work = w.Machine.Busy
		if cur.work == "" {
			cur.work = "nothing"
		}
		if w.Machine.Holding > 1 {
			cur.work += fmt.Sprintf(", %d grades", w.Machine.Holding)
		}
		if w.Machine.Ready {
			cur.work += ", trained and waiting for the exam"
		}
		cur.trouble = w.Machine.trouble()
	}
	if cur.phase != "" && (!had || prev.phase != cur.phase) {
		detail := r.Phase
		if r.Detail != "" && r.Detail != r.Phase && !strings.HasPrefix(r.Detail, "container up") {
			detail += ": " + r.Detail
		}
		f.event(r, w.Key.Handle, "phase", detail)
	}
	if had && prev.upFor > 0 && cur.upFor > 0 && cur.upFor+30*time.Second < prev.upFor {
		f.event(r, w.Key.Handle, "restarted", fmt.Sprintf("its container was started again: up %s, after %s", cur.upFor.Round(time.Second), prev.upFor.Round(time.Second)))
	}
	if !had || prev.state != cur.state {
		f.event(r, w.Key.Handle, "state", cur.state)
	}
	if cur.work != "" && had && prev.work != cur.work {
		f.event(r, w.Key.Handle, "work", cur.work)
	}
	if cur.trouble != "" && prev.trouble != cur.trouble {
		f.event(r, w.Key.Handle, "trouble", cur.trouble)
	}
	f.seen[key] = cur
}

// forget lets go of what was seen of machines no longer rented.
func (f *Fleet) forget(live map[string]bool) {
	for k := range f.seen {
		if !live[k] {
			delete(f.seen, k)
		}
	}
	for k := range f.lastLog {
		if !live[k] {
			delete(f.lastLog, k)
			delete(f.slowSaid, k)
		}
	}
}

// --- a machine slow to start -------------------------------------------------------------------

// slowLook is how long before the end of its grace a machine's log is first read, to have it as
// it was a look before.
const slowLook = 2 * time.Minute

// stillInstalling says whether a machine past its grace that has not asked for work is still
// getting there: its log has grown since the last look. Such a machine has up to twice its
// grace. In round 9 two Vast machines were ended at 11 minutes still downloading stet's
// libraries on a slow connection, where the rest asked within 5; a grace learned from the
// machines that started never sees those that were ended first. Only where the market keeps a
// machine's log.
func (f *Fleet) stillInstalling(ctx context.Context, w *Worker, now time.Time) bool {
	r := w.Rental
	key := ledgerKey(r.Market, r.ID)
	g, age := w.grace(f.o.Policy), now.Sub(r.Created)
	asked := w.Machine != nil && w.Machine.LastSeen >= r.Created.Unix()
	lg, ok := f.market(r.Market).(logger)
	if !ok || asked || r.State == "stopped" || age < g-slowLook || age > 2*g {
		delete(f.lastLog, key)
		return false
	}
	lctx, cancel := context.WithTimeout(ctx, 45*time.Second)
	lines, err := lg.Logs(lctx, r.ID, 30)
	cancel()
	if err != nil || len(lines) == 0 || failing(lines) {
		return false
	}
	if f.lastLog == nil {
		f.lastLog, f.slowSaid = map[string]string{}, map[string]bool{}
	}
	cur := strings.Join(lines, "\n")
	prev, had := f.lastLog[key]
	f.lastLog[key] = cur
	if !had {
		// Not read a look ago (the fleet started since): one more look, to see.
		return age > g
	}
	if prev == cur || age <= g {
		return false
	}
	if !f.slowSaid[key] {
		f.slowSaid[key] = true
		last := workerLine(lines)
		f.event(r, w.Key.Handle, "slow", fmt.Sprintf("past its grace of %s and still getting there (last it said: %s): given up to %s", g.Round(time.Minute), last, (2*g).Round(time.Minute)))
		f.say("%s on %s is past its grace of %s, and still getting there: given up to %s (last it said: %s)", w.Key.Handle, marketTitle(r.Market), g.Round(time.Minute), (2 * g).Round(time.Minute), last)
	}
	return true
}

// failing says whether what a machine said lately is of something that did not work: a log that
// grows with an install tried again and again is not one getting there.
func failing(lines []string) bool {
	all := strings.ToLower(strings.Join(lines, "\n"))
	for _, s := range []string{"did not work", "could not install", "no space left on device", "older than 580", "error:", "ending this"} {
		if strings.Contains(all, s) {
			return true
		}
	}
	return false
}

// --- why a machine failed ------------------------------------------------------------------------

// The causes a machine is ended for, or lost to, other than the work: kept in its record, and
// counted against its kind.
const (
	causeNoContainer   = "no container"   // its market never started its container
	causeInstall       = "install"        // its container ran, and stet could not be installed
	causeDriver        = "driver"         // its host's driver was too old
	causeSilent        = "stet silent"    // stet was installed and started, and never asked for work
	causeContainer     = "container ran"  // its container ran, and nothing it said says why it never asked
	causeMarketStopped = "market stopped" // its market stopped it
	causeQuiet         = "went quiet"     // it asked for work, and then said nothing
	causeGone          = "gone"           // its market took it away
	causeUnknown       = "unknown"
)

// inspector is a market whose record of one machine says more than its list.
type inspector interface {
	inspect(ctx context.Context, id string) (*Rental, error)
}

// workerLine is the last line the worker script said, or else the last line.
func workerLine(lines []string) string {
	for i := len(lines) - 1; i >= 0; i-- {
		if j := strings.Index(lines[i], "[stet-worker"); j >= 0 {
			return strings.TrimSpace(lines[i][j:])
		}
	}
	for i := len(lines) - 1; i >= 0; i-- {
		if s := strings.TrimSpace(lines[i]); s != "" {
			return s
		}
	}
	return ""
}

// causeOf is what a machine's last words say of why it never asked for work.
func causeOf(lines []string) string {
	all := strings.ToLower(strings.Join(lastLines(lines, 15), "\n"))
	last := strings.ToLower(workerLine(lines))
	switch {
	case strings.Contains(all, "older than 580") || strings.Contains(all, "cuda driver version is insufficient"):
		return causeDriver
	case strings.Contains(last, "install") || strings.Contains(all, "no space left on device") || strings.Contains(all, "could not be installed"):
		return causeInstall
	case strings.Contains(all, "stet verify") || strings.Contains(last, "] stet "):
		return causeSilent
	}
	return causeContainer
}

// diagnose says why a machine ended for not starting, or for going quiet, never asked for work
// or stopped asking: what its market says of it, and what it last said. Its logs, where its
// market keeps them, are kept in logs/.
func (f *Fleet) diagnose(ctx context.Context, w *Worker) (cause, said string) {
	r := w.Rental
	if in, ok := f.market(r.Market).(inspector); ok {
		// A market whose list says less than its record of one machine is asked for that.
		if fresh, err := in.inspect(ctx, r.ID); err == nil {
			fresh.Name, fresh.Created = r.Name, r.Created
			r = fresh
		}
	}
	title := marketTitle(r.Market)
	age := f.now().Sub(r.Created).Round(time.Minute)
	var lines []string
	var logErr error
	if lg, ok := f.market(r.Market).(logger); ok {
		lctx, cancel := context.WithTimeout(ctx, 45*time.Second)
		lines, logErr = lg.Logs(lctx, r.ID, 60)
		cancel()
		if len(lines) > 0 {
			f.keepLog(r, w.Key.Handle, lines)
		}
	}
	asked := w.Machine != nil && w.Machine.LastSeen >= r.Created.Unix()
	switch {
	case asked:
		cause, said = causeQuiet, fmt.Sprintf("it asked for work, then said nothing for %s", f.now().Sub(time.Unix(w.Machine.LastSeen, 0)).Round(time.Minute))
	case r.State == "stopped":
		cause, said = causeMarketStopped, fmt.Sprintf("%s stopped it (%s)", title, strings.ToLower(r.Status))
	case r.Up == "no":
		cause, said = causeNoContainer, fmt.Sprintf("%s never started its container in %s (%s)", title, age, r.Phase)
	case len(lines) > 0:
		cause = causeOf(lines)
		said = map[string]string{
			causeDriver:    "its host's driver is too old",
			causeInstall:   "stet could not be installed",
			causeSilent:    "stet started, and never asked for work",
			causeContainer: "its container ran, and stet never asked for work",
		}[cause]
	case r.Up == "yes":
		cause, said = causeContainer, "its container ran, and stet never asked for work"
	default:
		cause, said = causeUnknown, title+" does not say what became of it"
	}
	if last := workerLine(lines); last != "" {
		if len(last) > 160 {
			last = last[:160] + "…"
		}
		said += "; last it said: " + last
	} else if logErr != nil && f.market(r.Market) != nil {
		said += "; its log could not be had: " + strings.TrimSpace(firstLine(logErr.Error()))
	}
	return cause, said
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 120 {
		s = s[:120] + "…"
	}
	return s
}

func (f *Fleet) logPath(key string) string {
	return filepath.Join(f.o.Home, "logs", strings.ReplaceAll(key, ":", "-")+".log")
}

// keepLog keeps what a machine's market and container last said.
func (f *Fleet) keepLog(r *Rental, handle string, lines []string) {
	path := f.logPath(ledgerKey(r.Market, r.ID))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return
	}
	head := fmt.Sprintf("# %s on %s (%s), %s, rented %s UTC; kept %s UTC\n", handle, marketTitle(r.Market), r.ID, r.Card,
		r.Created.UTC().Format("Jan 2 15:04:05"), f.now().UTC().Format("Jan 2 15:04:05"))
	os.WriteFile(path, []byte(head+strings.Join(lines, "\n")+"\n"), 0o600)
}

// --- a machine's story -------------------------------------------------------------------------------

// cmdMachine says what was seen of the machines rented with one key, the latest last: each
// event, how it ended, and what it last said.
func cmdMachine(f *Fleet, handle string, last int) error {
	fh, err := os.Open(f.eventsPath())
	if errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("nothing has been seen of any machine yet (%s)", f.eventsPath())
	}
	if err != nil {
		return err
	}
	defer fh.Close()
	var keys []string
	byKey := map[string][]event{}
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		var e event
		if json.Unmarshal(sc.Bytes(), &e) != nil || e.Handle != handle {
			continue
		}
		if _, ok := byKey[e.Key]; !ok {
			keys = append(keys, e.Key)
		}
		byKey[e.Key] = append(byKey[e.Key], e)
	}
	if len(keys) == 0 {
		return fmt.Errorf("nothing has been seen of a machine rented with %s", handle)
	}
	for _, k := range keys[max(0, len(keys)-last):] {
		es := byKey[k]
		fmt.Fprintf(f.out, "%s on %s, %s (%s):\n", handle, marketTitle(es[0].Market), es[0].Card, k)
		t0 := es[0].T
		for _, e := range es {
			fmt.Fprintf(f.out, "  %s  +%-6s %-9s %s\n", e.T.Format("15:04:05"), e.T.Sub(t0).Round(time.Second).String(), e.Event, e.Detail)
		}
		if raw, err := os.ReadFile(f.logPath(k)); err == nil {
			lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
			fmt.Fprintf(f.out, "  what it last said (%s):\n", f.logPath(k))
			for _, l := range lastLines(lines, 15) {
				fmt.Fprintf(f.out, "    %s\n", l)
			}
		}
		fmt.Fprintln(f.out)
	}
	return nil
}
