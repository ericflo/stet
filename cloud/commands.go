package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

func cmdKeys(ctx context.Context, f *Fleet, add int, prefix string, vouch bool, revoke []string) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	if len(revoke) > 0 {
		return f.revoke(ctx, revoke)
	}
	if add > 0 {
		made, err := f.addKeys(add, prefix)
		if err != nil {
			return err
		}
		names := make([]string, len(made))
		for i, k := range made {
			names[i] = k.Handle
		}
		fmt.Fprintf(f.out, "Made %d keys: %s (in %s).\n", len(made), strings.Join(names, ", "), f.keyDir())
		vouch = true
	}
	if len(f.keys) == 0 {
		fmt.Fprintln(f.out, "No keys yet. Make some: stet-cloud keys --add 4")
		return nil
	}
	vouched, err := f.st.vouched(ctx, f.op)
	if err != nil {
		return err
	}
	if vouch {
		// Vouched for with the organisers' own key, and each given its account now: the
		// server needs nothing else to take them as the organisers' rented machines.
		for _, k := range f.keys {
			if vouched[k.Public].Counts {
				continue
			}
			if err := f.ready(ctx, k, vouched); err != nil {
				return err
			}
		}
		if vouched, err = f.st.vouched(ctx, f.op); err != nil {
			return err
		}
		fmt.Fprintln(f.out)
	}
	missing := 0
	for _, k := range f.keys {
		v, ok := vouched[k.Public]
		say := ""
		switch {
		case ok && v.Counts && v.Busy:
			say = fmt.Sprintf("vouched for by %s · at work now", v.By)
		case ok && v.Counts && v.LastSeen > 0:
			say = fmt.Sprintf("vouched for by %s · last seen %s ago", v.By, f.now().Sub(time.Unix(v.LastSeen, 0)).Round(time.Minute))
		case ok && v.Counts:
			say = fmt.Sprintf("vouched for by %s · not used yet", v.By)
		case ok:
			say, missing = fmt.Sprintf("vouched for by %s, whose key is not one of the organisers' any more", v.By), missing+1
		default:
			say, missing = "not vouched for", missing+1
		}
		fmt.Fprintf(f.out, "  %-14s %s…  %s\n", k.Handle, k.Public[:12], say)
	}
	if missing > 0 {
		fmt.Fprintf(f.out, "\n%d not vouched for: stet-cloud keys --vouch (they are vouched for when first rented, too).\n", missing)
	}
	return nil
}

// revoke takes the organisers' grant for these keys back and forgets them here.
func (f *Fleet) revoke(ctx context.Context, handles []string) error {
	byHandle := map[string]*Key{}
	for _, k := range f.keys {
		byHandle[k.Handle] = k
	}
	for _, h := range handles {
		k := byHandle[h]
		if k == nil {
			return fmt.Errorf("there is no key %q in the pool", h)
		}
		if err := f.st.revoke(ctx, f.op, k.Public); err != nil {
			return fmt.Errorf("taking back %s: %w", h, err)
		}
		if err := os.Remove(filepath.Join(f.keyDir(), h+".json")); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		f.say("took back %s: the server no longer takes it as the organisers', and its key is gone from here. A machine still running with it should be ended: stet-cloud down %s", h, h)
	}
	return nil
}

// ceiling is --max-price, as said in passing; long, --long-max-price.
func (f *Fleet) ceiling(long bool) string {
	if p := f.maxPrice(long); p > 0 {
		return fmt.Sprintf(", $%.2f an hour a machine at most", p)
	}
	return ""
}

// longCeiling is --long-max-price, as said in passing, where it is not --max-price.
func (f *Fleet) longCeiling() string {
	if !f.o.LongFarm || f.maxPrice(true) == f.maxPrice(false) {
		return ""
	}
	if f.maxPrice(true) <= 0 {
		return ", any price for the bigger students"
	}
	return fmt.Sprintf(", $%.2f for the bigger students", f.maxPrice(true))
}

func (f *Fleet) maxPrice(long bool) float64 {
	if long && f.o.LongMaxPrice > 0 {
		return f.o.LongMaxPrice
	}
	return f.o.MaxPrice
}

func cmdStatus(ctx context.Context, f *Fleet) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	v, err := f.look(ctx)
	if err != nil {
		return err
	}
	now := f.now()
	statusPol := f.o.Policy
	statusReserved := v.ExaminerLive
	if statusReserved == 0 && f.o.MaxWorkers > 0 && f.o.Examiner.Addr != "" && roundNeedingExam(v.Work) != "" {
		statusReserved = 1
	}
	statusPol.MaxWorkers = max(0, statusPol.MaxWorkers-statusReserved)
	d := statusPol.decideNext(v.Work, v.Workers, v.Today, now, f.nextGrades(ctx, v.Work))
	if len(v.Work.Rounds) == 0 {
		fmt.Fprintln(f.out, "No round is open.")
	}
	for _, r := range v.Work.Rounds {
		fmt.Fprintf(f.out, "Round %s: %d entries open · %d runs a new machine could take · %d running · %d scripts to grade · %d being graded\n",
			r.ID, r.Entries, r.RunsWanted, r.EditsUnderWay, r.GradesWanted, r.GradesUnderWay)
		if due := f.o.Policy.examDue(r, now); !due.IsZero() {
			fmt.Fprintf(f.out, "Exam due about %s UTC (%s): %d grades a grader could take before it, machines rented for them from %s UTC, each kind just in time to train by then\n",
				due.UTC().Format("15:04"), until(now, due), r.GradesEarly, due.Add(-f.o.Early).UTC().Format("15:04"))
		}
	}
	if v.Work.Paused > 0 {
		fmt.Fprintf(f.out, "Rounds paused since %s UTC: none is drawn (stet-cloud resume)\n", time.Unix(v.Work.Paused, 0).UTC().Format("Jan 2 15:04"))
	}
	if v.Work.BackoffUntil > 0 {
		at := time.Unix(v.Work.BackoffUntil, 0)
		fmt.Fprintf(f.out, "Automatic draws backed off until %s UTC (%s) after two empty rounds (`stet draw` overrides)\n", at.UTC().Format("15:04"), until(now, at))
	}
	if v.Work.NextDrawAt > 0 {
		at := time.Unix(v.Work.NextDrawAt, 0)
		fmt.Fprintf(f.out, "Next draw %s UTC (%s)\n", at.UTC().Format("15:04"), until(now, at))
	}
	if v.Work.FarmWaiting+v.Work.FarmUnderWay+v.Work.FarmLongWaiting > 0 {
		fmt.Fprintf(f.out, "The organisers' students: %d to train · %d bigger ones, hours each · %d being trained\n",
			v.Work.FarmWaiting, v.Work.FarmLongWaiting, v.Work.FarmUnderWay)
	}
	fmt.Fprintf(f.out, "%d of the organisers' machines running curators\n\n", v.Work.Runners)
	rented := 0
	for _, w := range v.Workers {
		st := w.state(now, f.o.Policy)
		if st == "free" {
			continue
		}
		rented++
		card := w.Rental.Card
		if w.Rental.Market != "runpod" {
			card += " (" + marketTitle(w.Rental.Market) + ")"
		}
		note := ""
		if st == "starting" || st == "stopped" || st == "quiet" || w.Rental.Up == "no" {
			note = "  · " + w.Rental.Detail // what its market says of it, while it is not at work
		}
		if w.NoCurators != "" {
			note += "  · grades only: " + w.NoCurators
		}
		if t := w.Machine.trouble(); t != "" && w.Machine.LastSeen >= w.Rental.Created.Unix() {
			note += "  · " + t
		}
		fmt.Fprintf(f.out, "  %-14s %-8s %-26s $%.2f/h  up %s%s\n", w.Key.Handle, st, card, w.Rental.Price, now.Sub(w.Rental.Created).Round(time.Minute), note)
	}
	if rented == 0 {
		fmt.Fprintf(f.out, "Nothing rented (%d keys in the pool).\n", len(v.Workers))
	}
	if ex := readExaminerState(f.o.Home); ex != nil && ex.InstanceID != "" {
		fmt.Fprintf(f.out, "Examiner GPU: machine %s for %s ($%.2f/h, up %s)\n", ex.InstanceID, ex.RoundID, ex.Price, now.Sub(ex.Created).Round(time.Minute))
	} else if ex != nil && ex.PendingRent {
		fmt.Fprintf(f.out, "Examiner GPU: rental reservation pending for %s ($%.2f/h)\n", ex.RoundID, ex.Price)
	} else if ex != nil && len(ex.Unresolved) > 0 {
		fmt.Fprintf(f.out, "Examiner GPU: %d unresolved instance(s) awaiting teardown\n", len(ex.Unresolved))
	} else if ex != nil && len(ex.PendingSpends) > 0 {
		fmt.Fprintf(f.out, "Examiner GPU: idle (%d pending spend record(s) awaiting spent.jsonl flush)\n", len(ex.PendingSpends))
	}
	fmt.Fprintf(f.out, "\n$%.2f an hour now · about $%.2f today (UTC)\n", v.Hourly, v.Today)
	if statusReserved > 0 {
		fmt.Fprintf(f.out, "The work wants %d worker machines now (at most %d workers + %d examiner = %d total)", d.Target, statusPol.MaxWorkers, statusReserved, f.o.MaxWorkers)
	} else {
		fmt.Fprintf(f.out, "The work wants %d machines now (at most %d)", d.Target, f.o.MaxWorkers)
	}
	if d.Launch > 0 || len(d.End) > 0 {
		fmt.Fprintf(f.out, ": `stet-cloud scale` would rent %d and end %d", d.Launch, len(d.End))
	}
	fmt.Fprintln(f.out, ".")
	if d.Note != "" {
		fmt.Fprintf(f.out, "(%s)\n", d.Note)
	}
	if svc := f.o.Service; svc != nil {
		fmt.Fprintf(f.out, "Judged as the service judges it: scale --follow %s (started %s UTC)\n", svc, time.Unix(svc.At, 0).UTC().Format("Jan 2 15:04"))
	}
	return nil
}

// cmdPause pauses the rounds, or resumes them.
func cmdPause(ctx context.Context, f *Fleet, paused bool) error {
	since, by, err := f.st.pause(ctx, f.op, paused)
	if err != nil {
		return err
	}
	if since > 0 {
		f.say("rounds paused since %s UTC, by %s: none is drawn, and the round under way finishes. The machines turn to the organisers' students", time.Unix(since, 0).UTC().Format("Jan 2 15:04"), by)
		return nil
	}
	f.say("rounds resumed: the next is drawn at its time")
	return nil
}

func until(now, t time.Time) string {
	d := t.Sub(now).Round(time.Minute)
	if d <= 0 {
		return "now"
	}
	return "in " + strings.TrimSuffix(d.String(), "0s")
}

// tick is one round of scale: look, decide, act. What it saw, decided and did is kept, a line a
// look (history.go).
func tick(ctx context.Context, f *Fleet) (err error) {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	start := f.now()
	rec := f.beginLook()
	defer func() { f.endLook(start, err) }()
	v, err := f.look(ctx)
	if err != nil {
		if f.examiner != nil {
			if v != nil && v.Work != nil {
				if exErr := f.examiner.Reconcile(ctx, v.Work, ExaminerBudget{AccountingErr: err}); exErr != nil && !errors.Is(exErr, err) {
					f.say("examiner: %v", exErr)
				}
			} else if exErr := f.examiner.EnforceTimeout(ctx); exErr != nil {
				f.say("examiner: %v", exErr)
			}
		}
		return err
	}
	rec.seenWork(v, f.o.Policy, f.now())
	f.roundsSettled(ctx, v.Work)
	f.keepBalances(ctx)
	f.calibrate(ctx)
	f.tickOffers = map[offerKey][]Offer{}
	defer func() { f.tickOffers = nil }()
	next := f.nextGrades(ctx, v.Work)
	if f.offersDue() {
		if got, err := f.choices(ctx, false, false); err == nil {
			f.keepOffers(got)
		}
		f.st.site(ctx) // and the builds the server says it runs and hands out
	}
	var exErr error
	var snapErr error
	if f.examiner != nil {
		liveWorkers := 0
		for _, wk := range v.Workers {
			if wk.Rental != nil && wk.Rental.live() {
				liveWorkers++
			}
		}
		budget := ExaminerBudget{
			LiveWorkers: liveWorkers,
			MaxWorkers:  f.o.MaxWorkers,
			HourlyNow:   max(0, v.Hourly-v.ExaminerHourly),
			MaxHourly:   f.o.MaxHourly,
			SpentToday:  v.Today,
			DailyBudget: f.o.DailyBudget,
			CurrentSpentToday: func() (float64, error) {
				spentBase, serr := f.spent(v.Ledger, startOfDay(f.now()))
				if serr != nil {
					return 0, fmt.Errorf("reading daily spend: %w", serr)
				}
				_, _, exAcc, aerr := f.examinerSnapshot()
				if aerr != nil {
					return 0, aerr
				}
				return spentBase + exAcc, nil
			},
		}
		if exErr = f.examiner.Reconcile(ctx, v.Work, budget); exErr != nil {
			f.say("examiner: %v", exErr)
		}
		exLive, exHourly, exAccrued, sErr := f.examinerSnapshot()
		snapErr = sErr
		if sErr != nil {
			f.say("examiner: %v", sErr)
		} else {
			v.Hourly = max(0, v.Hourly-v.ExaminerHourly) + exHourly
			if spentBase, serr := f.spent(v.Ledger, startOfDay(f.now())); serr == nil {
				v.Today = spentBase + exAccrued
			} else if exLive > 0 {
				v.Today = max(0, v.Today-v.ExaminerAccrued) + exAccrued
			}
			v.ExaminerLive, v.ExaminerHourly, v.ExaminerAccrued = exLive, exHourly, exAccrued
			if rec != nil {
				rec.Hourly, rec.Today = v.Hourly, v.Today
			}
		}
	}
	pol := f.o.Policy
	reserved := v.ExaminerLive
	if reserved == 0 && f.examiner != nil && roundNeedingExam(v.Work) != "" {
		reserved = 1
	}
	pol.MaxWorkers = max(0, pol.MaxWorkers-reserved)
	d := pol.decideNext(v.Work, v.Workers, v.Today, f.now(), next)
	rec.decided(d, next)
	ended := len(v.Ledger.Pods)
	if err := f.end(ctx, v, d.End); err != nil {
		return err
	}
	rec.Ended = max(ended-len(v.Ledger.Pods), 0)
	f.release(ctx, d.Release)
	if d.Note != "" {
		f.say("%s", d.Note)
	}
	var evicted map[string]bool
	if len(d.End) > 0 {
		evicted = make(map[string]bool, len(d.End))
		for _, e := range d.End {
			evicted[e.Worker.Key.Handle] = true
		}
	}
	postLive := 0
	for _, wk := range v.Workers {
		if wk.Rental != nil && wk.Rental.live() {
			postLive++
		}
	}
	launchN := min(d.Launch, max(0, pol.MaxWorkers-postLive))
	if exErr != nil || snapErr != nil {
		launchN = 0
	}
	rec.Rented, err = f.launchExcluding(ctx, v, launchN, d.LaunchLong, d.LaunchEarly, d.LaunchRuns, d.ExamDue, evicted)
	return err
}

func cmdScale(ctx context.Context, f *Fleet, follow bool) error {
	if !follow {
		return tick(ctx, f)
	}
	stopExaminer, err := f.startExaminer(ctx)
	if err != nil {
		return err
	}
	defer stopExaminer()
	f.say("scaling to the work every %s, at most %d machines and $%.2f an hour (%s cloud%s). Ctrl-C to stop; the machines stop by themselves.",
		f.o.Every, f.o.MaxWorkers, f.o.MaxHourly, strings.ToLower(f.o.Cloud), f.ceiling(false)+f.longCeiling())
	for {
		if err := tick(ctx, f); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			// The Stet server or RunPod away for a moment is no reason to stop watching.
			f.say("%v", err)
		}
		if err := sleepCtx(ctx, f.o.Every); err != nil {
			return nil
		}
	}
}

// reconcileExaminerOnce serializes standalone examiner budget checks, rentals, and state updates
// under the same fleet lock used by `up`, `down`, and `scale`, and defers new rentals whenever
// fleet accounting (fleet.json or spent.jsonl) cannot be read cleanly.
func reconcileExaminerOnce(ctx context.Context, f *Fleet) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		if ctx.Err() == nil {
			f.say("examiner: %v", err)
		}
		return err
	}
	defer unlock()
	w, err := f.st.work(ctx, f.op)
	if err != nil {
		if ctx.Err() != nil {
			return nil
		}
		f.say("examiner: %v", err)
		if exErr := f.examiner.EnforceTimeout(ctx); exErr != nil && ctx.Err() == nil {
			f.say("examiner: %v", exErr)
		}
		return err
	}
	var acctErr error
	l, lerr := f.readLedger()
	today, liveWorkers, hourlyWorkers := 0.0, 0, 0.0
	if lerr != nil {
		acctErr = fmt.Errorf("reading fleet ledger: %w", lerr)
	} else {
		t, serr := f.spent(l, startOfDay(f.now()))
		if serr != nil {
			acctErr = fmt.Errorf("reading daily spend: %w", serr)
		} else {
			today = t
		}
		for _, p := range l.Pods {
			liveWorkers++
			hourlyWorkers += p.Cost
		}
	}
	_, _, exAccrued, snapErr := f.examinerSnapshot()
	if snapErr != nil && acctErr == nil {
		acctErr = snapErr
	}
	budget := ExaminerBudget{
		LiveWorkers:   liveWorkers,
		MaxWorkers:    f.o.MaxWorkers,
		HourlyNow:     hourlyWorkers,
		MaxHourly:     f.o.MaxHourly,
		SpentToday:    today + exAccrued,
		DailyBudget:   f.o.DailyBudget,
		AccountingErr: acctErr,
	}
	if l != nil && acctErr == nil {
		budget.CurrentSpentToday = func() (float64, error) {
			spentBase, serr := f.spent(l, startOfDay(f.now()))
			if serr != nil {
				return 0, fmt.Errorf("reading daily spend: %w", serr)
			}
			_, _, exAcc, aerr := f.examinerSnapshot()
			if aerr != nil {
				return 0, aerr
			}
			return spentBase + exAcc, nil
		}
	}
	if err := f.examiner.Reconcile(ctx, w, budget); err != nil {
		if ctx.Err() == nil {
			f.say("examiner: %v", err)
		}
		return err
	}
	return nil
}

func cmdExaminer(ctx context.Context, f *Fleet) error {
	stopExaminer, err := f.startExaminer(ctx)
	if err != nil {
		return err
	}
	defer stopExaminer()
	every := min(f.o.Every, 15*time.Second)
	if every <= 0 {
		every = 15 * time.Second
	}
	for {
		_ = reconcileExaminerOnce(ctx, f)
		if ctx.Err() != nil {
			return nil
		}
		if err := sleepCtx(ctx, every); err != nil {
			return nil
		}
	}
}

func cmdUp(ctx context.Context, f *Fleet, n int) error {
	_, err := cmdUpRentals(ctx, f, n)
	return err
}

func cmdUpRentals(ctx context.Context, f *Fleet, n int) ([]*Rental, error) {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return nil, err
	}
	defer unlock()
	v, err := f.look(ctx)
	if err != nil {
		return nil, err
	}
	live := v.ExaminerLive
	for _, w := range v.Workers {
		if w.Rental != nil && w.Rental.live() {
			live++
		}
	}
	if room := f.o.MaxWorkers - live; n > room {
		f.say("%d rented already, and --max is %d: renting %d", live, f.o.MaxWorkers, max(room, 0))
		n = max(room, 0)
	}
	if v.Today >= f.o.DailyBudget && f.o.DailyBudget > 0 {
		return nil, fmt.Errorf("about $%.2f spent today, and the daily budget is $%.2f", v.Today, f.o.DailyBudget)
	}
	runs := 0
	if f.o.Edits {
		if v.Work == nil {
			runs = n
		} else {
			for _, r := range v.Work.Rounds {
				if r.RunsWanted > 0 || r.RunsWaiting > 0 {
					runs = n
					break
				}
			}
			if runs == 0 {
				if want := f.o.Policy.runners(v.Work); want > 0 {
					runs = min(n, want)
				}
			}
		}
	}
	rented, err := f.launchExcludingRentals(ctx, v, n, 0, 0, runs, time.Time{}, nil)
	if err == nil && len(rented) > 0 {
		f.say("each stops by itself after %d minutes with nothing to grade", f.o.Idle)
	}
	return rented, err
}

func cmdDown(ctx context.Context, f *Fleet, force bool, handles []string) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	v, err := f.look(ctx)
	if err != nil {
		return err
	}
	want, found := map[string]bool{}, map[string]bool{}
	for _, h := range handles {
		want[h] = true
	}
	var endings []Ending
	for _, w := range v.Workers {
		if w.Rental == nil || (len(want) > 0 && !want[w.Key.Handle]) {
			continue
		}
		found[w.Key.Handle] = true
		st := w.state(f.now(), f.o.Policy)
		if st == "grading" || st == "running" || st == "training" {
			if !force {
				f.say("left %s, which is %s: it stops by itself once there is nothing more for it. --force ends it now, and hands its job back", w.Key.Handle, st)
				continue
			}
			endings = append(endings, Ending{Worker: w, Why: "ended by hand, in the middle of a job; it hands the job back"})
			continue
		}
		endings = append(endings, Ending{Worker: w, Why: "ended by hand"})
	}
	for h := range want {
		if !found[h] {
			f.say("%s has no machine", h)
		}
	}
	if len(endings) == 0 {
		f.say("nothing to end")
	}
	return f.end(ctx, v, endings)
}

// cmdDownRental ends one specific rental by exact (market, id) under the fleet lock,
// retiring only that ledger entry and never touching a replacement machine that may have
// reused the same worker handle. An already-gone rental is treated as complete.
func cmdDownRental(ctx context.Context, f *Fleet, market, id string) error {
	if id == "" || id == "dry-run" {
		return nil
	}
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	if err := f.refresh(); err != nil {
		return err
	}
	m := f.market(market)
	if m == nil {
		return fmt.Errorf("could not end %s on %s: there is no key for %s now", id, marketTitle(market), marketTitle(market))
	}
	l, err := f.readLedger()
	if err != nil {
		return err
	}
	key := ledgerKey(market, id)
	rec := l.Pods[key]
	var found *Rental
	listedOK := false
	if rs, rerr := m.Rentals(ctx); rerr == nil {
		listedOK = true
		for i := range rs {
			if rs[i].ID == id && (rs[i].Market == "" || rs[i].Market == market) {
				found = &rs[i]
				break
			}
		}
	}
	if listedOK && found == nil {
		if rec != nil {
			rec.Why = "probe ended before cleanup"
			if err := f.retire(l, key, minTime(f.now(), rec.Alive.Add(2*f.o.Every))); err != nil {
				return err
			}
			return f.writeLedger(l)
		}
		return nil
	}
	if f.o.DryRun {
		f.say("would end probe machine %s on %s", id, marketTitle(market))
		return nil
	}
	if err := m.End(ctx, id); err != nil {
		return err
	}
	handle := ""
	if rec != nil {
		handle = rec.Handle
		rec.Why = "probe ended"
	} else if found != nil {
		handle = found.Name
	}
	r := Rental{Market: market, ID: id}
	if found != nil {
		r = *found
	} else if rec != nil {
		r.Card = rec.GPU
	}
	f.event(&r, handle, "ended", "probe ended")
	if rec != nil {
		if err := f.retire(l, key, f.now()); err != nil {
			return err
		}
		if err := f.writeLedger(l); err != nil {
			return err
		}
	}
	if handle != "" {
		f.say("ended probe %s (%s on %s)", handle, id, marketTitle(market))
	} else {
		f.say("ended probe %s on %s", id, marketTitle(market))
	}
	return nil
}

func cmdSSH(ctx context.Context, f *Fleet, handle string) error {
	rp := f.runpod()
	if rp == nil {
		return errors.New("ssh is for machines on RunPod, and there is no key for RunPod")
	}
	pods, err := rp.pods(ctx)
	if err != nil {
		return err
	}
	for i := range pods {
		p := &pods[i]
		if p.Name != handle || !p.live() {
			continue
		}
		if p.SSH == nil || p.SSH.Direct == nil || p.SSH.Direct.Command == "" {
			return fmt.Errorf("%s has no way in: it was rented without --ssh, or is still starting", handle)
		}
		args := strings.Fields(p.SSH.Direct.Command)
		bin, err := exec.LookPath(args[0])
		if err != nil {
			return err
		}
		fmt.Fprintf(f.out, "%s\n", p.SSH.Direct.Command)
		return syscall.Exec(bin, args, os.Environ())
	}
	return errors.New(handle + " has no pod running")
}

// cmdLogs prints what a machine's pod has said: the last lines, or everything as it comes.
func cmdLogs(ctx context.Context, f *Fleet, handle string, follow bool, last int) error {
	rp := f.runpod()
	if rp == nil {
		return errors.New("logs are for machines on RunPod, and there is no key for RunPod")
	}
	pods, err := rp.pods(ctx)
	if err != nil {
		return err
	}
	var pod *Pod
	for i := range pods {
		if pods[i].Name == handle && (pod == nil || pods[i].CreatedAt.After(pod.CreatedAt)) {
			pod = &pods[i]
		}
	}
	if pod == nil {
		return errors.New(handle + " has no pod")
	}
	show := func(l LogLine) {
		ts := l.TS
		if len(ts) >= 19 {
			ts = ts[11:19]
		}
		fmt.Fprintf(f.out, "%s  %s\n", ts, l.Line)
	}
	if follow {
		return rp.logs(ctx, pod.ID, true, show)
	}
	var kept []LogLine
	err = rp.logs(ctx, pod.ID, false, func(l LogLine) {
		if kept = append(kept, l); len(kept) > last {
			kept = kept[1:]
		}
	})
	for _, l := range kept {
		show(l)
	}
	return err
}
