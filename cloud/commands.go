package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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
	statusPol.MaxWorkers = max(0, statusPol.MaxWorkers-statusReserved-v.extraWorkerLiveCount())
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
		if st == "free" || st == "pending" || w.Rental == nil {
			continue
		}
		rented++
		card := w.Rental.Card
		if w.Rental.Market != "runpod" {
			card += " (" + marketTitle(w.Rental.Market) + ")"
		}
		note := ""
		if st == "terminating" {
			note = "  · outstanding termination (run `stet-cloud workers resolve --end`)"
		} else if st == "unconfirmed" || st == "starting" || st == "stopped" || st == "quiet" || w.Rental.Up == "no" {
			if msg := w.Rental.Detail; msg != "" {
				note = "  · " + msg // what its market says of it, while it is not at work
			} else if w.Rental.Status != "" {
				note = "  · " + w.Rental.Status
			}
		}
		if w.NoCurators != "" {
			note += "  · grades only: " + w.NoCurators
		}
		if t := w.Machine.trouble(); t != "" && w.Machine.LastSeen >= w.Rental.Created.Unix() {
			note += "  · " + t
		}
		rate := fmt.Sprintf("$%.2f/h", w.Rental.Price)
		if !validRate(w.Rental.Price) {
			rate = "unknown/h"
		}
		fmt.Fprintf(f.out, "  %-14s %-11s %-26s %-9s up %s%s\n", w.Key.Handle, st, card, rate, now.Sub(w.Rental.Created).Round(time.Minute), note)
		for _, ex := range w.ExtraRentals {
			if ex == nil {
				continue
			}
			rented++
			ecard := ex.Card
			if ex.Market != "runpod" {
				ecard += " (" + marketTitle(ex.Market) + ")"
			}
			erate := fmt.Sprintf("$%.2f/h", ex.Price)
			if !validRate(ex.Price) {
				erate = "unknown/h"
			}
			est := ex.State
			if ex.Terminating {
				est = "terminating"
			} else if ex.Unconfirmed && ex.State == "stopped" {
				est = "unconfirmed"
			}
			fmt.Fprintf(f.out, "  %-14s %-11s %-26s %-9s up %s  · extra contract %s\n",
				w.Key.Handle, est, ecard, erate, now.Sub(ex.Created).Round(time.Minute), ex.ID)
		}
	}
	for _, orph := range v.OrphanRentals {
		if orph == nil {
			continue
		}
		rented++
		card := orph.Card
		if orph.Market != "runpod" {
			card += " (" + marketTitle(orph.Market) + ")"
		}
		name := orph.Name
		if name == "" {
			name = orph.ID
		}
		orate := fmt.Sprintf("$%.2f/h", orph.Price)
		if !validRate(orph.Price) {
			orate = "unknown/h"
		}
		ost := orph.State
		if orph.Terminating {
			ost = "terminating"
		} else if orph.Unconfirmed && orph.State == "stopped" {
			ost = "unconfirmed"
		}
		fmt.Fprintf(f.out, "  %-14s %-11s %-26s %-9s up %s  · unpooled contract %s\n",
			name, ost, card, orate, now.Sub(orph.Created).Round(time.Minute), orph.ID)
	}
	for _, pr := range v.PendingRents {
		if pr == nil {
			continue
		}
		rented++
		card := pr.GPU + " (" + marketTitle(pr.Market) + ")"
		prate := fmt.Sprintf("$%.2f/h", pr.Cost)
		if !validRate(pr.Cost) {
			prate = "unknown/h"
		}
		fmt.Fprintf(f.out, "  %-14s %-11s %-26s %-9s up %s  · uncertain create %s (run `stet-cloud workers resolve`)\n",
			pr.Handle, "pending", card, prate, now.Sub(pr.Created).Round(time.Minute), pr.ID)
	}
	if rented == 0 {
		fmt.Fprintf(f.out, "Nothing rented (%d keys in the pool).\n", len(v.Workers))
	}
	if ex := readExaminerState(f.o.Home); ex != nil && ex.InstanceID != "" {
		fmt.Fprintf(f.out, "Examiner GPU: machine %s for %s ($%.2f/h, up %s)\n", ex.InstanceID, ex.RoundID, ex.Price, now.Sub(ex.Created).Round(time.Minute))
	} else if ex != nil && ex.PendingRent {
		fmt.Fprintf(f.out, "Examiner GPU: rental reservation %s pending for %s ($%.2f/h; run `stet-cloud examiner resolve` after verifying on Vast.ai)\n", ex.ReservedLabel, ex.RoundID, ex.Price)
	} else if ex != nil && len(ex.Unresolved) > 0 {
		fmt.Fprintf(f.out, "Examiner GPU: %d unresolved instance(s) awaiting teardown\n", len(ex.Unresolved))
	} else if ex != nil && len(ex.PendingSpends) > 0 {
		fmt.Fprintf(f.out, "Examiner GPU: idle (%d pending spend record(s) awaiting spent.jsonl flush)\n", len(ex.PendingSpends))
	}
	if v.hourlyUnknown() {
		fmt.Fprintf(f.out, "\n$%.2f an hour now (incomplete: rate unknown on %s) · about $%.2f today (UTC)\n", v.Hourly, v.unknownRateSummary(), v.Today)
		d.Launch = 0
		if d.Note == "" {
			d.Note = fmt.Sprintf("deferring new paid rentals until provider reports hourly rate for %s", v.unknownRateSummary())
		}
	} else {
		fmt.Fprintf(f.out, "\n$%.2f an hour now · about $%.2f today (UTC)\n", v.Hourly, v.Today)
	}
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

func (f *Fleet) examinerReserveHourly() float64 {
	if f.o.MaxHourly <= 0 {
		return 0
	}
	reserve := f.o.Examiner.MaxPrice
	if f.examiner != nil && validRate(f.examiner.cfg.MaxPrice) {
		reserve = f.examiner.cfg.MaxPrice
	}
	if !validRate(reserve) {
		reserve = 1.50
	}
	if reserve > f.o.MaxHourly {
		reserve = f.o.MaxHourly
	}
	return reserve
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
	reconcileTickExaminer := func() {
		if f.examiner == nil {
			return
		}
		var rateErr error
		if v.hourlyUnknown() {
			rateErr = fmt.Errorf("live hourly spend is incomplete (%s has no reported hourly rate)", v.unknownRateSummary())
		}
		budget := ExaminerBudget{
			LiveWorkers:   v.liveWorkerCount(),
			MaxWorkers:    f.o.MaxWorkers,
			HourlyNow:     v.liveWorkerHourly(),
			MaxHourly:     f.o.MaxHourly,
			SpentToday:    v.Today,
			DailyBudget:   f.o.DailyBudget,
			AccountingErr: rateErr,
			CurrentSpentToday: func() (float64, error) {
				spentBase, serr := f.spentFor(v.Ledger, startOfDay(f.now()), v.UnconfirmedKeys)
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
			if spentBase, serr := f.spentFor(v.Ledger, startOfDay(f.now()), v.UnconfirmedKeys); serr == nil {
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
	reconcileTickExaminer()
	now := f.now()
	pol := f.o.Policy
	needExaminer := roundNeedingExam(v.Work) != ""
	reserved := v.ExaminerLive
	if reserved == 0 && (f.examiner != nil || f.o.Examiner.Addr != "") && needExaminer {
		reserved = 1
	}
	pol.MaxWorkers = max(0, pol.MaxWorkers-reserved-v.extraWorkerLiveCount())
	d := pol.decideNext(v.Work, v.Workers, v.Today, now, next)
	for _, orph := range v.OrphanRentals {
		if orph != nil && (orph.Terminating || (orph.State == "stopped" && !orph.Unconfirmed)) {
			why := "stopped unpooled machine"
			if orph.Terminating {
				why = "outstanding termination on unpooled machine"
			}
			d.End = append(d.End, Ending{Rental: orph, Why: why})
		}
	}
	// If an available examiner offer was blocked solely by --max-hourly, reclaim headroom from
	// idle (waiting) workers so existing workers cannot strand a required examiner.
	if f.examiner != nil && v.ExaminerLive == 0 && needExaminer && exErr == nil && snapErr == nil && !v.hourlyUnknown() && f.o.MaxHourly > 0 {
		if needHourly := f.examiner.blockedHourly(); needHourly > 0 {
			endingWorkers := make(map[*Worker]bool, len(d.End))
			projectedHourly := v.Hourly
			for _, e := range d.End {
				if e.Worker != nil && (e.Rental == nil || e.Rental == e.Worker.Rental) {
					endingWorkers[e.Worker] = true
				}
				r := e.Rental
				if r == nil && e.Worker != nil {
					r = e.Worker.Rental
				}
				if r == nil {
					continue
				}
				switch {
				case r.live() || r.Unconfirmed:
					if validRate(r.Price) {
						projectedHourly = max(0, projectedHourly-r.Price)
					}
				case r.State == "stopped":
					diskRate := r.DiskCost
					if recPod := v.Ledger.Pods[ledgerKey(r.Market, r.ID)]; recPod != nil && recPod.stoppedHourlyRate() > diskRate {
						diskRate = recPod.stoppedHourlyRate()
					}
					if validRate(diskRate) {
						projectedHourly = max(0, projectedHourly-diskRate)
					}
				}
			}
			if projectedHourly+needHourly > f.o.MaxHourly+1e-9 {
				var idle []*Worker
				for _, wk := range v.Workers {
					if wk == nil || endingWorkers[wk] || wk.Rental == nil || !wk.Rental.live() || !validRate(wk.Rental.Price) {
						continue
					}
					if wk.state(now, f.o.Policy) == "waiting" {
						idle = append(idle, wk)
					}
				}
				sort.SliceStable(idle, func(i, j int) bool {
					if idle[i].Rental.Price != idle[j].Rental.Price {
						return idle[i].Rental.Price > idle[j].Rental.Price
					}
					if idle[i].Speed != idle[j].Speed {
						return idle[i].Speed < idle[j].Speed
					}
					return idle[i].Key.Handle < idle[j].Key.Handle
				})
				for _, wk := range idle {
					if projectedHourly+needHourly <= f.o.MaxHourly+1e-9 {
						break
					}
					d.End = append(d.End, Ending{Worker: wk, Why: "idle while hourly budget is needed for the round's examiner"})
					endingWorkers[wk] = true
					projectedHourly = max(0, projectedHourly-wk.Rental.Price)
					d.Live = max(0, d.Live-1)
				}
			}
		}
	}
	rec.decided(d, next)
	ended := len(v.Ledger.Pods)
	preLive := v.liveWorkerCount()
	if err := f.end(ctx, v, d.End); err != nil {
		return err
	}
	rec.Ended = max(ended-len(v.Ledger.Pods), 0)
	f.release(ctx, d.Release)
	// If ending idle/stopped workers freed a slot or hourly headroom while a required examiner is
	// still absent, give the examiner an immediate second chance before launching new workers.
	if f.examiner != nil && v.ExaminerLive == 0 && needExaminer && exErr == nil && snapErr == nil && !f.o.DryRun && (rec.Ended > 0 || v.liveWorkerCount() < preLive) && !v.hourlyUnknown() && v.liveWorkerCount() < f.o.MaxWorkers {
		reconcileTickExaminer()
		reserved = v.ExaminerLive
		if reserved == 0 && needExaminer {
			reserved = 1
		}
	}
	if reserved > v.ExaminerLive {
		v.ExaminerReservedHourly = f.examinerReserveHourly()
	} else {
		v.ExaminerReservedHourly = 0
	}
	if d.Note != "" {
		f.say("%s", d.Note)
	}
	var evicted map[string]bool
	if len(d.End) > 0 {
		evicted = make(map[string]bool, len(d.End))
		for _, e := range d.End {
			if e.Worker != nil && e.Worker.Key != nil {
				evicted[e.Worker.Key.Handle] = true
			}
		}
	}
	postLive := v.liveWorkerCount()
	launchN := min(d.Launch, max(0, (f.o.MaxWorkers-reserved)-postLive))
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
// under the same fleet lock used by `up`, `down`, and `scale`, reconciling live owned worker
// rentals with the configured markets before admitting a new examiner and deferring new rentals
// whenever fleet accounting (markets, fleet.json, or spent.jsonl) cannot be read cleanly while
// keeping teardown and timeout enforcement available.
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
	if !f.fixed && f.o.Credentials != "" {
		if rerr := f.refresh(); rerr != nil {
			acctErr = fmt.Errorf("reading market settings: %w", rerr)
		}
	}
	if (!f.fixed || len(f.keys) == 0) && f.o.Home != "" {
		if kerr := f.loadKeys(); kerr != nil && acctErr == nil {
			acctErr = fmt.Errorf("reading worker keys: %w", kerr)
		}
	}
	l, lerr := f.readLedger()
	today, liveWorkers, hourlyWorkers := 0.0, 0, 0.0
	var unconfirmed map[string]bool
	if lerr != nil {
		if acctErr == nil {
			acctErr = fmt.Errorf("reading fleet ledger: %w", lerr)
		}
	} else {
		var rentals []Rental
		var answered map[string]bool
		if acctErr == nil && len(f.markets) > 0 {
			rs, ans, rerr := f.rentals(ctx, l)
			if rerr != nil {
				acctErr = fmt.Errorf("listing market rentals: %w", rerr)
			} else {
				rentals, answered = rs, ans
				for _, m := range f.markets {
					if !answered[m.Name()] {
						acctErr = fmt.Errorf("%s did not list its rentals", marketTitle(m.Name()))
						break
					}
				}
				if acctErr == nil {
					var nerr error
					if rentals, nerr = f.note(ctx, l, rentals, answered); nerr != nil {
						acctErr = fmt.Errorf("reconciling fleet ledger: %w", nerr)
					}
				}
			}
		}
		spentSum, serr := f.readSpentSummary()
		if serr != nil && acctErr == nil {
			acctErr = fmt.Errorf("reading spend log: %w", serr)
		}
		seen := map[string]bool{}
		knownHandles := map[string]bool{}
		for _, k := range f.keys {
			knownHandles[k.Handle] = true
		}
		unconfirmed = map[string]bool{}
		confirmedListed := map[string]bool{}
		for i := range rentals {
			key := ledgerKey(rentals[i].Market, rentals[i].ID)
			if rentals[i].Unconfirmed {
				unconfirmed[key] = true
			} else {
				confirmedListed[key] = true
			}
		}
		for key := range confirmedListed {
			delete(unconfirmed, key)
		}
		var unknownRates []string
		for key, p := range l.Pods {
			if p == nil || spentSum.hasContract(p.market(), p.ID) {
				continue
			}
			seen[key] = true
			if !confirmedListed[key] {
				unconfirmed[key] = true
			}
			if p.Stopped && !unconfirmed[key] {
				if rate := p.stoppedHourlyRate(); rate > 0 {
					hourlyWorkers += rate
				} else if p.DiskCostUnknown {
					unknownRates = append(unknownRates, unknownRateLabel(p.market(), p.ID, p.Handle))
				}
				continue
			}
			liveWorkers++
			if validRate(p.Cost) {
				hourlyWorkers += p.Cost
			}
			if !validRate(p.Cost) || p.PartialCost {
				unknownRates = append(unknownRates, unknownRateLabel(p.market(), p.ID, p.Handle))
			}
		}
		for i := range rentals {
			r := &rentals[i]
			key := ledgerKey(r.Market, r.ID)
			if seen[key] {
				continue
			}
			if h := owner(l, r); h != "" && knownHandles[h] {
				if r.live() || r.Unconfirmed {
					seen[key] = true
					liveWorkers++
					if validRate(r.Price) {
						hourlyWorkers += r.Price
					}
					if !validRate(r.Price) || r.PartialRate {
						unknownRates = append(unknownRates, unknownRateLabel(r.Market, r.ID, h))
					}
				} else if r.State == "stopped" {
					seen[key] = true
					if validRate(r.DiskCost) {
						hourlyWorkers += r.DiskCost
					} else if r.DiskCostUnknown {
						unknownRates = append(unknownRates, unknownRateLabel(r.Market, r.ID, h))
					}
				}
			}
		}
		for _, pr := range l.Pending {
			if pr == nil || (spentSum != nil && spentSum.byPendingID[pr.ID]) {
				continue
			}
			liveWorkers++
			if validRate(pr.Cost) {
				hourlyWorkers += pr.Cost
			} else {
				unknownRates = append(unknownRates, unknownRateLabel(pr.Market, pr.ID, pr.Handle))
			}
		}
		if len(unknownRates) > 0 && acctErr == nil {
			sort.Strings(unknownRates)
			acctErr = fmt.Errorf("live hourly spend is incomplete (%s has no reported hourly rate)", strings.Join(unknownRates, ", "))
		}
		t, serr := f.spentFor(l, startOfDay(f.now()), unconfirmed)
		if serr != nil {
			if acctErr == nil {
				acctErr = fmt.Errorf("reading daily spend: %w", serr)
			}
		} else {
			today = t
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
			spentBase, serr := f.spentFor(l, startOfDay(f.now()), unconfirmed)
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

// cmdExaminerResolve lets an operator resolve an uncertain PendingRent examiner reservation in
// examiner.json after querying Vast.ai under the fleet lock: if Vast.ai lists the reserved
// instance it is adopted (or destroyed when end is true); if Vast.ai lists zero matching
// instances the reservation is kept unless the operator explicitly verified absence and passed
// clearLabel matching ReservedLabel (preserving PendingSpends and Unresolved).
func cmdExaminerResolve(ctx context.Context, f *Fleet, end bool, clearLabel string) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	if !f.fixed {
		if err := f.refresh(); err != nil {
			return err
		}
	}
	var prov ExaminerProvider
	if f.examiner != nil && f.examiner.prov != nil {
		prov = f.examiner.prov
	} else if vm, ok := f.market("vast").(*vastMarket); ok && vm != nil {
		prov = vm
	}
	if prov == nil {
		return fmt.Errorf("examiner resolve requires VAST_API_KEY in %s (or environment)", f.o.Credentials)
	}
	return resolvePendingExaminer(ctx, f.o.Home, prov, f.now, f.say, f.o.DryRun, end, clearLabel, func(rec podRecord) error {
		return appendExaminerSpend(f.spentLog(), rec)
	})
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
	live := v.totalLiveCount()
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
		if len(want) > 0 && !want[w.Key.Handle] {
			continue
		}
		if w.Rental == nil && len(w.ExtraRentals) == 0 {
			if w.Pending != nil {
				found[w.Key.Handle] = true
				f.say("%s has an unresolved pending worker create on %s: run `stet-cloud workers resolve --end`", w.Key.Handle, marketTitle(w.Pending.Market))
			}
			continue
		}
		found[w.Key.Handle] = true
		st := w.state(f.now(), f.o.Policy)
		if w.Rental != nil {
			if st == "grading" || st == "running" || st == "training" {
				if !force {
					f.say("left %s, which is %s: it stops by itself once there is nothing more for it. --force ends it now, and hands its job back", w.Key.Handle, st)
				} else {
					endings = append(endings, Ending{Worker: w, Rental: w.Rental, Why: "ended by hand, in the middle of a job; it hands the job back", Forced: true})
				}
			} else if st == "unconfirmed" && !force {
				f.say("left %s, whose provider state is unconfirmed (%s): --force ends it now", w.Key.Handle, w.Rental.Status)
			} else {
				endings = append(endings, Ending{Worker: w, Rental: w.Rental, Why: "ended by hand", Forced: force})
			}
		}
		for _, ex := range w.ExtraRentals {
			if ex == nil {
				continue
			}
			if (ex.live() || ex.Unconfirmed) && !ex.Terminating && !force && (ex.Unconfirmed || st == "grading" || st == "running" || st == "training") {
				whySt := st
				if ex.Unconfirmed {
					whySt = "unconfirmed"
				}
				f.say("left extra contract %s on %s while %s (--force ends it now)", ex.ID, w.Key.Handle, whySt)
				continue
			}
			endings = append(endings, Ending{Worker: w, Rental: ex, Why: "ended by hand", Forced: force})
		}
	}
	for _, orph := range v.OrphanRentals {
		if orph == nil {
			continue
		}
		if len(want) > 0 && !want[orph.Name] && !want[orph.ID] {
			continue
		}
		if orph.Name != "" {
			found[orph.Name] = true
		}
		found[orph.ID] = true
		if (orph.live() || orph.Unconfirmed) && !orph.Terminating && !force {
			name := orph.Name
			if name == "" {
				name = orph.ID
			}
			stateDesc := orph.State
			if orph.Unconfirmed {
				stateDesc = "unconfirmed"
			}
			f.say("left unpooled machine %s (%s on %s), which is still %s: --force ends it now", name, orph.ID, marketTitle(orph.Market), stateDesc)
			continue
		}
		endings = append(endings, Ending{Rental: orph, Why: "ended by hand", Forced: force})
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

// cmdWorkersResolve reconciles unresolved pending worker create reservations AND persisted
// outstanding terminations (`Terminating: true`) in fleet.json against each provider under the
// fleet lock:
//   - persisted terminating contract IDs are retried via m.End(ctx, id) when end is true until
//     provider deletion or confirmed absence, and reported as errors when end is false;
//   - pending reservations with a single unambiguous matching contract are adopted into fleet.json
//     (or destroyed when end is true, persisting Terminating=true on the exact contract ID if
//     deletion fails transiently so retrying `workers resolve --end` retries that contract);
//   - genuinely ambiguous pending matches (multiple candidate contracts or multiple competing
//     handles) are left pending without adopting or canceling any contract;
//   - unmatched reservations on an inconclusive provider listing are kept intact unless the
//     operator explicitly verified absence and scoped release to that reservation ID via --clear.
func cmdWorkersResolve(ctx context.Context, f *Fleet, end bool, clearIDs ...string) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	if !f.fixed {
		if err := f.refresh(); err != nil {
			return err
		}
	}
	l, err := f.readLedger()
	if err != nil {
		return err
	}
	clearSet := map[string]bool{}
	var clearList []string
	for _, raw := range clearIDs {
		for _, part := range strings.Split(raw, ",") {
			if cid := strings.TrimSpace(part); cid != "" && !clearSet[cid] {
				clearSet[cid] = true
				clearList = append(clearList, cid)
			}
		}
	}
	var termKeys []string
	for key, rec := range l.Pods {
		if rec != nil && rec.Terminating {
			termKeys = append(termKeys, key)
		}
	}
	sort.Strings(termKeys)
	pkeys := make([]string, 0, len(l.Pending))
	for id, pr := range l.Pending {
		if pr != nil {
			pkeys = append(pkeys, id)
		}
	}
	sort.Strings(pkeys)
	if len(pkeys) == 0 && len(termKeys) == 0 {
		if len(clearList) > 0 {
			return fmt.Errorf("no pending worker reservation matching --clear %s", strings.Join(clearList, ", "))
		}
		f.say("no pending worker reservations in fleet.json")
		return nil
	}
	spentSum, err := f.readSpentSummary()
	if err != nil {
		return fmt.Errorf("reading spend log: %w", err)
	}
	now := f.now()
	changed := false
	var firstErr error
	for _, cid := range clearList {
		if l.Pending[cid] == nil {
			e := fmt.Errorf("no pending worker reservation matching --clear %q", cid)
			f.say("%v", e)
			if firstErr == nil {
				firstErr = e
			}
		}
	}
	endedTermKeys := map[string]bool{}

	// 1. Reconcile any persisted outstanding terminations bound to exact contract IDs.
	fallbackListed := map[string][]Rental{}
	fallbackListErr := map[string]error{}
	lookupContract := func(m Market, id string) (fresh *Rental, exists bool, direct bool, err error) {
		if lu, ok := m.(rentalLookup); ok {
			f, ex, lerr := lu.Lookup(ctx, id)
			return f, ex, true, lerr
		}
		mName := m.Name()
		rs, loaded := fallbackListed[mName]
		lerr, hadErr := fallbackListErr[mName]
		if !loaded && !hadErr {
			rs, lerr = m.Rentals(ctx)
			if lerr != nil {
				fallbackListErr[mName] = lerr
			} else {
				fallbackListed[mName] = rs
			}
		}
		if lerr != nil {
			return nil, false, false, lerr
		}
		for i := range rs {
			if rs[i].ID == id {
				cp := rs[i]
				return &cp, true, false, nil
			}
		}
		return nil, false, false, nil
	}
	terminatedEndAt := func(rec *podRecord, fresh *Rental) (time.Time, bool) {
		minEnd := rec.currentRunStart()
		if !rec.Alive.IsZero() && (minEnd.IsZero() || rec.Alive.After(minEnd)) {
			minEnd = rec.Alive
		}
		if rec.Stopped {
			stoppedAt := rec.StoppedAt
			if stoppedAt.IsZero() {
				stoppedAt = rec.currentStopStart(now)
			}
			if !stoppedAt.IsZero() && (minEnd.IsZero() || stoppedAt.After(minEnd)) {
				minEnd = stoppedAt
			}
			if !rec.StoppedSeen.IsZero() && (minEnd.IsZero() || rec.StoppedSeen.After(minEnd)) {
				minEnd = rec.StoppedSeen
			}
		}
		var providerEndAt time.Time
		if fresh != nil {
			providerEndAt = fresh.EndedAt
		}
		endAt := resolveStopTime(minEnd, now, providerEndAt)
		trusted := !providerEndAt.IsZero() && endAt.Equal(providerEndAt)
		return endAt, trusted
	}

	for _, key := range termKeys {
		rec := l.Pods[key]
		if rec == nil {
			continue
		}
		mName := rec.market()
		if spentSum.hasContract(mName, rec.ID) {
			if f.o.DryRun {
				f.say("would clear already-retired terminating worker %s (%s on %s)", rec.Handle, rec.ID, marketTitle(mName))
				continue
			}
			delete(l.Pods, key)
			endedTermKeys[key] = true
			changed = true
			f.say("cleared terminating worker %s (%s on %s): already retired in spent.jsonl", rec.Handle, rec.ID, marketTitle(mName))
			continue
		}
		m := f.market(mName)
		if m == nil {
			e := fmt.Errorf("resolving terminating worker %s (%s on %s) requires credentials: %s", rec.Handle, rec.ID, marketTitle(mName), f.off[mName])
			f.say("%v", e)
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		endAt := now
		if !end {
			if lu, ok := m.(rentalLookup); ok {
				fresh, exists, lerr := lu.Lookup(ctx, rec.ID)
				if lerr == nil && !exists {
					if f.o.DryRun {
						f.say("would clear absent terminating worker %s (%s on %s)", rec.Handle, rec.ID, marketTitle(mName))
						continue
					}
					if rec.Stopped {
						// 404 confirms absence without proving the worker stayed stopped during
						// [StoppedSeen, endAt]; conservatively budget any unobserved gap at compute rate.
						rec.markLiveAt(endAt, time.Time{})
					}
					f.event(&Rental{Market: mName, ID: rec.ID, Name: rec.Handle, Host: rec.Host, Card: rec.GPU, Price: rec.Cost}, rec.Handle, "ended", rec.Why)
					if rerr := f.retire(l, key, endAt); rerr != nil && firstErr == nil {
						firstErr = rerr
					}
					endedTermKeys[key] = true
					changed = true
					f.say("cleared terminating worker %s (%s on %s): provider confirmed absent", rec.Handle, rec.ID, marketTitle(mName))
					continue
				}
				if lerr == nil && exists && fresh != nil && !f.o.DryRun {
					reconcileRentalRate(rec, fresh)
					if fresh.live() {
						rec.markLiveAt(now, fresh.ResumedAt)
						changed = true
					} else if fresh.State == "stopped" {
						rec.markStoppedAt(now, fresh.StoppedAt)
						changed = true
					}
				}
			}
			e := fmt.Errorf("worker %s (%s on %s) has an outstanding termination; run `stet-cloud workers resolve --end` to retry", rec.Handle, rec.ID, marketTitle(mName))
			f.say("%v", e)
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		if f.o.DryRun {
			f.say("would retry ending terminating worker %s (%s on %s)", rec.Handle, rec.ID, marketTitle(mName))
			continue
		}
		// Reconcile the exact contract BEFORE deleting it so a stopped worker that resumed during
		// controller downtime (or changed rate) applies the ordinary rate and transition logic
		// instead of retiring the stale stopped record at storage rates.
		fresh, exists, _, lerr := lookupContract(m, rec.ID)
		stateConfirmed := false
		if lerr == nil && exists && fresh != nil {
			reconcileRentalRate(rec, fresh)
			switch {
			case fresh.live():
				rec.markLiveAt(now, fresh.ResumedAt)
				stateConfirmed = true
				changed = true
			case fresh.State == "stopped":
				rec.markStoppedAt(now, fresh.StoppedAt)
				stateConfirmed = true
				changed = true
			default:
				var trusted bool
				endAt, trusted = terminatedEndAt(rec, fresh)
				stateConfirmed = trusted
			}
		}
		endErr := m.End(ctx, rec.ID)
		if endErr != nil {
			if lu, ok := m.(rentalLookup); ok {
				if fresh2, exists2, lerr2 := lu.Lookup(ctx, rec.ID); lerr2 == nil {
					if !exists2 {
						endErr = nil
					} else if fresh2 != nil {
						reconcileRentalRate(rec, fresh2)
						if fresh2.live() {
							rec.markLiveAt(now, fresh2.ResumedAt)
							changed = true
						} else if fresh2.State == "stopped" {
							rec.markStoppedAt(now, fresh2.StoppedAt)
							changed = true
						}
					}
				}
			}
		}
		if endErr != nil {
			f.say("could not end terminating worker %s (%s on %s): %v (still persisted for retry)", rec.Handle, rec.ID, marketTitle(mName), endErr)
			if firstErr == nil {
				firstErr = fmt.Errorf("ending %s on %s: %w", rec.ID, marketTitle(mName), endErr)
			}
			continue
		}
		if !stateConfirmed && rec.Stopped {
			// Lookup did not confirm the contract stayed stopped during [StoppedSeen, endAt];
			// retain confirmed storage through StoppedSeen and conservatively bill the unobserved
			// interval at the compute rate before retiring.
			rec.markLiveAt(endAt, time.Time{})
		}
		f.event(&Rental{Market: mName, ID: rec.ID, Name: rec.Handle, Host: rec.Host, Card: rec.GPU, Price: rec.Cost}, rec.Handle, "ended", rec.Why)
		if rerr := f.retire(l, key, endAt); rerr != nil && firstErr == nil {
			firstErr = rerr
		}
		endedTermKeys[key] = true
		changed = true
		f.say("ended terminating worker %s (%s on %s)", rec.Handle, rec.ID, marketTitle(mName))
	}

	// 2. Reconcile unresolved pending create reservations against provider listings.
	if len(spentSum.byPendingID) > 0 {
		rem := pkeys[:0]
		for _, id := range pkeys {
			if spentSum.byPendingID[id] {
				if f.o.DryRun {
					f.say("would clear already-retired pending worker reservation %s", id)
					continue
				}
				delete(l.Pending, id)
				changed = true
				f.say("cleared pending worker reservation %s (already retired in spent.jsonl)", id)
				continue
			}
			rem = append(rem, id)
		}
		pkeys = rem
	}
	marketsNeeded := map[string]bool{}
	for _, id := range pkeys {
		if pr := l.Pending[id]; pr != nil && pr.Market != "" {
			marketsNeeded[pr.Market] = true
		}
	}
	marketList := make([]string, 0, len(marketsNeeded))
	for mName := range marketsNeeded {
		marketList = append(marketList, mName)
	}
	sort.Strings(marketList)
	listedByMarket := map[string][]Rental{}
	for _, mName := range marketList {
		m := f.market(mName)
		if m == nil {
			e := fmt.Errorf("resolving pending workers on %s requires credentials: %s", marketTitle(mName), f.off[mName])
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		rs, rerr := m.Rentals(ctx)
		if rerr != nil {
			e := fmt.Errorf("listing rentals on %s: %w", marketTitle(mName), rerr)
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		if len(endedTermKeys) > 0 {
			filtered := rs[:0]
			for _, r := range rs {
				if !endedTermKeys[ledgerKey(r.Market, r.ID)] {
					filtered = append(filtered, r)
				}
			}
			rs = filtered
		}
		listedByMarket[mName] = rs
	}
	matchesByID, claimants := resolvePendingMatches(l, listedByMarket, pkeys)
	handled := map[string]bool{}
	for _, id := range pkeys {
		pr := l.Pending[id]
		if pr == nil {
			continue
		}
		if _, ok := listedByMarket[pr.Market]; !ok {
			continue
		}
		matches := matchesByID[id]
		if len(matches) == 0 {
			if !clearSet[id] {
				e := fmt.Errorf("pending worker reservation %s (%s on %s, $%.2f/h) is not listed yet; keeping reservation (verify on %s and re-run `stet-cloud workers resolve --clear %s` only if confirmed absent)",
					id, pr.Handle, marketTitle(pr.Market), pr.Cost, marketTitle(pr.Market), id)
				f.say("%v", e)
				if firstErr == nil {
					firstErr = e
				}
				continue
			}
			if f.o.DryRun {
				f.say("would clear verified pending worker reservation %s (%s on %s, $%.2f/h)", id, pr.Handle, marketTitle(pr.Market), pr.Cost)
				continue
			}
			delete(l.Pending, id)
			changed = true
			f.say("cleared pending worker reservation %s (%s on %s) via --clear", id, pr.Handle, marketTitle(pr.Market))
			continue
		}
		if len(matches) > 1 {
			e := fmt.Errorf("pending worker %s on %s ambiguously matches %d untracked contracts (%s); leaving pending without modifying any contract",
				pr.Handle, marketTitle(pr.Market), len(matches), rentalIDList(matches))
			f.say("%v", e)
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		p := matches[0]
		key := ledgerKey(p.Market, p.ID)
		if len(claimants[key]) > 1 {
			e := fmt.Errorf("pending worker %s on %s shares candidate contract %s with another pending handle; leaving pending without modifying the contract",
				pr.Handle, marketTitle(pr.Market), p.ID)
			f.say("%v", e)
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		if handled[key] {
			if !clearSet[id] {
				e := fmt.Errorf("pending worker reservation %s (%s on %s) has no remaining unmatched contract (contract %s was already resolved by another reservation); keeping reservation (re-run with `--clear %s` after verifying no second contract exists on %s)",
					id, pr.Handle, marketTitle(pr.Market), p.ID, id, marketTitle(pr.Market))
				f.say("%v", e)
				if firstErr == nil {
					firstErr = e
				}
				continue
			}
			if f.o.DryRun {
				f.say("would clear verified duplicate pending worker reservation %s (%s on %s)", id, pr.Handle, marketTitle(pr.Market))
				continue
			}
			delete(l.Pending, id)
			changed = true
			f.say("cleared pending worker reservation %s (%s on %s) via --clear", id, pr.Handle, marketTitle(pr.Market))
			continue
		}
		if clearSet[id] {
			e := fmt.Errorf("refusing --clear %q: %s lists matching contract %s for %s; run `stet-cloud workers resolve` to adopt or `--end` to terminate",
				id, marketTitle(pr.Market), p.ID, pr.Handle)
			f.say("%v", e)
			if firstErr == nil {
				firstErr = e
			}
			continue
		}
		handled[key] = true
		m := f.market(pr.Market)
		start := p.Created
		if !pr.Created.IsZero() && (start.IsZero() || start.After(pr.Created)) {
			start = pr.Created
		}
		if start.IsZero() || start.After(now) {
			start = now
		}
		alive := now
		endAt := now
		var stoppedAt, stoppedSeen time.Time
		if p.State == "stopped" {
			// Use a trustworthy provider stop timestamp when available; otherwise preserve the
			// conservative compute-rate accrual reserved while pending up through this first
			// confirmed stop, then accrue storage from now forward.
			stoppedAt = resolveStopTime(start, now, p.StoppedAt)
			stoppedSeen = now
		} else if !p.live() {
			alive = start
			endAt = start
		}
		gpu := p.Card
		if gpu == "" {
			gpu = pr.GPU
		}
		host := p.Host
		if host == "" {
			host = pr.Host
		}
		rec := &podRecord{
			ID: p.ID, Market: p.Market, Handle: pr.Handle, GPU: gpu, Cloud: pr.Cloud, PendingID: pr.ID,
			Cost: pr.Cost, DiskCost: pr.DiskCost, TransferCost: pr.TransferCost, PartialCost: !validRate(pr.Cost),
			Start: start, Alive: alive, StoppedAt: stoppedAt, StoppedSeen: stoppedSeen, Tracked: true, Stopped: p.State == "stopped",
			Grades: pr.Grades, Host: host, CPU: pr.CPU, Curators: pr.Curators,
			For: pr.For, Round: pr.Round, Report: pr.Report,
		}
		reconcileRentalRate(rec, p)
		if !end {
			if f.o.DryRun {
				f.say("would adopt matching contract %s for pending worker %s on %s", p.ID, pr.Handle, marketTitle(pr.Market))
				continue
			}
			l.Pods[key] = rec
			delete(l.Pending, id)
			changed = true
			f.event(p, pr.Handle, "rented", "adopted via workers resolve")
			f.say("adopted pending worker %s on %s (machine %s)", pr.Handle, marketTitle(p.Market), p.ID)
			continue
		}
		if f.o.DryRun {
			f.say("would end matching contract %s for pending worker %s on %s", p.ID, pr.Handle, marketTitle(pr.Market))
			continue
		}
		rec.Why = "ended via workers resolve --end"
		if err := m.End(ctx, p.ID); err != nil {
			// Persist outstanding termination against the exact contract ID so retrying
			// `workers resolve --end` (and `scale` ticks even while work wants GPUs) keeps
			// terminating this exact contract until deletion or confirmed absence.
			rec.Terminating = true
			l.Pods[key] = rec
			delete(l.Pending, id)
			changed = true
			f.say("could not end pending worker %s (%s on %s): %v (persisted as terminating in fleet.json; retry `stet-cloud workers resolve --end`)",
				pr.Handle, p.ID, marketTitle(p.Market), err)
			if firstErr == nil {
				firstErr = fmt.Errorf("ending %s on %s: %w", p.ID, marketTitle(p.Market), err)
			}
			continue
		}
		l.Pods[key] = rec
		f.event(p, pr.Handle, "ended", rec.Why)
		if rerr := f.retire(l, key, endAt); rerr != nil && firstErr == nil {
			firstErr = rerr
		}
		delete(l.Pending, id)
		changed = true
		f.say("ended pending worker %s (%s on %s)", pr.Handle, p.ID, marketTitle(p.Market))
	}
	if changed {
		if werr := f.writeLedger(l); werr != nil && firstErr == nil {
			firstErr = werr
		}
	}
	return firstErr
}

// cmdDownRental ends one specific rental by exact (market, id), attempting to acquire the fleet
// lock and update fleet.json but NEVER letting a lock failure, read-only filesystem, or corrupt
// ledger prevent issuing m.End(ctx, id) to the provider. Any lock or ledger repair error is
// reported AFTER provider teardown is confirmed.
func cmdDownRental(ctx context.Context, f *Fleet, market, id string) error {
	if id == "" || id == "dry-run" {
		return nil
	}
	unlock, lockErr := f.lock(time.Minute)
	if lockErr == nil {
		defer unlock()
	}
	m := f.market(market)
	var refreshErr error
	if lockErr == nil || m == nil {
		refreshErr = f.refresh()
		if refreshed := f.market(market); refreshed != nil {
			m = refreshed
		}
	}
	if m == nil {
		if refreshErr != nil {
			return refreshErr
		}
		if lockErr != nil {
			return lockErr
		}
		return fmt.Errorf("could not end %s on %s: there is no key for %s now", id, marketTitle(market), marketTitle(market))
	}
	if f.o.DryRun {
		f.say("would end probe machine %s on %s", id, marketTitle(market))
		return nil
	}
	var l *ledger
	var ledgerErr error
	var rec *podRecord
	key := ledgerKey(market, id)
	if lockErr == nil {
		l, ledgerErr = f.readLedger()
		if ledgerErr == nil && l != nil {
			rec = l.Pods[key]
			if rec != nil {
				if spentSum, serr := f.readSpentSummary(); serr == nil && spentSum.hasContract(market, id) {
					delete(l.Pods, key)
					if err := f.writeLedger(l); err != nil {
						return fmt.Errorf("clearing already-retired probe %s on %s from fleet ledger failed: %w", id, marketTitle(market), err)
					}
					f.say("cleared probe %s (%s on %s): already retired in spent.jsonl", rec.Handle, id, marketTitle(market))
					return nil
				}
			}
		}
	}
	stateConfirmed := false
	if rec != nil && lockErr == nil && ledgerErr == nil {
		if lu, ok := m.(rentalLookup); ok {
			if fresh, exists, lerr := lu.Lookup(ctx, id); lerr == nil && exists && fresh != nil {
				reconcileRentalRate(rec, fresh)
				now := f.now()
				if fresh.live() {
					rec.markLiveAt(now, fresh.ResumedAt)
					stateConfirmed = true
				} else if fresh.State == "stopped" {
					rec.markStoppedAt(now, fresh.StoppedAt)
					stateConfirmed = true
				}
			}
		}
	}
	if err := m.End(ctx, id); err != nil {
		return err
	}
	handle := ""
	r := Rental{Market: market, ID: id}
	if rec != nil {
		handle = rec.Handle
		rec.Why = "probe ended"
		r.Card = rec.GPU
	}
	f.event(&r, handle, "ended", "probe ended")
	if handle != "" {
		f.say("ended probe %s (%s on %s)", handle, id, marketTitle(market))
	} else {
		f.say("ended probe %s on %s", id, marketTitle(market))
	}
	if lockErr != nil {
		return fmt.Errorf("ended probe %s on %s, but fleet lock could not be acquired for ledger cleanup: %w", id, marketTitle(market), lockErr)
	}
	if ledgerErr != nil {
		return fmt.Errorf("ended probe %s on %s, but fleet ledger could not be read for cleanup: %w", id, marketTitle(market), ledgerErr)
	}
	if rec != nil {
		endAt := f.now()
		if !stateConfirmed && rec.Stopped {
			rec.markLiveAt(endAt, time.Time{})
		}
		if err := f.retire(l, key, endAt); err != nil {
			return fmt.Errorf("ended probe %s on %s, but retiring ledger entry failed: %w", id, marketTitle(market), err)
		}
		if err := f.writeLedger(l); err != nil {
			return fmt.Errorf("ended probe %s on %s, but writing fleet ledger failed: %w", id, marketTitle(market), err)
		}
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
