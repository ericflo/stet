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
		fmt.Fprintln(f.out, "No keys yet. Make some: stet-runpod keys --add 4")
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
		fmt.Fprintf(f.out, "\n%d not vouched for: stet-runpod keys --vouch (they are vouched for when first rented, too).\n", missing)
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
		f.say("took back %s: the server no longer takes it as the organisers', and its key is gone from here. A machine still running with it should be ended: stet-runpod down %s", h, h)
	}
	return nil
}

func cmdGPUs(ctx context.Context, f *Fleet) error {
	gpus, err := f.choices(ctx)
	if err != nil {
		return err
	}
	if len(gpus) == 0 {
		fmt.Fprintf(f.out, "RunPod has no GPU free now that would do (%s cloud).\n", strings.ToLower(f.o.Cloud))
		return nil
	}
	fmt.Fprintf(f.out, "What would be rented, in this order (%s cloud, CUDA %s or newer, hosts with %d GB and %d vCPUs a GPU):\n\n",
		strings.ToLower(f.o.Cloud), f.o.MinCUDA, f.o.MinRAM, f.o.MinCPU)
	for _, c := range gpus {
		fmt.Fprintf(f.out, "  $%.2f/h  %-9s  %-34s %3d GB  %s\n", c.Price, strings.ToLower(c.Cloud), shortGPU(c.GPU.ID), c.GPU.Memory, strings.ToLower(c.GPU.Availability))
	}
	fmt.Fprintln(f.out, "\nEach is tried in turn until RunPod has a machine of that kind free with room for the rest.")
	return nil
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
	d := f.o.Policy.decide(v.Work, v.Workers, v.Today, now)
	if len(v.Work.Rounds) == 0 {
		fmt.Fprintln(f.out, "No round is open.")
	}
	for _, r := range v.Work.Rounds {
		fmt.Fprintf(f.out, "Round %s: %d entries open · %d runs a new machine could take · %d running · %d scripts to grade · %d being graded\n",
			r.ID, r.Entries, r.RunsWanted, r.EditsUnderWay, r.GradesWanted, r.GradesUnderWay)
	}
	if v.Work.NextDrawAt > 0 {
		at := time.Unix(v.Work.NextDrawAt, 0)
		fmt.Fprintf(f.out, "Next draw %s UTC (%s)\n", at.UTC().Format("15:04"), until(now, at))
	}
	fmt.Fprintf(f.out, "%d of the organisers' machines running curators\n\n", v.Work.Runners)
	rented := 0
	for _, w := range v.Workers {
		st := w.state(now, f.o.Policy)
		if st == "free" {
			continue
		}
		rented++
		fmt.Fprintf(f.out, "  %-14s %-8s %-26s $%.2f/h  up %s\n", w.Key.Handle, st, w.Pod.gpuName(), w.Pod.Cost, now.Sub(w.Pod.CreatedAt).Round(time.Minute))
	}
	if rented == 0 {
		fmt.Fprintf(f.out, "Nothing rented (%d keys in the pool).\n", len(v.Workers))
	}
	fmt.Fprintf(f.out, "\n$%.2f an hour now · about $%.2f today (UTC)\n", v.Hourly, v.Today)
	fmt.Fprintf(f.out, "The work wants %d machines now (at most %d)", d.Target, f.o.MaxWorkers)
	if d.Launch > 0 || len(d.End) > 0 {
		fmt.Fprintf(f.out, ": `stet-runpod scale` would rent %d and end %d", d.Launch, len(d.End))
	}
	fmt.Fprintln(f.out, ".")
	if d.Note != "" {
		fmt.Fprintf(f.out, "(%s)\n", d.Note)
	}
	return nil
}

func until(now, t time.Time) string {
	d := t.Sub(now).Round(time.Minute)
	if d <= 0 {
		return "now"
	}
	return "in " + strings.TrimSuffix(d.String(), "0s")
}

// tick is one round of scale: look, decide, act.
func tick(ctx context.Context, f *Fleet) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	v, err := f.look(ctx)
	if err != nil {
		return err
	}
	d := f.o.Policy.decide(v.Work, v.Workers, v.Today, f.now())
	if err := f.end(ctx, v, d.End); err != nil {
		return err
	}
	if d.Note != "" && d.Launch < d.Target-d.Live {
		f.say("%s", d.Note)
	}
	_, err = f.launch(ctx, v, d.Launch)
	return err
}

func cmdScale(ctx context.Context, f *Fleet, follow bool) error {
	if !follow {
		return tick(ctx, f)
	}
	f.say("scaling to the work every %s, at most %d machines and $%.2f an hour (%s cloud). Ctrl-C to stop; the machines stop by themselves.",
		f.o.Every, f.o.MaxWorkers, f.o.MaxHourly, strings.ToLower(f.o.Cloud))
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

func cmdUp(ctx context.Context, f *Fleet, n int) error {
	unlock, err := f.lock(time.Minute)
	if err != nil {
		return err
	}
	defer unlock()
	v, err := f.look(ctx)
	if err != nil {
		return err
	}
	live := 0
	for _, w := range v.Workers {
		if w.Pod != nil && w.Pod.live() {
			live++
		}
	}
	if room := f.o.MaxWorkers - live; n > room {
		f.say("%d rented already, and --max is %d: renting %d", live, f.o.MaxWorkers, max(room, 0))
		n = max(room, 0)
	}
	if v.Today >= f.o.DailyBudget && f.o.DailyBudget > 0 {
		return fmt.Errorf("about $%.2f spent today, and the daily budget is $%.2f", v.Today, f.o.DailyBudget)
	}
	got, err := f.launch(ctx, v, n)
	if err == nil && got > 0 {
		f.say("each stops by itself after %d minutes with nothing to grade", f.o.Idle)
	}
	return err
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
		if w.Pod == nil || (len(want) > 0 && !want[w.Key.Handle]) {
			continue
		}
		found[w.Key.Handle] = true
		st := w.state(f.now(), f.o.Policy)
		if st == "grading" || st == "running" {
			if !force {
				f.say("left %s, which is %s: it stops by itself once there is nothing more for it. --force ends it now, and hands its job back", w.Key.Handle, st)
				continue
			}
			endings = append(endings, Ending{w, "ended by hand, in the middle of a job; it hands the job back"})
			continue
		}
		endings = append(endings, Ending{w, "ended by hand"})
	}
	for h := range want {
		if !found[h] {
			f.say("%s has no pod", h)
		}
	}
	if len(endings) == 0 {
		f.say("nothing to end")
	}
	return f.end(ctx, v, endings)
}

func cmdSSH(ctx context.Context, f *Fleet, handle string) error {
	pods, err := f.rp.pods(ctx)
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
	pods, err := f.rp.pods(ctx)
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
		return f.rp.logs(ctx, pod.ID, true, show)
	}
	var kept []LogLine
	err = f.rp.logs(ctx, pod.ID, false, func(l LogLine) {
		if kept = append(kept, l); len(kept) > last {
			kept = kept[1:]
		}
	})
	for _, l := range kept {
		show(l)
	}
	return err
}
