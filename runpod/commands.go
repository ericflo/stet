package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

func cmdKeys(ctx context.Context, f *Fleet, add int, prefix string) error {
	if add > 0 {
		unlock, err := f.lock(time.Minute)
		if err != nil {
			return err
		}
		defer unlock()
		made, err := f.addKeys(add, prefix)
		if err != nil {
			return err
		}
		names := make([]string, len(made))
		for i, k := range made {
			names[i] = k.Handle
		}
		fmt.Fprintf(f.out, "Made %d keys: %s (in %s).\n\n", len(made), strings.Join(names, ", "), f.keyDir())
	}
	if len(f.keys) == 0 {
		fmt.Fprintln(f.out, "No keys yet. Make some: stet-runpod keys --add 4")
		return nil
	}
	// What the server knows of each, when the organisers' key is to hand to ask with.
	var op *Key
	if k, err := loadKey(f.o.OperatorKey); err == nil {
		op = k
	}
	known := 0
	for _, k := range f.keys {
		say := ""
		if op != nil {
			exists, worker, err := f.st.lookup(ctx, op, k.Handle)
			switch {
			case err != nil:
				say = "the server could not be asked: " + err.Error()
			case worker:
				say, known = "a rented machine to the server", known+1
			case exists:
				say = "has an account, and the server does not know the key as a rented machine yet"
			default:
				say = "no account yet: it is made when the machine is first rented"
			}
		}
		fmt.Fprintf(f.out, "  %-14s %s…  %s\n", k.Handle, k.Public[:12], say)
	}
	fmt.Fprintf(f.out, "\nThe server takes these keys as the organisers' rented machines once they are in\n"+
		"STET_WORKER_KEYS (Epsilon: apps/stet/deployment.yaml) and it has been started again.\n"+
		"Each machine counts as one of the organisers', and is given no held-out exam.\n\n"+
		"STET_WORKER_KEYS=%s\n", f.workerKeys())
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
	fmt.Fprintf(f.out, "What would be rented, in this order (%s cloud, CUDA %s or newer):\n\n", strings.ToLower(f.o.Cloud), f.o.MinCUDA)
	for _, g := range gpus {
		fmt.Fprintf(f.out, "  $%.2f/h  %-34s %3d GB  %s\n", g.price(f.o.Cloud), shortGPU(g.ID), g.Memory, strings.ToLower(g.Availability))
	}
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
	want := map[string]bool{}
	for _, h := range handles {
		want[h] = true
	}
	var endings []Ending
	for _, w := range v.Workers {
		if w.Pod == nil || (len(want) > 0 && !want[w.Key.Handle]) {
			continue
		}
		delete(want, w.Key.Handle)
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
		f.say("%s has no pod", h)
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
