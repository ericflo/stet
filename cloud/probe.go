package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// --- proving stet's container sandbox on a market ---------------------------------------------------
//
// Machines run curators only where stet's container sandbox is known to hold. A probe proves it:
// one machine rented by hand (up --market), with tools/cage_check.py in its environment. It
// takes no work. It installs stet, says what the machine is (GPU, driver, kernel, memory, disk)
// and what stet doctor says, runs the check, a curator that wants out tried every way the check
// knows, and waits. What it found is read back by logging in where the market lets a key in
// (Lium, Clore), or from the market's log (Vast, Salad), and the machine is ended. Where every
// case held, `markets curators M --note …` lets machines there run curators; each still checks its
// own sandbox before it takes one, and only grades where it cannot be made.

// sshable is a market a machine there can be logged in to: "root@host", and the port.
type sshable interface {
	sshTo(ctx context.Context, id string) (string, int, error)
}

// probeHomes are where the worker keeps its files on each market.
var probeHomes = map[string]string{"lium": "/workspace/root"}

func cmdProbe(ctx context.Context, f *Fleet, check string, wait time.Duration) error {
	raw, err := os.ReadFile(check)
	if err != nil {
		return fmt.Errorf("the sandbox's check (--check, run from stet's repository): %w", err)
	}
	f.probe = gzipB64(string(raw))
	f.o.Cheapest, f.o.Idle = true, 5
	market := f.o.Market
	m := f.market(market)
	if m == nil {
		return fmt.Errorf("%s has no key: %s", marketTitle(market), f.off[market])
	}
	if err := cmdUp(ctx, f, 1); err != nil {
		return err
	}
	// The machine just rented on the market: the newest of the pool's there.
	l, err := f.readLedger()
	if err != nil {
		return err
	}
	var rec *podRecord
	for _, r := range l.Pods {
		if r.market() == market && (rec == nil || r.Start.After(rec.Start)) {
			rec = r
		}
	}
	if rec == nil {
		return fmt.Errorf("nothing was rented on %s", marketTitle(market))
	}
	f.say("probing %s on %s (%s, %s): waiting up to %s for what it finds", rec.Handle, marketTitle(market), rec.ID, rec.GPU, wait)
	deadline := f.now().Add(wait)
	var found []string
	for {
		found = f.probeFound(ctx, m, rec.ID)
		if probeDone(found) || f.now().After(deadline) || ctx.Err() != nil {
			break
		}
		if err := sleepCtx(ctx, time.Minute); err != nil {
			break
		}
	}
	for _, line := range found {
		fmt.Fprintf(f.out, "  %s\n", line)
	}
	held := false
	for _, line := range found {
		held = held || strings.Contains(line, "every case held")
	}
	if err := cmdDown(context.WithoutCancel(ctx), f, true, []string{rec.Handle}); err != nil {
		f.say("ending the probe: %v", err)
	}
	switch {
	case held:
		f.say("every case held on %s: `stet-cloud markets curators %s --note \"…\"` lets machines there run curators", marketTitle(market), market)
	case probeDone(found):
		f.say("the sandbox did not hold on this %s machine: its machines go on grading only", marketTitle(market))
	default:
		f.say("the probe said nothing more in %s", wait)
	}
	return nil
}

// probeDone says whether a probe has finished.
func probeDone(lines []string) bool {
	for _, l := range lines {
		if strings.Contains(l, "done: ") {
			return true
		}
	}
	return false
}

// probeFound is what a probe has said so far: its lines, read where it keeps them, or from its
// market's log.
func (f *Fleet) probeFound(ctx context.Context, m Market, id string) []string {
	if s, ok := m.(sshable); ok {
		if who, port, err := s.sshTo(ctx, id); err == nil {
			home := probeHomes[m.Name()]
			if home == "" {
				home = "/root"
			}
			c, cancel := context.WithTimeout(ctx, 40*time.Second)
			out, err := exec.CommandContext(c, "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
				"-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "LogLevel=ERROR", "-p", strconv.Itoa(port), who,
				"cat "+home+"/probe.txt").Output()
			cancel()
			if err == nil {
				return strings.Split(strings.TrimRight(string(out), "\n"), "\n")
			}
		}
	}
	if lg, ok := m.(logger); ok {
		lines, err := lg.Logs(ctx, id, 400)
		if err == nil {
			var out []string
			for _, l := range lines {
				if i := strings.Index(l, "[probe] "); i >= 0 {
					out = append(out, l[i:])
				}
			}
			return out
		}
		if !errors.Is(err, context.Canceled) {
			return []string{"(its log could not be had: " + firstLine(err.Error()) + ")"}
		}
	}
	return nil
}
