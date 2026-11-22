package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// --- where curators cannot be run ----------------------------------------------------------------
//
// stet runs curators only on a machine at the season's floor (stet bench): its GPU, its memory,
// and its CPU, a mix of text work on six processes at once. The sandbox may hold on a market and
// a machine there still be under the floor: the cheapest hosts on Vast and Clore are old
// servers' CPUs (in round 9 an EPYC 7K62 measured 95 MB a second, against a floor of 100), and
// stet there only grades. The server says of each machine whether, as it last asked, it could be
// handed curators to run, and if not why (/admin/work). One that could not, rented to run them,
// grades only: it is not counted among the machines the curators want, and is ended if no grade
// wants it either (policy.go). Its host is kept here (curators.json), and its CPU too where its
// market names it and the server says its speed was short, and no machine is rented on either to
// run curators for a week.

// refusal is a host or a CPU curators were not run on, and what showed it.
type refusal struct {
	At     int64  `json:"at"`
	Why    string `json:"why"`
	Market string `json:"market"`
	Host   string `json:"host,omitempty"`
	CPU    string `json:"cpu,omitempty"`
	Card   string `json:"card,omitempty"`
	Handle string `json:"handle,omitempty"`
}

type refusals struct {
	Hosts map[string]refusal `json:"hosts"` // by market:host, "vast:157631"
	CPUs  map[string]refusal `json:"cpus"`  // by cpuKey
}

// refusedFor is how long a host or a CPU is not rented to run curators.
const refusedFor = 7 * 24 * time.Hour

func (f *Fleet) refusalsPath() string { return filepath.Join(f.o.Home, "curators.json") }

func (f *Fleet) readRefusals() (*refusals, error) {
	r := &refusals{Hosts: map[string]refusal{}, CPUs: map[string]refusal{}}
	raw, err := os.ReadFile(f.refusalsPath())
	if errors.Is(err, os.ErrNotExist) {
		return r, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, r); err != nil {
		return nil, fmt.Errorf("%s: %w", f.refusalsPath(), err)
	}
	if r.Hosts == nil {
		r.Hosts = map[string]refusal{}
	}
	if r.CPUs == nil {
		r.CPUs = map[string]refusal{}
	}
	return r, nil
}

// refuse keeps a machine's host and CPU as ones curators are not run on, those past their week
// let go.
func (f *Fleet) refuse(rec *podRecord, why string) error {
	r, err := f.readRefusals()
	if err != nil {
		return err
	}
	now := f.now()
	for k, v := range r.Hosts {
		if now.Sub(time.Unix(v.At, 0)) > refusedFor {
			delete(r.Hosts, k)
		}
	}
	for k, v := range r.CPUs {
		if now.Sub(time.Unix(v.At, 0)) > refusedFor {
			delete(r.CPUs, k)
		}
	}
	it := refusal{At: now.Unix(), Why: why, Market: rec.market(), Host: rec.Host, CPU: rec.CPU, Card: rec.GPU, Handle: rec.Handle}
	if rec.Host != "" {
		r.Hosts[rec.market()+":"+rec.Host] = it
	}
	if k := cpuKey(rec.CPU); k != "" && slowCPU(why) {
		r.CPUs[k] = it
	}
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	tmp := f.refusalsPath() + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, f.refusalsPath())
}

// noCurators says, once, that a machine rented to run curators may not, and keeps its host and
// CPU from being rented for them again.
func (f *Fleet) noCurators(r *Rental, rec *podRecord, why string) {
	f.event(r, rec.Handle, "no curators", why)
	host, cpu := rec.Host != "", rec.CPU != "" && slowCPU(why)
	more := ""
	switch {
	case host && cpu:
		more = fmt.Sprintf("; neither its host nor any %s is rented to run curators for a week", strings.TrimSpace(rec.CPU))
	case host:
		more = "; its host is not rented to run curators for a week"
	case cpu:
		more = fmt.Sprintf("; no %s is rented to run curators for a week", strings.TrimSpace(rec.CPU))
	}
	f.say("%s on %s (%s) may not run curators: %s. It grades only%s", rec.Handle, marketTitle(rec.market()), rec.GPU, why, more)
	if host || cpu {
		if err := f.refuse(rec, why); err != nil {
			f.say("keeping where curators may not run: %v", err)
		}
	}
}

// refused says why an offer is not rented to run curators, if it is not: its host, or its CPU,
// was under the floor within the week.
func (r *refusals) refused(o Offer, now time.Time) (string, bool) {
	if r == nil {
		return "", false
	}
	if v, ok := r.Hosts[o.Market+":"+o.Host]; ok && o.Host != "" && now.Sub(time.Unix(v.At, 0)) <= refusedFor {
		return fmt.Sprintf("its host ran no curators on %s: %s", time.Unix(v.At, 0).UTC().Format("Jan 2"), v.Why), true
	}
	if v, ok := r.CPUs[cpuKey(o.CPU)]; ok && o.CPU != "" && now.Sub(time.Unix(v.At, 0)) <= refusedFor {
		return fmt.Sprintf("its CPU, %s, ran no curators on %s: %s", strings.TrimSpace(o.CPU), time.Unix(v.At, 0).UTC().Format("Jan 2"), v.Why), true
	}
	return "", false
}

// cpuShort is the server's word that a machine's CPU was too slow for the season's floor:
// "CPU 95 MB/s, floor 100", among its shortfalls.
var cpuShort = regexp.MustCompile(`\bCPU [\d.]+ MB/s`)

// slowCPU says whether why a machine may not run curators is its CPU's speed. Only that is
// the CPU's: a machine may be refused for its memory, or for too few of the host's cores, or
// refuse itself for its sandbox, on a CPU that would do.
func slowCPU(why string) bool { return cpuShort.MatchString(why) }

var (
	cpuNoise  = regexp.MustCompile(`(?i)\(r\)|\(tm\)|®|™|\bcpu\b|\bprocessor\b|@\s*[\d.]+\s*ghz|\bintel\b|\bamd\b`)
	cpuSpaces = regexp.MustCompile(`\s+`)
)

// cpuKey is a CPU's model as the markets' names for it agree: Vast's "Xeon® E5-2680 v4" and
// Clore's "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz" are both "xeon e5-2680 v4".
func cpuKey(name string) string {
	s := cpuNoise.ReplaceAllString(name, " ")
	return strings.ToLower(strings.TrimSpace(cpuSpaces.ReplaceAllString(s, " ")))
}
