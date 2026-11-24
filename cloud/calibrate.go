package main

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

// --- what the machines have been seen to take ----------------------------------------------------
//
// A grade trains the student on the round's number of tokens: one training pass. A machine
// asked to measure the round's unedited baseline trains that first, on as many: in Season 1
// every machine did on its first grade of a round, in Season 2 only four a round do
// (baselines_wanted in /admin/work). How long a pass takes
// on each kind of GPU is learned from the grades of settled rounds, which say the GPU that
// trained and for how long; how long a machine takes to start and first ask for work is
// learned from the machines this service rents. Until a kind has been seen, its pass is a
// 3090's over its speed in the table (speed.go).

// passOn3090 is a training pass on an RTX 3090 at the standard scale (Season 3 Qwen3-0.6B-Base).
const passOn3090 = 40 * time.Minute

// Between booting and training, and after: fetching the round's corpus and making the two
// scripts before it trains, and sitting the exam once it has.
const fetchAndScripts, sitting = 2 * time.Minute, time.Minute

// bootUnseen is how long a machine is taken to start until enough have been seen to say, on
// each market: Lium's start in seconds, Salad's pull images over people's own connections.
const bootUnseen = 8 * time.Minute

// The others' are what the smoke test saw (Nov 21): Vast 1¾ minutes, Salad 2, Lium 13, Clore
// 20, each rounded up.
var bootUnseenOn = map[string]time.Duration{"runpod": bootUnseen, "vast": 4 * time.Minute, "lium": 15 * time.Minute, "clore": 20 * time.Minute, "salad": 5 * time.Minute}

// graceAtLeast is the least a machine is given to first ask for work before it is taken for
// dead.
const graceAtLeast = 10 * time.Minute

// Calibration is what has been seen, kept in calibration.json in the home directory.
type Calibration struct {
	// Rounds are the settled rounds whose grades are counted in Pass.
	Rounds []string `json:"rounds"`
	// Pass is the minutes a training pass took, by kind (as shortGPU names it), the latest last.
	Pass map[string][]float64 `json:"pass"`
	// Boot is the minutes machines took from being rented to first asking for work, the latest
	// last.
	Boot []BootSample `json:"boot"`
}

type BootSample struct {
	Pod     string  `json:"pod"`
	Minutes float64 `json:"minutes"`
}

const keepPasses, keepBoots, bootsCounted = 10, 100, 30

func (f *Fleet) calibrationPath() string { return filepath.Join(f.o.Home, "calibration.json") }

// calibration is what has been seen, read once.
func (f *Fleet) calibration() *Calibration {
	if f.cal != nil {
		return f.cal
	}
	f.cal = &Calibration{Pass: map[string][]float64{}}
	data, err := os.ReadFile(f.calibrationPath())
	if err == nil {
		if json.Unmarshal(data, f.cal) != nil || f.cal.Pass == nil {
			f.cal = &Calibration{Pass: map[string][]float64{}}
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		f.say("reading %s: %v", f.calibrationPath(), err)
	}
	return f.cal
}

func (f *Fleet) saveCalibration() error {
	data, err := json.MarshalIndent(f.calibration(), "", "  ")
	if err != nil {
		return err
	}
	tmp := f.calibrationPath() + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, f.calibrationPath())
}

// pass is how long a training pass takes on a kind, and whether that was seen.
func (f *Fleet) pass(id string) (time.Duration, bool) {
	if seen := f.calibration().Pass[shortGPU(id)]; len(seen) > 0 {
		return time.Duration(median(seen) * float64(time.Minute)), true
	}
	s := speed(id)
	if s <= 0 {
		s = 0.5
	}
	return time.Duration(float64(passOn3090) / s), false
}

// speedOf is a kind's speed against a 3090, as seen where it has been.
func (f *Fleet) speedOf(id string) float64 {
	p, _ := f.pass(id)
	return float64(passOn3090) / float64(p)
}

// boot is how long a machine on RunPod takes from being rented to first asking for work.
func (f *Fleet) boot() time.Duration { return f.bootOn("runpod") }

// bootOn is how long a machine on a market takes from being rented to first asking for work:
// of the latest bootsCounted seen there, longer than four in five took. A sample's pod is
// known by its key in the record, which is RunPod's pod ID alone, or "market:ID".
func (f *Fleet) bootOn(market string) time.Duration {
	m := f.bootsOn(market)
	if len(m) < 3 {
		if d, ok := bootUnseenOn[market]; ok {
			return d
		}
		return bootUnseen
	}
	m = m[max(0, len(m)-bootsCounted):]
	sort.Float64s(m)
	return max(2*time.Minute, time.Duration(m[len(m)*4/5]*float64(time.Minute)))
}

// bootsOn are the minutes machines on a market were seen to take to first ask for work.
func (f *Fleet) bootsOn(market string) []float64 {
	if market == "" {
		market = "runpod"
	}
	var m []float64
	for _, s := range f.calibration().Boot {
		on := "runpod"
		if i := strings.Index(s.Pod, ":"); i > 0 {
			on = s.Pod[:i]
		}
		if on == market {
			m = append(m, s.Minutes)
		}
	}
	return m
}

// grace is how long a machine on a market may take to first ask for work before it is taken
// for dead and ended: twice what four in five took there, and ten minutes at least. A
// replacement usually starts sooner than a slow one would have. Until three have been seen
// there, it is the policy's Grace (25 minutes), or twice the boot taken for the market if
// that is longer: a market whose machines all took longer than a short Grace would have them
// all ended, and never be seen to start.
func (f *Fleet) grace(market string) time.Duration {
	if len(f.bootsOn(market)) < 3 {
		return max(f.o.Policy.Grace, 2*f.bootOn(market))
	}
	return max(graceAtLeast, 2*f.bootOn(market))
}

// lead is how long before a machine of this kind is to report a grade it must be rented,
// for its first grade of a round: to start, fetch the corpus, train the round's baseline
// while one is wanted, train the student, and sit the exam.
func (f *Fleet) lead(c Offer) time.Duration {
	p, _ := f.pass(c.Card)
	passes := time.Duration(2)
	if f.onePass {
		passes = 1
	}
	return f.bootOn(c.Market) + fetchAndScripts + passes*p + sitting
}

// baselinesWanted says whether a machine rented now may be asked to measure a round's unedited
// baseline with its first grade, training twice: while a round open wants one, or does not
// say (every machine did in Season 1), and while none is open, since the next one will want
// its baselines first.
func baselinesWanted(w *Work) bool {
	if w == nil || len(w.Rounds) == 0 {
		return true
	}
	for _, r := range w.Rounds {
		if r.BaselinesWanted == nil || *r.BaselinesWanted > 0 {
			return true
		}
	}
	return false
}

// noteBoot counts how long a machine took to first ask for work, once.
func (f *Fleet) noteBoot(pod string, took time.Duration) {
	cal := f.calibration()
	for _, s := range cal.Boot {
		if s.Pod == pod {
			return
		}
	}
	cal.Boot = append(cal.Boot, BootSample{pod, took.Minutes()})
	if len(cal.Boot) > keepBoots {
		cal.Boot = cal.Boot[len(cal.Boot)-keepBoots:]
	}
	f.calDirty = true
}

// calibrate counts the grades of rounds settled since it last looked, at most every ten
// minutes, and keeps what has been seen. Failing to ask is not fatal: what was seen before
// still counts.
func (f *Fleet) calibrate(ctx context.Context) {
	if now := f.now(); now.Sub(f.calAt) >= 10*time.Minute {
		f.calAt = now
		if err := f.countGrades(ctx); err != nil && f.calErr != err.Error() {
			f.calErr = err.Error()
			f.say("could not learn what grades took from the settled rounds: %v", err)
		}
	}
	if f.calDirty {
		if err := f.saveCalibration(); err != nil {
			f.say("keeping what was seen: %v", err)
			return
		}
		f.calDirty = false
	}
}

func (f *Fleet) countGrades(ctx context.Context) error {
	var list struct {
		Rounds []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
			Scale  string `json:"scale"`
		} `json:"rounds"`
	}
	if err := f.st.do(ctx, "GET", "/api/v1/rounds", nil, nil, &list); err != nil {
		return err
	}
	cal := f.calibration()
	slices.Reverse(list.Rounds) // the oldest first, so that the latest grades are last
	for _, r := range list.Rounds {
		if r.Status != "settled" || r.Scale != "standard" || slices.Contains(cal.Rounds, r.ID) {
			continue
		}
		var round struct {
			Entries []struct {
				Detail struct {
					Hardware struct {
						GPU string `json:"gpu"`
					} `json:"hardware"`
					Train struct {
						Seconds float64 `json:"seconds"`
					} `json:"train"`
				} `json:"detail"`
			} `json:"entries"`
		}
		if err := f.st.do(ctx, "GET", "/api/v1/rounds/"+url.PathEscape(r.ID), nil, nil, &round); err != nil {
			return err
		}
		for _, e := range round.Entries {
			if g, s := e.Detail.Hardware.GPU, e.Detail.Train.Seconds; g != "" && s > 0 {
				k := shortGPU(g)
				cal.Pass[k] = append(cal.Pass[k], s/60)
				if n := len(cal.Pass[k]); n > keepPasses {
					cal.Pass[k] = cal.Pass[k][n-keepPasses:]
				}
			}
		}
		cal.Rounds = append(cal.Rounds, r.ID)
		f.calDirty = true
	}
	return nil
}

func median(xs []float64) float64 {
	s := slices.Clone(xs)
	sort.Float64s(s)
	if n := len(s); n%2 == 1 {
		return s[n/2]
	} else {
		return (s[n/2-1] + s[n/2]) / 2
	}
}
