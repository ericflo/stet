package main

import (
	"fmt"
	"strings"
	"time"
)

// --- how often each kind of machine starts -----------------------------------------------------
//
// Some cheap machines never start: in round 7, 12 of 18 RunPod community pods never asked for
// work, and none of 8 secure ones failed. So each kind is ranked by its price over how often
// machines of its kind have started (Offer.Rate). Each machine rented is recorded with whether
// it ever asked for work (podRecord.Asked). One that lived ten minutes without asking counts as
// one that did not start, and so does one its market took away before it asked (Gone: round 7
// had community pods vanish minutes after they were rented); one the fleet ended sooner is
// not counted.
//
// What a kind has been seen to do is weighed with what is expected of it: first of its class
// of host (prior), then of its market's hosts of that class, then of the kind itself, each
// counting as startWeight machines until more of its own have been seen.

// startPrior is how often a machine is expected to start, by the market's word for its hosts.
var startPrior = expectedStart

func expectedStart(trust string) float64 {
	switch strings.ToLower(trust) {
	case "secure", "datacenter":
		return 0.95
	case "verified":
		return 0.9
	}
	return 0.75 // community hosts, people's own PCs
}

const startWeight = 4

// startCount is how many machines were seen to start, of those whose start was seen or missed,
// and why those that did not.
type startCount struct {
	started, rented int
	causes          map[string]int
}

// failures says why a kind's machines did not start: "2 no container, 1 gone".
func (c startCount) failures() string {
	var parts []string
	for _, cause := range []string{causeNoContainer, causeGone, causeInstall, causeDriver, causeSilent, causeContainer, causeMarketStopped, causeUnknown, "never asked"} {
		if n := c.causes[cause]; n > 0 {
			parts = append(parts, fmt.Sprintf("%d %s", n, cause))
		}
	}
	return strings.Join(parts, ", ")
}

// kind is what was seen of an offer's kind.
func (s startRates) kind(o Offer) startCount {
	market := o.Market
	if market == "" {
		market = "runpod"
	}
	return s.byKind[kindKey(market, o.Card, o.Trust)]
}

type startRates struct {
	byKind, byClass map[string]startCount
}

func kindKey(market, card, trust string) string {
	return market + "|" + card + "|" + strings.ToLower(trust)
}

func classKey(market, trust string) string { return market + "|" + strings.ToLower(trust) }

// of is how often an offer's kind is taken to start.
func (s startRates) of(o Offer) float64 {
	market := o.Market
	if market == "" {
		market = "runpod"
	}
	class := s.byClass[classKey(market, o.Trust)]
	classRate := (startPrior(o.Trust)*startWeight + float64(class.started)) / float64(startWeight+class.rented)
	kind := s.byKind[kindKey(market, o.Card, o.Trust)]
	return (classRate*startWeight + float64(kind.started)) / float64(startWeight+kind.rented)
}

// learnStarts says whether starts are counted at all (a test of ordering by price alone turns
// it off).
var learnStarts = true

// startRates counts, from the record of machines live and ended, the starts seen.
func (f *Fleet) startRates() startRates {
	s := startRates{byKind: map[string]startCount{}, byClass: map[string]startCount{}}
	if !learnStarts {
		return s
	}
	count := func(r podRecord, end time.Time) {
		if !r.Tracked {
			return // rented before it was kept
		}
		started := r.Asked
		if !started && r.Cause == "" && !r.Gone && end.Sub(r.Start) < graceAtLeast {
			return // ended by the fleet before it could be said
		}
		for _, k := range []string{kindKey(r.market(), r.GPU, r.Cloud)} {
			c := s.byKind[k]
			c.rented++
			if started {
				c.started++
			} else {
				if c.causes == nil {
					c.causes = map[string]int{}
				}
				cause := r.Cause
				if cause == "" {
					cause = "never asked"
				}
				c.causes[cause]++
			}
			s.byKind[k] = c
		}
		k := classKey(r.market(), r.Cloud)
		c := s.byClass[k]
		c.rented++
		if started {
			c.started++
		}
		s.byClass[k] = c
	}
	seen := map[string]bool{}
	if sum, err := f.readSpentSummary(); err == nil {
		for _, r := range sum.records {
			if r.ID != "" {
				seen[ledgerKey(r.market(), r.ID)] = true
			}
			end := r.End
			if end.IsZero() {
				end = r.Alive
			}
			count(r, end)
		}
	}
	if l, err := f.readLedger(); err == nil {
		for key, r := range l.Pods {
			if r == nil || (r.ID != "" && seen[ledgerKey(r.market(), r.ID)]) || seen[key] {
				continue
			}
			count(*r, f.now())
		}
	}
	return s
}
