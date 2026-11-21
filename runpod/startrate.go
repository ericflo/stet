package main

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"time"
)

// --- how often each kind of machine starts -----------------------------------------------------
//
// Some cheap machines never start: in round 7, 12 of 18 RunPod community pods never asked for
// work, and none of 8 secure ones failed. So each kind is ranked by its price over how often
// machines of its kind have started (Offer.Rate). Each machine rented is recorded with whether
// it ever asked for work (podRecord.Asked). One that lived ten minutes without asking counts as
// one that did not start; one ended sooner than that is not counted.
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

// startCount is how many machines were seen to start, of those whose start was seen or missed.
type startCount struct{ started, rented int }

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

// startRates counts, from the record of machines live and ended, the starts seen.
func (f *Fleet) startRates() startRates {
	s := startRates{byKind: map[string]startCount{}, byClass: map[string]startCount{}}
	count := func(r podRecord, end time.Time) {
		if !r.Tracked {
			return // rented before it was kept
		}
		started := r.Asked
		if !started && end.Sub(r.Start) < graceAtLeast {
			return // not long enough to say
		}
		for _, k := range []string{kindKey(r.market(), r.GPU, r.Cloud)} {
			c := s.byKind[k]
			c.rented++
			if started {
				c.started++
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
	if fh, err := os.Open(f.spentLog()); err == nil {
		sc := bufio.NewScanner(fh)
		for sc.Scan() {
			var r podRecord
			if json.Unmarshal(sc.Bytes(), &r) == nil {
				end := r.End
				if end.IsZero() {
					end = r.Alive
				}
				count(r, end)
			}
		}
		fh.Close()
	}
	if l, err := f.readLedger(); err == nil {
		for _, r := range l.Pods {
			count(*r, f.now())
		}
	}
	return s
}
