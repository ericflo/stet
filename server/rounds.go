package main

// The life of a round: scheduled (announced, with its draw time and its crawl) -> open
// (drawn; anchors edit, verifiers grade) -> settled.
//
// A round is drawn at a moment on the season's public grid, from the first beacon round
// emitted at or after it. The crawl is named when the round is scheduled. So everything
// that goes into the seed except the beacon's randomness is public before the randomness
// exists, and every verifier checks the grid for itself.

import (
	"context"
	"errors"
	"log/slog"
	"math/rand/v2"
	"sort"
	"time"
)

const commitMargin = 30 // seconds an editor must precede the draw by

// beaconSource is where a round's randomness comes from. Tests replace it.
var beaconSource = fetchBeacon

// beaconIsFake is set by tests, which do not wait for real beacon rounds.
var beaconIsFake = false

// description is the public statement of a round: enough for anyone to rebuild its corpus
// and exam, and to check that the seed follows from the beacon.
func (r *Round) description(s *Season) map[string]any {
	d := map[string]any{
		"id": roundID(s.ID, r.Idx), "season": s.ID, "kind": "official", "index": r.Idx, "status": r.Status, "scale": r.Scale,
		"draw_at": r.DrawAt, "drand_round": r.DrandRound,
	}
	if r.Crawl != nil {
		d["crawl"] = r.Crawl
	}
	if r.Status != "scheduled" {
		d["seed"] = r.Seed
		d["beacon"] = map[string]any{"round": r.DrandRound, "randomness": r.Randomness}
		d["drawn_at"] = r.DrawnAt
	}
	return d
}

// nextSlot is the next scheduled draw time after a moment: a multiple of the round
// interval, which is itself a multiple of the season's grid.
func (a *App) nextSlot(after int64) int64 {
	iv := int64(a.cfg.RoundInterval.Seconds())
	return (after/iv + 1) * iv
}

// savedCrawl is the crawl last adopted, without touching the network.
func (a *App) savedCrawl(s *State) *Crawl {
	if a.cfg.FixedCrawl != nil {
		return a.cfg.FixedCrawl
	}
	if s.Crawl != nil && s.Crawl.Crawl.Crawl != "" {
		return &s.Crawl.Crawl
	}
	return nil
}

// refreshCrawl looks for a newer Common Crawl crawl when the last look is stale. It
// consults the network, so it runs outside any change to the state.
func (a *App) refreshCrawl(ctx context.Context) {
	if a.cfg.FixedCrawl != nil {
		return
	}
	var saved *SavedCrawl
	a.st.view(func(s *State) { saved = s.Crawl })
	if saved != nil && now()-saved.CheckedAt < 6*3600 {
		return
	}
	c, err := discoverCrawl(ctx, a.season.Pools["official"])
	if err != nil {
		slog.Warn("crawl discovery failed", "err", err)
		return
	}
	a.st.update(ctx, func(tx *Tx) error {
		if tx.s.Crawl == nil || tx.s.Crawl.Crawl.Crawl != c.Crawl {
			slog.Info("adopted crawl", "crawl", c.Crawl, "files", c.NShards)
			tx.event("crawl", map[string]any{"crawl": c.Crawl, "files": c.NShards})
		}
		tx.saveCrawl(&SavedCrawl{Crawl: *c, CheckedAt: now()})
		return nil
	})
}

// tick advances everything that moves on its own. Safe to call at any time.
func (a *App) tick(ctx context.Context) {
	a.refreshCrawl(ctx)
	a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		t := now()
		for _, c := range s.Claims {
			if c.Status == "claimed" && c.ExpiresAt <= t {
				c.Status = "expired"
				tx.saveClaim(c)
			}
		}
		// Claims of rounds long settled have nothing left to say.
		for _, c := range s.Claims {
			if r := s.Rounds[c.RoundIdx]; r != nil && r.Status == "settled" && r.SettledAt < t-86400 {
				tx.dropClaim(c)
			}
		}
		a.schedule(tx)
		a.settleRounds(tx)
		return nil
	})
	a.maybeDraw(ctx, false)
	a.st.flush(ctx)
}

// schedule announces the next round, if none is announced: when it will be drawn (and so
// from which beacon round) and from which crawl.
func (a *App) schedule(tx *Tx) {
	s := tx.s
	if crawl := a.savedCrawl(s); s.roundWithStatus("scheduled") == nil && crawl != nil {
		at := a.nextSlot(now())
		tx.saveRound(&Round{Idx: s.maxRound() + 1, Status: "scheduled", Scale: a.cfg.Scale, DrawAt: at, DrandRound: a.season.Beacon.drandRoundAt(at), Crawl: crawl})
	}
}

// reschedule moves the scheduled round to a later draw time: a new announcement, made
// before that time's beacon value exists. It takes the newest crawl with it.
func (a *App) reschedule(tx *Tx, r *Round, at int64) {
	if c := a.savedCrawl(tx.s); c != nil {
		r.Crawl = c
	}
	r.DrawAt, r.DrandRound = at, a.season.Beacon.drandRoundAt(at)
	tx.saveRound(r)
}

type entrant struct {
	EditorID string
	Why      string // ladder | qualifying | new
}

// selectEntrants decides who runs in a round: the ladder (so that the best editors always
// share rounds and can be compared head to head), editors still qualifying, and new
// entries, with people who have graded for others going first.
func (a *App) selectEntrants(s *State, cutoff int64) []entrant {
	seen := map[string]bool{}
	var out []entrant
	add := func(id, why string) {
		if !seen[id] {
			seen[id] = true
			out = append(out, entrant{id, why})
		}
	}
	for _, row := range a.computeBoard(s).Ladder {
		add(row.ID, "ladder")
	}
	var qualifying, queue []*Editor
	for _, e := range s.Editors {
		switch {
		case e.Status == "running" && len(s.ofEditor[e.ID]) < a.cfg.QualifyRounds:
			qualifying = append(qualifying, e)
		case e.Status == "queued" && e.CreatedAt <= cutoff-commitMargin && !s.owner(e).Banned:
			queue = append(queue, e)
		}
	}
	sort.Slice(qualifying, func(i, j int) bool {
		return qualifying[i].CreatedAt < qualifying[j].CreatedAt || (qualifying[i].CreatedAt == qualifying[j].CreatedAt && qualifying[i].ID < qualifying[j].ID)
	})
	for _, e := range qualifying {
		add(e.ID, "qualifying")
	}
	priority := map[int64]int{} // grades for others in the last two weeks
	for _, r := range s.Results {
		if r.Verdict == "agreed" && r.ReceivedAt > now()-14*86400 {
			priority[r.AccountID]++
		}
	}
	sort.SliceStable(queue, func(i, j int) bool {
		pi, pj := priority[queue[i].AccountID], priority[queue[j].AccountID]
		if pi != pj {
			return pi > pj
		}
		if queue[i].CreatedAt != queue[j].CreatedAt {
			return queue[i].CreatedAt < queue[j].CreatedAt
		}
		return queue[i].ID < queue[j].ID
	})
	for i, e := range queue {
		if i >= a.cfg.MaxNewPerRound {
			break
		}
		add(e.ID, "new")
	}
	return out
}

// bringForward moves the scheduled round to the next moment on the grid (an anchor asked
// for a draw sooner than the schedule's). In development the draw is immediate and off the
// grid, which real verifiers would refuse.
func (a *App) bringForward(ctx context.Context) (*Round, error) {
	var out Round
	err := a.st.update(ctx, func(tx *Tx) error {
		r := tx.s.roundWithStatus("scheduled")
		if r == nil {
			return fail(409, "no round is scheduled")
		}
		if tx.s.roundWithStatus("open") != nil {
			return fail(409, "the previous round is still being graded")
		}
		at := a.season.nextGrid(now() + commitMargin)
		if a.cfg.Dev {
			at = now()
		}
		if at < r.DrawAt || a.cfg.Dev {
			a.reschedule(tx, r, at)
		}
		out = *r
		return nil
	})
	return &out, err
}

// maybeDraw draws the scheduled round if its time has come.
func (a *App) maybeDraw(ctx context.Context, force bool) error {
	var r Round
	var problem error
	due := false
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		cur := s.roundWithStatus("scheduled")
		if cur == nil {
			problem = errors.New("no round is scheduled")
			return nil
		}
		r = *cur
		due = now() >= a.season.Beacon.timeOf(r.DrandRound)+2 || (force && beaconIsFake)
		switch {
		case !due:
		case s.roundWithStatus("open") != nil:
			// Grading of the previous round is still under way. Rounds never overlap, so
			// that the ladder is always complete; this one moves to the next slot.
			a.reschedule(tx, cur, a.nextSlot(now()))
			problem = errors.New("the previous round is still being graded")
		case len(a.selectEntrants(s, r.DrawAt)) == 0:
			a.reschedule(tx, cur, a.nextSlot(now()))
			problem = errors.New("no editors are waiting to run")
		case r.Crawl == nil:
			problem = errors.New("the round has no crawl")
		}
		return nil
	})
	if err != nil {
		return err
	}
	if problem != nil {
		return problem
	}
	if !due {
		if force && a.cfg.Dev {
			time.Sleep(time.Until(time.Unix(a.season.Beacon.timeOf(r.DrandRound)+2, 0)))
			return a.maybeDraw(ctx, false)
		}
		return nil
	}

	// The beacon is consulted without holding the state.
	randomness, err := beaconSource(ctx, a.season.Beacon, r.DrandRound)
	if err != nil {
		slog.Warn("beacon not available yet", "round", r.DrandRound, "err", err)
		return err
	}
	entries := 0
	err = a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		cur := s.Rounds[r.Idx]
		if cur == nil || cur.Status != "scheduled" || cur.DrandRound != r.DrandRound || cur.Crawl == nil {
			return errors.New("the round changed while it was being drawn")
		}
		entrants := a.selectEntrants(s, cur.DrawAt)
		if len(entrants) == 0 {
			a.reschedule(tx, cur, a.nextSlot(now()))
			problem = errors.New("no editors are waiting to run")
			return nil
		}
		// The crawl is the one announced with the round, never one chosen now.
		cur.Status, cur.Randomness, cur.DrawnAt = "open", randomness, now()
		cur.Seed = hexOf(officialSeed(a.season.ID, cur.Idx, cur.DrandRound, randomness, *cur.Crawl))
		tx.saveRound(cur)
		for _, e := range entrants {
			tx.saveEntry(&Entry{RoundIdx: cur.Idx, EditorID: e.EditorID, Status: "open", Spot: rand.Float64() < a.cfg.SpotRate})
			// Once an editor has been drawn into a round it can no longer be changed, and
			// everyone is entitled to read what is being scored.
			ed := s.Editors[e.EditorID]
			ed.Public = true
			if ed.Status == "queued" {
				ed.Status = "running"
			}
			tx.saveEditor(ed)
		}
		entries = len(entrants)
		tx.event("draw", map[string]any{"round": roundID(a.season.ID, cur.Idx), "entries": entries, "crawl": cur.Crawl.Crawl})
		a.schedule(tx) // the next draw is announced the moment this one is made
		return nil
	})
	if err != nil {
		return err
	}
	if problem != nil {
		return problem
	}
	slog.Info("round drawn", "round", roundID(a.season.ID, r.Idx), "entries", entries, "crawl", r.Crawl.Crawl)
	return nil
}

// settleRounds closes every open round whose entries have all settled.
func (a *App) settleRounds(tx *Tx) {
	s := tx.s
	var due []*Round
	for _, r := range s.Rounds {
		if r.Status != "open" || len(s.inRound[r.Idx]) == 0 {
			continue
		}
		open := false
		for _, e := range s.inRound[r.Idx] {
			open = open || e.Status == "open"
		}
		if !open {
			due = append(due, r)
		}
	}
	sort.Slice(due, func(i, j int) bool { return due[i].Idx < due[j].Idx })
	for _, r := range due {
		a.refreshBaseline(tx, r)
		a.regain(tx, r)
		r.Status, r.SettledAt = "settled", now()
		tx.saveRound(r)
		// An editor that has had its qualifying rounds stops being scheduled, unless it is on
		// the ladder (the ladder is recomputed at every draw).
		for _, e := range s.inRound[r.Idx] {
			ed := s.Editors[e.EditorID]
			if ed.Status == "running" && len(s.ofEditor[ed.ID]) >= a.cfg.QualifyRounds {
				ed.Status = "retired"
				tx.saveEditor(ed)
			}
		}
		best := 0.0
		for _, e := range s.inRound[r.Idx] {
			if e.Gain > best {
				best = e.Gain
			}
		}
		tx.event("settle", map[string]any{"round": roundID(a.season.ID, r.Idx), "entries": len(s.inRound[r.Idx]), "best_gain": best})
		slog.Info("round settled", "round", roundID(a.season.ID, r.Idx), "entries", len(s.inRound[r.Idx]))
		a.checkRecord(tx, r.Idx)
	}
}

func (a *App) runScheduler(ctx context.Context) {
	t := time.NewTicker(15 * time.Second)
	defer t.Stop()
	a.tick(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			a.tick(ctx)
		}
	}
}
