package main

// The life of a round: scheduled (announced, with its draw time and its crawl) -> open
// (drawn; verifiers run the editors and grade what they wrote) -> settled.
//
// A round is drawn at a moment on the season's public grid, from the first beacon round
// emitted at or after it. The crawl is named when the round is scheduled. So everything
// that goes into the seed except the beacon's randomness is public before the randomness
// exists, and every verifier checks the grid for itself.

import (
	"context"
	"errors"
	"log/slog"
	"sort"
	"time"
)

const commitMargin = 30 // seconds an editor must precede the draw by

// drawNotice is how far ahead a draw that is moved is announced, at least. Verifiers hear
// of the next draw when they ask for work, and an idle one asks every five minutes; a
// draw moved with less notice than that is one some of them were never told of.
const drawNotice = 6 * 60

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
	if len(r.Was) > 0 {
		// Beacon rounds this round was announced for before its draw moved. Public, so
		// that a draw skipped after its beacon value was known cannot go unnoticed.
		d["moved_from"] = r.Was
	}
	if r.Status != "scheduled" {
		d["seed"] = r.Seed
		d["beacon"] = map[string]any{"round": r.DrandRound, "randomness": r.Randomness}
		d["drawn_at"] = r.DrawnAt
	}
	if examPosted(r) {
		d["exam_digest"], d["exam_at"] = r.ExamSHA, r.ExamAt
	}
	return d
}

// nextSlot is the next scheduled draw time after a moment: a multiple of the round
// interval, which is itself a multiple of the season's grid.
func (a *App) nextSlot(after int64) int64 {
	iv := int64(a.cfg.RoundInterval.Seconds())
	return (after/iv + 1) * iv
}

// laterSlot is where a draw goes that is being announced now: the next usual time, or,
// when that is less than the notice away, the first place on the grid that is not. No
// draw is ever announced with less than drawNotice to go.
func (a *App) laterSlot() int64 {
	at := a.nextSlot(now())
	if at < now()+drawNotice {
		at = a.season.nextGrid(now() + drawNotice)
	}
	return at
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
	// Both of these hold the state while they talk to the database, so both get a deadline.
	bounded, cancel := context.WithTimeout(ctx, storeBudget)
	defer cancel()
	a.st.sync(bounded)
	a.refreshCrawl(ctx)
	a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		t := now()
		for _, c := range s.Claims {
			if c.Status == "claimed" && c.ExpiresAt <= t {
				if rd := s.Rounds[c.RoundIdx]; c.Kind == "grade" && rd != nil && rd.Status == "open" && !examPosted(rd) {
					// A grade taken before the exam cannot be finished without it: it waits as
					// long as the exam does.
					c.ExpiresAt = t + int64(a.cfg.Lease.Seconds())
					tx.saveClaim(c)
					continue
				}
				c.Status = "lapsed" // out of time, with no report and no release
				tx.saveClaim(c)
				if rd := s.Rounds[c.RoundIdx]; rd != nil && rd.Status == "open" && a.established(s.Accounts[c.AccountID]) {
					a.moved(tx, rd)
				}
			}
		}
		// Claims of rounds long settled have nothing left to say.
		for _, c := range s.Claims {
			if r := s.Rounds[c.RoundIdx]; r != nil && r.Status == "settled" && r.SettledAt < t-86400 {
				tx.dropClaim(c)
			}
		}
		a.schedule(tx)
		a.onDemand(tx)
		a.abandonStuck(tx)
		a.settleReady(tx)
		return nil
	})
	a.archiveExam(ctx)
	a.maybeDraw(ctx, false)
	a.st.flush(bounded)
}

// schedule announces the next round, if none is announced: when it will be drawn (and so
// from which beacon round) and from which crawl.
func (a *App) schedule(tx *Tx) {
	if a.cfg.ArchiveSeed != "" { // a finished season: no round comes next
		return
	}
	s := tx.s
	if crawl := a.savedCrawl(s); s.roundWithStatus("scheduled") == nil && crawl != nil {
		// The usual time, or the first free place on the grid after it: a slot some round
		// was once announced for costs the next round ten minutes, not a whole interval.
		taken := announcedSlots(s, nil)
		at := a.laterSlot()
		for taken[a.season.Beacon.drandRoundAt(at)] {
			at = a.season.nextGrid(at)
		}
		tx.saveRound(&Round{Idx: s.maxRound() + 1, Status: "scheduled", Scale: a.cfg.Scale, DrawAt: at, DrandRound: a.season.Beacon.drandRoundAt(at), Crawl: crawl})
	}
}

// onDemand brings the next draw forward when an entry is waiting for it. Left to the
// interval alone, somebody who submits just after a draw waits most of six hours to learn
// anything, and three rounds of that is most of a day. So when an entry is waiting to run,
// or part-way through its qualifying rounds, the next round is drawn as soon as it fairly
// can be: at the first mark on the grid that gives verifiers their notice.
//
// A round is paid for in volunteers' GPU time, and three things hold this back. Nothing is
// drawn while a round is open. Draws are at least RoundMinGap apart. And somebody who can
// run editors must have asked for work a short while ago: a round nobody is there to run
// only uses up a round number, and that can wait for the usual time. So can the next round
// when the last two both came to nothing: whatever is wrong, hurrying will not mend it.
func (a *App) onDemand(tx *Tx) {
	if r, at := a.earlyDraw(tx.s); r != nil && a.runnerAround(tx.s) {
		a.reschedule(tx, r, at)
		slog.Info("draw brought forward: an entry is waiting", "round", roundID(a.season.ID, r.Idx), "at", r.DrawAt)
	}
}

// earlyDraw is the scheduled round and the earlier time onDemand would draw it at once
// somebody who runs curators is around, or nil when it would not bring the draw forward:
// rounds keep to the clock (RoundMinGap 0), no entry is waiting, a round is open, or the
// draw is already as early as it can be.
func (a *App) earlyDraw(s *State) (*Round, int64) {
	gap := int64(a.cfg.RoundMinGap.Seconds())
	r := s.roundWithStatus("scheduled")
	if gap <= 0 || r == nil || r.Crawl == nil || s.Meta.Paused != 0 || s.roundWithStatus("open") != nil || !a.waiting(s) {
		return nil, 0
	}
	at, empty := now()+drawNotice, 0
	for _, rd := range s.roundsDesc(3) {
		if rd.DrawnAt == 0 {
			continue // the scheduled round itself
		}
		if empty >= 0 && rd.Status == "settled" && rd.BaselineLoss == nil {
			empty++ // nothing in it could be graded
		} else {
			empty = -1 << 30
		}
		if rd.DrawnAt+gap > at {
			at = rd.DrawnAt + gap
		}
	}
	if empty >= 2 {
		return nil, 0
	}
	if at = a.season.nextGrid(at - 1); at < r.DrawAt {
		return r, at
	}
	return nil, 0
}

// waiting reports whether an entry is waiting for a round: one that has not run, or one
// part-way through its qualifying rounds.
func (a *App) waiting(s *State) bool {
	for _, e := range s.Editors {
		if e.Status == "queued" || (e.Status == "running" && roundsRun(s, e.ID) < a.cfg.QualifyRounds) {
			return true
		}
	}
	return false
}

// runnerAround reports whether somebody who runs editors asked for work a short while ago.
func (a *App) runnerAround(s *State) bool {
	since := now() - int64(a.cfg.ActiveWindow.Seconds())
	for _, acct := range s.Accounts {
		if acct.LastSeen > since && acct.RunnerSeen != 0 && a.established(acct) {
			return true
		}
	}
	return false
}

// announcedSlots is every beacon round that a draw has ever been announced for, by any
// round, other than the one `except` holds now. A verifier remembers the first thing it
// was told about a slot, so a slot is used for one announcement and never another.
func announcedSlots(s *State, except *Round) map[int64]bool {
	taken := map[int64]bool{}
	for _, r := range s.Rounds {
		if r != except {
			taken[r.DrandRound] = true
		}
		for _, was := range r.Was {
			taken[was] = true
		}
	}
	return taken
}

// reschedule moves the scheduled round to a later draw time: a new announcement, made
// before that time's beacon value exists. It takes the newest crawl with it.
func (a *App) reschedule(tx *Tx, r *Round, at int64) {
	// A slot this round has been announced for and left is not gone back to, and neither
	// is one that was ever another round's: by now the crawl may be a newer one, and a
	// verifier holds the announcement it saw first.
	for taken := announcedSlots(tx.s, r); taken[a.season.Beacon.drandRoundAt(at)]; {
		at = a.season.nextGrid(at)
	}
	next := a.season.Beacon.drandRoundAt(at)
	if next == r.DrandRound {
		return // what was announced for this slot stands, crawl and all
	}
	if c := a.savedCrawl(tx.s); c != nil {
		r.Crawl = c
	}
	if r.Was = append(r.Was, r.DrandRound); len(r.Was) > 64 {
		r.Was = r.Was[len(r.Was)-64:]
	}
	r.DrawAt, r.DrandRound = at, next
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
		case e.Status == "running" && roundsRun(s, e.ID) < a.cfg.QualifyRounds:
			qualifying = append(qualifying, e)
		case e.Status == "queued" && e.CreatedAt <= cutoff-commitMargin:
			queue = append(queue, e)
		}
	}
	sort.Slice(qualifying, func(i, j int) bool {
		return qualifying[i].CreatedAt < qualifying[j].CreatedAt || (qualifying[i].CreatedAt == qualifying[j].CreatedAt && qualifying[i].ID < qualifying[j].ID)
	})
	for _, e := range qualifying {
		add(e.ID, "qualifying")
	}
	// One person may have several entries waiting. Each person's first goes before anyone's
	// second, so that nobody fills a round alone while others wait; an entry of theirs that
	// is already qualifying counts as a turn taken.
	sort.Slice(queue, func(i, j int) bool {
		return queue[i].CreatedAt < queue[j].CreatedAt || (queue[i].CreatedAt == queue[j].CreatedAt && queue[i].ID < queue[j].ID)
	})
	turns, turn := map[string]int{}, map[string]int{}
	for _, e := range qualifying {
		turns[person(e)]++
	}
	for _, e := range queue {
		turn[e.ID] = turns[person(e)]
		turns[person(e)]++
	}
	priority := recentWork(s) // runs and grades for others in the last two weeks
	sort.SliceStable(queue, func(i, j int) bool {
		if ti, tj := turn[queue[i].ID], turn[queue[j].ID]; ti != tj {
			return ti < tj
		}
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

// expectedRow is a curator a scheduled round's draw would enter if it were made now, or a
// queued one it would leave for a later round.
type expectedRow struct {
	EditorID string `json:"editor_id"`
	Name     string `json:"name"`
	Version  int    `json:"version"`
	Owner    string `json:"owner"`
	Why      string `json:"why"`              // ladder | qualifying | new | later
	Played   int    `json:"played,omitempty"` // rounds a qualifying curator has played
	Public   bool   `json:"public"`           // its code is public: it has been drawn before
}

// expected is what the draw of a scheduled round would do if it were made now: who it would
// enter, and which queued curators it would leave for a later round. It is not a promise:
// an entry submitted, replaced or withdrawn before the draw changes it. Nothing about a
// curator that has never been drawn is shown but its name and its owner.
func (a *App) expected(s *State, rd *Round) (in, later []expectedRow) {
	if rd == nil || rd.Status != "scheduled" {
		return nil, nil
	}
	row := func(ed *Editor, why string) expectedRow {
		return expectedRow{EditorID: ed.ID, Name: ed.Name, Version: ed.Version, Owner: s.owner(ed).Handle, Why: why, Public: ed.Public}
	}
	chosen := map[string]bool{}
	for _, e := range a.selectEntrants(s, rd.DrawAt) {
		ed := s.Editors[e.EditorID]
		if ed == nil {
			continue
		}
		chosen[ed.ID] = true
		r := row(ed, e.Why)
		if e.Why == "qualifying" {
			r.Played = roundsRun(s, ed.ID)
		}
		in = append(in, r)
	}
	var queued []*Editor
	for _, ed := range s.Editors {
		if ed.Status == "queued" && !chosen[ed.ID] {
			queued = append(queued, ed)
		}
	}
	sort.Slice(queued, func(i, j int) bool {
		return queued[i].CreatedAt < queued[j].CreatedAt || (queued[i].CreatedAt == queued[j].CreatedAt && queued[i].ID < queued[j].ID)
	})
	for _, ed := range queued {
		later = append(later, row(ed, "later"))
	}
	return in, later
}

// bringForward moves the scheduled round to the next moment on the grid (an organiser asked
// for a draw sooner than the schedule's). In development the draw is immediate and off the
// grid, which real verifiers would refuse.
func (a *App) bringForward(ctx context.Context) (*Round, error) {
	var out Round
	err := a.st.update(ctx, func(tx *Tx) error {
		r := tx.s.roundWithStatus("scheduled")
		if r == nil {
			return fail(409, "no round is scheduled")
		}
		if tx.s.Meta.Paused != 0 {
			return fail(409, "the rounds are paused; resume them first")
		}
		if tx.s.roundWithStatus("open") != nil {
			return fail(409, "the previous round is still being graded")
		}
		at := a.season.nextGrid(now() + drawNotice)
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
		case s.Meta.Paused != 0:
			// The organisers have paused the rounds: the slot passes, as when nobody is waiting.
			a.reschedule(tx, cur, a.laterSlot())
			problem = errors.New("the organisers have paused the rounds")
		case s.roundWithStatus("open") != nil:
			// Grading of the previous round is still under way. Rounds never overlap, so
			// that the ladder is always complete. This one moves to the next slot for now,
			// and is drawn sooner if the round before settles sooner (settleRounds).
			cur.Postponed = true
			a.reschedule(tx, cur, a.laterSlot())
			tx.saveRound(cur)
			problem = errors.New("the previous round is still being graded")
		case len(a.selectEntrants(s, r.DrawAt)) == 0:
			a.reschedule(tx, cur, a.laterSlot())
			problem = errors.New("no curators are waiting to run")
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
			a.reschedule(tx, cur, a.laterSlot())
			problem = errors.New("no curators are waiting to run")
			return nil
		}
		// The crawl is the one announced with the round, never one chosen now.
		cur.Status, cur.Randomness, cur.DrawnAt, cur.ActiveAt = "open", randomness, now(), now()
		cur.Seed = hexOf(officialSeed(a.season.ID, cur.Idx, cur.DrandRound, randomness, *cur.Crawl))
		tx.saveRound(cur)
		for _, e := range entrants {
			tx.saveEntry(&Entry{RoundIdx: cur.Idx, EditorID: e.EditorID, Status: "open"})
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
		a.refreshScores(tx, r)
		a.regain(tx, r)
		r.Status, r.SettledAt = "settled", now()
		tx.saveRound(r)
		// An editor that has had its qualifying rounds stops being scheduled, unless it is on
		// the ladder (the ladder is recomputed at every draw).
		for _, e := range s.inRound[r.Idx] {
			ed := s.Editors[e.EditorID]
			if ed.Status == "running" && roundsRun(s, ed.ID) >= a.cfg.QualifyRounds {
				ed.Status = "retired"
				tx.saveEditor(ed)
			}
			// Whether an entry the verifiers could not run is the entry's trouble or theirs
			// depends on whether they ran the others, which is known only now.
			if e.Void || !e.Valid {
				a.afterInvalid(tx, e.EditorID)
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
		// A round that ran over cost the next one its slot. It does not cost it six hours:
		// the draw moves to the next mark on the grid.
		if next := s.roundWithStatus("scheduled"); next != nil && next.Postponed {
			next.Postponed = false
			if at := a.season.nextGrid(now() + drawNotice); at < next.DrawAt {
				a.reschedule(tx, next, at)
			}
			tx.saveRound(next) // reschedule keeps the slot it has when no earlier one is free
		}
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
