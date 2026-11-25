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
	"slices"
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
		if len(r.Hours) > 0 {
			d["exam_hours"] = r.Hours
			if r.HoursLate {
				d["exam_hours_late"] = true
			}
		}
		d["seed"] = r.Seed
		d["beacon"] = map[string]any{"round": r.DrandRound, "randomness": r.Randomness}
		d["drawn_at"] = r.DrawnAt
	}
	if examPosted(r) {
		d["exam_digest"], d["exam_at"] = r.ExamSHA, r.ExamAt
	}
	return d
}

// onDemandOnly reports whether rounds are drawn only when an entry is waiting (RoundInterval 0),
// with no idle rounds on a fixed clock.
func (a *App) onDemandOnly() bool { return a.cfg.RoundInterval == 0 }

// nextSlot is the next scheduled draw time after a moment: a multiple of the round
// interval, which is itself a multiple of the season's grid.
func (a *App) nextSlot(after int64) int64 {
	if a.onDemandOnly() {
		return a.season.nextGrid(after)
	}
	iv := int64(a.cfg.RoundInterval.Seconds())
	if iv <= 0 {
		return a.season.nextGrid(after)
	}
	return (after/iv + 1) * iv
}

// laterSlot is where a draw goes that is being announced now: the next usual time, or,
// when that is less than the notice away, the first place on the grid that is not. No
// draw is ever announced with less than drawNotice to go.
func (a *App) laterSlot() int64 {
	if a.onDemandOnly() {
		return a.season.nextGrid(now() + drawNotice)
	}
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

const emptyBackoff = 3 * time.Hour // retry cadence after two consecutive rounds settled with no baseline

// emptyStreak returns how many consecutive latest drawn rounds settled with no baseline, and
// when the latest of them finished.
func emptyStreak(s *State) (empty int, lastEmpty int64) {
	for _, rd := range s.roundsDesc(3) {
		if rd.DrawnAt == 0 {
			continue // the scheduled round itself
		}
		if rd.Status == "settled" && rd.BaselineLoss == nil {
			empty++
			if done := max(rd.SettledAt, rd.DrawnAt); done > lastEmpty {
				lastEmpty = done
			}
		} else {
			break
		}
	}
	return empty, lastEmpty
}

// backoffUntil reports when on-demand automatic draws will resume after two consecutive
// rounds settled with no baseline, or 0 when draws are not backed off.
func (a *App) backoffUntil(s *State) int64 {
	if !a.onDemandOnly() {
		return 0
	}
	empty, lastEmpty := emptyStreak(s)
	if empty < 2 {
		return 0
	}
	if retry := lastEmpty + int64(emptyBackoff.Seconds()); now() < retry {
		return retry
	}
	return 0
}

// demandSlot is the earliest grid moment an on-demand draw may take, respecting drawNotice
// and RoundMinGap (when set), and reports whether the last two drawn rounds both came to
// nothing. In fixed-clock mode that holds back early draws so the fixed schedule retries; in
// on-demand-only mode it backs off for emptyBackoff so a transient outage neither spins GPUs
// nor deadlocks automatic draws forever.
func (a *App) demandSlot(s *State) (at int64, blocked bool) {
	gap := int64(a.cfg.RoundMinGap.Seconds())
	at = now() + drawNotice
	for _, rd := range s.roundsDesc(3) {
		if rd.DrawnAt == 0 {
			continue // the scheduled round itself
		}
		if gap > 0 && rd.DrawnAt+gap > at {
			at = rd.DrawnAt + gap
		}
	}
	empty, lastEmpty := emptyStreak(s)
	if empty >= 2 {
		if !a.onDemandOnly() {
			return 0, true
		}
		if retry := lastEmpty + int64(emptyBackoff.Seconds()); now() < retry {
			return 0, true
		}
	}
	return a.season.nextGrid(at - 1), false
}

// takePendingWas consumes any moved_from trail left by a previously dropped scheduled round
// for this round index so the public audit trail survives on-demand unscheduling.
func takePendingWas(tx *Tx) []int64 {
	if len(tx.s.Meta.PendingWas) == 0 {
		return nil
	}
	was := append([]int64(nil), tx.s.Meta.PendingWas...)
	tx.s.Meta.PendingWas = nil
	tx.put(metaKey, &tx.s.Meta)
	return was
}

// schedule announces the next round, if none is announced: when it will be drawn (and so
// from which beacon round) and from which crawl. In on-demand-only mode (RoundInterval 0),
// a round is announced only once the previous round has settled, an entry is waiting, and
// somebody who runs curators is around.
func (a *App) schedule(tx *Tx) {
	if a.cfg.ArchiveSeed != "" { // a finished season: no round comes next
		return
	}
	s := tx.s
	crawl := a.savedCrawl(s)
	if s.roundWithStatus("scheduled") != nil || crawl == nil {
		return
	}
	var at int64
	if a.onDemandOnly() {
		if s.Meta.Paused != 0 || s.roundWithStatus("open") != nil || !a.waiting(s) || !a.runnerAround(s) {
			return
		}
		slot, blocked := a.demandSlot(s)
		if blocked {
			return
		}
		at = slot
	} else {
		at = a.laterSlot()
	}
	// The usual time, or the first free place on the grid after it: a slot some round
	// was once announced for costs the next round ten minutes, not a whole interval.
	taken := announcedSlots(s, nil)
	for taken[a.season.Beacon.drandRoundAt(at)] {
		at = a.season.nextGrid(at)
	}
	rd := &Round{Idx: s.maxRound() + 1, Status: "scheduled", Scale: a.cfg.Scale, DrawAt: at, DrandRound: a.season.Beacon.drandRoundAt(at), Crawl: crawl, Was: takePendingWas(tx)}
	tx.saveRound(rd)
	if a.onDemandOnly() {
		slog.Info("round scheduled on demand: an entry is waiting", "round", roundID(a.season.ID, rd.Idx), "at", rd.DrawAt)
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
	if a.onDemandOnly() {
		a.schedule(tx)
		if r := tx.s.roundWithStatus("scheduled"); r != nil && tx.s.Meta.Paused == 0 && tx.s.roundWithStatus("open") == nil && a.waiting(tx.s) && a.runnerAround(tx.s) {
			if at, blocked := a.demandSlot(tx.s); !blocked && at < r.DrawAt {
				a.reschedule(tx, r, at)
				slog.Info("draw brought forward: an entry is waiting", "round", roundID(a.season.ID, r.Idx), "at", r.DrawAt)
			}
		}
		return
	}
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
	at, blocked := a.demandSlot(s)
	if blocked {
		return nil, 0
	}
	if at < r.DrawAt {
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
	for _, slot := range s.Meta.UsedSlots {
		taken[slot] = true
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

// resting is which of the ladder's editors sit round idx out. The organisers' reference
// editors are yardsticks, and every one of them in every round was half of all the training
// a season did. So the few the season names (references.every_round) run in every round,
// as the ladder does: they are what ties the rounds' difficulties together. Each of the
// others takes one round in references.one_in, and they take different ones: the k-th of
// them, in the order they were entered, runs in the rounds where idx+k is a multiple of it.
//
// It is the draw that leaves them out, not the board (computeBoard): a reference that rests
// keeps its place on the ladder, and its rating is from the rounds it ran. A round it sat
// out is no entry of its, so nothing that counts an editor's rounds or its failures
// (roundsRun, failing) ever sees one. The record holder never rests, whoever entered it: the
// record changes hands only on rounds shared with it. Nor does an editor still in its
// qualifying rounds, which selectEntrants enters for that reason whatever the ladder says.
func (a *App) resting(b *Board, idx int) map[string]bool {
	rule := a.season.References
	out := map[string]bool{}
	if rule.OneIn < 2 {
		return out
	}
	var turns []*BoardRow
	for _, row := range b.Ladder {
		if row.Organisers && row.ID != b.RecordID && !slices.Contains(rule.EveryRound, row.Name) {
			turns = append(turns, row)
		}
	}
	sort.Slice(turns, func(i, j int) bool {
		return turns[i].CreatedAt < turns[j].CreatedAt || (turns[i].CreatedAt == turns[j].CreatedAt && turns[i].ID < turns[j].ID)
	})
	for k, row := range turns {
		if (idx+k)%rule.OneIn != 0 {
			out[row.ID] = true
		}
	}
	return out
}

// selectEntrants decides who runs in round idx: the ladder (so that the best editors always
// share rounds and can be compared head to head) less the reference editors whose turn it
// is not (resting), editors still qualifying, and every new entry waiting.
func (a *App) selectEntrants(s *State, cutoff int64, idx int) []entrant {
	seen := map[string]bool{}
	var out []entrant
	add := func(id, why string) {
		if !seen[id] {
			seen[id] = true
			out = append(out, entrant{id, why})
		}
	}
	board := a.computeBoard(s)
	rest := a.resting(board, idx)
	for _, row := range board.Ladder {
		if !rest[row.ID] {
			add(row.ID, "ladder")
		}
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
	// Every curator waiting at the draw enters it, in the order they were submitted. A round
	// turns nobody away: what one person may have waiting or qualifying at once is settled
	// when an entry is submitted (EntriesInPlay), and rounds never overlap, so a great many
	// entries make a long round and not a lost one.
	sort.Slice(queue, func(i, j int) bool {
		return queue[i].CreatedAt < queue[j].CreatedAt || (queue[i].CreatedAt == queue[j].CreatedAt && queue[i].ID < queue[j].ID)
	})
	for _, e := range queue {
		add(e.ID, "new")
	}
	if len(out) == 0 {
		// Nobody but references whose turn it is not. A round with no entrant is not drawn,
		// and it keeps its number until it is, so their turn would never come: they all run.
		for _, row := range board.Ladder {
			add(row.ID, "ladder")
		}
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
// enter, and which queued curators it would leave for the next round (those submitted in the
// last seconds before the draw, after its entries close). It is not a promise:
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
	for _, e := range a.selectEntrants(s, rd.DrawAt, rd.Idx) {
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
		r := tx.s.roundWithStatus("scheduled")
		if r == nil {
			crawl := a.savedCrawl(tx.s)
			if !a.onDemandOnly() || crawl == nil {
				return fail(409, "no round is scheduled")
			}
			taken := announcedSlots(tx.s, nil)
			for taken[a.season.Beacon.drandRoundAt(at)] {
				at = a.season.nextGrid(at)
			}
			r = &Round{Idx: tx.s.maxRound() + 1, Status: "scheduled", Scale: a.cfg.Scale, DrawAt: at, DrandRound: a.season.Beacon.drandRoundAt(at), Crawl: crawl, Was: takePendingWas(tx), Manual: true}
			tx.saveRound(r)
			out = *r
			return nil
		}
		r.Manual = true
		if at < r.DrawAt || a.cfg.Dev {
			a.reschedule(tx, r, at)
		} else {
			tx.saveRound(r)
		}
		out = *r
		return nil
	})
	return &out, err
}

// hasEligibleWaiting reports whether any editor eligible for a draw at cutoff is waiting
// to start (queued before cutoff - commitMargin) or still finishing its qualifying rounds
// (running with fewer than QualifyRounds completed), independent of whether it already holds
// a place on the ladder (e.g. when RatedRounds < QualifyRounds).
func (a *App) hasEligibleWaiting(s *State, cutoff int64) bool {
	for _, e := range s.Editors {
		switch {
		case e.Status == "running" && roundsRun(s, e.ID) < a.cfg.QualifyRounds:
			return true
		case e.Status == "queued" && e.CreatedAt <= cutoff-commitMargin:
			return true
		}
	}
	return false
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
			// The organisers have paused the rounds: in on-demand-only mode unschedule the passed
			// slot cleanly so resuming draws at the next grid mark; on a clock, move to laterSlot.
			if a.onDemandOnly() {
				tx.dropRound(cur)
			} else {
				a.reschedule(tx, cur, a.laterSlot())
			}
			problem = errors.New("the organisers have paused the rounds")
		case s.roundWithStatus("open") != nil:
			// Grading of the previous round is still under way. Rounds never overlap, so
			// that the ladder is always complete. This one moves to the next slot for now,
			// and is drawn sooner if the round before settles sooner (settleRounds).
			cur.Postponed = true
			a.reschedule(tx, cur, a.laterSlot())
			tx.saveRound(cur)
			problem = errors.New("the previous round is still being graded")
		case len(a.selectEntrants(s, r.DrawAt, r.Idx)) == 0:
			if a.onDemandOnly() {
				if a.waiting(s) {
					a.reschedule(tx, cur, a.season.nextGrid(now()+drawNotice))
				} else {
					tx.dropRound(cur)
				}
			} else {
				a.reschedule(tx, cur, a.laterSlot())
			}
			problem = errors.New("no curators are waiting to run")
		case a.onDemandOnly() && !force && !cur.Manual && !a.hasEligibleWaiting(s, r.DrawAt):
			// In on-demand-only mode do not draw a round for the ladder alone if the sole waiting
			// entry withdrew or resubmitted inside the last commitMargin seconds.
			if a.waiting(s) {
				a.reschedule(tx, cur, a.season.nextGrid(now()+drawNotice))
			} else {
				tx.dropRound(cur)
			}
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
		entrants := a.selectEntrants(s, cur.DrawAt, cur.Idx)
		if len(entrants) == 0 {
			if a.onDemandOnly() {
				if a.waiting(s) {
					a.reschedule(tx, cur, a.season.nextGrid(now()+drawNotice))
				} else {
					tx.dropRound(cur)
				}
			} else {
				a.reschedule(tx, cur, a.laterSlot())
			}
			problem = errors.New("no curators are waiting to run")
			return nil
		}
		if a.onDemandOnly() && !force && !cur.Manual && !a.hasEligibleWaiting(s, cur.DrawAt) {
			if a.waiting(s) {
				a.reschedule(tx, cur, a.season.nextGrid(now()+drawNotice))
			} else {
				tx.dropRound(cur)
			}
			problem = errors.New("no curators are waiting to run")
			return nil
		}
		// The crawl is the one announced with the round, never one chosen now.
		cur.Status, cur.Randomness, cur.DrawnAt, cur.ActiveAt = "open", randomness, now(), now()
		cur.Seed = hexOf(officialSeed(a.season.ID, cur.Idx, cur.DrandRound, randomness, *cur.Crawl))
		cur.Hours = a.roundHours(s, cur) // the writing its exam is made from: see harvest.go
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
		if a.onDemandOnly() {
			a.schedule(tx)
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
