package main

// Ratings, the board and the record.
//
// An editor's rating starts from its mean gain over the rounds it ran in, adjusted for which
// rounds those happened to be. Some rounds' corpora are easier to improve than others, and
// editors do not all run in the same rounds, so a plain mean would reward luck of the draw.
// The model is the simplest one that removes it:
//
//     gain(editor, round) = skill(editor) + difficulty(round) + noise
//
// fitted by least squares. The ladder's editors that run in every round (the entrants' best,
// and the reference editors the season names) tie the rounds together, so difficulties are
// well determined.
//
// That adjusted mean is the rating, and nothing else is in it: what an editor's rating says is
// what its own rounds said. A few rounds say little, and three lucky ones would otherwise take
// a place from thirty steady ones. So every rating has a standard error, wider the fewer rounds
// it rests on (fitRatings); the ladder is filled by what each editor has shown it is at least
// worth, its rating less 1.28 standard errors; and ratings too close to tell apart are marked
// as tied.
//
// The rating was once drawn towards the mean of every editor on the board as well, by as much
// as one round's worth. That mean is set by whoever happens to be on the board, the organisers'
// weakest reference editors among them: in Season 2's second round it stood at -0.11, and
// three new editors that had each gained 0.07 were shown at -0.02, below editors they had
// beaten. The standard error already says how little one round is worth.
//
// An entry that failed counts as the lowest gain any valid entry got in its round, or zero
// if that is lower (failedGain): failing is never the better move.

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// entryRow is one round as it counts for one editor. Gain is the entry's gain, or for an
// entry that failed what the round counts as instead (failedGain); Valid says which.
type entryRow struct {
	RoundIdx int
	EditorID string
	Gain     float64
	Valid    bool
}

type Rating struct {
	N        int     // settled rounds
	Mean     float64 // plain mean of what its rounds count as
	Adjusted float64 // that, adjusted for which rounds it happened to run: the rating
	SE       float64 // the rating's standard error
	Invalid  int
}

const (
	// The variance of one round's gain to assume while neither the entries nor the season's
	// file can say: a spread of 0.01 bits per byte.
	noiseFloor = 0.0001
	// A rating's lower bound is this many standard errors under it: nine times in ten the
	// editor is worth at least that.
	lowerZ = 1.28
)

// fitRatings solves the additive model by alternating means, which converges in a handful
// of passes because the ladder connects the rounds. An editor's rating is its adjusted mean,
// and the rating's standard error is sqrt(s2/n) over its n rounds.
//
// s2 is how much one round's gain moves about an editor's own mean. It is pooled over every
// editor with two measured rounds or more (a failed round is a penalty on its editor's rating,
// and no reading of the noise), together with noise: what a round's gain was taken to move by
// before the season began, counted as prior rounds' worth of evidence. A season's first few
// rounds say little about their own spread, and would otherwise put a standard error on
// every rating from two or three numbers.
func fitRatings(entries []entryRow, noise, prior float64) (map[string]*Rating, map[int]float64) {
	byEditor := map[string][]entryRow{}
	byRound := map[int][]entryRow{}
	for _, e := range entries {
		byEditor[e.EditorID] = append(byEditor[e.EditorID], e)
		byRound[e.RoundIdx] = append(byRound[e.RoundIdx], e)
	}
	// skill is what an editor does in the rounds where it ran: its mean over valid entries
	// only. It is what explains a round's difficulty. The rating published below counts a
	// failed round as the round's lowest gain, but that must not be used here: it is a
	// penalty and not a measurement, and taken for one it would make the editor's good
	// rounds look easy and mark down everyone who shared them.
	skill := map[string]float64{}
	diff := map[int]float64{}
	for i := 0; i < 200; i++ {
		delta := 0.0
		for id, es := range byEditor {
			s, n := 0.0, 0
			for _, e := range es {
				if e.Valid {
					s += e.Gain - diff[e.RoundIdx]
					n++
				}
			}
			if n > 0 {
				s /= float64(n)
			}
			delta = math.Max(delta, math.Abs(s-skill[id]))
			skill[id] = s
		}
		mean := 0.0
		for r, es := range byRound {
			// Only entries that were really measured say anything about the round. What a
			// failed entry counts as is a penalty, not a reading: counted here, one editor's
			// timeout would make the round look hard and lift everyone else in it.
			s, n := 0.0, 0
			for _, e := range es {
				if e.Valid {
					s += e.Gain - skill[e.EditorID]
					n++
				}
			}
			diff[r] = 0
			if n > 0 {
				diff[r] = s / float64(n)
			}
			mean += diff[r]
		}
		mean /= math.Max(1, float64(len(byRound)))
		for r := range diff {
			diff[r] -= mean // difficulty is relative: it averages to zero over the season
		}
		if delta < 1e-9 && i > 0 {
			break
		}
	}
	// In the order of their ids, so that the sums below come out the same every time.
	ids := make([]string, 0, len(byEditor))
	for id := range byEditor {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := map[string]*Rating{}
	ss, df := 0.0, 0 // squared residuals, and their degrees of freedom
	for _, id := range ids {
		es := byEditor[id]
		r := &Rating{N: len(es)}
		var resid, measured []float64
		for _, e := range es {
			r.Mean += e.Gain / float64(len(es))
			resid = append(resid, e.adjusted(diff))
			if e.Valid {
				measured = append(measured, e.adjusted(diff))
			} else {
				r.Invalid++
			}
		}
		r.Adjusted = mean(resid)
		// How much a round's gain moves is read from the rounds that were measured, about
		// their own mean. What a failed round counts as is far from them by design: it marks
		// its editor's rating down, and counted as noise it would widen everybody's.
		if len(measured) >= 2 {
			own := mean(measured)
			for _, x := range measured {
				ss += (x - own) * (x - own)
			}
			df += len(measured) - 1
		}
		out[id] = r
	}
	s2 := noise
	if df > 0 {
		s2 = (ss + prior*noise) / (float64(df) + prior)
	}
	for _, r := range out {
		r.SE = math.Sqrt(s2 / float64(r.N))
	}
	return out, diff
}

// adjusted is what a round counts as for an editor with the round's difficulty taken out:
// a failed entry's like any other's.
func (e entryRow) adjusted(diff map[int]float64) float64 {
	return e.Gain - diff[e.RoundIdx]
}

func mean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func stddev(v []float64) float64 {
	m := mean(v)
	s := 0.0
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return math.Sqrt(s / float64(len(v)-1))
}

// pairedP is the one-sided p-value that the mean of d is greater than zero (Student's t).
func pairedP(d []float64) float64 {
	n := len(d)
	if n < 2 {
		return 1
	}
	sd := stddev(d)
	m := mean(d)
	if sd == 0 {
		if m > 0 {
			return 0
		}
		return 1
	}
	t := m / (sd / math.Sqrt(float64(n)))
	df := float64(n - 1)
	p := 0.5 * incBeta(df/2, 0.5, df/(df+t*t))
	if t < 0 {
		return 1 - p
	}
	return p
}

// incBeta is the regularised incomplete beta function I_x(a, b), by continued fraction.
func incBeta(a, b, x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	lg := func(v float64) float64 { r, _ := math.Lgamma(v); return r }
	front := math.Exp(lg(a+b) - lg(a) - lg(b) + a*math.Log(x) + b*math.Log(1-x))
	if x > (a+1)/(a+b+2) {
		return 1 - incBeta(b, a, 1-x)
	}
	const tiny = 1e-300
	c, d := 1.0, 1-(a+b)*x/(a+1)
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1 / d
	h := d
	for m := 1; m <= 300; m++ {
		fm := float64(m)
		num := fm * (b - fm) * x / ((a + 2*fm - 1) * (a + 2*fm))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		h *= d * c
		num = -(a + fm) * (a + b + fm) * x / ((a + 2*fm) * (a + 2*fm + 1))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < 1e-12 {
			break
		}
	}
	return front * h / a
}

// --- the board ------------------------------------------------------------------------------

type BoardRow struct {
	Rank        int       `json:"rank,omitempty"`
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Version     int       `json:"version"`
	Owner       string    `json:"owner"`
	Github      string    `json:"github,omitempty"`
	Technique   string    `json:"technique,omitempty"`
	Description string    `json:"description,omitempty"`
	Parent      string    `json:"parent,omitempty"`
	Status      string    `json:"status"`
	Rounds      int       `json:"rounds"`
	Rating      *float64  `json:"rating"`    // the mean gain adjusted for its rounds: what the board ranks by
	Adjusted    *float64  `json:"adjusted"`  // the same number, under the name it had when the rating was something more
	MeanGain    *float64  `json:"mean_gain"` // the plain mean of what its rounds count as
	SE          *float64  `json:"se"`        // the rating's standard error
	Lower       *float64  `json:"lower"`     // the rating less lowerZ standard errors: what the ladder is filled by
	Multiplier  *float64  `json:"multiplier"`
	Capped      bool      `json:"capped,omitempty"` // the multiplier is "more than" this: the gain runs off the measured curve
	Invalid     int       `json:"invalid,omitempty"`
	Provisional bool      `json:"provisional,omitempty"`
	Record      bool      `json:"record,omitempty"`
	Ladder      bool      `json:"ladder,omitempty"`
	Tied        bool      `json:"tied,omitempty"` // too close to the row its run of tied rows starts at to tell apart
	TiedAt      int       `json:"-"`              // the rank its run of tied rows starts at; 0 when it is tied with nobody
	Organisers  bool      `json:"organisers"`     // entered by the organisers: a reference curator, a yardstick
	CreatedAt   int64     `json:"created_at"`
	AccountID   int64     `json:"-"`
	Gains       []float64 `json:"-"` // per settled round, oldest first, for sparklines
}

type Board struct {
	Rated       []*BoardRow          // ranked
	Provisional []*BoardRow          // fewer than RatedRounds settled rounds
	Ladder      []*BoardRow          // top editors, one per account: run in every round, but for the references that take turns (resting); see computeBoard
	ByID        map[string]*BoardRow // every editor that has settled at least one round
	Entries     []entryRow
	Difficulty  map[int]float64
	RecordID    string
}

func (a *App) curveKey() string { return "official/" + a.cfg.Scale }

// hasMultiplier says whether a rating can be read as a multiplier at all: whether the season
// has a reference curve for the scale rounds are run at.
func (a *App) hasMultiplier() bool {
	_, _, ok := a.season.multiplier(a.curveKey(), 0)
	return ok
}

// rated reports whether an entry is one the ratings and the record count: settled, run, in
// a round that has settled with a baseline.
func (s *State) rated(e *Entry) bool {
	// Whole rounds only. An entry with no script settles the moment that is known, while
	// the rest of its round is still sealed; alone, it would stand for the whole round.
	rd := s.Rounds[e.RoundIdx]
	return e.Status == "settled" && !e.Void && rd != nil && rd.BaselineLoss != nil && rd.Status == "settled"
}

// failedGain is what a failed entry counts as in a round: the lowest gain any valid entry
// got in it, or zero if that is lower, as it is when no entry was valid. A failure that
// counted as zero, in a round where finishing scored below zero, was the better move.
func failedGain(s *State, idx int) float64 {
	low := 0.0
	for _, e := range s.inRound[idx] {
		if s.rated(e) && e.Valid && e.Gain < low {
			low = e.Gain
		}
	}
	return low
}

// countsAs is what a round counts as for an entry that failed in it, and whether the entry
// is one: an entry that was not run, or whose round counts for nobody, counts as nothing.
func countsAs(s *State, e *Entry) (float64, bool) {
	if e.Valid || !s.rated(e) {
		return 0, false
	}
	return failedGain(s, e.RoundIdx), true
}

// settledEntries are the entries that count: settled, in a round whose baseline is known. A
// failed entry's row carries what it counts as, and is still marked as not valid: it says
// nothing about how hard the round was, or how good the editor is when it works.
func settledEntries(s *State) []entryRow {
	var out []entryRow
	failed := map[int]float64{}
	for _, e := range s.Entries {
		if !s.rated(e) {
			continue
		}
		row := entryRow{RoundIdx: e.RoundIdx, EditorID: e.EditorID, Gain: e.Gain, Valid: e.Valid}
		if !e.Valid {
			if _, ok := failed[e.RoundIdx]; !ok {
				failed[e.RoundIdx] = failedGain(s, e.RoundIdx)
			}
			row.Gain = failed[e.RoundIdx]
		}
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].RoundIdx < out[j].RoundIdx || (out[i].RoundIdx == out[j].RoundIdx && out[i].EditorID < out[j].EditorID)
	})
	return out
}

// roundNoise is the variance of one round's gain to assume until the season's own rounds
// can say: the square of how far the season's file takes a curator's gain to move from round
// to round (noise.<scale>.round_sd), or failing that between two trainings (gain_sd), or
// noiseFloor where neither is given.
func (a *App) roundNoise() float64 {
	if m, ok := a.season.Noise[a.cfg.Scale].(map[string]any); ok {
		for _, key := range []string{"round_sd", "gain_sd"} {
			if sd, ok := m[key].(float64); ok && sd > 0 {
				return sd * sd
			}
		}
	}
	return noiseFloor
}

// priorRounds is how many rounds' worth of evidence roundNoise counts as, beside the season's
// own rounds, when the spread of a round's gain is worked out.
const priorRounds = 8

func (a *App) computeBoard(s *State) *Board {
	entries := settledEntries(s)
	ratings, diff := fitRatings(entries, a.roundNoise(), priorRounds)
	b := &Board{ByID: map[string]*BoardRow{}, Entries: entries, Difficulty: diff}
	if h := s.holder(); h != nil {
		b.RecordID = h.EditorID
	}
	for _, e := range s.Editors {
		rt := ratings[e.ID]
		if rt == nil || !e.Public {
			continue
		}
		owner := s.owner(e)
		r := &BoardRow{ID: e.ID, Name: e.Name, Version: e.Version, Owner: owner.Handle, Github: owner.Github, Technique: e.Technique,
			Description: e.Description, Parent: e.Parent, Status: e.Status, CreatedAt: e.CreatedAt, AccountID: e.AccountID, Organisers: owner.Anchor}
		r.Rounds, r.Invalid = rt.N, rt.Invalid
		rating, adj, mg, se, lower := rt.Adjusted, rt.Adjusted, rt.Mean, rt.SE, rt.Adjusted-lowerZ*rt.SE
		r.Rating, r.Adjusted, r.MeanGain, r.SE, r.Lower = &rating, &adj, &mg, &se, &lower
		if m, capped, ok := a.season.multiplier(a.curveKey(), rating); ok {
			r.Multiplier, r.Capped = &m, capped
		}
		r.Record = r.ID == b.RecordID
		b.ByID[r.ID] = r
		if e.Status == "withdrawn" {
			// Replaced by its owner. It keeps its page and its results, but not a rank: a rank
			// kept by an editor that will never run again is a lucky start made permanent.
			continue
		}
		if rt.N >= a.cfg.RatedRounds {
			b.Rated = append(b.Rated, r)
		} else {
			r.Provisional = true
			b.Provisional = append(b.Provisional, r)
		}
	}
	for _, e := range entries {
		if r := b.ByID[e.EditorID]; r != nil {
			r.Gains = append(r.Gains, e.Gain)
		}
	}
	byRating := func(rows []*BoardRow) {
		sort.Slice(rows, func(i, j int) bool {
			if *rows[i].Rating != *rows[j].Rating {
				return *rows[i].Rating > *rows[j].Rating
			}
			if rows[i].CreatedAt != rows[j].CreatedAt {
				return rows[i].CreatedAt < rows[j].CreatedAt
			}
			return rows[i].ID < rows[j].ID
		})
	}
	byRating(b.Rated)
	byRating(b.Provisional)
	// Ranks, and ties. A row is tied with the first of the run of tied rows above it when
	// the gap between their ratings is within two standard errors of that gap: the order of
	// the two could be chance. A run of tied rows shares the rank it starts at. It is the
	// run's first row a row is held against, and not the one just above: a long field in
	// which each is a little behind the last would otherwise all be tied with its best.
	var first *BoardRow // of the run the row above is in
	for i, r := range b.Rated {
		r.Rank = i + 1
		if first != nil && *first.Rating-*r.Rating <= 2*math.Sqrt(*first.SE**first.SE+*r.SE**r.SE) {
			first.TiedAt = first.Rank
			r.Tied, r.TiedAt = true, first.Rank
			continue
		}
		first = r
	}
	// The ladder: the record holder, then the best editor of each person. The organisers'
	// reference editors are fixed points for everyone and all stay on: the ladder's size is
	// how many places everyone else's editors take, the holder's among them, and no entrant's
	// editor takes a reference's.
	//
	// Best by the lower bound of its rating, not by the rating: a place on the ladder is a
	// run in every round, on every verifier's machine, and an editor has to have shown it
	// is worth one. Three good rounds have not; thirty steady ones a little lower have.
	//
	// The holder keeps its place whatever it does. A challenger can only take the record on
	// rounds it shares with the holder, so a holder that stopped running would hold the
	// record for ever. One that has started failing counts as the worst of each round and
	// is soon beaten. Its place is its own: its owner's best other editor may still take
	// one, which is how a person improves on their own record.
	seen := map[string]bool{}
	taken := 0 // places held by editors that are not the organisers'
	if h := b.ByID[b.RecordID]; h != nil {
		h.Ladder = true
		b.Ladder = append(b.Ladder, h)
		if !h.Organisers {
			taken++
		}
	}
	byLower := append([]*BoardRow(nil), b.Rated...) // rows level on it stay in the board's order
	sort.SliceStable(byLower, func(i, j int) bool { return *byLower[i].Lower > *byLower[j].Lower })
	for _, r := range byLower {
		if r.Ladder || r.Status == "withdrawn" || failing(s, r.ID) {
			continue
		}
		if !r.Organisers {
			who := person(s.Editors[r.ID])
			if taken >= a.cfg.LadderSize || seen[who] {
				continue
			}
			seen[who] = true
			taken++
		}
		r.Ladder = true
		b.Ladder = append(b.Ladder, r)
	}
	return b
}

// boardCache holds the board for the state it was computed from.
type boardCache struct {
	mu      sync.Mutex
	board   *Board
	version uint64
}

// board is the current standings, computed once per change to the state.
func (a *App) board() *Board {
	a.boards.mu.Lock()
	defer a.boards.mu.Unlock()
	a.st.mu.RLock()
	defer a.st.mu.RUnlock()
	if a.boards.board == nil || a.boards.version != a.st.version {
		a.boards.board, a.boards.version = a.computeBoard(a.st.s), a.st.version
	}
	return a.boards.board
}

// --- records --------------------------------------------------------------------------------

func gainsByRound(entries []entryRow, editorID string) map[int]float64 {
	out := map[int]float64{}
	for _, e := range entries {
		if e.EditorID == editorID {
			out[e.RoundIdx] = e.Gain
		}
	}
	return out
}

// checkRecord looks for a new record holder after a round settles.
//
// Taking the record means beating the holder head to head, on the rounds both ran, by more
// than a margin, at a significance that survives the fact that challengers keep coming. A
// rule that looked after every round would sooner or later hand the record to a lucky copy
// of the holder, so:
//
//   - A challenger gets a fixed number of looks at a holder, at the shared-round counts the
//     season lists (5, 10, 20 and 40). After the last, that challenger has had its chance
//     at that holder. tools/record_power.py simulates what this can and cannot tell apart.
//   - At a look, the challenger's advantage over the holder (gain minus gain, round by
//     round) must exceed the season's margin, by a one-sided paired t-test at the season's
//     alpha divided by the number of looks. The margin is one standard deviation of the
//     difference between an editor and a trivially different copy of itself, measured.
//   - Every number in the test is a settled score: the median of established accounts'
//     grades, which nobody else's grade can move. A round either of them failed in counts
//     for it as the lowest settled score of that round, or zero (failedGain).
//
// The first record goes to the first editor that beats the unedited baseline on the same
// terms.
func (a *App) checkRecord(tx *Tx, roundIdx int) {
	s := tx.s
	rule := a.season.Record
	if len(rule.Looks) == 0 {
		return
	}
	b := a.computeBoard(s)
	margin := rule.Margin[a.cfg.Scale]
	alpha := rule.Alpha / float64(len(rule.Looks))
	holder, holderN := "", 0
	if h := s.holder(); h != nil {
		holder, holderN = h.EditorID, h.N
	}
	held := gainsByRound(b.Entries, holder)
	candidates := append(append([]*BoardRow(nil), b.Rated...), b.Provisional...) // best rating first within each
	for _, c := range candidates {
		if c.ID == holder {
			continue
		}
		var d []float64
		for _, e := range b.Entries {
			if e.EditorID != c.ID {
				continue
			}
			if holder == "" {
				d = append(d, e.Gain-margin)
			} else if h, ok := held[e.RoundIdx]; ok {
				d = append(d, e.Gain-h-margin)
			}
		}
		look := 0
		for _, l := range rule.Looks {
			if len(d) >= l && s.Looks[lookKey(c.ID, holder, l)] == nil {
				look = l
			}
		}
		if look == 0 {
			continue // nothing new to look at
		}
		p := pairedP(d) // H0: the mean advantage is no more than the margin
		advantage := mean(d) + margin
		for _, l := range rule.Looks { // a look at 10 rounds uses up the look at 5
			if l <= look && s.Looks[lookKey(c.ID, holder, l)] == nil {
				tx.saveLook(&Look{Challenger: c.ID, Holder: holder, Look: l, Shared: len(d), Advantage: advantage, P: p, At: now()})
			}
		}
		if p >= alpha {
			continue
		}
		rec := &Record{N: holderN + 1, EditorID: c.ID, RoundIdx: roundIdx, At: now(), Rating: *c.Rating, Multiplier: c.Multiplier,
			Capped: c.Capped, Rounds: len(d), Advantage: advantage, P: p, Beat: holder}
		tx.addRecord(rec)
		tx.event("record", map[string]any{"n": rec.N, "editor": c.ID, "name": c.Name, "owner": c.Owner, "rating": *c.Rating,
			"multiplier": c.Multiplier, "capped": c.Capped, "technique": c.Technique, "advantage": advantage})
		return
	}
}

// person is who an editor belongs to for "one per entrant": the GitHub identity that
// entered it, so that a second key linked to the same identity is still the same person.
// Editors from before identities were recorded fall back to the account.
func person(e *Editor) string {
	if e == nil {
		return ""
	}
	if e.GithubID != 0 {
		return fmt.Sprintf("gh:%d", e.GithubID)
	}
	return fmt.Sprintf("acct:%d", e.AccountID)
}
