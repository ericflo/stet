package main

// Ratings, the board and the record.
//
// An editor's rating is its mean gain over the rounds it ran in, adjusted for which rounds
// those happened to be. Some rounds' corpora are easier to improve than others, and editors
// do not all run in the same rounds, so a plain mean would reward luck of the draw. The
// model is the simplest one that removes it:
//
//     gain(editor, round) = skill(editor) + difficulty(round) + noise
//
// fitted by least squares. The ladder (the top editors, who run in every round) ties the
// rounds together, so difficulties are well determined.

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

type entryRow struct {
	RoundIdx int
	EditorID string
	Gain     float64
	Valid    bool
}

type Rating struct {
	N        int     // settled rounds
	Mean     float64 // plain mean gain
	Adjusted float64 // mean gain adjusted for which rounds it happened to run
	SE       float64
	Invalid  int
}

// fitRatings solves the additive model by alternating means. It converges in a handful of
// passes because the ladder connects the rounds.
func fitRatings(entries []entryRow) (map[string]*Rating, map[int]float64) {
	byEditor := map[string][]entryRow{}
	byRound := map[int][]entryRow{}
	for _, e := range entries {
		byEditor[e.EditorID] = append(byEditor[e.EditorID], e)
		byRound[e.RoundIdx] = append(byRound[e.RoundIdx], e)
	}
	// skill is what an editor does in the rounds where it ran: its mean over valid entries
	// only. It is what explains a round's difficulty. The rating published below counts an
	// invalid round as zero, but that zero must not be used here: an editor with one
	// timeout would look much better than its "skill" in its good rounds, those rounds
	// would be taken for easy ones, and everyone who shared them would be marked down.
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
			// Only entries that were really measured say anything about the round. An
			// invalid entry's zero is a penalty, not a reading: counted here, one editor's
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
	out := map[string]*Rating{}
	for id, es := range byEditor {
		r := &Rating{N: len(es)}
		var resid []float64
		for _, e := range es {
			r.Mean += e.Gain / float64(len(es))
			resid = append(resid, e.adjusted(diff))
			if !e.Valid {
				r.Invalid++
			}
		}
		r.Adjusted = mean(resid)
		if len(resid) >= 2 {
			r.SE = stddev(resid) / math.Sqrt(float64(len(resid)))
		}
		out[id] = r
	}
	return out, diff
}

// adjusted is an entry's gain with its round's difficulty taken out. An invalid entry
// counts as zero whatever the round was like.
func (e entryRow) adjusted(diff map[int]float64) float64 {
	if !e.Valid {
		return 0
	}
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
	Rating      *float64  `json:"rating"`
	MeanGain    *float64  `json:"mean_gain"`
	SE          *float64  `json:"se"`
	Multiplier  *float64  `json:"multiplier"`
	Capped      bool      `json:"capped,omitempty"` // the multiplier is "more than" this: the gain runs off the measured curve
	Invalid     int       `json:"invalid,omitempty"`
	Provisional bool      `json:"provisional,omitempty"`
	Record      bool      `json:"record,omitempty"`
	Ladder      bool      `json:"ladder,omitempty"`
	CreatedAt   int64     `json:"created_at"`
	AccountID   int64     `json:"-"`
	Gains       []float64 `json:"-"` // per settled round, oldest first, for sparklines
}

type Board struct {
	Rated       []*BoardRow          // ranked
	Provisional []*BoardRow          // fewer than RatedRounds settled rounds
	Ladder      []*BoardRow          // top editors, one per account: run in every round
	ByID        map[string]*BoardRow // every editor that has settled at least one round
	Entries     []entryRow
	Difficulty  map[int]float64
	RecordID    string
}

func (a *App) curveKey() string { return "official/" + a.cfg.Scale }

// settledEntries are the entries that count: settled, in a round whose baseline is known.
func settledEntries(s *State) []entryRow {
	var out []entryRow
	for _, e := range s.Entries {
		// Whole rounds only. An entry with no script settles the moment that is known, while
		// the rest of its round is still sealed; alone, it would stand for the whole round.
		if rd := s.Rounds[e.RoundIdx]; e.Status != "settled" || e.Void || rd.BaselineLoss == nil || rd.Status != "settled" {
			continue
		}
		out = append(out, entryRow{RoundIdx: e.RoundIdx, EditorID: e.EditorID, Gain: e.Gain, Valid: e.Valid})
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].RoundIdx < out[j].RoundIdx || (out[i].RoundIdx == out[j].RoundIdx && out[i].EditorID < out[j].EditorID)
	})
	return out
}

func (a *App) computeBoard(s *State) *Board {
	entries := settledEntries(s)
	ratings, diff := fitRatings(entries)
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
			Description: e.Description, Parent: e.Parent, Status: e.Status, CreatedAt: e.CreatedAt, AccountID: e.AccountID}
		r.Rounds, r.Invalid = rt.N, rt.Invalid
		adj, mg := rt.Adjusted, rt.Mean
		r.Rating, r.MeanGain = &adj, &mg
		if rt.N >= 2 {
			se := rt.SE
			r.SE = &se
		}
		if m, capped, ok := a.season.multiplier(a.curveKey(), adj); ok {
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
	// The ladder: the record holder, then the best-rated editor of each person. The
	// organisers' reference editors are fixed points for everyone and all stay on.
	//
	// The holder keeps its place whatever it does. A challenger can only take the record on
	// rounds it shares with the holder, so a holder that stopped running would hold the
	// record for ever. One that has started failing scores zero and is soon beaten. Its
	// place is its own: its owner's best other editor may still take one, which is how a
	// person improves on their own record.
	seen := map[string]bool{}
	if h := b.ByID[b.RecordID]; h != nil {
		h.Ladder = true
		b.Ladder = append(b.Ladder, h)
	}
	for i, r := range b.Rated {
		r.Rank = i + 1
		owner := s.Accounts[r.AccountID]
		who := person(s.Editors[r.ID])
		if r.Ladder || r.Status == "withdrawn" || failing(s, r.ID) || len(b.Ladder) >= a.cfg.LadderSize || (seen[who] && (owner == nil || !owner.Anchor)) {
			continue
		}
		seen[who] = true
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
//   - Every number in the test is an anchor's grade: nobody else's can set a score.
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
