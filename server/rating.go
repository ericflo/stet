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
	"math"
	"sort"
	"sync"
)

type entryRow struct {
	RoundIdx int
	EditorID string
	Gain     float64
	Valid    bool
	Audited  bool
}

type Rating struct {
	N        int     // settled rounds
	Mean     float64 // plain mean gain
	Adjusted float64 // mean gain adjusted for which rounds it happened to run
	SE       float64
	Invalid  int
	Audited  int
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
	skill := map[string]float64{}
	diff := map[int]float64{}
	for i := 0; i < 200; i++ {
		delta := 0.0
		for id, es := range byEditor {
			s := 0.0
			for _, e := range es {
				s += e.Gain - diff[e.RoundIdx]
			}
			s /= float64(len(es))
			delta = math.Max(delta, math.Abs(s-skill[id]))
			skill[id] = s
		}
		mean := 0.0
		for r, es := range byRound {
			s := 0.0
			for _, e := range es {
				s += e.Gain - skill[e.EditorID]
			}
			diff[r] = s / float64(len(es))
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
			resid = append(resid, e.Gain-diff[e.RoundIdx])
			if !e.Valid {
				r.Invalid++
			}
			if e.Audited {
				r.Audited++
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
		if e.Status != "settled" || s.Rounds[e.RoundIdx].BaselineLoss == nil {
			continue
		}
		out = append(out, entryRow{RoundIdx: e.RoundIdx, EditorID: e.EditorID, Gain: e.Gain, Valid: e.Valid, Audited: e.Detail != nil && e.Detail.Audited})
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
	// The ladder: the record holder (a challenger can only take the record by sharing rounds
	// with it, so it must keep running), then the best-rated editor of each account. The
	// organisers' reference editors are fixed points for everyone and all stay on.
	seen := map[int64]bool{}
	if h := b.ByID[b.RecordID]; h != nil && h.Status != "withdrawn" {
		h.Ladder = true
		b.Ladder = append(b.Ladder, h)
	}
	for i, r := range b.Rated {
		r.Rank = i + 1
		owner := s.Accounts[r.AccountID]
		if r.Ladder || r.Status == "withdrawn" || len(b.Ladder) >= a.cfg.LadderSize || (seen[r.AccountID] && (owner == nil || !owner.Anchor)) {
			continue
		}
		seen[r.AccountID] = true
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
//   - A challenger gets a fixed number of looks at a holder: when the two have shared 5,
//     10 and 20 rounds. After the last, that challenger has had its chance at that holder.
//   - At a look, the challenger's advantage over the holder (gain minus gain, round by
//     round) must exceed the season's margin, by a one-sided paired t-test at the season's
//     alpha divided by the number of looks. The margin is one standard deviation of the
//     difference between an editor and a trivially different copy of itself, measured.
//   - At least two of the challenger's rounds must have been graded by an anchor.
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
		audited := 0
		for _, e := range b.Entries {
			if e.EditorID != c.ID {
				continue
			}
			if e.Audited {
				audited++
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
		if look == 0 || audited < rule.MinAudited {
			continue // nothing new to look at, or not yet audited enough for the look to count
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
