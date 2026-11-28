package main

import (
	"errors"
	"math"
	"sort"
)

// A round's exam score, from what each item is and its gain: the server's copy of
// score_from_table in src/stet/exam/aggregate.py (what round_score gives), which it must
// match to the last few bits, since it is the official score.
//
// The season declares what the score is made of: groups, each a fixed share of the score, and
// in each group pillars that count the same. A pillar the exam does not have gives its share
// to the others of its group. Inside a pillar each part (a family of generators, or a kind of
// writing) counts the same; inside a part each language counts by its share of the round's
// corpus, the small ones measured together as "other", programming languages together as one,
// taking code_share beside natural languages (default 0.5; 0.15 in Season 3); inside a language
// each stratum (format/kind) counts the same, each a trimmed mean of gains clipped to a few
// robust widths. A language or a stratum with fewer than the season's "small" items counts in
// proportion to how many it has. The season names every part: an item of any other part is not
// scored, it is refused. A group declared with 0.0 weight is scored per pillar for diagnostics
// without entering the headline score.
//
// To give the harness's floating point, not just its arithmetic, everything is added in the
// order the harness adds it, with the compensated sum Python's sum() does, and no product is
// fused into a sum. testdata/aggregate_vectors.json holds the harness's own answers
// (tools/aggregate_vectors.py writes it).

const (
	otherShare = 0.05    // the share of a language the round's shares leave out
	codeLang   = "code"  // every programming language, as one language
	otherLang  = "other" // a part's small languages, measured together
)

// FreshPlan is how the season takes an exam's writing from the hours the examiner harvests.
type FreshPlan struct {
	Settle int64 `json:"settle_seconds"` // an hour is harvested this long after it ends
	Reach  int64 `json:"reach_seconds"`  // a round is given hours that began no longer than this before its draw
	Hours  int   `json:"hours"`          // and this many of them
}

// ExamPlan is what a season's score is made of: exam.groups, exam.pillars and exam.aggregate of
// its file, as tools/export_season.py writes them. The pillars name every part of the exam.
type ExamPlan struct {
	Groups             map[string]float64 `json:"groups"`    // each group's share of the score
	Pillars            []PlanPillar       `json:"pillars"`   // in the order they are shown
	Generated          []string           `json:"generated"` // the parts that are families of generated problems; the others are kinds of writing
	Clip               float64            `json:"clip"`      // robust widths
	Trim               float64            `json:"trim"`      // from each end, within a stratum
	Small              int                `json:"small"`     // items a language or a stratum needs to count in full
	WeightOtherByShare bool               `json:"weight_other_by_share,omitempty"`
	CodeShare          *float64           `json:"code_share,omitempty"` // share given to code beside natural languages (default 0.5)
	Floor              int                `json:"answer_floor_bytes"`
	MinPillarItems     int                `json:"min_pillar_items,omitempty"`
	Fresh              FreshPlan          `json:"fresh"`
}

func (p *ExamPlan) codeShare() float64 {
	if p.CodeShare != nil {
		return *p.CodeShare
	}
	return 0.5
}

type PlanPillar struct {
	Name  string   `json:"name"`
	Group string   `json:"group"`
	Parts []string `json:"parts"`
	About string   `json:"about"`
}

// pillarOf is the pillar a part belongs to, or "" for a part the season does not name: one
// that is not on the exam.
func (p *ExamPlan) pillarOf(part string) string {
	for _, pl := range p.Pillars {
		for _, x := range pl.Parts {
			if x == part {
				return pl.Name
			}
		}
	}
	return ""
}

// generated reports whether a part is a family of generated problems, not a kind of writing.
func (p *ExamPlan) generated(part string) bool {
	for _, x := range p.Generated {
		if x == part {
			return true
		}
	}
	return false
}

// weights is each pillar's share of the score, among the pillars an exam has: its group's
// share over the shares of the groups that are there, split evenly among the group's pillars
// that are there.
func (p *ExamPlan) weights(present map[string]bool) map[string]float64 {
	var groups []string // in the order the plan's pillars first name them
	byGroup := map[string][]string{}
	for _, pl := range p.Pillars {
		if !present[pl.Name] {
			continue
		}
		if _, ok := byGroup[pl.Group]; !ok {
			groups = append(groups, pl.Group)
		}
		byGroup[pl.Group] = append(byGroup[pl.Group], pl.Name)
	}
	shares := make([]float64, len(groups))
	for i, g := range groups {
		shares[i] = p.Groups[g]
	}
	total := pySum(shares)
	out := map[string]float64{}
	if total <= 0 {
		for _, g := range groups {
			for _, name := range byGroup[g] {
				out[name] = 0.0
			}
		}
		return out
	}
	for _, g := range groups {
		for _, name := range byGroup[g] {
			out[name] = p.Groups[g] / total / float64(len(byGroup[g]))
		}
	}
	return out
}

// ItemInfo is all the score needs to know of an item: its row of the harness's item_table.
type ItemInfo struct {
	Pillar  string `json:"pillar"`
	Part    string `json:"part"`
	Lang    string `json:"lang"`
	Stratum string `json:"stratum"`
}

// ScoreItems is a round's score on its exam and each of its pillars' scores, from its items,
// a gain per item (in order), and the round's shares of languages (lang -> percent; a language
// left out, or with no share above zero, gets otherShare). An item whose part the season does
// not name, or names under another pillar, is refused. The harness falls back to a published
// table of shares when it is given none; the server is always given the round's, and refuses
// to score without them.
func (p *ExamPlan) ScoreItems(items []ItemInfo, gains []float64, shares map[string]float64) (score float64, pillars map[string]float64, err error) {
	if len(items) != len(gains) {
		return 0, nil, errors.New("one gain per item")
	}
	if len(items) == 0 {
		return 0, nil, errors.New("no items to score")
	}
	if len(shares) == 0 {
		return 0, nil, errors.New("no shares of languages to weigh the items by")
	}
	under := map[string]string{} // each part's pillar
	for _, pl := range p.Pillars {
		for _, part := range pl.Parts {
			under[part] = pl.Name
		}
	}
	for _, it := range items {
		if pillar, ok := under[it.Part]; !ok || pillar != it.Pillar {
			return 0, nil, errors.New("an item of a part the season's exam does not have")
		}
	}
	pillars = p.pillarScores(items, gains, shares)
	present := make(map[string]bool, len(pillars))
	for name := range pillars {
		present[name] = true
	}
	w := p.weights(present)
	names := make([]string, 0, len(w))
	for name := range w {
		names = append(names, name)
	}
	sort.Strings(names)
	terms := make([]float64, len(names))
	for i, name := range names {
		terms[i] = float64(w[name] * pillars[name])
	}
	return pySum(terms), pillars, nil
}

// pillarScores is each pillar's score: the mean of its parts.
func (p *ExamPlan) pillarScores(items []ItemInfo, gains []float64, shares map[string]float64) map[string]float64 {
	idx := make([]int, len(items))
	for i := range idx {
		idx[i] = i
	}
	names, byPillar := groupInOrder(idx, func(i int) string { return items[i].Pillar })
	out := make(map[string]float64, len(names))
	for _, name := range names {
		parts, byPart := groupInOrder(byPillar[name], func(i int) string { return items[i].Part })
		sort.Strings(parts)
		scores := make([]float64, len(parts))
		for k, part := range parts {
			scores[k] = p.part(items, gains, byPart[part], shares)
		}
		out[name] = pySum(scores) / float64(len(scores))
	}
	return out
}

// groupInOrder groups indices of items by a key, with the keys in the order they first appear,
// as a Python dict keeps them.
func groupInOrder(idx []int, key func(int) string) ([]string, map[string][]int) {
	var keys []string
	groups := map[string][]int{}
	for _, i := range idx {
		k := key(i)
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], i)
	}
	return keys, groups
}

// part is a part's score: each language cell's balanced score by its weight, added in the
// cells' sorted order. Inside the pooled small-language cell "other", each item is weighted by
// its language's crawl share.
func (p *ExamPlan) part(items []ItemInfo, gains []float64, idx []int, shares map[string]float64) float64 {
	_, byLang := groupInOrder(idx, func(i int) string { return items[i].Lang })
	counts := make(map[string]int, len(byLang))
	for lang, ix := range byLang {
		counts[lang] = len(ix)
	}
	cell, w := languageCells(counts, shares, p.Small, p.codeShare())
	names, byCell := groupInOrder(idx, func(i int) string { return cell[items[i].Lang] })
	sort.Strings(names)
	terms := make([]float64, len(names))
	for j, c := range names {
		var iw map[string]float64
		if c == otherLang && p.WeightOtherByShare {
			iw = shares
		}
		terms[j] = float64(w[c] * p.balanced(items, gains, byCell[c], iw))
	}
	return pySum(terms)
}

// languageCells is where each language of a part is measured, and each cell's weight. A
// natural language with at least `small` items is a cell of its own; the others share the cell
// "other". A cell's weight is its share of the round's corpus, a pooled cell's the sum of its
// languages', and a cell with fewer than `small` items counts in proportion. Programming
// languages together, beside natural languages, take codeShare (in proportion, when they are few).
func languageCells(counts map[string]int, shares map[string]float64, small int, codeShare float64) (cell map[string]string, weight map[string]float64) {
	var natural []string
	for lang := range counts {
		if lang != codeLang {
			natural = append(natural, lang)
		}
	}
	sort.Strings(natural)
	cell = make(map[string]string, len(counts))
	w, n := map[string]float64{}, map[string]int{}
	for _, lang := range natural {
		c := otherLang
		if counts[lang] >= small {
			c = lang
		}
		cell[lang] = c
		s, ok := shares[lang]
		if !ok {
			s = otherShare
		}
		if 0.0 > s { // max(share, 0.0)
			s = 0.0
		}
		if s == 0 { // ... or OTHER
			s = otherShare
		}
		w[c] += s
		n[c] += counts[lang]
	}
	cells := make([]string, 0, len(w))
	for c := range w {
		cells = append(cells, c)
	}
	sort.Strings(cells)
	full := func(k int) float64 {
		if k > small {
			k = small
		}
		return float64(k) / float64(small)
	}
	terms := make([]float64, len(cells))
	for j, c := range cells {
		terms[j] = float64(w[c] * full(n[c]))
	}
	total := pySum(terms)
	weight = make(map[string]float64, len(cells)+1)
	for j, c := range cells {
		weight[c] = terms[j] / total
	}
	if k, ok := counts[codeLang]; ok {
		cell[codeLang] = codeLang
		if len(weight) == 0 {
			return cell, map[string]float64{codeLang: 1.0}
		}
		half := codeShare * float64(min(k, small)) / float64(small)
		for c, v := range weight {
			weight[c] = float64(v * (1 - half))
		}
		weight[codeLang] = half
	}
	return cell, weight
}

func langShare(lang string, shares map[string]float64) float64 {
	s, ok := shares[lang]
	if !ok || s <= 0 {
		return otherShare
	}
	return s
}

// balanced is a language cell's score: its strata, in sorted order, each a trimmed mean of
// its clipped gains (or crawl-share weighted trimmed mean of clipped gains inside the pooled
// "other" cell), with equal weight; one with fewer than `small` items in proportion.
func (p *ExamPlan) balanced(items []ItemInfo, gains []float64, idx []int, poolShares map[string]float64) float64 {
	strata, byStratum := groupInOrder(idx, func(i int) string { return items[i].Stratum })
	sort.Strings(strata)
	terms := make([]float64, len(strata))
	total := 0
	for j, s := range strata {
		sidx := byStratum[s]
		vals := make([]float64, len(sidx))
		for k, i := range sidx {
			vals[k] = gains[i]
		}
		cl := clipped(vals, p.Clip)
		var mean float64
		if poolShares == nil {
			mean = trimmedMean(cl, p.Trim)
		} else {
			ws := make([]float64, len(sidx))
			for k, i := range sidx {
				ws[k] = langShare(items[i].Lang, poolShares)
			}
			mean = weightedTrimmedMean(cl, ws, p.Trim)
		}
		w := min(len(vals), p.Small)
		total += w
		terms[j] = float64(float64(w) * mean)
	}
	return pySum(terms) / float64(total)
}

// clipped clips values to k robust widths (1.4826 median absolute deviations) of their
// median; fewer than three values, or no spread, are left as they are.
func clipped(vals []float64, k float64) []float64 {
	out := append([]float64(nil), vals...)
	if len(vals) < 3 {
		return out
	}
	med := pyMedian(vals)
	dev := make([]float64, len(vals))
	for i, v := range vals {
		dev[i] = math.Abs(v - med)
	}
	width := 1.4826 * pyMedian(dev)
	if width <= 0 {
		return out
	}
	lo, hi := med-float64(k*width), med+float64(k*width)
	for i, v := range vals {
		x := lo // max(lo, v)
		if v > lo {
			x = v
		}
		out[i] = hi // min(hi, x)
		if x < hi {
			out[i] = x
		}
	}
	return out
}

// weightedTrimmedMean cuts `trim` of the total weight from each tail of the values sorted
// ascending, matching weighted_trimmed_mean in src/stet/exam/aggregate.py.
func weightedTrimmedMean(vals, weights []float64, trim float64) float64 {
	n := len(vals)
	totalW := pySum(weights)
	plain := func() float64 {
		num := make([]float64, n)
		for i := range vals {
			num[i] = float64(weights[i] * vals[i])
		}
		return pySum(num) / totalW
	}
	if n < 3 || trim <= 0 || trim >= 0.5 || totalW <= 0 {
		return plain()
	}
	lo, hi := float64(totalW*trim), float64(totalW*(1.0-trim))
	if hi <= lo {
		return plain()
	}
	type pair struct{ v, w float64 }
	pairs := make([]pair, n)
	for i := range vals {
		pairs[i] = pair{vals[i], weights[i]}
	}
	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].v < pairs[j].v })
	cum := 0.0
	var num, den []float64
	for _, p := range pairs {
		nxt := cum + p.w
		top := nxt
		if hi < top {
			top = hi
		}
		bot := cum
		if lo > bot {
			bot = lo
		}
		if kept := top - bot; kept > 0.0 {
			num = append(num, float64(kept*p.v))
			den = append(den, kept)
		}
		cum = nxt
	}
	denSum := pySum(den)
	if denSum <= 0.0 {
		return plain()
	}
	return pySum(num) / denSum
}

// trimmedMean is the mean of the values with int(n*trim) cut from each end.
func trimmedMean(vals []float64, trim float64) float64 {
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	cut := int(float64(len(s)) * trim)
	kept := s
	if len(s)-2*cut >= 1 {
		kept = s[cut : len(s)-cut]
	}
	return pySum(kept) / float64(len(kept))
}

// pyMedian is Python's statistics.median: the middle value, or the mean of the middle two.
func pyMedian(vals []float64) float64 {
	s := append([]float64(nil), vals...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// pySum adds floats as Python's sum() does since 3.12: Neumaier's compensated summation, from
// the first value.
func pySum(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s, c := 0.0+vals[0], 0.0
	for _, x := range vals[1:] {
		t := s + x
		if math.Abs(s) >= math.Abs(x) {
			c += (s - t) + x
		} else {
			c += (x - t) + s
		}
		s = t
	}
	if c != 0 && !math.IsInf(c, 0) && !math.IsNaN(c) {
		s += c
	}
	return s
}
