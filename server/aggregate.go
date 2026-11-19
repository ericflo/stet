package main

import (
	"errors"
	"math"
	"sort"
)

// A round's exam score, from what each item is and its gain: the server's copy of
// score_from_table in src/stet/exam/aggregate.py (what round_score gives), which it must
// match to the last few bits, since in season 1 it is the official score.
//
// The score is the mean of the pillars. Inside a pillar each part (a family of generators, or a
// fresh source's kind of writing) counts the same; inside a part each language counts by its
// share of the round's corpus, programming languages together as one, which takes half beside
// natural languages; inside a language each stratum (format/kind) counts the same, each a
// trimmed mean of gains clipped to a few robust widths.
//
// To give the harness's floating point, not just its arithmetic, everything is added in the
// order the harness adds it (sorted where it sorts, in order of first appearance where it keeps
// a dict's order), with the compensated sum Python's sum() does, and no product is fused into a
// sum. testdata/aggregate_vectors.json holds the harness's own answers
// (tools/aggregate_vectors.py writes it).

const (
	nextClip  = 4.0    // robust widths
	nextTrim  = 0.1    // from each end, within a stratum
	nextOther = 0.05   // the share of a language the round's shares leave out
	nextCode  = "code" // every programming language, as one language
)

// ItemInfo is all the score needs to know of an item: its row of the harness's item_table.
type ItemInfo struct {
	Pillar  string `json:"pillar"`
	Part    string `json:"part"`
	Lang    string `json:"lang"`
	Stratum string `json:"stratum"`
}

// NextScore is a round's score on its exam and each of its pillars' scores, from its items,
// a gain per item (in order), and the round's shares of languages (lang -> percent; a language
// left out, or with no share above zero, gets nextOther). The harness falls back to a published
// table of shares when it is given none; the server is always given the round's, and refuses
// to score without them.
func NextScore(items []ItemInfo, gains []float64, shares map[string]float64) (score float64, pillars map[string]float64, err error) {
	if len(items) != len(gains) {
		return 0, nil, errors.New("one gain per item")
	}
	if len(items) == 0 {
		return 0, nil, errors.New("no items to score")
	}
	if len(shares) == 0 {
		return 0, nil, errors.New("no shares of languages to weigh the items by")
	}
	all := make([]int, len(items))
	for i := range all {
		all[i] = i
	}
	names, byPillar := groupInOrder(all, func(i int) string { return items[i].Pillar })
	sort.Strings(names)
	pillars = make(map[string]float64, len(names))
	means := make([]float64, len(names))
	for j, p := range names {
		parts, byPart := groupInOrder(byPillar[p], func(i int) string { return items[i].Part })
		sort.Strings(parts)
		scores := make([]float64, len(parts))
		for k, part := range parts {
			scores[k] = nextByLanguage(items, gains, byPart[part], shares)
		}
		means[j] = pySum(scores) / float64(len(scores))
		pillars[p] = means[j]
	}
	return pySum(means) / float64(len(means)), pillars, nil
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

// nextByLanguage is a part's score: each language's balanced score by its weight, added in the
// order the languages first appear.
func nextByLanguage(items []ItemInfo, gains []float64, idx []int, shares map[string]float64) float64 {
	langs, byLang := groupInOrder(idx, func(i int) string { return items[i].Lang })
	w := nextLanguageWeights(langs, shares)
	terms := make([]float64, len(langs))
	for j, lang := range langs {
		terms[j] = float64(w[lang] * nextBalanced(items, gains, byLang[lang]))
	}
	return pySum(terms)
}

// nextLanguageWeights is each language's weight among some items: its share over the shares of
// the natural languages there, halved when programming languages are there too, which take the
// other half.
func nextLanguageWeights(langs []string, shares map[string]float64) map[string]float64 {
	var natural []string
	code := false
	for _, lang := range langs {
		if lang == nextCode {
			code = true
		} else {
			natural = append(natural, lang)
		}
	}
	sort.Strings(natural)
	w := make([]float64, len(natural))
	for j, lang := range natural {
		s, ok := shares[lang]
		if !ok {
			s = nextOther
		}
		if 0.0 > s { // max(share, 0.0)
			s = 0.0
		}
		if s == 0 { // ... or OTHER
			s = nextOther
		}
		w[j] = s
	}
	total := pySum(w)
	out := make(map[string]float64, len(langs))
	for j, lang := range natural {
		out[lang] = w[j] / total
	}
	if code {
		if len(out) == 0 {
			return map[string]float64{nextCode: 1.0}
		}
		for lang, v := range out {
			out[lang] = v / 2
		}
		out[nextCode] = 0.5
	}
	return out
}

// nextBalanced is a language's score: its strata, in sorted order, with equal weight, each a
// trimmed mean of its clipped gains.
func nextBalanced(items []ItemInfo, gains []float64, idx []int) float64 {
	strata, byStratum := groupInOrder(idx, func(i int) string { return items[i].Stratum })
	sort.Strings(strata)
	means := make([]float64, len(strata))
	for j, s := range strata {
		vals := make([]float64, len(byStratum[s]))
		for k, i := range byStratum[s] {
			vals[k] = gains[i]
		}
		means[j] = nextTrimmedMean(nextClipped(vals, nextClip), nextTrim)
	}
	return pySum(means) / float64(len(means))
}

// nextClipped clips values to k robust widths (1.4826 median absolute deviations) of their
// median; fewer than three values, or no spread, are left as they are.
func nextClipped(vals []float64, k float64) []float64 {
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

// nextTrimmedMean is the mean of the values with int(n*trim) cut from each end.
func nextTrimmedMean(vals []float64, trim float64) float64 {
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
