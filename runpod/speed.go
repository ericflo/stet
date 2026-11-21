package main

import (
	"math"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
)

// speeds say roughly how quickly each kind of GPU trains, against an RTX 3090: from
// published training benchmarks, and the RTX A4000's from the students it trained. A grade
// trains a model as a student does, so they order the kinds tried, where an hour of a quick
// card can cost less than the hours of a cheap one, and a rough figure does.
var speeds = map[string]float64{
	"RTX 3080": 0.85, "RTX 3080 Ti": 0.95, "RTX 3090": 1, "RTX 3090 Ti": 1.1,
	"RTX 4070 Ti": 0.95, "RTX 4080": 1.3, "RTX 4080 SUPER": 1.35, "RTX 4090": 1.8,
	"RTX 5080": 1.4, "RTX 5090": 2.5,
	"RTX A4000": 0.64, "RTX A4500": 0.75, "RTX A5000": 0.85, "RTX A6000": 1, "A40": 0.95,
	"RTX 2000 Ada": 0.35, "RTX 4000 Ada": 0.6, "RTX 5000 Ada": 1.1, "RTX 6000 Ada": 1.8,
	"L4": 0.45, "L40": 1.6, "L40S": 1.9,
	"RTX PRO 4000 Blackwell": 0.8, "RTX PRO 4500 Blackwell": 1, "RTX PRO 5000 Blackwell": 1.4, "RTX PRO 6000 Blackwell": 2.6,
	"A100": 2, "H100": 3.3, "H200": 3.6, "B200": 5, "B300": 5,
}

// speed is a kind of GPU's, by the longest of the names above that its RunPod ID holds as
// whole words (an RTX A4000 is not an A40, nor an L40S an L40); 0 when it is not known.
func speed(id string) float64 {
	best, s := 0, 0.0
	for name, v := range speeds {
		if len(name) > best && holdsWords(id, name) {
			best, s = len(name), v
		}
	}
	return s
}

func holdsWords(s, words string) bool {
	for i := 0; ; {
		j := strings.Index(s[i:], words)
		if j < 0 {
			return false
		}
		start, end := i+j, i+j+len(words)
		if (start == 0 || !alnum(s[start-1])) && (end == len(s) || !alnum(s[end])) {
			return true
		}
		i = start + 1
	}
}

func alnum(b byte) bool { return b < 128 && (unicode.IsLetter(rune(b)) || unicode.IsDigit(rune(b))) }

// byStudent orders the kinds to try for the bigger students by what a student costs on each,
// its price over its speed, leaning a little to the quicker: of two that cost about the same,
// the one done sooner comes first. Kinds whose speed is not known come last, cheapest first.
func byStudent(cs []Choice, speed func(string) float64) {
	score := func(c Choice) float64 {
		if s := speed(c.GPU.ID); s > 0 {
			return c.Price / math.Pow(s, 1.25)
		}
		return math.Inf(1)
	}
	sort.SliceStable(cs, func(i, j int) bool { return score(cs[i]) < score(cs[j]) })
}

// byPace orders the kinds to try the fastest first, the cheapest first where they are as
// fast, and kinds whose speed is not known last, cheapest first: for grades wanted once the
// exam is posted, which the round waits on.
func byPace(cs []Choice, speed func(string) float64) {
	sort.SliceStable(cs, func(i, j int) bool {
		si, sj := speed(cs[i].GPU.ID), speed(cs[j].GPU.ID)
		if si != sj {
			return si > sj
		}
		return cs[i].Price < cs[j].Price
	})
}

// forExam is the kinds to rent now for grades taken before an exam due in left, the
// cheapest a round first, where lead is how long before it must report a machine of a kind
// is rented. Every machine rented for one waits for the exam once it has trained, so what a
// grade costs on it is its price over its lead, and a kind is rented just in time: when
// renting it now has it report between one look before the exam and settle after it.
// Sooner, it would only wait; later, the round would. A kind is rented now only if none whose
// time is still to come would cost less: the cheapest to come is waited for, and once its
// time has passed, the next. Once no kind can report in time, the quickest come first.
func forExam(cs []Choice, left, settle, every time.Duration, lead func(Choice) time.Duration) []Choice {
	cost := func(c Choice) float64 { return c.Price * lead(c).Hours() }
	var due, later []Choice
	for _, c := range cs {
		switch l := lead(c); {
		case l < left-every:
			later = append(later, c)
		case l <= left+settle:
			due = append(due, c)
		}
	}
	if len(due) == 0 && len(later) == 0 {
		out := slices.Clone(cs)
		sort.SliceStable(out, func(i, j int) bool { return lead(out[i]) < lead(out[j]) })
		return out
	}
	best := math.Inf(1)
	for _, c := range later {
		best = min(best, cost(c))
	}
	var out []Choice
	for _, c := range due {
		if cost(c) <= best {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return cost(out[i]) < cost(out[j]) })
	return out
}

// waitingFor is the kind whose time is still to come that forExam waits for: the cheapest a
// round of those.
func waitingFor(cs []Choice, left, every time.Duration, lead func(Choice) time.Duration) (Choice, bool) {
	var best Choice
	found := false
	for _, c := range cs {
		if lead(c) < left-every && (!found || c.Price*lead(c).Hours() < best.Price*lead(best).Hours()) {
			best, found = c, true
		}
	}
	return best, found
}

// byGrade orders the kinds to try (with --cheapest) by what a grade costs on each, its price over its speed:
// a grade trains on a fixed number of tokens, so a quicker card holds its machine for less
// of the hour, and is done sooner. A kind whose speed is not known counts as half a 3090.
func byGrade(cs []Choice, speed func(string) float64) {
	score := func(c Choice) float64 {
		s := speed(c.GPU.ID)
		if s <= 0 {
			s = 0.5
		}
		return c.Price / s
	}
	sort.SliceStable(cs, func(i, j int) bool { return score(cs[i]) < score(cs[j]) })
}
