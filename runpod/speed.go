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
// published training benchmarks, and the RTX A4000's from the students it trained. The
// consumer cards the other markets offer most (3060, 4060 Ti, 4070s, 5060 Ti, 5070s) are
// from the same benchmarks; what grades take on each is learned as they are seen. A grade
// trains a model as a student does, so they order the kinds tried, where an hour of a quick
// card can cost less than the hours of a cheap one, and a rough figure does.
var speeds = map[string]float64{
	"RTX 3080": 0.85, "RTX 3080 Ti": 0.95, "RTX 3090": 1, "RTX 3090 Ti": 1.1,
	"RTX 4070 Ti": 0.95, "RTX 4080": 1.3, "RTX 4080 SUPER": 1.35, "RTX 4090": 1.8,
	"RTX 5080": 1.4, "RTX 5090": 2.5,
	"RTX 3060": 0.4, "RTX 4060 Ti": 0.55, "RTX 4070": 0.8, "RTX 4070 SUPER": 0.9, "RTX 4070 Ti SUPER": 1.05,
	"RTX 5060 Ti": 0.6, "RTX 5070": 0.95, "RTX 5070 Ti": 1.15, "A10": 0.8, "A30": 1.1,
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
func byStudent(cs []Offer, speed func(string) float64) {
	score := func(c Offer) float64 {
		if s := speed(c.Card); s > 0 {
			return c.rankPrice() / math.Pow(s, 1.25)
		}
		return math.Inf(1)
	}
	sort.SliceStable(cs, func(i, j int) bool { return score(cs[i]) < score(cs[j]) })
}

// byPace orders the kinds to try the fastest first, the cheapest first where they are as
// fast, and kinds whose speed is not known last, cheapest first: for grades wanted once the
// exam is posted, which the round waits on.
func byPace(cs []Offer, speed func(string) float64) {
	sort.SliceStable(cs, func(i, j int) bool {
		si, sj := speed(cs[i].Card), speed(cs[j].Card)
		if si != sj {
			return si > sj
		}
		return cs[i].rankPrice() < cs[j].rankPrice()
	})
}

// examPlan is what is known of a kind when it is rented for grades taken before an exam: how
// long before it is to report its first grade it must be rented (lead), how long it takes to
// train one more student (pass), and the most grades one machine may hold before the exam
// (most; 0 or 1 when the server lets a machine hold only one).
type examPlan struct {
	lead func(Offer) time.Duration
	pass func(Offer) time.Duration
	most int
}

// extraMargin is how long before the exam a machine must be able to have trained one more
// student for the server to hand it one.
const extraMargin = 2 * time.Minute

// grades is how many grades a machine of a kind rented with left to go until the exam takes
// before it: its first, and one more for each further student it can have trained by two
// minutes before the exam, up to the most one may hold.
func (pl examPlan) grades(c Offer, left time.Duration) int {
	if pl.most <= 1 || pl.pass == nil {
		return 1
	}
	p := pl.pass(c)
	spare := left - (pl.lead(c) - sitting) - extraMargin
	if p <= 0 || spare < 0 {
		return 1
	}
	return min(pl.most, 1+int(spare/p))
}

// full is how long before the exam a kind is best rented: just in time to take the most
// grades it may. Sooner, it would only wait; later, it would take fewer.
func (pl examPlan) full(c Offer) time.Duration {
	if pl.most <= 1 || pl.pass == nil {
		return pl.lead(c)
	}
	return pl.lead(c) - sitting + extraMargin + time.Duration(pl.most-1)*pl.pass(c)
}

// cost is what a grade costs on a kind rented with left to go: its price for the time it is
// held, until it reports its last grade, shared among the grades it takes.
func (pl examPlan) cost(c Offer, left time.Duration) float64 {
	if pl.most <= 1 || pl.pass == nil {
		return c.rankPrice() * pl.lead(c).Hours()
	}
	held := max(left+sitting, pl.lead(c))
	return c.rankPrice() * held.Hours() / float64(pl.grades(c, left))
}

// forExam is the kinds to rent now for grades taken before an exam due in left, the cheapest a
// grade first. Every machine rented for one waits for the exam once it has trained, so what a
// grade costs on it is its price for the time it is held over the grades it takes, and a kind
// is rented just in time: when renting it now has it take the most grades it may (only one,
// unless the server lets a machine hold several) and report between one look before the exam
// and settle after it. Sooner, it would only wait; later, it would take fewer, or the round
// would wait. A kind is rented now only if none whose time is still to come would cost less: the
// cheapest to come is waited for, and once its time has passed, the next. Once no kind can
// report in time, the quickest come first.
func forExam(cs []Offer, left, settle, every time.Duration, pl examPlan) []Offer {
	var due, later []Offer
	for _, c := range cs {
		switch {
		case pl.full(c) < left-every:
			later = append(later, c)
		case pl.lead(c) <= left+settle:
			due = append(due, c)
		}
	}
	if len(due) == 0 && len(later) == 0 {
		out := slices.Clone(cs)
		sort.SliceStable(out, func(i, j int) bool { return pl.lead(out[i]) < pl.lead(out[j]) })
		return out
	}
	best := math.Inf(1)
	for _, c := range later {
		best = min(best, pl.cost(c, pl.full(c)))
	}
	var out []Offer
	for _, c := range due {
		if pl.cost(c, left) <= best {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return pl.cost(out[i], left) < pl.cost(out[j], left) })
	return out
}

// waitingFor is the kind whose time is still to come that forExam waits for: the cheapest a
// grade of those, at its time.
func waitingFor(cs []Offer, left, every time.Duration, pl examPlan) (Offer, bool) {
	var best Offer
	found := false
	for _, c := range cs {
		if pl.full(c) < left-every && (!found || pl.cost(c, pl.full(c)) < pl.cost(best, pl.full(best))) {
			best, found = c, true
		}
	}
	return best, found
}

// byGrade orders the kinds to try (with --cheapest) by what a grade costs on each, its price over its speed:
// a grade trains on a fixed number of tokens, so a quicker card holds its machine for less
// of the hour, and is done sooner. A kind whose speed is not known counts as half a 3090.
func byGrade(cs []Offer, speed func(string) float64) {
	score := func(c Offer) float64 {
		s := speed(c.Card)
		if s <= 0 {
			s = 0.5
		}
		return c.rankPrice() / s
	}
	sort.SliceStable(cs, func(i, j int) bool { return score(cs[i]) < score(cs[j]) })
}
