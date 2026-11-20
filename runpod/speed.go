package main

import (
	"math"
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
func byStudent(cs []Choice) {
	score := func(c Choice) float64 {
		if s := speed(c.GPU.ID); s > 0 {
			return c.Price / math.Pow(s, 1.25)
		}
		return math.Inf(1)
	}
	sort.SliceStable(cs, func(i, j int) bool { return score(cs[i]) < score(cs[j]) })
}

// byPace orders the kinds to try the fastest first, the cheapest first where they are as
// fast, and kinds whose speed is not known last, cheapest first. A round settles only once
// its slowest grade is in, and a machine's first grade of a round trains the round's
// baseline as well: on an RTX A4000 that is most of an hour, on an RTX 4090 a quarter.
func byPace(cs []Choice) {
	sort.SliceStable(cs, func(i, j int) bool {
		si, sj := speed(cs[i].GPU.ID), speed(cs[j].GPU.ID)
		if si != sj {
			return si > sj
		}
		return cs[i].Price < cs[j].Price
	})
}

// lead is how long before a round's exam a machine of this kind is rented to take one of its
// grades: its boot, the install and the round's corpus (--boot), and its first grade of the
// round, the round's baseline and the student: --first-grade on a 3090, the others by their
// speed, and one of unknown speed as half a 3090.
func lead(c Choice, boot, first time.Duration) time.Duration {
	s := speed(c.GPU.ID)
	if s <= 0 {
		s = 0.5
	}
	return boot + time.Duration(float64(first)/s)
}

// forExam is the kinds to rent now for grades taken before an exam due in left, in order.
// Every machine rented for one waits for the exam once it has trained, so what a grade costs
// on it is its price over the time it is held, and a kind is rented just in time: when its
// lead is about what is left (one look sooner, or up to five minutes later), the cheapest a
// round first. A kind whose time has not come would only wait, and is left until it has;
// one whose time has passed is left while a quicker kind can still be on time. Once none
// can, the quickest come first.
func forExam(cs []Choice, left, boot, first, every time.Duration) []Choice {
	const slack = 5 * time.Minute
	held := func(c Choice) float64 { return c.Price * lead(c, boot, first).Hours() }
	quickest := time.Duration(math.MaxInt64)
	for _, c := range cs {
		quickest = min(quickest, lead(c, boot, first))
	}
	var out []Choice
	if left+slack < quickest {
		out = append(out, cs...)
		sort.SliceStable(out, func(i, j int) bool { return lead(out[i], boot, first) < lead(out[j], boot, first) })
		return out
	}
	for _, c := range cs {
		if l := lead(c, boot, first); l >= left-every && l <= left+slack {
			out = append(out, c)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return held(out[i]) < held(out[j]) })
	return out
}

// byGrade orders the kinds to try (with --cheapest) by what a grade costs on each, its price over its speed:
// a grade trains on a fixed number of tokens, so a quicker card holds its machine for less
// of the hour, and is done sooner. A kind whose speed is not known counts as half a 3090.
func byGrade(cs []Choice) {
	score := func(c Choice) float64 {
		s := speed(c.GPU.ID)
		if s <= 0 {
			s = 0.5
		}
		return c.Price / s
	}
	sort.SliceStable(cs, func(i, j int) bool { return score(cs[i]) < score(cs[j]) })
}
