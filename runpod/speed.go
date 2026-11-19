package main

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// speeds say roughly how quickly each kind of GPU trains one of the organisers' students,
// against an RTX 3090: from published training benchmarks, and the RTX A4000's from the
// students it trained. They only order the kinds tried for the bigger students, where an
// hour of a quick card can cost less than the hours of a cheap one, so a rough figure does.
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
