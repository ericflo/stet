package main

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// The harness's answers: tools/aggregate_vectors.py writes them, and a harness test fails when
// they are not what it writes now.
type aggregateVectors struct {
	Tables map[string][]ItemInfo `json:"tables"`
	Cases  []struct {
		Name    string             `json:"name"`
		Table   string             `json:"table"`
		Shares  map[string]float64 `json:"shares"`
		Gains   []float64          `json:"gains"`
		Score   float64            `json:"score"`
		Pillars map[string]float64 `json:"pillars"`
		Watch   map[string]float64 `json:"watch"`
	} `json:"cases"`
}

// The server scores a round's exam as the harness does, to within 1e-9: every case
// the harness wrote, from single items to a whole round of generated and fresh items.
func TestTheScoreItemsIsTheHarnesss(t *testing.T) {
	raw, err := os.ReadFile("testdata/aggregate_vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var v aggregateVectors
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	if len(v.Cases) < 30 {
		t.Fatalf("only %d cases", len(v.Cases))
	}
	const tol = 1e-9
	worst := 0.0
	plan := &loadSeason().Exam
	for _, c := range v.Cases {
		items, ok := v.Tables[c.Table]
		if !ok || len(items) == 0 {
			t.Fatalf("%s: no table %q", c.Name, c.Table)
		}
		for _, it := range items {
			if it.Pillar == "" || it.Part == "" || it.Lang == "" || it.Stratum == "" {
				t.Fatalf("%s: an item without all it says: %+v", c.Name, it)
			}
			if it.Pillar != plan.pillarOf(it.Part) {
				t.Fatalf("%s: the harness puts %s under %s, the server under %s", c.Name, it.Part, it.Pillar, plan.pillarOf(it.Part))
			}
		}
		score, pillars, watch, err := plan.ScoreItems(items, c.Gains, c.Shares)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		if len(watch) != len(c.Watch) {
			t.Errorf("%s: watched %v, the harness %v", c.Name, watch, c.Watch)
		}
		for part, want := range c.Watch {
			got, ok := watch[part]
			d := math.Abs(got - want)
			worst = math.Max(worst, d)
			if !ok || !(d <= tol) {
				t.Errorf("%s: watched %s %.17g, the harness %.17g", c.Name, part, got, want)
			}
		}
		d := math.Abs(score - c.Score)
		worst = math.Max(worst, d)
		if !(d <= tol) {
			t.Errorf("%s: score %.17g, the harness %.17g", c.Name, score, c.Score)
		}
		if len(pillars) != len(c.Pillars) {
			t.Errorf("%s: pillars %v, the harness %v", c.Name, pillars, c.Pillars)
		}
		for p, want := range c.Pillars {
			got, ok := pillars[p]
			d := math.Abs(got - want)
			worst = math.Max(worst, d)
			if !ok || !(d <= tol) {
				t.Errorf("%s: pillar %s %.17g, the harness %.17g", c.Name, p, got, want)
			}
		}
	}
	t.Logf("%d cases, the largest difference from the harness %.3g", len(v.Cases), worst)
}

// Programming languages take half of a part beside natural languages, which share the rest by
// the round's shares; a language without a share, or with none above zero, gets otherShare; and
// languages with too few items are measured together.
func TestLanguagesWeighByTheRoundsShares(t *testing.T) {
	plan := &loadSeason().Exam
	var items []ItemInfo
	var gains []float64
	add := func(lang, stratum string, n int, g float64) {
		for i := 0; i < n; i++ {
			items = append(items, ItemInfo{"technical", "technical", lang, stratum})
			gains = append(gains, g)
		}
	}
	add("en", "prose/passage", plan.Small, 1)
	add("code", "code/passage", plan.Small, 10)
	add("sw", "prose/passage", 1, 100) // sw and ru are small: one cell, "other", of two items
	add("ru", "prose/passage", 1, 1000)
	score, pillars, watch, err := plan.ScoreItems(items, gains, map[string]float64{"en": 0.85, "ru": -1})
	if err != nil {
		t.Fatal(err)
	}
	// en 0.85 in full; sw and ru 0.05 each, together, a quarter counted (2 of 8 items): halved, beside code's half
	other := 0.1 * 2 / float64(plan.Small)
	want := 0.5*(0.85*1+other*550)/(0.85+other) + 0.5*10
	if math.Abs(score-want) > 1e-12 || math.Abs(pillars["technical"]-want) > 1e-12 || len(watch) != 0 {
		t.Fatalf("score %v, pillars %v, want %v", score, pillars, want)
	}
}

// A group's share is the season's, whatever the exam has of it; what is watched is scored and
// does not count.
func TestGroupsKeepTheirSharesAndWatchedPartsDoNotCount(t *testing.T) {
	plan := &loadSeason().Exam
	var items []ItemInfo
	var gains []float64
	add := func(pillar, part string, g float64) {
		for i := 0; i < plan.Small; i++ {
			items = append(items, ItemInfo{pillar, part, "en", "prose/answer"})
			gains = append(gains, g)
		}
	}
	add("reference", "reference", 0.4)
	add("news", "news", 0.2)
	add("agents", "agent", -1)
	add("watch", "arith", 50)
	score, pillars, watch, err := plan.ScoreItems(items, gains, map[string]float64{"en": 43})
	if err != nil {
		t.Fatal(err)
	}
	want := plan.Groups["writing"]*(0.4+0.2)/2 + plan.Groups["skills"]*-1
	if math.Abs(score-want) > 1e-12 || len(pillars) != 3 || watch["arith"] != 50 {
		t.Fatalf("score %v (want %v), pillars %v, watched %v", score, want, pillars, watch)
	}
	if plan.pillarOf("arith") != watchPillar || plan.pillarOf("agent") != "agents" || plan.pillarOf("social") != "conversation" {
		t.Fatal("the season's parts are not under its pillars")
	}
}

// What cannot be scored is refused, not scored as nothing.
func TestTheScoreItemsRefusesWhatItCannotScore(t *testing.T) {
	plan := &loadSeason().Exam
	one := []ItemInfo{{"logic", "logic", "en", "prose/answer"}}
	shares := map[string]float64{"en": 43}
	for name, call := range map[string]func() error{
		"more gains than items": func() error { _, _, _, err := plan.ScoreItems(one, []float64{0.1, 0.2}, shares); return err },
		"no gains":              func() error { _, _, _, err := plan.ScoreItems(one, nil, shares); return err },
		"no items":              func() error { _, _, _, err := plan.ScoreItems(nil, nil, shares); return err },
		"no shares":             func() error { _, _, _, err := plan.ScoreItems(one, []float64{0.1}, nil); return err },
		"a pillar the season does not have": func() error {
			_, _, _, err := plan.ScoreItems([]ItemInfo{{"reasoning", "arith", "en", "math/answer"}}, []float64{0.1}, shares)
			return err
		},
	} {
		if call() == nil {
			t.Errorf("%s: scored", name)
		}
	}
	// One item, alone in its stratum and its language: an eighth of a full cell, and all there is.
	if s, p, _, err := plan.ScoreItems(one, []float64{0.25}, shares); err != nil || s != 0.25 || p["logic"] != 0.25 {
		t.Fatalf("one item: %v %v %v", s, p, err)
	}
}
