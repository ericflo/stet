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
	for _, c := range v.Cases {
		items, ok := v.Tables[c.Table]
		if !ok || len(items) == 0 {
			t.Fatalf("%s: no table %q", c.Name, c.Table)
		}
		for _, it := range items {
			if it.Pillar == "" || it.Part == "" || it.Lang == "" || it.Stratum == "" {
				t.Fatalf("%s: an item without all it says: %+v", c.Name, it)
			}
		}
		score, pillars, err := ScoreItems(items, c.Gains, c.Shares)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
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
// the round's shares; a language without a share, or with none above zero, gets otherShare.
func TestLanguagesWeighByTheRoundsShares(t *testing.T) {
	items := []ItemInfo{
		{"writing", "technical", "en", "prose/passage"},
		{"writing", "technical", "code", "code/passage"},
		{"writing", "technical", "sw", "prose/passage"},
		{"writing", "technical", "ru", "prose/passage"},
	}
	gains := []float64{1, 10, 100, 1000}
	score, pillars, err := ScoreItems(items, gains, map[string]float64{"en": 0.85, "ru": -1})
	if err != nil {
		t.Fatal(err)
	}
	// en 0.85, sw and ru 0.05 each: halved, beside code's half
	want := 0.5*(0.85/0.95*1+0.05/0.95*100+0.05/0.95*1000) + 0.5*10
	if math.Abs(score-want) > 1e-12 || math.Abs(pillars["writing"]-want) > 1e-12 {
		t.Fatalf("score %v, pillars %v, want %v", score, pillars, want)
	}
}

// What cannot be scored is refused, not scored as nothing.
func TestTheScoreItemsRefusesWhatItCannotScore(t *testing.T) {
	one := []ItemInfo{{"reasoning", "arith", "en", "math/answer"}}
	shares := map[string]float64{"en": 43}
	for name, call := range map[string]func() error{
		"more gains than items": func() error { _, _, err := ScoreItems(one, []float64{0.1, 0.2}, shares); return err },
		"no gains":              func() error { _, _, err := ScoreItems(one, nil, shares); return err },
		"no items":              func() error { _, _, err := ScoreItems(nil, nil, shares); return err },
		"no shares":             func() error { _, _, err := ScoreItems(one, []float64{0.1}, nil); return err },
	} {
		if call() == nil {
			t.Errorf("%s: scored", name)
		}
	}
	if s, p, err := ScoreItems(one, []float64{0.25}, shares); err != nil || s != 0.25 || p["reasoning"] != 0.25 {
		t.Fatalf("one item: %v %v %v", s, p, err)
	}
}
