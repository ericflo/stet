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
		score, pillars, err := plan.ScoreItems(items, c.Gains, c.Shares)
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

// Programming languages take code_share of a part beside natural languages, which share the rest by
// the round's shares; a language without a share, or with none above zero, gets otherShare; and
// languages with too few items are measured together.
func TestLanguagesWeighByTheRoundsShares(t *testing.T) {
	plan := &loadSeason().Exam
	var items []ItemInfo
	var gains []float64
	add := func(pillar, part, lang, stratum string, n int, g float64) {
		for i := 0; i < n; i++ {
			items = append(items, ItemInfo{pillar, part, lang, stratum})
			gains = append(gains, g)
		}
	}
	add("technical", "technical", "en", "prose/passage", plan.Small, 1)
	add("technical", "technical", "code", "code/passage", plan.Small, 10)
	add("technical", "technical", "sw", "prose/passage", 1, 100) // sw and ru are small: one cell, "other", of two items
	add("technical", "technical", "ru", "prose/passage", 1, 1000)
	score, pillars, err := plan.ScoreItems(items, gains, map[string]float64{"en": 0.85, "ru": -1})
	if err != nil {
		t.Fatal(err)
	}
	// en 0.85 in full; sw and ru 0.05 each, together, a quarter counted (2 of 8 items); beside code's codeShare
	cs := plan.codeShare()
	other := 0.1 * 2 / float64(plan.Small)
	wantTech := (1-cs)*(0.85*1+other*550)/(0.85+other) + cs*10
	// technical is in the 0%-weight diagnostic comprehension group, so score is 0 while its pillar score matches wantTech.
	if math.Abs(score) > 1e-12 || math.Abs(pillars["technical"]-wantTech) > 1e-12 || len(pillars) != 1 {
		t.Fatalf("score %v, pillars %v, want technical=%v", score, pillars, wantTech)
	}
	// And on a scored reasoning pillar, score equals the pillar's language-weighted aggregate directly.
	items, gains = nil, nil
	add("explain", "explain", "en", "prose/answer", plan.Small, 1)
	add("explain", "explain", "sw", "prose/answer", 1, 100)
	add("explain", "explain", "ru", "prose/answer", 1, 1000)
	score, pillars, err = plan.ScoreItems(items, gains, map[string]float64{"en": 0.85, "ru": -1})
	if err != nil {
		t.Fatal(err)
	}
	wantExp := (0.85*1 + other*550) / (0.85 + other)
	if math.Abs(score-wantExp) > 1e-12 || math.Abs(pillars["explain"]-wantExp) > 1e-12 {
		t.Fatalf("explain score %v, pillars %v, want %v", score, pillars, wantExp)
	}
}

// A group's share is the season's, whatever the exam has of it.
func TestGroupsKeepTheirShares(t *testing.T) {
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
	add("explain", "explain", -1)
	score, pillars, err := plan.ScoreItems(items, gains, map[string]float64{"en": 43})
	if err != nil {
		t.Fatal(err)
	}
	want := plan.Groups["comprehension"]*(0.4+0.2)/2 + plan.Groups["reasoning"]*-1
	if math.Abs(score-want) > 1e-12 || len(pillars) != 3 {
		t.Fatalf("score %v (want %v), pillars %v", score, want, pillars)
	}
	if plan.pillarOf("explain") != "explain" || plan.pillarOf("social") != "conversation" || plan.pillarOf("no such part") != "" {
		t.Fatal("the season's parts are not under its pillars")
	}
}

// The season says which of its parts are generated, and every one of them is a part it names
// (Season 3 is 100% fresh so Generated is empty).
func TestTheSeasonsGeneratedPartsAreParts(t *testing.T) {
	plan := &loadSeason().Exam
	for _, part := range plan.Generated {
		if plan.pillarOf(part) == "" {
			t.Errorf("%s is generated, and is no part of the exam", part)
		}
	}
}

// What cannot be scored is refused, not scored as nothing.
func TestTheScoreItemsRefusesWhatItCannotScore(t *testing.T) {
	plan := &loadSeason().Exam
	one := []ItemInfo{{"explain", "explain", "en", "prose/answer"}}
	shares := map[string]float64{"en": 43}
	for name, call := range map[string]func() error{
		"more gains than items": func() error { _, _, err := plan.ScoreItems(one, []float64{0.1, 0.2}, shares); return err },
		"no gains":              func() error { _, _, err := plan.ScoreItems(one, nil, shares); return err },
		"no items":              func() error { _, _, err := plan.ScoreItems(nil, nil, shares); return err },
		"no shares":             func() error { _, _, err := plan.ScoreItems(one, []float64{0.1}, nil); return err },
		"a pillar the season does not have": func() error {
			_, _, err := plan.ScoreItems([]ItemInfo{{"skills", "explain", "en", "prose/answer"}}, []float64{0.1}, shares)
			return err
		},
		"a part the season does not name": func() error {
			_, _, err := plan.ScoreItems([]ItemInfo{{"explain", "riddles", "en", "prose/answer"}}, []float64{0.1}, shares)
			return err
		},
		"a part under another part's pillar": func() error {
			_, _, err := plan.ScoreItems([]ItemInfo{{"explain", "infer", "en", "prose/answer"}}, []float64{0.1}, shares)
			return err
		},
	} {
		if call() == nil {
			t.Errorf("%s: scored", name)
		}
	}
	// One item, alone in its stratum and its language: an eighth of a full cell, and all there is.
	if s, p, err := plan.ScoreItems(one, []float64{0.25}, shares); err != nil || s != 0.25 || p["explain"] != 0.25 {
		t.Fatalf("one item: %v %v %v", s, p, err)
	}
}

// Review 7 Finding 4: Season 2 replay preserves its original unweighted pooled-language rule
// (`WeightOtherByShare: false`) across all 47 base-branch reference vectors to within 1e-9,
// while Season 3 enables `WeightOtherByShare: true` explicitly in its season plan.
func TestSeason2AggregateVectorsReplayUnchanged(t *testing.T) {
	if !loadSeason().Exam.WeightOtherByShare {
		t.Fatal("Season 3 ExamPlan must enable WeightOtherByShare")
	}
	s2Plan := &ExamPlan{
		Groups: map[string]float64{"writing": 0.75, "skills": 0.25},
		Pillars: []PlanPillar{
			{Name: "reference", Group: "writing", Parts: []string{"reference"}},
			{Name: "news", Group: "writing", Parts: []string{"news"}},
			{Name: "technical", Group: "writing", Parts: []string{"technical"}},
			{Name: "conversation", Group: "writing", Parts: []string{"discussion", "social"}},
			{Name: "agents", Group: "skills", Parts: []string{"agent"}},
			{Name: "reading", Group: "skills", Parts: []string{"context"}},
			{Name: "logic", Group: "skills", Parts: []string{"logic"}},
			{Name: "grammar", Group: "skills", Parts: []string{"grammar"}},
			{Name: "numbers", Group: "skills", Parts: []string{"arith-shots", "units-shots", "word-shots", "algebra-shots"}},
			{Name: "programs", Group: "skills", Parts: []string{"algo-shots"}},
			{Name: "knowledge", Group: "skills", Parts: []string{"knowledge"}},
		},
		Clip:               4.0,
		Trim:               0.1,
		Small:              8,
		WeightOtherByShare: false,
	}
	raw, err := os.ReadFile("testdata/aggregate_vectors_s2.json")
	if err != nil {
		t.Fatal(err)
	}
	var v aggregateVectors
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	if len(v.Cases) != 47 {
		t.Fatalf("expected 47 Season 2 vectors, got %d", len(v.Cases))
	}
	const tol = 1e-9
	sawEverySource := false
	for _, c := range v.Cases {
		items := v.Tables[c.Table]
		score, pillars, err := s2Plan.ScoreItems(items, c.Gains, c.Shares)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		if math.Abs(score-c.Score) > tol {
			t.Errorf("%s: score %.17g, base Season 2 %.17g", c.Name, score, c.Score)
		}
		for p, want := range c.Pillars {
			if math.Abs(pillars[p]-want) > tol {
				t.Errorf("%s/%s: %.17g, base Season 2 %.17g", c.Name, p, pillars[p], want)
			}
		}
		if c.Name == "every source of writing" {
			sawEverySource = true
			if math.Abs(score-(-0.008952)) > 1e-5 {
				t.Fatalf("every source of writing = %.6f, want -0.008952", score)
			}
		}
	}
	if !sawEverySource {
		t.Fatal("missing 'every source of writing' case in Season 2 vectors")
	}
}
