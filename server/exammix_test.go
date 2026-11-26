package main

import (
	"strings"
	"testing"
)

// A round's page says what its exam was made of: how many items, how many of them people
// wrote and in how many languages, and how many each part has. It is kept once the round
// settles, when the table it is counted from is not.
func TestARoundsPageSaysWhatItsExamWasMadeOf(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.drawWithoutExam()
	rid := roundID(w.app.season.ID, idx)
	if _, page := w.page("/rounds/" + rid); !strings.Contains(page, "This one is being made now") {
		t.Fatal("before its exam is posted, the page says it is being made")
	}
	if code, msg := w.postExam(anchor, idx, testTable()); code != 200 {
		t.Fatalf("posting the exam: HTTP %d %s", code, msg)
	}
	var mix *ExamMix
	w.app.st.view(func(s *State) { mix = s.Rounds[idx].ExamMix })
	if mix == nil || mix.Items != 140 || mix.Fresh != 140 || mix.Langs != 2 || len(mix.Kinds) != 7 {
		t.Fatalf("the exam's mix: %+v", mix)
	}
	// Four comprehension pillars split 50% (12.5% each); three reasoning pillars split 50% (~16.67% each).
	for _, p := range mix.Parts {
		want := 0.50 / 4.0
		if p.Pillar == "explain" || p.Pillar == "infer" || p.Pillar == "synthesis" {
			want = 0.50 / 3.0
		}
		if diff := p.Share - want; diff < -1e-4 || diff > 1e-4 {
			t.Fatalf("%s is %.4f of the score, not %.4f", p.Pillar, p.Share, want)
		}
	}
	_, page := w.page("/rounds/" + rid)
	for _, want := range []string{"What its exam was made of", "<b>140 items</b>", "all 140 built from writing people published, in 2 languages",
		`<dt>reference</dt><dd>20 written by people<span class="why">12.5% of the score</span></dd>`, `<dt>explain</dt><dd>20 harvest-derived reasoning<span class="why">16.7% of the score</span></dd>`,
		`<dt>infer</dt>`, `<dt>synthesis</dt>`, "Graders have it now"} {
		if !strings.Contains(page, want) {
			t.Fatalf("the round's page does not say %q", want)
		}
	}
	// Every item is in a part the season names, so the pillars listed hold the whole exam.
	listed := 0
	for _, p := range mix.Parts {
		listed += p.Items
	}
	if listed != mix.Items {
		t.Fatalf("%d of the exam's %d items are under a pillar", listed, mix.Items)
	}
	w.check()
}
