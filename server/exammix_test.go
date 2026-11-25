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
	if mix == nil || mix.Items != 18 || mix.Fresh != 18 || mix.Langs != 2 || len(mix.Kinds) != 2 || mix.Kinds[0] != (ExamKind{"explain", 12}) || mix.Kinds[1] != (ExamKind{"reference", 6}) {
		t.Fatalf("the exam's mix: %+v", mix)
	}
	// What each part counts for in this exam: the one comprehension pillar takes half, the one reasoning pillar takes half.
	for _, p := range mix.Parts {
		if want := map[string]float64{"reference": 0.50, "explain": 0.50}[p.Pillar]; p.Share != want {
			t.Fatalf("%s is %.2f of the score, not %.2f", p.Pillar, p.Share, want)
		}
	}
	_, page := w.page("/rounds/" + rid)
	for _, want := range []string{"What its exam was made of", "<b>18 items</b>", "all 18 built from writing people published, in 2 languages",
		`<dt>reference</dt><dd>6 written by people<span class="why">50% of the score</span></dd>`, `<dt>explain</dt><dd>12 harvest-derived reasoning<span class="why">50% of the score</span></dd>`,
		"Graders have it now"} {
		if !strings.Contains(page, want) {
			t.Fatalf("the round's page does not say %q", want)
		}
	}
	if strings.Contains(page, "<dt>infer</dt>") {
		t.Fatal("a part with no items is listed")
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
