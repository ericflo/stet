package main

import (
	"strings"
	"testing"
)

// A round's page says what its exam was made of: how many items, how many of them people
// wrote after the draw and in how many languages, and how many each capability has. It is
// kept once the round settles, when the table it is counted from is not.
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
	if mix == nil || mix.Items != 22 || mix.Fresh != 6 || mix.Langs != 2 || len(mix.Kinds) != 1 || mix.Kinds[0] != (ExamKind{"reference", 6}) {
		t.Fatalf("the exam's mix: %+v", mix)
	}
	_, page := w.page("/rounds/" + rid)
	for _, want := range []string{"What its exam was made of", "<b>22 items</b>", "6 passages people wrote in the hour after the draw, in 2 languages", "16 problems generated",
		"<dt>writing</dt><dd>6 items written by people</dd>", "<dt>reasoning</dt><dd>12 items generated</dd>", "Graders have it now"} {
		if !strings.Contains(page, want) {
			t.Fatalf("the round's page does not say %q", want)
		}
	}
	if strings.Contains(page, "<dt>agents</dt>") {
		t.Fatal("a capability with no items is listed")
	}
	w.check()
}
