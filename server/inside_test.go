package main

import (
	"fmt"
	"strings"
	"testing"
)

// A curator's page shows the inside of its latest scored round, and of any other it was
// scored in when asked; each round's page links to it.
func TestACuratorsPageShowsTheInsideOfAnyOfItsRounds(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	id := w.submit(anchor, "tidy", "print(1)")
	w.play(anchor, 3, func(_ string, round int) float64 { return 5.5 + 0.01*float64(round) })
	var rounds []int // newest first
	w.app.st.view(func(s *State) {
		for _, e := range s.ofEditor[id] {
			rounds = append([]int{e.RoundIdx}, rounds...)
		}
	})
	if len(rounds) != 3 {
		t.Fatalf("it played %d rounds, not 3", len(rounds))
	}
	path := "/editors/" + id[:12]
	pick := func(idx int) string {
		return fmt.Sprintf(`class="pill on" href="?round=%d#inside" aria-label="round %d" aria-current="page">%d<`, idx, idx, idx)
	}
	sees := func(page string, want ...string) {
		t.Helper()
		for _, s := range want {
			if !strings.Contains(page, s) {
				t.Errorf("the page does not say %q", s)
			}
		}
	}

	_, page := w.page(path)
	sees(page, "Inside its latest round", pick(rounds[0]), `id="inside">`,
		fmt.Sprintf("who ran and graded it in round %d", rounds[0]))
	for _, idx := range rounds[1:] {
		sees(page, fmt.Sprintf(`href="?round=%d#inside" aria-label="round %d">%d<`, idx, idx, idx))
	}

	// An earlier round, asked for: open, and that round's.
	oldest := rounds[len(rounds)-1]
	_, page = w.page(fmt.Sprintf("%s?round=%d", path, oldest))
	sees(page, "Inside one of its rounds", pick(oldest), `id="inside" open>`,
		fmt.Sprintf(`href="/rounds/%s#e-%s">who ran and graded it in round %d`, roundID(w.app.season.ID, oldest), id[:12], oldest))

	// A round it was not in shows its latest, closed.
	_, page = w.page(path + "?round=999")
	sees(page, "Inside its latest round", pick(rounds[0]), `id="inside">`)

	// Each round's page links to the inside of each curator's run in it.
	_, page = w.page("/rounds/" + roundID(w.app.season.ID, oldest))
	sees(page, fmt.Sprintf(`href="/editors/%s?round=%d#inside">inside this round`, id[:12], oldest))
	w.check()
}
