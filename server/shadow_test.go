package main

import (
	"strings"
	"testing"
)

// The next exam's verdict on a settled round is kept when an organiser's machine sends it, is
// refused from anyone else and when it does not add up, and is shown beside the season's
// scores, ranking nothing.
func TestTheNextExamIsShownBesideTheSeasonsAndRanksNothing(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	alice := w.join("alice", nil)
	if code, page := w.page("/next"); code != 200 || !strings.Contains(page, "No round has been measured on the next exam yet") {
		t.Fatalf("the next exam's page before anything is measured: HTTP %d", code)
	}
	a1 := w.submit(alice, "sieve", "print(1)")
	b1 := w.submit(anchor, "ref-a", "print(2)")
	w.play(anchor, 1, func(name string, round int) float64 {
		if name == "sieve" {
			return 5.2
		}
		return 5.5
	})
	idx, next := 0, 0
	w.app.st.view(func(s *State) {
		for i, rd := range s.Rounds {
			if rd.Status == "settled" && i > idx {
				idx = i
			}
			if rd.Status != "settled" {
				next = i
			}
		}
	})
	if next != 0 {
		// A round still open or scheduled takes no shadow.
		sh := map[string]any{"round": next, "items": 10, "corners": 10, "generated": 10, "fresh": 0, "items_digest": strings.Repeat("ab", 32),
			"generator_version": 2, "entries": []map[string]any{{"editor": a1, "score": 0.1}}}
		if code := anchor.do("POST", "/api/v1/admin/shadow", sh, nil); code != 409 {
			t.Fatalf("a shadow for a round that has not settled: HTTP %d", code)
		}
	}
	gen := 0.3
	good := map[string]any{"round": idx, "items": 900, "corners": 700, "generated": 620, "fresh": 280, "items_digest": strings.Repeat("cd", 32),
		"generator_version": 2, "languages": map[string]float64{"en": 40, "ru": 12},
		"entries": []map[string]any{{"editor": b1, "score": 0.05, "families": map[string]float64{"arith": 0.1, "fresh": 0.0, "agent": -0.1}},
			{"editor": a1, "score": 0.21, "generated": gen, "families": map[string]float64{"arith": 0.4, "fresh": 0.2, "agent": 0.01, "grammar": 0.05}}}}
	if code := alice.do("POST", "/api/v1/admin/shadow", good, nil); code != 403 {
		t.Fatalf("a shadow from a volunteer: HTTP %d", code)
	}
	for _, bad := range []map[string]any{
		{"round": idx, "items": 900, "corners": 700, "generated": 600, "fresh": 280, "items_digest": strings.Repeat("cd", 32), "entries": good["entries"]},
		{"round": idx, "items": 900, "corners": 700, "generated": 620, "fresh": 280, "items_digest": "nope", "entries": good["entries"]},
		{"round": idx, "items": 900, "corners": 700, "generated": 620, "fresh": 280, "items_digest": strings.Repeat("cd", 32),
			"entries": []map[string]any{{"editor": strings.Repeat("e", 64), "score": 0.1}}},
		{"round": idx, "items": 900, "corners": 700, "generated": 620, "fresh": 280, "items_digest": strings.Repeat("cd", 32),
			"entries": []map[string]any{{"editor": a1, "score": 0.1}, {"editor": a1, "score": 0.2}}},
	} {
		if code := anchor.do("POST", "/api/v1/admin/shadow", bad, nil); code != 400 {
			t.Fatalf("a shadow that does not add up (%v): HTTP %d", bad, code)
		}
	}
	if code := anchor.do("POST", "/api/v1/admin/shadow", good, nil); code != 200 {
		t.Fatalf("the organisers' shadow: HTTP %d", code)
	}
	var round struct {
		Shadow *Shadow `json:"shadow"`
	}
	anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &round)
	if round.Shadow == nil || len(round.Shadow.Entries) != 2 || round.Shadow.Entries[0].EditorID != a1 || round.Shadow.By != "anchor-one" {
		t.Fatalf("the round's shadow, best first: %+v", round.Shadow)
	}
	// It is kept: a restart finds it.
	w.check()
	again := startWorld(t, w.cfg, w.kv, w.blobs)
	again.app.st.view(func(s *State) {
		if s.Shadows[idx] == nil || s.Shadows[idx].Items != 900 || len(s.Shadows[idx].Entries) != 2 {
			t.Fatal("the shadow was not kept")
		}
	})
	_, page := w.page("/rounds/" + roundID(w.app.season.ID, idx))
	for _, want := range []string{`id="next"`, "On the next exam", "620 generated, 280 published after the draw", "1st on the season's exam"} {
		if !strings.Contains(page, want) {
			t.Fatalf("the round's page does not show %q", want)
		}
	}
	_, page = w.page("/next")
	if !strings.Contains(page, "sieve") || !strings.Contains(page, "ranks nothing yet") || strings.Index(page, "sieve") > strings.Index(page, "ref-a") {
		t.Fatal("the next exam's standings are missing, or out of order")
	}
	_, page = w.page("/editors/" + a1[:12])
	if !strings.Contains(page, "the next exam</a>, which ranks nothing yet") || !strings.Contains(page, "Strongest at arithmetic and fresh text") {
		t.Fatal("the curator's page does not say how it does on the next exam")
	}
	// The season's standings do not move.
	b := w.app.board()
	if b.ByID[b1] == nil || b.ByID[a1] == nil {
		t.Fatal("both are on the season's board")
	}
}
