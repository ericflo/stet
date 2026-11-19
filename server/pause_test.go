package main

import (
	"context"
	"strings"
	"testing"
)

// The organisers pause the rounds while they work on the next exam: no round is drawn, by
// the clock or by hand, nobody is told one is coming, and the pages say they are paused.
// Resumed, the rounds go on as before.
func TestTheOrganisersPauseTheRoundsAndResumeThem(t *testing.T) {
	w, anchor := demandWorld(t, nil)
	alice := w.join("alice", nil)
	w.submit(alice, "first", "print(1)")

	pause := func(by *actor, paused bool) (int, map[string]any) {
		var out map[string]any
		return by.do("POST", "/api/v1/admin/pause", map[string]any{"paused": paused}, &out), out
	}
	if code, _ := pause(alice, true); code != 403 {
		t.Fatalf("an entrant paused the rounds: HTTP %d", code)
	}
	if code := anchor.do("POST", "/api/v1/admin/pause", map[string]any{}, nil); code != 400 {
		t.Fatalf("a pause that says neither: HTTP %d", code)
	}
	if code, out := pause(anchor, true); code != 200 || out["paused"] == float64(0) || out["paused_by"] != "anchor-one" {
		t.Fatalf("pausing: HTTP %d %v", code, out)
	}

	// No round is drawn, though one is due and a curator is waiting.
	ctx := context.Background()
	w.app.tick(ctx)
	if err := w.app.maybeDraw(ctx, true); err == nil || !strings.Contains(err.Error(), "paused") {
		t.Fatalf("a round was drawn while paused (%v)", err)
	}
	w.app.st.view(func(s *State) {
		if s.roundWithStatus("open") != nil {
			t.Fatal("a round is open")
		}
	})
	if code := anchor.do("POST", "/api/v1/admin/draw", map[string]any{}, nil); code != 409 {
		t.Fatalf("a draw by hand while paused: HTTP %d", code)
	}

	// Nobody is told a round is coming: not a volunteer, not the organisers' machines.
	var season map[string]any
	alice.do("GET", "/api/v1/season", nil, &season)
	if season["next_round"] != nil || season["paused"] == nil {
		t.Fatalf("the season says next_round %v, paused %v", season["next_round"], season["paused"])
	}
	var work Work
	if code := anchor.do("GET", "/api/v1/admin/work", nil, &work); code != 200 || work.Paused == 0 || work.NextDrawAt != 0 {
		t.Fatalf("the work says paused %d, next draw %d (HTTP %d)", work.Paused, work.NextDrawAt, code)
	}
	if _, home := w.page("/"); !strings.Contains(home, "Rounds paused") {
		t.Fatal("the home page does not say the rounds are paused")
	}

	// Resumed (twice: the second changes nothing), the round is drawn.
	for range 2 {
		if code, out := pause(anchor, false); code != 200 || out["paused"] != float64(0) {
			t.Fatalf("resuming: HTTP %d %v", code, out)
		}
	}
	w.draw()
	if _, home := w.page("/"); strings.Contains(home, "Rounds paused") {
		t.Fatal("the home page still says the rounds are paused")
	}
	w.check()
}
