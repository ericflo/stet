package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// The organisers pause the rounds while they work on something: no round is drawn, by
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

// Review 20 P2 #2: an organizer pausing the rounds while maybeDraw's beacon lookup is in flight
// MUST NOT leave both an active pause and a newly opened round in the database; the draw
// transaction MUST re-check s.Meta.Paused and apply the paused-slot behavior (dropping the passed
// on-demand slot, or rescheduling in clock mode).
func TestReview20PauseDuringInFlightBeaconLookupPreventsDraw(t *testing.T) {
	t.Run("on_demand_mode_drops_slot_and_does_not_open_round", func(t *testing.T) {
		w, anchor := demandWorld(t, func(cfg *Config) {
			cfg.RoundInterval = 0
			cfg.RoundMinGap = 0
		})
		alice := w.join("alice", nil)
		w.submit(alice, "first", "print(1)")

		// Seed crawl + runner check-in so onDemand schedules the on-demand round.
		w.app.tick(context.Background())
		_ = w.claim(anchor, true)
		w.app.tick(context.Background())

		var drawAt int64
		w.app.st.view(func(s *State) {
			r := s.roundWithStatus("scheduled")
			if r == nil {
				t.Fatal("expected scheduled on-demand round")
			}
			drawAt = w.app.season.Beacon.timeOf(r.DrandRound) + 5
		})
		// Advance server clock past the scheduled beacon round so maybeDraw(ctx, false) is due.
		if err := w.app.st.update(context.Background(), func(tx *Tx) error {
			r := tx.s.roundWithStatus("scheduled")
			if r == nil {
				return nil
			}
			r.DrandRound = w.app.season.Beacon.drandRoundAt(now() - 10)
			r.DrawAt = now() + 60
			tx.saveRound(r)
			return nil
		}); err != nil {
			t.Fatalf("make scheduled round due: %v", err)
		}
		_ = drawAt

		origBeacon := beaconSource
		t.Cleanup(func() { beaconSource = origBeacon })
		entered := make(chan struct{})
		proceed := make(chan struct{})
		beaconSource = func(ctx context.Context, b Beacon, round int64) (string, error) {
			close(entered)
			<-proceed
			return origBeacon(ctx, b, round)
		}

		drawErr := make(chan error, 1)
		go func() {
			drawErr <- w.app.maybeDraw(context.Background(), false)
		}()

		select {
		case <-entered:
		case err := <-drawErr:
			t.Fatalf("maybeDraw exited before beacon lookup: %v", err)
		}
		var out map[string]any
		if code := anchor.do("POST", "/api/v1/admin/pause", map[string]any{"paused": true}, &out); code != 200 || out["paused"] == float64(0) {
			close(proceed)
			t.Fatalf("pause during in-flight beacon lookup: HTTP %d %v", code, out)
		}
		close(proceed)

		if err := <-drawErr; err == nil || !strings.Contains(err.Error(), "paused") {
			t.Fatalf("expected maybeDraw to abort with paused error, got %v", err)
		}
		w.app.st.view(func(s *State) {
			if s.Meta.Paused == 0 {
				t.Fatal("expected pause to remain active")
			}
			if open := s.roundWithStatus("open"); open != nil {
				t.Fatalf("acknowledged pause MUST NOT open a round, got open round %d", open.Idx)
			}
			if sched := s.roundWithStatus("scheduled"); sched != nil {
				t.Fatalf("expected paused on-demand slot to be dropped cleanly, got scheduled round %d", sched.Idx)
			}
			for _, ed := range s.Editors {
				if ed.Status != "queued" {
					t.Fatalf("editor %s status=%q, want queued", ed.Name, ed.Status)
				}
			}
		})
		w.check()
	})

	t.Run("clock_mode_reschedules_slot_and_does_not_open_round", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.cfg.RoundInterval = 3 * time.Hour
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		w.submit(alice, "first", "print(1)")
		w.app.tick(context.Background())

		var origDrand int64
		if err := w.app.st.update(context.Background(), func(tx *Tx) error {
			r := tx.s.roundWithStatus("scheduled")
			if r == nil {
				t.Fatal("expected scheduled round")
			}
			r.DrandRound = w.app.season.Beacon.drandRoundAt(now() - 10)
			r.DrawAt = now() + 60
			origDrand = r.DrandRound
			tx.saveRound(r)
			return nil
		}); err != nil {
			t.Fatalf("make scheduled round due: %v", err)
		}

		origBeacon := beaconSource
		t.Cleanup(func() { beaconSource = origBeacon })
		entered := make(chan struct{})
		proceed := make(chan struct{})
		beaconSource = func(ctx context.Context, b Beacon, round int64) (string, error) {
			close(entered)
			<-proceed
			return origBeacon(ctx, b, round)
		}

		drawErr := make(chan error, 1)
		go func() {
			drawErr <- w.app.maybeDraw(context.Background(), false)
		}()

		select {
		case <-entered:
		case err := <-drawErr:
			t.Fatalf("maybeDraw exited before beacon lookup: %v", err)
		}
		if code := anchor.do("POST", "/api/v1/admin/pause", map[string]any{"paused": true}, nil); code != 200 {
			close(proceed)
			t.Fatalf("pause: HTTP %d", code)
		}
		close(proceed)

		if err := <-drawErr; err == nil || !strings.Contains(err.Error(), "paused") {
			t.Fatalf("expected paused error, got %v", err)
		}
		w.app.st.view(func(s *State) {
			if s.Meta.Paused == 0 {
				t.Fatal("expected pause active")
			}
			if open := s.roundWithStatus("open"); open != nil {
				t.Fatalf("acknowledged pause MUST NOT open a round, got %d", open.Idx)
			}
			sched := s.roundWithStatus("scheduled")
			if sched == nil || sched.DrandRound <= origDrand {
				t.Fatalf("expected clock-mode paused slot rescheduled past %d, got %+v", origDrand, sched)
			}
		})
		w.check()
	})
}
