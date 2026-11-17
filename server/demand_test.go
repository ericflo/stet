package main

import (
	"context"
	"html"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A world where rounds are drawn on demand and one person may have three entries in play.
func demandWorld(t *testing.T, change func(*Config)) (*world, *actor) {
	t.Helper()
	pub, priv := newKey()
	cfg := testConfig(t, pub)
	cfg.RoundMinGap, cfg.EntriesInPlay, cfg.MaxNewPerRound = time.Hour, 3, 3
	if change != nil {
		change(&cfg)
	}
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	return w, w.join("anchor-one", priv)
}

func (w *world) scheduledAt() (at int64) {
	w.app.st.view(func(s *State) {
		if r := s.roundWithStatus("scheduled"); r != nil {
			at = r.DrawAt
		}
	})
	return at
}

func TestARoundIsDrawnSoonWhenAnEntryIsWaiting(t *testing.T) {
	w, anchor := demandWorld(t, nil)
	alice := w.join("alice", nil)
	grid := w.app.season.DrawGrid
	w.app.tick(context.Background())
	usual := w.scheduledAt()
	if usual == 0 || usual%int64(w.cfg.RoundInterval.Seconds()) != 0 && usual > now()+drawNotice+grid {
		t.Fatalf("with nobody waiting the draw is not at the usual time: %d", usual)
	}

	// Nobody who runs editors is around: a round drawn now would only use up its number.
	w.submit(alice, "first", "print(1)")
	w.app.tick(context.Background())
	if at := w.scheduledAt(); at != usual {
		t.Fatalf("the draw moved from %d to %d with nobody there to run editors", usual, at)
	}

	// The organisers' machine asks for work. The entry that is waiting gets its round at
	// the first mark on the grid that leaves verifiers their notice.
	if c := w.claim(anchor, true); c.Job != nil {
		t.Fatalf("a job before any round: %+v", c.Job)
	}
	w.app.tick(context.Background())
	at := w.scheduledAt()
	if at < now()+drawNotice-2 || at > now()+drawNotice+grid || at%grid != 0 {
		t.Fatalf("the draw is %ds away, want between %d and %d and on the grid", at-now(), drawNotice, drawNotice+grid)
	}
	w.check()

	// Whoever submits next is told the draw they will be in, not the one six hours off.
	var out struct {
		Enters struct {
			DrawAt int64 `json:"draw_at"`
		} `json:"enters"`
		InPlay int `json:"in_play"`
		Limit  int `json:"in_play_limit"`
	}
	if code := alice.do("POST", "/api/v1/editors", bundle(t, "second", "print(2)"), &out); code != 201 {
		t.Fatalf("HTTP %d", code)
	}
	if out.Enters.DrawAt != at || out.InPlay != 2 || out.Limit != 3 {
		t.Fatalf("the reply says draw %d, %d of %d in play; the draw is at %d", out.Enters.DrawAt, out.InPlay, out.Limit, at)
	}
}

func TestDrawsOnDemandKeepAGapAndStopWhenNobodyIsWaiting(t *testing.T) {
	w, anchor := demandWorld(t, func(c *Config) { c.QualifyRounds = 2 })
	alice := w.join("alice", nil)
	grid := w.app.season.DrawGrid
	w.submit(alice, "first", "print(1)")
	loss := func(string, int) float64 { return 3.0 }

	// One round in, the entry is still qualifying: the next is brought forward, but not to
	// less than the gap after the last draw.
	w.play(anchor, 1, loss)
	var drawn int64
	w.app.st.view(func(s *State) { drawn = s.Rounds[s.maxRound()-1].DrawnAt })
	w.app.tick(context.Background())
	at, gap := w.scheduledAt(), int64(w.cfg.RoundMinGap.Seconds())
	if at > drawn+gap+grid {
		t.Fatalf("the next draw is %ds after the last; an entry is waiting and the gap is %ds", at-drawn, gap)
	}
	if usual := w.app.nextSlot(now()); usual >= drawn+gap && at < drawn+gap {
		t.Fatalf("the next draw is %ds after the last, inside the gap of %ds", at-drawn, gap)
	}

	// Its second round is its last. Then nobody is waiting, and the schedule is the usual one.
	w.play(anchor, 1, loss)
	w.app.tick(context.Background())
	w.app.st.view(func(s *State) {
		if w.app.waiting(s) {
			t.Fatal("an editor that has had its rounds still counts as waiting")
		}
	})
	// (A slot once announced for another round is not used again, so "the usual time" may
	// be a mark or two after the interval's own.)
	if at, usual := w.scheduledAt(), w.app.laterSlot(); at < usual || at > usual+6*grid {
		t.Fatalf("with nobody waiting the draw is at %d, not the usual %d", at, usual)
	}
	w.check()
}

func TestOnePersonMayHaveSeveralEntriesInPlay(t *testing.T) {
	w, _ := demandWorld(t, nil)
	alice := w.join("alice", nil)
	a1 := w.submit(alice, "idea-a", "print('a1')")
	b := w.submit(alice, "idea-b", "print('b')")
	c := w.submit(alice, "idea-c", "print('c')")
	status := func(id string) (st string) {
		w.app.st.view(func(s *State) { st = s.Editors[id].Status })
		return st
	}
	for _, id := range []string{a1, b, c} {
		if status(id) != "queued" {
			t.Fatalf("three entries with three names: %s is %s", id[:8], status(id))
		}
	}
	// Spread them out in time, oldest first: a, b, c.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for i, id := range []string{a1, b, c} {
			tx.s.Editors[id].CreatedAt -= int64(300 - 100*i)
			tx.saveEditor(tx.s.Editors[id])
		}
		return nil
	})

	// A new version of an entry takes the old version's place, and nothing else's.
	a2 := w.submit(alice, "idea-a", "print('a2')")
	if status(a1) != "withdrawn" || status(b) != "queued" || status(c) != "queued" || status(a2) != "queued" {
		t.Fatalf("after a new version of a: a1 %s, b %s, c %s, a2 %s", status(a1), status(b), status(c), status(a2))
	}
	// A fourth idea: the oldest of the others makes way.
	d := w.submit(alice, "idea-d", "print('d')")
	if status(b) != "withdrawn" || status(c) != "queued" || status(a2) != "queued" || status(d) != "queued" {
		t.Fatalf("after a fourth entry: b %s, c %s, a2 %s, d %s", status(b), status(c), status(a2), status(d))
	}
	w.check()
}

func TestEachPersonsFirstEntryGoesBeforeAnyonesSecond(t *testing.T) {
	w, anchor := demandWorld(t, nil)
	alice, bob, carol := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	for _, n := range []string{"alice-1", "alice-2", "alice-3"} {
		w.submit(alice, n, "print('"+n+"')")
	}
	// Alice's three are the oldest in the queue by a long way.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, e := range tx.s.Editors {
			e.CreatedAt -= 600
			tx.saveEditor(e)
		}
		return nil
	})
	w.submit(bob, "bob-1", "print('b1')")
	w.submit(carol, "carol-1", "print('c1')")
	w.claim(anchor, true)
	idx := w.draw()
	owners := map[string]int{}
	w.app.st.view(func(s *State) {
		for _, e := range s.inRound[idx] {
			owners[s.owner(s.Editors[e.EditorID]).Handle]++
		}
	})
	if owners["alice"] != 1 || owners["bob"] != 1 || owners["carol"] != 1 {
		t.Fatalf("a round takes three new entries and three people are waiting; it took %v", owners)
	}
	w.check()
}

// What the operating guide says about the early days, when the only established machines
// are the organisers' own.
func TestTwoOrganisersMachinesConfirmOrGoTwiceAsFast(t *testing.T) {
	for _, runs := range []int{2, 1} {
		pubA, privA := newKey()
		pubB, privB := newKey()
		cfg := testConfig(t, pubA, pubB)
		cfg.RunsPerEntry = runs
		w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
		a, b := w.join("anchor-one", privA), w.join("anchor-two", privB)
		alice, bob := w.join("alice", nil), w.join("bob", nil)
		w.submit(alice, "first", "print(1)")
		w.submit(bob, "second", "print(2)")
		idx := w.draw()
		ran := map[string]map[string]bool{"anchor-one": {}, "anchor-two": {}}
		for step := 0; step < 40; step++ {
			busy := false
			for _, who := range []*actor{a, b} {
				c := w.claim(who, true)
				if c.Job == nil {
					continue
				}
				busy = true
				if c.Job.Kind == "edit" {
					ran[who.handle][c.Job.Editor.Name] = true
					// The two machines write different scripts for one entry, as two real runs may.
					if code, status := w.edit(who, c, "{\"doc\": 1, \"by\": \""+c.Job.Editor.Name+"\"}\n"); code != 200 {
						t.Fatal(status)
					}
				} else if code, status := w.report(who, c, 3.0); code != 200 {
					t.Fatalf("HTTP %d %s", code, status)
				}
			}
			if !busy {
				break
			}
		}
		if w.settled(idx) != 2 {
			t.Fatalf("runs per entry %d: the round did not settle with two organisers' machines", runs)
		}
		w.app.st.view(func(s *State) {
			for _, e := range s.inRound[idx] {
				name := s.Editors[e.EditorID].Name
				both := ran["anchor-one"][name] && ran["anchor-two"][name]
				switch {
				case !e.Valid:
					t.Fatalf("runs per entry %d: %s did not score: %s", runs, name, e.Reason)
				case runs == 2 && (!both || !e.Confirmed):
					t.Fatalf("runs per entry 2: %s run by both: %v, confirmed: %v", name, both, e.Confirmed)
				case runs == 1 && (both || len(e.Runs) != 1):
					t.Fatalf("runs per entry 1: %s was run %d times", name, len(e.Runs))
				}
			}
		})
		if runs == 1 && (len(ran["anchor-one"]) != 1 || len(ran["anchor-two"]) != 1) {
			t.Fatalf("runs per entry 1: the two machines did not share the entries: %v", ran)
		}
		w.check()
	}
}

// The site has a page for each thing a visitor may have come to do, and each says what to
// type. The home page leads to all three.
func TestEachWayOfTakingPartHasAPageOfItsOwn(t *testing.T) {
	w, _ := demandWorld(t, nil)
	for path, want := range map[string][]string{
		"/":       {`href="/enter"`, `href="/verify"`, `href="/run"`, "Three ways to take part"},
		"/enter":  {"stet init my-editor", "stet run", "stet submit", "3 entries in play at once", "6 a day", "/AGENTS.md"},
		"/verify": {"stet verify", "stet service install", "--no-upgrade", "STET_AUTO_UPGRADE=0", "stet upgrade", "Nobody else's code runs on your machine"},
		"/run":    {"stet verify --edits", "stet service install --edits", "stet bench", "bubblewrap", "strangers' programs", "could reinstall"},
	} {
		code, body := w.page(path)
		if code != 200 {
			t.Fatalf("%s: HTTP %d", path, code)
		}
		for _, s := range want {
			if !strings.Contains(body, s) && !strings.Contains(body, html.EscapeString(s)) {
				t.Errorf("%s does not say %q", path, s)
			}
		}
	}
}
