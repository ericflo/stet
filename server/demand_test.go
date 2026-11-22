package main

import (
	"context"
	"html"
	"os"
	"os/exec"
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
	cfg.RoundMinGap, cfg.EntriesInPlay = time.Hour, 3
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
		t.Fatalf("the draw moved from %d to %d with nobody there to run curators", usual, at)
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
	// It is the first mark on the grid at least the gap after the last draw that no other
	// round was announced for: a slot once announced is never another round's, and near a
	// usual time (every six hours) the first mark after the gap is often the one the last
	// round was announced for before it was drawn early.
	var taken map[int64]bool
	w.app.st.view(func(s *State) { taken = announcedSlots(s, s.roundWithStatus("scheduled")) })
	for m := w.app.season.nextGrid(drawn + gap - 1); m < at; m = w.app.season.nextGrid(m) {
		if !taken[w.app.season.Beacon.drandRoundAt(m)] {
			t.Fatalf("the next draw is %ds after the last, though %ds after it was free; an entry is waiting and the gap is %ds", at-drawn, m-drawn, gap)
		}
	}
	if usual := w.app.nextSlot(now()); usual >= drawn+gap && at < drawn+gap {
		t.Fatalf("the next draw is %ds after the last, inside the gap of %ds", at-drawn, gap)
	}

	// Its second round is its last. Then nobody is waiting, and the schedule is the usual one.
	w.play(anchor, 1, loss)
	w.app.tick(context.Background())
	w.app.st.view(func(s *State) {
		if w.app.waiting(s) {
			t.Fatal("a curator that has had its rounds still counts as waiting")
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

// A round turns nobody away: one person's three and everybody else's all run in the next draw.
// What bounds a round is what one person may have in play, settled when an entry is submitted.
func TestARoundTakesEveryCuratorWaiting(t *testing.T) {
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
	if owners["alice"] != 3 || owners["bob"] != 1 || owners["carol"] != 1 {
		t.Fatalf("five curators were waiting, three of them one person's, and the round took %v", owners)
	}
	w.check()
}

// What the operating guide says about the early days, when the only established machines
// are the organisers' own.
func TestTwoOrganisersMachinesAgreeOrGoTwiceAsFast(t *testing.T) {
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
				// Two machines that agree, and one person's: worth knowing, and not called confirmed.
				case runs == 2 && (!both || e.NRuns != 2 || e.RunsDiffer || e.Confirmed || !e.OneGrader):
					t.Fatalf("runs per entry 2: %s run by both: %v, runs %d, confirmed: %v, one grader: %v", name, both, e.NRuns, e.Confirmed, e.OneGrader)
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
		// The first screen says what this is, a competition, and what one enters, by a name
		// that says what it does.
		"/":       {`href="/enter"`, `href="/verify"`, `href="/run"`, "Three ways to take part", "Stet is a competition to make the best training data", "You enter a <strong>curator</strong>", "Enter the competition", "See the leaderboard", ">Compete</a>", ">Run entries</a>"},
		"/enter":  {"stet init my-curator", "stet run", "stet submit", "3 entries in play at once", "submit as often as you like", "/AGENTS.md", "<h1>Enter the competition</h1>", "writing a <b>curator</b>"},
		"/verify": {"stet verify", "stet service install", "--no-upgrade", "STET_AUTO_UPGRADE=0", "stet upgrade", "Nobody else's code runs on your machine", "cleans up after itself", "stet clean"},
		"/run":    {"<h1>Run entries</h1>", "is a curator", "stet verify --edits", "stet service install --edits", "stet bench", "bubblewrap", "strangers' programs", "could reinstall"},
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

// A script is copies of the round's pages and no more. The pages an entrant reads say so,
// with the season's shortest copy, and promise no allowance of text of a curator's own.
func TestThePagesSayAScriptOnlyCopies(t *testing.T) {
	w, _ := demandWorld(t, nil)
	for path, want := range map[string][]string{
		"/enter":    {"none: a script only copies", "pieces of at least 64 bytes", "so a script holds no text of its own"},
		"/glossary": {"copied byte for byte from the round's pages, and a script holds no text of its own"},
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
		for _, old := range []string{"Almost none", "99.99%", "<no value>"} {
			if strings.Contains(body, old) {
				t.Errorf("%s still says %q", path, old)
			}
		}
	}
	if w.app.season.MinCopyBytes != 64 {
		t.Fatalf("the season's shortest copy is %d bytes", w.app.season.MinCopyBytes)
	}
}

// While the organisers' machines are the only established ones, every score is one
// machine's word, and the site says so where a visitor will see it. It stops saying so
// when an established volunteer has been around.
func TestThePagesSayWhenTheOrganisersAreAlone(t *testing.T) {
	w, anchor := demandWorld(t, nil)
	w.claim(anchor, true)
	for _, path := range []string{"/", "/verify"} {
		if _, body := w.page(path); !strings.Contains(body, "note early") {
			t.Fatalf("%s does not say that only the organisers are verifying", path)
		}
	}
	bob := w.join("bob", nil)
	w.established("bob")
	w.claim(bob, false)
	for _, path := range []string{"/", "/verify"} {
		if _, body := w.page(path); strings.Contains(body, "note early") {
			t.Fatalf("%s still says the organisers are alone, with an established volunteer around", path)
		}
	}
}

// Every verifier that upgrades itself runs this script with nobody watching: it has to be
// a script a shell will take, and it must not leave its scratch environment behind.
func TestTheInstallerIsAScriptAShellAcceptsAndClearsUpAfterItself(t *testing.T) {
	w, _ := demandWorld(t, nil)
	w.app.cfg.FilesDir = t.TempDir()
	if err := os.WriteFile(filepath.Join(w.app.cfg.FilesDir, "stet_cli-9.9.9-py3-none-any.whl"), []byte("a wheel"), 0o600); err != nil {
		t.Fatal(err)
	}
	code, body := w.page("/install.sh")
	if code != 200 {
		t.Fatalf("/install.sh: HTTP %d", code)
	}
	for _, s := range []string{`PRE="${STET_CACHE:-$HOME/.cache/stet}/upgrade-env"`, `trap 'rm -rf "$TMP" ${PRE:+"$PRE"}' EXIT`, "stet init my-curator"} {
		if !strings.Contains(body, s) {
			t.Errorf("the installer does not have %q", s)
		}
	}
	for _, sh := range []string{"sh", "dash", "bash"} {
		bin, err := exec.LookPath(sh)
		if err != nil {
			continue
		}
		path := filepath.Join(t.TempDir(), "install.sh")
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		if out, err := exec.Command(bin, "-n", path).CombinedOutput(); err != nil {
			t.Errorf("%s does not accept the installer: %v\n%s", sh, err, out)
		}
	}
}

// The work says an entry is waiting only when a runner turning up would bring its round's
// draw forward: not while a round is open, not once the draw is as early as it can be, and
// never when rounds keep to the clock. A runner rented for anything else would only wait.
func TestTheWorkSaysWaitingOnlyWhenARunnerWouldDrawEarly(t *testing.T) {
	waiting := func(w *world, anchor *actor) bool {
		t.Helper()
		w.app.tick(context.Background())
		var work Work
		if code := anchor.do("GET", "/api/v1/admin/work", nil, &work); code != 200 {
			t.Fatalf("HTTP %d", code)
		}
		return work.Waiting
	}
	w, anchor := demandWorld(t, nil)
	alice := w.join("alice", nil)
	if waiting(w, anchor) {
		t.Fatal("waiting, with no entry")
	}
	w.submit(alice, "first", "print(1)")
	// In the last minutes before the usual draw there is nothing left to bring forward: the
	// draw is already at the first mark on the grid that leaves verifiers their notice.
	room := w.app.season.nextGrid(now()+drawNotice-1) < w.scheduledAt()
	if got := waiting(w, anchor); got != room {
		t.Fatalf("waiting is %v with an entry and nobody around to run it, and the draw %ds off", got, w.scheduledAt()-now())
	}
	// A runner asks: the draw comes forward, and nothing more is gained by another.
	w.claim(anchor, true)
	if waiting(w, anchor) {
		t.Fatalf("still waiting once the draw was brought forward to %d", w.scheduledAt())
	}
	// While its round is open, the next one is not drawn early for anybody.
	w.draw()
	w.submit(alice, "second", "print(2)")
	if waiting(w, anchor) {
		t.Fatal("waiting while a round is open")
	}

	// On the clock alone, an entry waiting brings nothing forward.
	w, anchor = demandWorld(t, func(c *Config) { c.RoundMinGap = 0 })
	w.submit(w.join("bob", nil), "third", "print(3)")
	if waiting(w, anchor) {
		t.Fatal("waiting, with rounds on the clock")
	}
	w.check()
}
