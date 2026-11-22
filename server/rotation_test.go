package main

import (
	"context"
	"path/filepath"
	"sort"
	"testing"
)

// --- the organisers' reference curators take turns ---------------------------------------------

// ranIn is which editors a round entered, by id.
func (w *world) ranIn(idx int) map[string]bool {
	out := map[string]bool{}
	w.app.st.view(func(s *State) {
		for _, e := range s.inRound[idx] {
			out[e.EditorID] = true
		}
	})
	return out
}

// rounds plays whole rounds with the organisers' one machine doing all the work (play), an
// hour apart as far as the server's limit on one key's uploads goes: it uploads every script.
func (w *world) rounds(anchor *actor, n int, loss func(name string, round int) float64) {
	w.t.Helper()
	for i := 0; i < n; i++ {
		w.app.limits = newLimiter()
		w.play(anchor, 1, loss)
	}
}

// inTurn orders editors as the rotation does: by when they were entered, then by id.
func (w *world) inTurn(ids []string) []string {
	out := append([]string(nil), ids...)
	w.app.st.view(func(s *State) {
		sort.Slice(out, func(i, j int) bool {
			x, y := s.Editors[out[i]], s.Editors[out[j]]
			return x.CreatedAt < y.CreatedAt || (x.CreatedAt == y.CreatedAt && x.ID < y.ID)
		})
	})
	return out
}

// Season 1 ran all seven reference curators in every round, and half of all its training was
// theirs. The two the season names still run in every round and tie the rounds together; each
// of the others runs in one round in eight, a different one each, and keeps its place on the
// ladder with the rating its rounds gave it. An entrant's curator on the ladder is untouched.
func TestReferenceCuratorsTakeTurnsAndTheNamedOnesRunEveryRound(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	rule := w.app.season.References
	if rule.OneIn != 8 || len(rule.EveryRound) != 2 || rule.EveryRound[0] != "fineweb2" || rule.EveryRound[1] != "dclm" {
		t.Fatalf("this test is written for fineweb2 and dclm in every round and the rest one in 8: %+v", rule)
	}
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	every := []string{w.submit(anchor, "fineweb2", "print('fineweb2')"), w.submit(anchor, "dclm", "print('dclm')")}
	var turns []string
	for _, name := range []string{"ref-a", "ref-b", "ref-c", "ref-d", "ref-e"} {
		turns = append(turns, w.submit(anchor, name, "print('"+name+"')"))
	}
	mine := w.submit(alice, "tidy", "print('tidy')")
	all := append(append(append([]string(nil), every...), turns...), mine)

	// Good enough to be ranked, not far enough from the baseline to set a record. A curator
	// still in its qualifying rounds plays every one of them, a reference like any other: one
	// round in eight would make qualifying take most of a season.
	qualify := w.app.cfg.QualifyRounds
	w.rounds(anchor, qualify, func(string, int) float64 { return 5.99 })
	for idx := 1; idx <= qualify; idx++ {
		if in := w.ranIn(idx); len(in) != len(all) {
			t.Fatalf("qualifying round %d entered %d of the %d curators", idx, len(in), len(all))
		}
	}
	onLadder := func() map[string]*BoardRow {
		out := map[string]*BoardRow{}
		for _, r := range w.app.board().Ladder {
			out[r.ID] = r
		}
		return out
	}
	if l := onLadder(); len(l) != len(all) {
		t.Fatalf("the ladder after qualifying: %d of %d", len(l), len(all))
	}

	const rounds = 16
	turns = w.inTurn(turns)
	w.rounds(anchor, rounds, func(string, int) float64 { return 5.99 })
	ran := map[string]int{}
	for idx := qualify + 1; idx <= qualify+rounds; idx++ {
		in := w.ranIn(idx)
		for id := range in {
			ran[id]++
		}
		for _, id := range append(append([]string(nil), every...), mine) {
			if !in[id] {
				t.Fatalf("round %d left out %s, which runs in every round", idx, id[:12])
			}
		}
		taking := 0
		for k, id := range turns {
			if due := (idx+k)%rule.OneIn == 0; in[id] != due {
				t.Fatalf("round %d: the reference at place %d of the rotation ran: %v, and its turn says %v", idx, k, in[id], due)
			}
			if in[id] {
				taking++
			}
		}
		if taking > 1 {
			t.Fatalf("round %d ran %d of the references that take turns", idx, taking)
		}
	}
	for _, id := range append(append([]string(nil), every...), mine) {
		if ran[id] != rounds {
			t.Fatalf("%s ran in %d of %d rounds", id[:12], ran[id], rounds)
		}
	}
	// Sitting a round out is no entry at all: it is not a round that failed, not one of the
	// rounds an editor has had, and nothing takes it off the ladder.
	ladder := onLadder()
	w.app.st.view(func(s *State) {
		for _, id := range turns {
			if ran[id] != 2 {
				t.Fatalf("%s ran in %d of %d rounds, and its turn comes twice", id[:12], ran[id], rounds)
			}
			row := ladder[id]
			if row == nil || row.Rounds != qualify+2 || row.Rating == nil || row.Rank == 0 {
				t.Fatalf("a reference that sat rounds out lost its place on the board: %+v", row)
			}
			if failing(s, id) || roundsRun(s, id) != qualify+2 || s.Editors[id].Status != "retired" {
				t.Fatalf("sitting rounds out was held against %s: failing %v, %d rounds, %s", id[:12], failing(s, id), roundsRun(s, id), s.Editors[id].Status)
			}
		}
	})
	if len(ladder) != len(all) {
		t.Fatalf("the ladder after %d rounds: %d of %d", rounds, len(ladder), len(all))
	}

	// The round announced next says who it would enter: the same turn-taking, before the draw.
	w.app.tick(context.Background())
	next := qualify + rounds + 1
	var in []expectedRow
	w.app.st.view(func(s *State) {
		rd := s.Rounds[next]
		if rd == nil || rd.Status != "scheduled" {
			t.Fatalf("round %d is not the one announced: %+v", next, rd)
		}
		in, _ = w.app.expected(s, rd)
	})
	expected := map[string]bool{}
	for _, r := range in {
		expected[r.EditorID] = true
		if r.Why != "ladder" {
			t.Fatalf("expected to play, and not for the ladder: %+v", r)
		}
	}
	want := 3
	for k, id := range turns {
		due := (next+k)%rule.OneIn == 0
		if expected[id] != due {
			t.Fatalf("round %d expects the reference at place %d: %v, and its turn says %v", next, k, expected[id], due)
		}
		if due {
			want++
		}
	}
	if len(in) != want || !expected[every[0]] || !expected[every[1]] || !expected[mine] {
		t.Fatalf("round %d expects %d curators, not %d: %+v", next, len(in), want, in)
	}
	w.check()
}

// A reference that holds the record does not sit rounds out: the record changes hands only
// in rounds a challenger shares with its holder.
func TestAReferenceThatHoldsTheRecordRunsEveryRound(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	best, other := w.submit(anchor, "ref-best", "print('best')"), w.submit(anchor, "ref-other", "print('other')")
	mine := w.submit(alice, "tidy", "print('tidy')")
	loss := func(name string, _ int) float64 {
		if name == "ref-best" {
			return 5.0
		}
		return 5.99
	}
	qualify := w.app.cfg.QualifyRounds
	w.rounds(anchor, qualify, loss)
	w.app.st.view(func(s *State) {
		if h := s.holder(); h == nil || h.EditorID != best {
			t.Fatalf("the record: %+v", h)
		}
	})
	one := w.app.season.References.OneIn
	w.rounds(anchor, one, loss)
	ran := map[string]int{}
	for idx := qualify + 1; idx <= qualify+one; idx++ {
		for id := range w.ranIn(idx) {
			ran[id]++
		}
	}
	if ran[best] != one || ran[mine] != one || ran[other] != 1 {
		t.Fatalf("over %d rounds the holder ran %d times, the entrant's curator %d, the other reference %d", one, ran[best], ran[mine], ran[other])
	}
	w.check()
}

// With nobody but references whose turn it is not, a round would have no entrant, would not
// be drawn, and would keep its number: their turn would never come. They all run instead.
func TestReferencesAloneDoNotWaitForATurnThatCannotCome(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	refs := w.inTurn([]string{w.submit(anchor, "ref-a", "print('a')"), w.submit(anchor, "ref-b", "print('b')")})
	qualify, one := w.app.cfg.QualifyRounds, w.app.season.References.OneIn
	w.rounds(anchor, qualify+one, func(string, int) float64 { return 5.99 })
	both := 0
	for idx := qualify + 1; idx <= qualify+one; idx++ {
		in := w.ranIn(idx)
		due := 0
		for k, id := range refs {
			if (idx+k)%one == 0 {
				due++
				if !in[id] || len(in) != 1 {
					t.Fatalf("round %d is the turn of the reference at place %d, alone: %+v", idx, k, in)
				}
			}
		}
		if due == 0 {
			both++
			if !in[refs[0]] || !in[refs[1]] {
				t.Fatalf("round %d, nobody's turn and nobody else to run: %+v", idx, in)
			}
		}
	}
	if both != one-2 {
		t.Fatalf("of %d rounds, %d were nobody's turn", one, both)
	}
	w.check()
}

// The ladder's size is how many places the entrants' curators take. The organisers' reference
// curators all stay on beside them, however many entrants' curators have shown they are worth
// more: the ones the season names go on tying the rounds together, and the others keep the
// turns they had.
func TestEntrantsCuratorsDoNotTakeTheReferencesPlacesOnTheLadder(t *testing.T) {
	pub, priv := newKey()
	cfg := testConfig(t, pub)
	cfg.LadderSize = 2
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	anchor := w.join("anchor-one", priv)
	every := []string{w.submit(anchor, "fineweb2", "print('fineweb2')"), w.submit(anchor, "dclm", "print('dclm')")}
	turn := w.submit(anchor, "ref-a", "print('ref-a')")
	mine := map[string]string{}
	for _, name := range []string{"first", "second", "third"} {
		mine[name] = w.submit(w.join("owner-of-"+name, nil), name, "print('"+name+"')")
	}
	// Every entrant's curator is better than every reference in every round, and none beats
	// the unedited data: nobody takes the record, and a place on the ladder with it.
	loss := func(name string, _ int) float64 {
		switch name {
		case "first":
			return 6.10
		case "second":
			return 6.12
		case "third":
			return 6.14
		}
		return 6.30
	}
	qualify, one := w.app.cfg.QualifyRounds, w.app.season.References.OneIn
	w.rounds(anchor, qualify, loss)
	b := w.app.board()
	if b.RecordID != "" || len(b.Rated) != 6 {
		t.Fatalf("this test is written for six ranked curators and no record: %d ranked, record %q", len(b.Rated), b.RecordID)
	}
	for _, id := range append(append([]string(nil), every...), turn) {
		if row := b.ByID[id]; !row.Ladder || !(*row.Lower < *b.ByID[mine["third"]].Lower) {
			t.Fatalf("the reference %s (at least %v): on the ladder %v, below every entrant's curator %v", row.Name, *row.Lower, row.Ladder, *row.Lower < *b.ByID[mine["third"]].Lower)
		}
	}
	if len(b.Ladder) != 5 || !b.ByID[mine["first"]].Ladder || !b.ByID[mine["second"]].Ladder || b.ByID[mine["third"]].Ladder {
		t.Fatalf("a ladder of two places and three references has %d curators: first %v, second %v, third %v", len(b.Ladder),
			b.ByID[mine["first"]].Ladder, b.ByID[mine["second"]].Ladder, b.ByID[mine["third"]].Ladder)
	}

	w.rounds(anchor, one, loss)
	ran := map[string]int{}
	for idx := qualify + 1; idx <= qualify+one; idx++ {
		in := w.ranIn(idx)
		for id := range in {
			ran[id]++
		}
		if !in[every[0]] || !in[every[1]] {
			t.Fatalf("round %d left out a reference that runs in every round: %v", idx, in)
		}
		if due := idx%one == 0; in[turn] != due {
			t.Fatalf("round %d: the one reference that takes turns ran: %v, and its turn says %v", idx, in[turn], due)
		}
	}
	if ran[mine["first"]] != one || ran[mine["second"]] != one || ran[mine["third"]] != 0 || ran[turn] != 1 {
		t.Fatalf("over %d rounds: first ran %d times, second %d, third %d, the reference that takes turns %d", one,
			ran[mine["first"]], ran[mine["second"]], ran[mine["third"]], ran[turn])
	}
	w.check()
}
