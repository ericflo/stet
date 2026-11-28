package main

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// A rating is the curator's own mean, adjusted for its rounds, and its standard error says how
// many rounds that rests on: in numbers that can be checked by hand.
func TestARatingIsACuratorsOwnMeanAndSaysHowSureItIs(t *testing.T) {
	near := func(got, want float64) bool { return math.Abs(got-want) < 1e-9 }
	// Three editors, two rounds each, their means d apart around 0.10. In each round the
	// three are +0.01, +0.01 and -0.02 from their own means, then the reverse: nothing is
	// left over for the round, so neither is easier than the other.
	//
	// s2 is the squared residuals, 2(0.01²) + 2(0.01²) + 2(0.02²) = 0.0012, over three
	// degrees of freedom: 0.0004. However far apart the three are, each is rated its own
	// mean, give or take sqrt(0.0004/2).
	for _, d := range []float64{0.01, 0.015, 0.02, 1} {
		var es []entryRow
		for id, g := range map[string][2]float64{"a": {0.10 - d + 0.01, 0.10 - d - 0.01}, "b": {0.11, 0.09}, "c": {0.10 + d - 0.02, 0.10 + d + 0.02}} {
			es = append(es, entryRow{RoundIdx: 1, EditorID: id, Gain: g[0], Valid: true}, entryRow{RoundIdx: 2, EditorID: id, Gain: g[1], Valid: true})
		}
		rt, diff := fitRatings(es, 1, 0)
		if !near(diff[1], 0) || !near(diff[2], 0) {
			t.Fatalf("d = %v: the rounds are taken to differ: %v", d, diff)
		}
		for id, off := range map[string]float64{"a": -d, "b": 0, "c": d} {
			if r := rt[id]; !near(r.Adjusted, 0.10+off) || !near(r.SE, math.Sqrt(0.0004/2)) || r.N != 2 {
				t.Fatalf("d = %v: %s is %+v, want its own mean %.4f, ±%.5f", d, id, *r, 0.10+off, math.Sqrt(0.0004/2))
			}
		}
	}

	// Two editors, four rounds each, far apart: each is rated where it stands. s2 is
	// 8(0.01²) over 6 degrees of freedom, and a rating rests on four rounds.
	var es []entryRow
	for r := 1; r <= 4; r++ {
		wobble := 0.01 * float64(2*(r%2)-1)
		es = append(es, entryRow{RoundIdx: r, EditorID: "a", Gain: 0.10 + wobble, Valid: true}, entryRow{RoundIdx: r, EditorID: "b", Gain: 0.30 - wobble, Valid: true})
	}
	rt, _ := fitRatings(es, 1, 0)
	if se := math.Sqrt(0.0008 / 6 / 4); !near(rt["a"].Adjusted, 0.10) || !near(rt["b"].Adjusted, 0.30) || !near(rt["a"].SE, se) || !near(rt["b"].SE, se) {
		t.Fatalf("two editors, four rounds each: %+v %+v", *rt["a"], *rt["b"])
	}

	// One round each says nothing about how much a round's gain moves: that is then what the
	// season measured before it began, which the caller gives. A rating that rests on one
	// round is that round's gain, and is that unsure.
	rt, _ = fitRatings([]entryRow{{RoundIdx: 1, EditorID: "a", Valid: true}, {RoundIdx: 1, EditorID: "b", Gain: 0.5, Valid: true}}, 0.0004, 0)
	if !near(rt["a"].Adjusted, 0) || !near(rt["b"].Adjusted, 0.5) || !near(rt["a"].SE, 0.02) || !near(rt["b"].SE, 0.02) {
		t.Fatalf("one round each: %+v %+v", *rt["a"], *rt["b"])
	}
	// What the season took a round to move by counts beside its own rounds, as so many rounds'
	// worth: here 6 degrees of freedom at 0.0008/6 and a prior of 2 at 0.0005 make 0.0018/8.
	rt, _ = fitRatings(es, 0.0005, 2)
	if se := math.Sqrt(0.0018 / 8 / 4); !near(rt["a"].SE, se) {
		t.Fatalf("the prior spread, counted as two rounds: ±%v, want ±%v", rt["a"].SE, se)
	}
	// The season's file says it for the scales it was measured at, as a spread.
	if got := (&App{cfg: Config{Scale: "micro"}, season: &Season{Noise: map[string]any{"micro": map[string]any{"gain_sd": 0.02}}}}).roundNoise(); !near(got, 0.0004) {
		t.Fatalf("the noise the season measured: %v", got)
	}
	if got := (&App{cfg: Config{Scale: "micro"}, season: &Season{Noise: map[string]any{"micro": map[string]any{"gain_sd": 0.02, "round_sd": 0.03}}}}).roundNoise(); !near(got, 0.0009) {
		t.Fatalf("what a round moves by, when the season says: %v", got)
	}
	if got := (&App{cfg: Config{Scale: "nano"}, season: &Season{Noise: map[string]any{"micro": map[string]any{"gain_sd": 0.02}}}}).roundNoise(); got != noiseFloor {
		t.Fatalf("the noise at a scale the season did not measure: %v", got)
	}
}

// Who else is on the board is no part of a rating. These are Season 2's first two rounds as
// they were played: the organisers' eight reference curators in both, four of them far below
// the unedited pages, and three new curators in the second, each of which gained about 0.07
// and beat every reference but one. Drawn towards the mean of that board, as ratings once
// were, the three stood at -0.02, under references they had beaten.
func TestANewCuratorIsRatedWhatItsRoundSaidWhoeverElseIsOnTheBoard(t *testing.T) {
	refs := map[string][2]float64{
		"mimic": {0.05111, 0.08577}, "fineweb2": {0.05168, 0.06570}, "tidy": {0.03664, 0.04610}, "identity": {-0.00375, 0.00273},
		"english": {-0.24329, -0.23121}, "dclm": {-0.30176, -0.32550}, "fineweb-edu": {-0.32799, -0.32509}, "heuristic": {-0.46182, -0.47836},
	}
	news := map[string]float64{"glide-native-excerpts": 0.07463, "glide-exact-replay": 0.07198, "multilingual-structure-mix": 0.06790}
	board := func(without ...string) map[string]*Rating {
		var es []entryRow
		for name, g := range refs {
			if !strings.Contains(strings.Join(without, " "), name) {
				es = append(es, entryRow{RoundIdx: 1, EditorID: name, Gain: g[0], Valid: true}, entryRow{RoundIdx: 2, EditorID: name, Gain: g[1], Valid: true})
			}
		}
		for name, g := range news {
			es = append(es, entryRow{RoundIdx: 2, EditorID: name, Gain: g, Valid: true})
		}
		rt, _ := fitRatings(es, 0.0003, priorRounds)
		return rt
	}
	rt := board()
	for name, g := range news {
		r := rt[name]
		// Its one round, less the little that round was easier than the first.
		if math.Abs(r.Adjusted-g) > 0.005 || r.N != 1 {
			t.Fatalf("%s gained %.5f in its one round and is rated %.5f", name, g, r.Adjusted)
		}
		// It beat these in the round they shared, and stands above them.
		for _, beaten := range []string{"fineweb2", "tidy", "identity", "english"} {
			if r.Adjusted <= rt[beaten].Adjusted {
				t.Fatalf("%s (%.5f) is rated under %s (%.5f), which it beat", name, r.Adjusted, beaten, rt[beaten].Adjusted)
			}
		}
		// One round is less to go on than two, and its ± says so.
		if !(r.SE > rt["mimic"].SE) || math.Abs(r.SE/rt["mimic"].SE-math.Sqrt2) > 1e-9 {
			t.Fatalf("%s rests on one round (±%.5f) and mimic on two (±%.5f)", name, r.SE, rt["mimic"].SE)
		}
	}
	// With the four weakest references off the board, a new curator's rating moves only by
	// what they said of how hard the two rounds were: a few thousandths, and not a tenth.
	lean := board("english", "dclm", "fineweb-edu", "heuristic")
	for name := range news {
		if d := math.Abs(lean[name].Adjusted - rt[name].Adjusted); d > 0.01 {
			t.Fatalf("%s is rated %.5f with the weak references on the board and %.5f without", name, rt[name].Adjusted, lean[name].Adjusted)
		}
	}
}

// A failed round marks its own curator down, and says nothing about how much a round's gain
// moves: nobody else's standard error is any wider for it.
func TestAFailedRoundIsAPenaltyAndNotNoise(t *testing.T) {
	// Two editors, six rounds, in which they wobble by a thousandth of a bit; in the last, b
	// fails, and counts as zero.
	var without []entryRow
	for r := 1; r <= 6; r++ {
		wobble := 0.001 * float64(2*(r%2)-1)
		without = append(without, entryRow{RoundIdx: r, EditorID: "a", Gain: 0.10 + wobble, Valid: true})
		if r < 6 {
			without = append(without, entryRow{RoundIdx: r, EditorID: "b", Gain: 0.08 - wobble, Valid: true})
		}
	}
	with := append(append([]entryRow(nil), without...), entryRow{RoundIdx: 6, EditorID: "b", Gain: 0, Valid: false})
	for _, prior := range []float64{0, priorRounds} {
		clean, _ := fitRatings(without, noiseFloor, prior)
		failed, _ := fitRatings(with, noiseFloor, prior)
		if math.Abs(failed["a"].SE-clean["a"].SE) > 1e-12 {
			t.Fatalf("prior %v: b's failed round moved a's standard error from %v to %v", prior, clean["a"].SE, failed["a"].SE)
		}
		// The noise is the same, and b's rating rests on one round more.
		if want := clean["b"].SE * math.Sqrt(5.0/6.0); math.Abs(failed["b"].SE-want) > 1e-12 {
			t.Fatalf("prior %v: b's standard error with its failed round is %v, want %v", prior, failed["b"].SE, want)
		}
		if b := failed["b"]; b.N != 6 || b.Invalid != 1 || !(b.Adjusted < clean["b"].Adjusted-0.01) {
			t.Fatalf("prior %v: the failed round was not held against b: %+v, and without it %+v", prior, *b, *clean["b"])
		}
	}
}

// A place on the ladder goes to what a curator has shown it is at least worth. Three rounds
// with a better mean than sixteen steady ones stand higher on the board, are marked as too
// close to call, and do not take the place.
func TestThreeLuckyRoundsDoNotTakeALadderPlaceFromManySteadyOnes(t *testing.T) {
	pub, priv := newKey()
	cfg := testConfig(t, pub)
	cfg.LadderSize = 2
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	anchor := w.join("anchor-one", priv)
	ids := map[string]string{}
	enter := func(names ...string) {
		for _, name := range names {
			ids[name] = w.submit(w.join("owner-of-"+name, nil), name, "print('"+name+"')")
		}
	}
	// Two curators hold the ladder, and take turns having the better round, a tenth of a bit
	// either way, so that no round is easy or hard. Three more play their five rounds far
	// below and leave: they are the field. Nothing here beats the unedited data, so nobody
	// takes the record, and a place on the ladder with it.
	gain := func(name string, round int) float64 {
		turn := 0.10 * float64(2*(round%2)-1)
		switch name {
		case "best":
			return -0.10 + turn
		case "steady":
			return -0.20 - turn
		case "lucky":
			return -0.18
		}
		return -0.60
	}
	loss := func(name string, round int) float64 { return 6.0 - gain(name, round) }
	// One machine runs every curator of every round, which is more scripts than an account
	// may upload in an hour: each round is played in an hour of its own.
	play := func(rounds int) {
		for ; rounds > 0; rounds-- {
			w.app.limits = newLimiter()
			w.play(anchor, 1, loss)
		}
	}
	enter("best", "steady", "field-a", "field-b", "field-c")
	play(13)
	// A newcomer plays three rounds, each better than the steady curator's average.
	enter("lucky")
	play(3)

	b := w.app.board()
	best, steady, lucky := b.ByID[ids["best"]], b.ByID[ids["steady"]], b.ByID[ids["lucky"]]
	if steady.Rounds != 16 || lucky.Rounds != 3 || lucky.Provisional || len(b.Rated) != 6 {
		t.Fatalf("the board: steady %+v, lucky %+v, %d ranked", *steady, *lucky, len(b.Rated))
	}
	if !(*lucky.MeanGain > *steady.MeanGain && *lucky.Adjusted > *steady.Adjusted) {
		t.Fatalf("this test is not testing anything: lucky's mean %v, steady's %v", *lucky.Adjusted, *steady.Adjusted)
	}
	// Each is rated its own mean, whatever the field below them does, and the newcomer's is
	// the better: it stands above the steady curator on the board.
	if !(*lucky.Rating == *lucky.Adjusted && *steady.Rating == *steady.Adjusted && *lucky.Rating > *steady.Rating && lucky.Rank == 2 && steady.Rank == 3) {
		t.Fatalf("lucky is rated %v from a mean of %v (rank %d), steady %v from %v (rank %d)", *lucky.Rating, *lucky.Adjusted, lucky.Rank, *steady.Rating, *steady.Adjusted, steady.Rank)
	}
	// But what it has shown it is at least worth is less, and that is what fills the ladder.
	if !(*lucky.SE > *steady.SE && *lucky.Lower < *steady.Lower) || math.Abs(*lucky.Lower-(*lucky.Rating-1.28**lucky.SE)) > 1e-12 {
		t.Fatalf("lucky is at least %v (±%v), steady at least %v (±%v)", *lucky.Lower, *lucky.SE, *steady.Lower, *steady.SE)
	}
	if len(b.Ladder) != 2 || !best.Ladder || !steady.Ladder || lucky.Ladder {
		t.Fatalf("the ladder: best %v, steady %v, lucky %v", best.Ladder, steady.Ladder, lucky.Ladder)
	}
	// So the next round runs the steady curator again, from the ladder; the newcomer goes on
	// qualifying.
	var next struct {
		Expected []expectedRow `json:"expected"`
	}
	var scheduled int
	w.app.st.view(func(s *State) { scheduled = s.roundWithStatus("scheduled").Idx })
	anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, scheduled), nil, &next)
	why := map[string]string{}
	for _, row := range next.Expected {
		why[row.Name] = row.Why
	}
	if len(why) != 3 || why["best"] != "ladder" || why["steady"] != "ladder" || why["lucky"] != "qualifying" {
		t.Fatalf("who the next round would enter: %v", why)
	}

	// The newcomer is too close to the best to call, and the board says so: it is within two
	// standard errors of the gap. The steady curator is as close as that to the newcomer, the
	// row above it, and not to the best, which is the one a place at the top is shared with:
	// it is third. Ties do not chain. The field is far below, and its three are level.
	if best.Tied || !lucky.Tied || steady.Tied || best.TiedAt != 1 || lucky.TiedAt != 1 || steady.TiedAt != 0 || steady.Rank != 3 {
		t.Fatalf("the ties at the top: best %v %d, lucky %v %d, steady %v %d (rank %d)", best.Tied, best.TiedAt, lucky.Tied, lucky.TiedAt, steady.Tied, steady.TiedAt, steady.Rank)
	}
	within := func(above, below *BoardRow) bool {
		return *above.Rating-*below.Rating <= 2*math.Sqrt(*above.SE**above.SE+*below.SE**below.SE)
	}
	if !within(best, lucky) || !within(lucky, steady) || within(best, steady) {
		t.Fatalf("this test is written for a chain: best and lucky too close to call %v, lucky and steady %v, best and steady %v",
			within(best, lucky), within(lucky, steady), within(best, steady))
	}
	for i, name := range []string{"field-a", "field-b", "field-c"} {
		if r := b.ByID[ids[name]]; r.Rank < 4 || r.TiedAt != 4 || r.Tied != (r.Rank > 4) {
			t.Fatalf("%s (%d): rank %d, tied %v at %d", name, i, r.Rank, r.Tied, r.TiedAt)
		}
	}
	var lb struct {
		Editors []map[string]any `json:"editors"`
	}
	anchor.do("GET", "/api/v1/leaderboard", nil, &lb)
	if len(lb.Editors) != 6 {
		t.Fatalf("the leaderboard has %d rows", len(lb.Editors))
	}
	for i, want := range []struct {
		name string
		tied any
	}{{"best", nil}, {"lucky", true}, {"steady", nil}} {
		row := lb.Editors[i]
		if row["name"] != want.name || row["tied"] != want.tied || row["rank"] != float64(i+1) || row["adjusted"] == nil || row["lower"] == nil || row["se"] == nil {
			t.Fatalf("row %d of the leaderboard: %v", i+1, row)
		}
		if lower := row["rating"].(float64) - 1.28*row["se"].(float64); math.Abs(row["lower"].(float64)-lower) > 2e-5 {
			t.Fatalf("%s: lower is %v, and its rating less 1.28 standard errors is %v", want.name, row["lower"], lower)
		}
	}
	var ed struct {
		Editor map[string]any `json:"editor"`
	}
	anchor.do("GET", "/api/v1/editors/"+ids["lucky"], nil, &ed)
	if e := ed.Editor; e["tied"] != true || e["ladder"] != false || e["adjusted"] != places5(*lucky.Adjusted) || e["lower"] != places5(*lucky.Lower) {
		t.Fatalf("the curator's own JSON: %v", e)
	}
	_, page := w.page("/leaderboard")
	if n := strings.Count(page, `<span class="rank">=1</span>`); n != 2 {
		t.Fatalf("the leaderboard marks %d curators as level at the top, not 2", n)
	}
	if n := strings.Count(page, `<span class="rank">3</span>`); n != 1 {
		t.Fatalf("the leaderboard has %d curators third alone, not 1", n)
	}
	if n := strings.Count(page, `<span class="rank">=4</span>`); n != 3 {
		t.Fatalf("the leaderboard marks %d curators as level in fourth place, not 3", n)
	}
	if !strings.Contains(page, "means too close to call") {
		t.Fatal("the leaderboard does not say what = means")
	}
	if _, page := w.page("/editors/" + ids["lucky"][:12]); !strings.Contains(page, "<b>=1</b>") {
		t.Fatal("the curator's page does not show its place as level")
	}
	if _, page := w.page("/editors/" + ids["steady"][:12]); !strings.Contains(page, "<b>3</b>") {
		t.Fatal("the page of the curator in third place does not show it there")
	}
	w.check()
}

// Failing is never the better move: a failed entry counts as the lowest gain any entry that
// finished got in its round, or zero if that is lower.
func TestAFailedEntryCountsAsTheLowestGainOfItsRound(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, carol := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	w.submit(alice, "good", "print(1)")
	w.submit(bob, "poor", "print(2)")
	id := w.submit(carol, "broken", "raise SystemExit(1)")
	// round plays a round in which the curators named finish, with these losses against an
	// unedited 6.0, and every other one writes no script.
	round := func(losses map[string]float64) int {
		idx := w.draw()
		for {
			c := w.claim(anchor, true)
			if c.Job == nil {
				break
			}
			loss, finishes := losses[c.Job.Editor.Name]
			switch {
			case c.Job.Kind == "grade":
				if code, status := w.report(anchor, c, loss); code != 200 {
					t.Fatalf("HTTP %d %s", code, status)
				}
			case finishes:
				w.edit(anchor, c, fmt.Sprintf("{\"doc\": %d, \"by\": %q}\n", idx, c.Job.Editor.Name))
			default:
				w.edit(anchor, c, "")
			}
		}
		if w.settled(idx) != 3 {
			t.Fatalf("round %d did not settle", idx)
		}
		return idx
	}
	row := func() *BoardRow {
		if r := w.app.board().ByID[id]; r != nil {
			return r
		}
		t.Fatal("the curator is not on the board")
		return nil
	}

	// Both curators that finished did worse than the unedited data. Counted as zero, the one
	// that failed would have come first.
	first := round(map[string]float64{"good": 6.10, "poor": 6.30})
	e := w.entry(first, "broken")
	if e.Valid || e.Gain == nil || *e.Gain != 0 || e.CountsAs == nil || *e.CountsAs != -0.3 {
		t.Fatalf("the entry that failed: gain %v, counts as %v", e.Gain, e.CountsAs)
	}
	if es := w.entries(carol, first); es[0].Name != "good" || es[1].Name != "poor" || es[2].Name != "broken" || es[0].CountsAs != nil {
		t.Fatalf("the round lists %s, %s, %s", es[0].Name, es[1].Name, es[2].Name)
	}
	if r := row(); r.Rounds != 1 || r.Invalid != 1 || math.Abs(*r.MeanGain+0.3) > 1e-9 || math.Abs(*r.Adjusted+0.3) > 1e-9 {
		t.Fatalf("after failing in a round whose lowest gain was -0.3: %+v", *r)
	}
	// And it says nothing about the round: the two that finished are where they would be
	// without it.
	for name, want := range map[string]float64{"good": -0.1, "poor": -0.3} {
		for _, r := range w.app.board().Provisional {
			if r.Name == name && math.Abs(*r.Adjusted-want) > 1e-9 {
				t.Fatalf("%s's mean is %v, not %v", name, *r.Adjusted, want)
			}
		}
	}
	rid := roundID(w.app.season.ID, first)
	for _, path := range []string{"/rounds/" + rid, "/editors/" + id[:12]} {
		if _, page := w.page(path); !strings.Contains(page, `<span class="neg">failed</span>: counts as −0.300, the round's lowest`) {
			t.Fatalf("%s does not say what the failed entry counts as", path)
		}
	}

	// Both did better than the unedited data: zero is lower than the round's lowest.
	second := round(map[string]float64{"good": 5.80, "poor": 5.90})
	if e := w.entry(second, "broken"); e.Valid || e.CountsAs == nil || *e.CountsAs != 0 {
		t.Fatalf("the entry that failed in a round nobody lost: counts as %v", e.CountsAs)
	}
	if r := row(); r.Rounds != 2 || r.Invalid != 2 || math.Abs(*r.MeanGain+0.15) > 1e-9 {
		t.Fatalf("after failing twice: %+v", *r)
	}
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, second)); !strings.Contains(page, `<span class="neg">failed</span>: counts as zero`) {
		t.Fatal("the second round's page does not say the failed entry counts as zero")
	}
	// Its owner is told with its latest result, and anyone with every result.
	var me struct {
		Editors []struct {
			Last map[string]any `json:"last"`
		} `json:"editors"`
	}
	carol.do("GET", "/api/v1/me", nil, &me)
	if len(me.Editors) != 1 || me.Editors[0].Last["valid"] != false || me.Editors[0].Last["counts_as"] != 0.0 {
		t.Fatalf("/api/v1/me: %+v", me.Editors)
	}
	var ed struct {
		Results []entryJ `json:"results"`
	}
	carol.do("GET", "/api/v1/editors/"+id, nil, &ed)
	if len(ed.Results) != 2 || ed.Results[1].CountsAs == nil || *ed.Results[1].CountsAs != -0.3 || *ed.Results[0].CountsAs != 0 {
		t.Fatalf("the curator's results: %+v", ed.Results)
	}
	w.check()
}

// A failed run is never better than a finished one inside an entry either: beside a finished
// run below zero it counts as that run, and beside one above zero as zero.
func TestAFailedRunStandsAsTheEntrysWorstFinishedRun(t *testing.T) {
	bad, worse, failed := &Run{Valid: true, Gain: -0.2}, &Run{Valid: true, Gain: -0.3}, &Run{}
	if got := standIns([]*Run{failed, bad}); got[failed] != bad || len(got) != 1 {
		t.Fatalf("a failed run beside one at -0.2: %v", got)
	}
	if got := standIns([]*Run{bad, failed, worse}); got[failed] != worse {
		t.Fatalf("a failed run beside two below zero stands as the worse: %v", got)
	}
	good := &Run{Valid: true, Gain: 0.2}
	if got := standIns([]*Run{failed, good}); len(got) != 0 {
		t.Fatalf("a failed run beside one above zero counts zero: %v", got)
	}
	if got := standIns([]*Run{failed, {}}); len(got) != 0 {
		t.Fatalf("nothing finished: %v", got)
	}
}

// A curator that has failed every round has no number to show: what its rating would be is
// the field's mean, with its penalties drawn towards it. It is listed last.
func TestACuratorThatNeverFinishedIsListedLastWithNoNumber(t *testing.T) {
	f := func(x float64) *float64 { return &x }
	good := &BoardRow{Name: "good", Rounds: 3, Rating: f(0.05)}
	low := &BoardRow{Name: "low", Rounds: 3, Rating: f(0.01)}
	never := &BoardRow{Name: "never", Rounds: 2, Invalid: 2, Rating: f(0.02), Provisional: true}
	once := &BoardRow{Name: "once", Rounds: 2, Invalid: 1, Rating: f(0.03), Provisional: true}
	if never.Never() == false || once.Never() || good.Never() || (&BoardRow{}).Never() {
		t.Fatal("which curators have never finished a round")
	}
	var names []string
	for _, r := range standing([]*BoardRow{good, low}, []*BoardRow{never, once}) {
		names = append(names, r.Name)
	}
	if got := strings.Join(names, " "); got != "good once low never" {
		t.Fatalf("the board's order: %s", got)
	}
}

// Review 33 P1 #1: checkRecord (and recordRace.Clear) MUST apply each contributing round's
// recorded scale margin rather than the current server process's STET_SCALE, keeping published
// Advantage in raw gain units across both restart directions and mixed-scale histories.
func TestReview33RecordUsesPerRoundMarginAcrossRestartsAndMixedScaleHistory(t *testing.T) {
	runCheck := func(t *testing.T, serverScale string, roundScales []string, gains []float64) (awarded bool, adv, p float64, clear bool) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.cfg.Scale = serverScale
		anchor := w.join("anchor-one", priv)
		edID := w.submit(anchor, "challenger", "print(1)")

		err := w.app.st.update(context.Background(), func(tx *Tx) error {
			if ed := tx.s.Editors[edID]; ed != nil {
				ed.Public = true
				ed.Status = "active"
				tx.saveEditor(ed)
			}
			baseLoss := 5.5
			for i, sc := range roundScales {
				idx := i + 1
				tx.saveRound(&Round{
					Idx: idx, Status: "settled", Scale: sc, BaselineLoss: &baseLoss,
					DrawnAt: now() - int64(3600*(len(roundScales)-i)),
					ExamAt:  now() - int64(3500*(len(roundScales)-i)),
				})
				tx.saveEntry(&Entry{
					RoundIdx: idx, EditorID: edID, Status: "settled",
					Valid: true, Gain: gains[i], SettledAt: now() - int64(3400*(len(roundScales)-i)),
				})
			}
			w.app.checkRecord(tx, len(roundScales))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		w.app.st.view(func(s *State) {
			awarded = s.holder() != nil && s.holder().EditorID == edID
			lk := s.Looks[lookKey(edID, "", 5)]
			if lk == nil {
				t.Fatalf("expected look@5 recorded")
			}
			adv, p = lk.Advantage, lk.P
			if awarded {
				rec := s.holder()
				if math.Abs(rec.Advantage-adv) > 1e-12 || math.Abs(rec.P-p) > 1e-12 {
					t.Fatalf("record Advantage/P mismatch with look: rec=(%v,%v) look=(%v,%v)", rec.Advantage, rec.P, adv, p)
				}
			}
			// Also check recordRace.Clear before any holder existed (when not awarded).
			b := w.app.computeBoard(s)
			if !awarded {
				race := w.app.recordRace(s, b, 10)
				if len(race.Rows) != 1 {
					t.Fatalf("expected 1 challenger row in recordRace, got %+v", race.Rows)
				}
				clear = race.Rows[0].Clear
			} else {
				clear = true
			}
		})
		return awarded, adv, p, clear
	}

	// 1. Exact reviewer repro: 5 settled `nano` rounds with gains 0.0098..0.0102.
	//    nano margin = 0.030 (no record), standard margin = 0.005 (would falsely award record!).
	//    MUST yield NO record under BOTH STET_SCALE=nano AND STET_SCALE=standard!
	nanoScales := []string{"nano", "nano", "nano", "nano", "nano"}
	gains01 := []float64{0.0098, 0.0099, 0.0100, 0.0101, 0.0102}
	awN, advN, pN, clN := runCheck(t, "nano", nanoScales, gains01)
	awS, advS, pS, clS := runCheck(t, "standard", nanoScales, gains01)
	if awN || awS {
		t.Fatalf("5 nano rounds at 0.010 gain MUST NOT award record on either scale: nano=%v standard=%v", awN, awS)
	}
	if clN || clS {
		t.Fatalf("recordRace.Clear MUST be false on 5 nano rounds at 0.010 gain: nano=%v standard=%v", clN, clS)
	}
	if math.Abs(advN-0.0100) > 1e-9 || math.Abs(advS-advN) > 1e-12 || math.Abs(pS-pN) > 1e-12 {
		t.Fatalf("nano->standard restart changed record test: nano=(adv=%v,p=%v) standard=(adv=%v,p=%v)", advN, pN, advS, pS)
	}

	// 2. Reverse restart direction: 5 settled `standard` rounds with gains 0.0098..0.0102.
	//    MUST award record under BOTH STET_SCALE=standard AND STET_SCALE=nano with identical Advantage & P!
	stdScales := []string{"standard", "standard", "standard", "standard", "standard"}
	awSS, advSS, pSS, _ := runCheck(t, "standard", stdScales, gains01)
	awSN, advSN, pSN, _ := runCheck(t, "nano", stdScales, gains01)
	if !awSS || !awSN {
		t.Fatalf("5 standard rounds at 0.010 gain MUST award record on both scales: standard=%v nano=%v", awSS, awSN)
	}
	if math.Abs(advSS-0.0100) > 1e-9 || math.Abs(advSN-advSS) > 1e-12 || math.Abs(pSN-pSS) > 1e-12 {
		t.Fatalf("standard->nano restart changed record test: standard=(adv=%v,p=%v) nano=(adv=%v,p=%v)", advSS, pSS, advSN, pSN)
	}

	// 3. Genuinely mixed-scale history: 2 nano (0.030) + 3 standard (0.005) -> effective margin 0.015.
	mixedScales := []string{"nano", "nano", "standard", "standard", "standard"}
	// 3a. Mean gain 0.012 beats standard's 0.005 but NOT mixed 0.015 -> NO record on either scale, raw Advantage = 0.012!
	gains12 := []float64{0.0118, 0.0119, 0.0120, 0.0121, 0.0122}
	awM1S, advM1S, pM1S, clM1S := runCheck(t, "standard", mixedScales, gains12)
	awM1N, advM1N, pM1N, clM1N := runCheck(t, "nano", mixedScales, gains12)
	if awM1S || awM1N || clM1S || clM1N {
		t.Fatalf("mixed history at 0.012 (< 0.015 mixed margin) must not award/clear: std=(%v,%v) nano=(%v,%v)", awM1S, clM1S, awM1N, clM1N)
	}
	if math.Abs(advM1S-0.0120) > 1e-9 || math.Abs(advM1N-advM1S) > 1e-12 || math.Abs(pM1N-pM1S) > 1e-12 {
		t.Fatalf("mixed history (below) differed across scales: std=(adv=%v,p=%v) nano=(adv=%v,p=%v)", advM1S, pM1S, advM1N, pM1N)
	}

	// 3b. Mean gain 0.045 (exceeds both per-round margins cleanly) -> awards record on BOTH scales
	//     with raw Advantage = 0.0450 (NOT corrupted by adding back a.cfg.Scale's margin!).
	gains45 := []float64{0.0448, 0.0449, 0.0450, 0.0451, 0.0452}
	awM2S, advM2S, pM2S, _ := runCheck(t, "standard", mixedScales, gains45)
	awM2N, advM2N, pM2N, _ := runCheck(t, "nano", mixedScales, gains45)
	if !awM2S || !awM2N {
		t.Fatalf("mixed history at 0.045 must award record on both scales: std=%v nano=%v", awM2S, awM2N)
	}
	if math.Abs(advM2S-0.0450) > 1e-9 || math.Abs(advM2N-advM2S) > 1e-12 || math.Abs(pM2N-pM2S) > 1e-12 {
		t.Fatalf("mixed history (above) raw Advantage or P differed across scales: std=(adv=%v,p=%v) nano=(adv=%v,p=%v)", advM2S, pM2S, advM2N, pM2N)
	}
}
