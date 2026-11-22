package main

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// A rating is the adjusted mean drawn towards the field's, in numbers that can be checked by
// hand.
func TestARatingIsDrawnTowardsTheFieldsTheMoreTheFewerItsRounds(t *testing.T) {
	near := func(got, want float64) bool { return math.Abs(got-want) < 1e-9 }
	// Three editors, two rounds each, their means d apart around 0.10. In each round the
	// three are +0.01, +0.01 and -0.02 from their own means, then the reverse: nothing is
	// left over for the round, so neither is easier than the other.
	//
	// s2 is the squared residuals, 2(0.01²) + 2(0.01²) + 2(0.02²) = 0.0012, over three
	// degrees of freedom: 0.0004. The means have variance d², of which s2/n = 0.0002 is
	// noise, so tau2 = d² - 0.0002 and k = 0.0004/tau2, kept within 1 and 8.
	for _, c := range []struct {
		d, k float64
		why  string
	}{
		{0.02, 2, "tau2 = 0.0002, so k = 2"},
		{1, 1, "editors far apart: k would be next to nothing, and is 1"},
		{0.015, 8, "tau2 = 0.000025: k would be 16, and is 8"},
		{0.01, 4, "the means are no further apart than noise would put them: tau2 is 0, and k the usual 4"},
	} {
		var es []entryRow
		for id, g := range map[string][2]float64{"a": {0.10 - c.d + 0.01, 0.10 - c.d - 0.01}, "b": {0.11, 0.09}, "c": {0.10 + c.d - 0.02, 0.10 + c.d + 0.02}} {
			es = append(es, entryRow{RoundIdx: 1, EditorID: id, Gain: g[0], Valid: true}, entryRow{RoundIdx: 2, EditorID: id, Gain: g[1], Valid: true})
		}
		rt, diff := fitRatings(es, 1, 0)
		if !near(diff[1], 0) || !near(diff[2], 0) {
			t.Fatalf("d = %v: the rounds are taken to differ: %v", c.d, diff)
		}
		shrink, se := 2/(2+c.k), math.Sqrt(0.0004/(2+c.k))
		for id, off := range map[string]float64{"a": -c.d, "b": 0, "c": c.d} {
			r := rt[id]
			if !near(r.Adjusted, 0.10+off) || !near(r.Shrunk, 0.10+shrink*off) || !near(r.SE, se) || r.N != 2 {
				t.Fatalf("d = %v (%s): %s is %+v, want its mean %.4f drawn to %.5f, ±%.5f", c.d, c.why, id, *r, 0.10+off, 0.10+shrink*off, se)
			}
		}
	}

	// With fewer than three editors to say how far apart editors are, k is 4: after four
	// rounds a rating is midway between the editor's own mean and the field's. s2 is
	// 8(0.01²) over 6 degrees of freedom.
	var es []entryRow
	for r := 1; r <= 4; r++ {
		wobble := 0.01 * float64(2*(r%2)-1)
		es = append(es, entryRow{RoundIdx: r, EditorID: "a", Gain: 0.10 + wobble, Valid: true}, entryRow{RoundIdx: r, EditorID: "b", Gain: 0.30 - wobble, Valid: true})
	}
	rt, _ := fitRatings(es, 1, 0)
	if se := math.Sqrt(0.0008 / 6 / 8); !near(rt["a"].Shrunk, 0.15) || !near(rt["b"].Shrunk, 0.25) || !near(rt["a"].SE, se) || !near(rt["b"].SE, se) {
		t.Fatalf("two editors, four rounds each: %+v %+v", *rt["a"], *rt["b"])
	}

	// The field's mean counts every round once: three rounds at 0 and one at 0.4 make 0.1,
	// not 0.2. With three rounds the first is 3/7 of the way from it to its own mean, and
	// with one the second is 1/5.
	rt, _ = fitRatings([]entryRow{{RoundIdx: 1, EditorID: "long", Valid: true}, {RoundIdx: 2, EditorID: "long", Valid: true}, {RoundIdx: 3, EditorID: "long", Valid: true},
		{RoundIdx: 1, EditorID: "new", Gain: 0.4, Valid: true}}, 1, 0)
	if !near(rt["long"].Shrunk, 0.1-0.1*3/7) || !near(rt["new"].Shrunk, 0.1+0.3/5) {
		t.Fatalf("three rounds and one: %+v %+v", *rt["long"], *rt["new"])
	}

	// One round each says nothing about how much a round's gain moves: that is then what the
	// season measured before it began, which the caller gives.
	rt, _ = fitRatings([]entryRow{{RoundIdx: 1, EditorID: "a", Valid: true}, {RoundIdx: 1, EditorID: "b", Gain: 0.5, Valid: true}}, 0.0005, 0)
	if !near(rt["a"].Shrunk, 0.20) || !near(rt["b"].Shrunk, 0.30) || !near(rt["a"].SE, 0.01) {
		t.Fatalf("one round each: %+v %+v", *rt["a"], *rt["b"])
	}
	// What the season took a round to move by counts beside its own rounds, as so many rounds'
	// worth: here 6 degrees of freedom at 0.0008/6 and a prior of 2 at 0.0005 make 0.0018/8.
	rt, _ = fitRatings(es, 0.0005, 2)
	if se := math.Sqrt(0.0018 / 8 / 8); !near(rt["a"].SE, se) {
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
			return -0.16
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
	// Its three rounds count for less than its own mean says: it is drawn further towards
	// the field than the curator with sixteen. It still stands above it.
	if !(*lucky.Rating < *lucky.Adjusted && *lucky.Adjusted-*lucky.Rating > *steady.Adjusted-*steady.Rating && *lucky.Rating > *steady.Rating && lucky.Rank == 2 && steady.Rank == 3) {
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

	// The three are too close to call, and the board says so: each is within two standard
	// errors of the gap of the one above. The field is not, and its three are level.
	if best.Tied || !lucky.Tied || !steady.Tied || best.TiedAt != 1 || lucky.TiedAt != 1 || steady.TiedAt != 1 {
		t.Fatalf("the ties at the top: best %v %d, lucky %v %d, steady %v %d", best.Tied, best.TiedAt, lucky.Tied, lucky.TiedAt, steady.Tied, steady.TiedAt)
	}
	if gap, two := *best.Rating-*lucky.Rating, 2*math.Sqrt(*best.SE**best.SE+*lucky.SE**lucky.SE); gap > two {
		t.Fatalf("lucky is %v behind best, and tied within %v", gap, two)
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
	}{{"best", nil}, {"lucky", true}, {"steady", true}} {
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
	if n := strings.Count(page, `<span class="rank">=1</span>`); n != 3 {
		t.Fatalf("the leaderboard marks %d curators as level at the top, not 3", n)
	}
	if n := strings.Count(page, `<span class="rank">=4</span>`); n != 3 {
		t.Fatalf("the leaderboard marks %d curators as level in fourth place, not 3", n)
	}
	if !strings.Contains(page, "means too close to call") {
		t.Fatal("the leaderboard does not say what = means")
	}
	if _, page := w.page("/editors/" + ids["steady"][:12]); !strings.Contains(page, "<b>=1</b>") {
		t.Fatal("the curator's page does not show its place as level")
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
