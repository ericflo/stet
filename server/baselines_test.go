package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- a few machines measure a round's baseline -------------------------------------------------

// toGrade is a world with a round of n entries open, each run once by the organisers' machine:
// n scripts, each waiting for one grade. It returns the organisers' actor and the round.
func toGrade(t *testing.T, n int) (*world, *actor, int) {
	t.Helper()
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("entry-%d", i)
		w.submit(w.join("owner-"+name, nil), name, fmt.Sprintf("print(%d)", i))
	}
	idx := w.draw()
	if got := w.editAll(anchor); got != n {
		t.Fatalf("runs: %d of %d", got, n)
	}
	return w, anchor, idx
}

// volunteers joins established volunteers.
func (w *world) volunteers(handles ...string) []*actor {
	w.t.Helper()
	var out []*actor
	for _, h := range handles {
		out = append(out, w.join(h, nil))
	}
	w.established(handles...)
	return out
}

// asked has a verifier take a grade job, and fails the test unless the job says what it
// should about the round's baseline.
func (w *world) asked(a *actor, baseline bool) *claimed {
	w.t.Helper()
	c := w.claim(a, false)
	if c.Job == nil || c.Job.Kind != "grade" {
		w.t.Fatalf("%s should be grading: %+v %s", a.handle, c.Job, c.Message)
	}
	if c.Job.Baseline == nil || *c.Job.Baseline != baseline {
		w.t.Fatalf("%s's grade job should say baseline: %v, and says %v", a.handle, baseline, c.Job.Baseline)
	}
	return c
}

// send reports a grade whose student scores -loss, with the machine's own baseline (-baseline)
// or, with baseline 0, without one.
func (w *world) send(a *actor, c *claimed, loss, baseline float64) (int, string) {
	w.t.Helper()
	p := w.result(c, loss, baseline)
	if baseline == 0 {
		delete(p, "baseline")
	}
	var ack struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
	if code != 200 {
		return code, ack.Error
	}
	return code, ack.Status
}

// wanted is how many more machines the open round would ask for its baseline.
func (w *world) wanted(anchor *actor) int {
	w.t.Helper()
	work := w.work(anchor)
	if len(work.Rounds) != 1 {
		w.t.Fatalf("the work: %+v", work)
	}
	return work.Rounds[0].BaselinesWanted
}

// lapse lets a job run out: no report, and not handed back.
func (w *world) lapse(cs ...*claimed) {
	w.t.Helper()
	ctx := context.Background()
	w.app.st.update(ctx, func(tx *Tx) error {
		for _, c := range cs {
			cl := tx.s.Claims[c.Job.ID]
			cl.ExpiresAt = now() - 10
			tx.saveClaim(cl)
		}
		return nil
	})
	w.app.tick(ctx)
	w.app.st.view(func(s *State) {
		for _, c := range cs {
			if s.Claims[c.Job.ID].Status != "lapsed" {
				w.t.Fatalf("a job whose time ran out: %+v", s.Claims[c.Job.ID])
			}
		}
	})
}

// Season 3 sets baselines.per_round = 2: deterministic student init + corpus means two
// established verifiers cross-check the unedited baseline while saving two trainings per round.
func TestSeason3DefaultTwoBaselinesPerRound(t *testing.T) {
	w, anchor, idx := toGrade(t, 3)
	if w.app.season.Baselines.PerRound != 2 {
		t.Fatalf("expected Season 3 default baselines.per_round == 2, got %d", w.app.season.Baselines.PerRound)
	}
	g := w.volunteers("bob", "carol", "dave")
	if n := w.wanted(anchor); n != 2 {
		t.Fatalf("a round nobody has graded wants its two baselines: %d", n)
	}
	c0 := w.asked(g[0], true)
	c1 := w.asked(g[1], true)
	c2 := w.asked(g[2], false)
	if n := w.wanted(anchor); n != 0 {
		t.Fatalf("after 2 promises the round should want 0 more, got %d", n)
	}
	if code, status := w.send(g[2], c2, 5.5, 0); code != 200 || status != "pending" {
		t.Fatalf("dave's grade without baseline: HTTP %d %s", code, status)
	}
	if code, status := w.send(g[0], c0, 5.6, 6.00); code != 200 || status != "pending" {
		t.Fatalf("bob's grade: HTTP %d %s", code, status)
	}
	if code, status := w.send(g[1], c1, 5.6, 6.02); code != 200 || status != "settled" {
		t.Fatalf("carol's grade: HTTP %d %s", code, status)
	}
	w.check()
	if e := w.entry(idx, c2.Job.Editor.Name); !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.51) > 1e-9 {
		t.Fatalf("dave's grade against 2-machine baseline (6.01): %+v", e)
	}
}

// Review 9 P1 #2: when Season 3's two baseline graders disagree beyond tolerance (e.g. 6.00 vs
// 6.08 against 0.015 tol), zero verifiers agree with the 6.04 midpoint. The server must NOT settle
// rated against 6.04: it requests a third independent baseline even when all ordinary grading jobs
// are complete, resolves cleanly if one arrives, and closes the round unrated (with zero curator
// penalty and BaselineLoss == nil) if the disagreement cannot be resolved before ReplicaWait.
func TestDisputedTwoBaselinesRequestsThirdAndResolvesOrClosesUnrated(t *testing.T) {
	t.Run("resolves_with_third_baseline", func(t *testing.T) {
		w, anchor, idx := toGrade(t, 3)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		g := w.volunteers("bob", "carol", "dave", "erin")
		c0 := w.asked(g[0], true)
		c1 := w.asked(g[1], true)
		c2 := w.asked(g[2], false)
		if code, status := w.send(g[2], c2, 5.5, 0); code != 200 || status != "pending" {
			t.Fatalf("dave's grade: HTTP %d %s", code, status)
		}
		if code, status := w.send(g[0], c0, 5.6, 6.00); code != 200 || status != "pending" {
			t.Fatalf("bob's grade: HTTP %d %s", code, status)
		}
		// Carol reports 6.08: midpoint is 6.04 (0.04 > 0.015 from BOTH bob and carol -> 0 agree!).
		if code, status := w.send(g[1], c1, 5.6, 6.08); code != 200 || status != "pending" {
			t.Fatalf("carol's disagreeing baseline must NOT settle round: HTTP %d %s", code, status)
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "open" || rd.BaselineN != 0 || !w.app.baselineDisputed(s, rd) {
				t.Fatalf("disputed 2-baseline round state: status=%s baseN=%d disputed=%v", rd.Status, rd.BaselineN, w.app.baselineDisputed(s, rd))
			}
		})
		if wk := w.work(anchor).Rounds[0]; wk.BaselinesWanted == 0 || wk.GradesWanted != 1 {
			t.Fatalf("work must request exactly 1 tie-breaking baseline grade (not N scripts): %+v", wk)
		}
		// Erin (4th established verifier) claims: must be handed a required grade job asking for baseline!
		c3 := w.asked(g[3], true)
		erinLoss := 5.6
		if c3.Job.Editor.Name == c2.Job.Editor.Name {
			erinLoss = 5.5
		}
		if code, status := w.send(g[3], c3, erinLoss, 6.01); code != 200 {
			t.Fatalf("erin's tie-breaking baseline (6.01): HTTP %d %s", code, status)
		}
		// Now bob (6.00) and erin (6.01) agree at 6.005, while carol (6.08) is oddOut.
		// If erin did not already grade carol's script, bob re-grades it and settles the round.
		if c3.Job.Editor.Name != c1.Job.Editor.Name {
			cRegrade := w.claim(g[0], false)
			if cRegrade.Job == nil || cRegrade.Job.Editor.Name != c1.Job.Editor.Name {
				t.Fatalf("expected bob to re-grade carol's oddOut script (%s), got %+v", c1.Job.Editor.Name, cRegrade.Job)
			}
			if code, status := w.send(g[0], cRegrade, 5.6, 6.00); code != 200 || status != "settled" {
				t.Fatalf("bob's re-grade of carol's script: HTTP %d %s", code, status)
			}
		}
		w.check()
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "settled" || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.005) > 1e-6 || rd.BaselineN != 2 {
				t.Fatalf("resolved round baseline: status=%s loss=%v n=%d", rd.Status, rd.BaselineLoss, rd.BaselineN)
			}
			for _, en := range s.inRound[idx] {
				if !s.rated(en) || !en.Valid {
					t.Fatalf("resolved round entry should be rated & valid: %+v", en)
				}
			}
		})
	})

	t.Run("closes_unrated_when_unresolved", func(t *testing.T) {
		w, _, idx := toGrade(t, 3)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		g := w.volunteers("bob", "carol", "dave")
		beforeAgreed := map[string]int{}
		for _, a := range g {
			beforeAgreed[a.handle] = w.account(a.handle).Agreed
		}
		c0 := w.asked(g[0], true)
		c1 := w.asked(g[1], true)
		c2 := w.asked(g[2], false)
		if code, _ := w.send(g[2], c2, 5.5, 0); code != 200 {
			t.Fatalf("dave's grade: HTTP %d", code)
		}
		if code, _ := w.send(g[0], c0, 5.6, 6.00); code != 200 {
			t.Fatalf("bob's grade: HTTP %d", code)
		}
		if code, status := w.send(g[1], c1, 5.6, 6.08); code != 200 || status != "pending" {
			t.Fatalf("carol's disagreeing baseline must keep round open: HTTP %d %s", code, status)
		}
		// Nobody comes forward with a 3rd baseline: advance past ReplicaWait so the round closes unrated.
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "settled" || rd.BaselineLoss != nil || rd.BaselineN != 0 {
				t.Fatalf("unresolved disputed-baseline round must settle with nil baseline: status=%s loss=%v n=%d", rd.Status, rd.BaselineLoss, rd.BaselineN)
			}
			for _, en := range s.inRound[idx] {
				if s.rated(en) || !en.Void || !en.Unattended || en.Valid || en.Gain != 0 {
					t.Fatalf("unresolved baseline entry must be void+unattended+unrated: %+v", en)
				}
				if failing(s, en.EditorID) || roundsRun(s, en.EditorID) != 0 {
					t.Fatalf("unresolved baseline round held against curator: failing=%v roundsRun=%d", failing(s, en.EditorID), roundsRun(s, en.EditorID))
				}
				if en.Detail == nil || len(en.Detail.Verifiers) == 0 || en.NResults == 0 {
					t.Fatalf("unresolved baseline entry should record verifiers in detail: %+v", en.Detail)
				}
				for _, run := range en.Runs {
					for _, gr := range run.Grades {
						if gr.Verdict != "disputed" || gr.Gain != nil {
							t.Fatalf("unresolved baseline grade must be disputed with nil gain (no midpoint gain): %+v", gr)
						}
					}
				}
				for _, r := range s.results(idx, en.EditorID) {
					if r.Score != nil || len(r.BySource) > 0 {
						t.Fatalf("unresolved baseline result kept midpoint score in DB: score=%v bySource=%v", r.Score, r.BySource)
					}
				}
			}
		})
		if b := w.app.board(); b.RecordID != "" || len(b.Entries) != 0 {
			t.Fatalf("unresolved baseline round leaked into board/record: record=%q entries=%d", b.RecordID, len(b.Entries))
		}
		for _, a := range g {
			acct := w.account(a.handle)
			if acct.Agreed != beforeAgreed[a.handle] || acct.Strikes != 0 {
				t.Fatalf("grader %s got credit or strike on unresolved baseline round: %+v", a.handle, acct)
			}
		}
		if code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); code != 200 || !strings.Contains(strings.ToLower(page), "disputed") {
			t.Fatalf("round page for disputed baseline: HTTP %d", code)
		}
		w.check()
	})

	t.Run("three_way_baseline_disagreement_closes_uniformly", func(t *testing.T) {
		w, _, idx := toGrade(t, 3)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		g := w.volunteers("bob", "carol", "erin")
		c0 := w.asked(g[0], true)
		c1 := w.asked(g[1], true)
		if code, _ := w.send(g[0], c0, 5.6, 6.00); code != 200 {
			t.Fatalf("bob's grade: HTTP %d", code)
		}
		if code, _ := w.send(g[1], c1, 5.6, 6.08); code != 200 {
			t.Fatalf("carol's grade: HTTP %d", code)
		}
		// Erin supplies a 3rd baseline at 6.04: median(6.00, 6.04, 6.08) = 6.04, so only 1
		// baseline agrees within 0.015 (bob and carol are both oddOut -> still disputed!).
		c2 := w.asked(g[2], true)
		if code, status := w.send(g[2], c2, 5.6, 6.04); code != 200 || status != "pending" {
			t.Fatalf("erin's 3-way-splitting baseline must keep round open: HTTP %d %s", code, status)
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "settled" || rd.BaselineLoss != nil || rd.BaselineN != 0 {
				t.Fatalf("3-way disputed baseline round: status=%s loss=%v n=%d", rd.Status, rd.BaselineLoss, rd.BaselineN)
			}
			for _, en := range s.inRound[idx] {
				if !en.Void || !en.Unattended || en.Valid || !strings.Contains(en.Reason, "disagreed on the round's unedited baseline") {
					t.Fatalf("entry %s must close uniformly as disputed baseline (not giveUp): %+v", en.EditorID, en)
				}
				for _, run := range en.Runs {
					for _, gr := range run.Grades {
						if gr.Verdict != "disputed" || gr.Gain != nil {
							t.Fatalf("3-way disputed baseline grade leaked midpoint gain: %+v", gr)
						}
					}
				}
				for _, r := range s.results(idx, en.EditorID) {
					if r.Score != nil || len(r.BySource) > 0 {
						t.Fatalf("3-way disputed baseline result kept score in DB: %+v", r)
					}
				}
			}
		})
		w.check()
	})
}

// Gate 4 regression coverage for tokens_sha256 tie interactions with failed runs, second valid runs,
// and strict invalid majorities.
func TestTokensTieEdgeCases(t *testing.T) {
	t.Run("unvouched_failed_edit_plus_tokens_tie_voids_unattended", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.cfg.RunsPerEntry = 2
		_ = w.join("anchor-one", priv)
		runners := w.volunteers("run-a", "run-b")
		graders := w.volunteers("gr1", "gr2")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()

		// Volunteer run-a writes a valid script; volunteer run-b reports a failed edit.
		cA := w.claim(runners[0], true)
		if cA.Job == nil || cA.Job.Kind != "edit" {
			t.Fatalf("run-a edit claim: %+v", cA.Job)
		}
		if code, _ := w.edit(runners[0], cA, "0\n1\n2\n"); code != 200 {
			t.Fatalf("run-a script: HTTP %d", code)
		}
		cB := w.claim(runners[1], true)
		if cB.Job == nil || cB.Job.Kind != "edit" {
			t.Fatalf("run-b edit claim: %+v", cB.Job)
		}
		if code, _ := w.edit(runners[1], cB, ""); code != 200 {
			t.Fatalf("run-b failed edit: HTTP %d", code)
		}
		g1 := w.claimEarly(graders[0])
		g2 := w.claimEarly(graders[1])
		w.exam(idx)

		// Two established graders disagree on tokens_sha256 for run-a's script.
		p1 := w.result(g1, 5.5, 6.0)
		p1["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("a", 64)
		p2 := w.result(g2, 5.5, 6.0)
		p2["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("b", 64)
		if code := graders[0].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", g1.Job.ID), p1, nil); code != 200 {
			t.Fatalf("g1 grade: HTTP %d", code)
		}
		if code := graders[1].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", g2.Job.ID), p2, nil); code != 200 {
			t.Fatalf("g2 grade: HTTP %d", code)
		}

		// Time runs out unresolved: dropping run-a must NOT leave volunteer run-b's failed edit
		// as a rated invalid entry that penalizes the curator.
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			en := s.inRound[idx][0]
			if en.Status != "settled" || !en.Void || !en.Unattended || en.Valid || s.rated(en) {
				t.Fatalf("expected void+unattended entry, got %+v", en)
			}
			if failing(s, en.EditorID) || roundsRun(s, en.EditorID) != 0 {
				t.Fatalf("curator penalized by unvouched failed edit after tokens tie: failing=%v roundsRun=%d", failing(s, en.EditorID), roundsRun(s, en.EditorID))
			}
		})
		w.check()
	})

	t.Run("valid_run_not_polluted_by_dropped_tokens_tie_run", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.cfg.RunsPerEntry = 2
		_ = w.join("anchor-one", priv)
		runners := w.volunteers("run-a", "run-b")
		graders := w.volunteers("gr1", "gr2", "gr3")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()

		// Run 1 posts S1; claim 2 early grades on S1 before Run 2 posts S2 so S1 deterministically
		// gets g1 & g2 (conflicting tokens_sha256) and S2 gets g3 (clean grade).
		cA := w.claim(runners[0], true)
		if code, _ := w.edit(runners[0], cA, "0\n1\n"); code != 200 {
			t.Fatalf("run-a script: HTTP %d", code)
		}
		g1 := w.claimEarly(graders[0])
		g2 := w.claimEarly(graders[1])
		cB := w.claim(runners[1], true)
		if code, _ := w.edit(runners[1], cB, "2\n3\n"); code != 200 {
			t.Fatalf("run-b script: HTTP %d", code)
		}
		g3 := w.claimEarly(graders[2])
		if g3.Job == nil || g3.Job.Script.SHA256 == g1.Job.Script.SHA256 {
			t.Fatalf("expected g3 to grade S2, got %+v", g3.Job)
		}
		w.exam(idx)

		p1 := w.result(g1, 5.5, 6.0)
		p1["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("a", 64)
		p2 := w.result(g2, 5.5, 6.0)
		p2["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("b", 64)
		p3 := w.result(g3, 5.4, 6.0)
		p3["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("c", 64)
		for i, pair := range []struct {
			a *actor
			c *claimed
			p map[string]any
		}{{graders[0], g1, p1}, {graders[1], g2, p2}, {graders[2], g3, p3}} {
			if code := pair.a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", pair.c.Job.ID), pair.p, nil); code != 200 {
				t.Fatalf("grade %d: HTTP %d", i, code)
			}
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			en := s.inRound[idx][0]
			if en.Status != "settled" || !en.Valid || en.Void || en.NRuns != 1 {
				t.Fatalf("expected valid 1-run entry from S2, got %+v", en)
			}
			if en.Detail == nil || en.Detail.DisputedTogether || len(en.Detail.Verifiers) != 1 || en.Detail.Verifiers[0] != "gr3" {
				t.Fatalf("dropped tokens-tie run leaked into valid entry detail: %+v", en.Detail)
			}
		})
		w.check()
	})

	t.Run("invalid_majority_beats_tokens_tie", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		_ = w.join("anchor-one", priv)
		g := w.volunteers("gr1", "gr2", "gr3", "gr4", "gr5", "gr6")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		cEdit := w.claim(g[0], true)
		if cEdit.Job == nil || cEdit.Job.Kind != "edit" {
			t.Fatalf("edit claim: %+v", cEdit.Job)
		}
		if code, _ := w.edit(g[0], cEdit, "0\n1\n"); code != 200 {
			t.Fatalf("script: HTTP %d", code)
		}
		// Claim 4 early grades (max early replicas = 1 + maxReplicas = 4) before posting exam.
		graders := g[1:5]
		var cs []*claimed
		for _, a := range graders {
			c := w.claimEarly(a)
			if c.Job == nil || c.Job.Kind != "grade" {
				t.Fatalf("early grade claim for %s: %+v", a.handle, c.Job)
			}
			cs = append(cs, c)
		}
		w.exam(idx)
		// First 2 valid with conflicting tokens_sha256 (tokA, tokB), next 2 invalid:
		// 2 valid vs 2 invalid is a 50/50 validity tie (stays pending and requests a tie-breaker).
		for i, a := range graders {
			var p map[string]any
			if i < 2 {
				p = w.result(cs[i], 5.5, 6.0)
				if i == 0 {
					p["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("a", 64)
				} else {
					p["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("b", 64)
				}
			} else {
				p = w.result(cs[i], 5.5, 6.0)
				p["valid"] = false
				p["invalid_reason"] = "bad script syntax"
				delete(p, "exam")
				delete(p, "baseline")
			}
			if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cs[i].Job.ID), p, nil); code != 200 {
				t.Fatalf("grader %d result: HTTP %d", i, code)
			}
		}
		if w.settled(idx) != 0 {
			t.Fatalf("2 valid (tokA/tokB) vs 2 invalid must stay pending for a tie-breaker")
		}
		// 5th established grader submits a 3rd invalid grade: strict invalid majority (3 vs 2)
		// beats the tokens disagreement and settles rated invalid!
		c5 := w.claim(g[5], false)
		if c5.Job == nil || c5.Job.Kind != "grade" {
			t.Fatalf("tie-breaker claim for %s: %+v", g[5].handle, c5.Job)
		}
		p5 := w.result(c5, 5.5, 6.0)
		p5["valid"] = false
		p5["invalid_reason"] = "bad script syntax"
		delete(p5, "exam")
		delete(p5, "baseline")
		if code := g[5].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c5.Job.ID), p5, nil); code != 200 {
			t.Fatalf("grader 5 result: HTTP %d", code)
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			en := s.inRound[idx][0]
			if en.Status != "settled" || en.Void || en.Valid {
				t.Fatalf("strict invalid majority (3 vs 2) with tokens disagreement must settle rated invalid (not void): %+v", en)
			}
		})
		w.check()
	})
}

// Season 1 had every grader train the round's unedited student, and a third of all training
// was that. Four machines' measurements make a round's baseline as well as forty: the first
// four established graders are asked for it, and the fifth trains the script's student alone.
// Its grade comes without a baseline and is scored against the mean of the four.
func TestFourMachinesMeasureARoundsBaselineAndEveryGradeIsScoredAgainstIt(t *testing.T) {
	w, anchor, idx := toGrade(t, 5)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	if n := w.wanted(anchor); n != 4 {
		t.Fatalf("a round nobody has graded wants its four baselines: %d", n)
	}
	var four []*claimed
	for i, a := range g[:4] {
		four = append(four, w.asked(a, true))
		if n := w.wanted(anchor); n != 3-i {
			t.Fatalf("after %d promises the round wants %d more", i+1, n)
		}
	}
	fifth := w.asked(g[4], false)
	// A job read again says what it said when it was handed out.
	if again := w.asked(g[4], false); again.Job.ID != fifth.Job.ID {
		t.Fatalf("the fifth grader was handed another job: %+v", again.Job)
	}
	if again := w.asked(g[0], true); again.Job.ID != four[0].Job.ID {
		t.Fatalf("the first grader was handed another job: %+v", again.Job)
	}

	// The fifth has one pass to train and is done first. Its grade is taken, with nothing
	// yet to score it against, and the round waits for the baseline it was promised.
	if code, status := w.send(g[4], fifth, 5.5, 0); code != 200 || status != "pending" {
		t.Fatalf("a grade from a job that asked for no baseline: HTTP %d %s", code, status)
	}
	// A job that asked for the baseline is not reported without it.
	if code, msg := w.send(g[0], four[0], 5.5, 0); code != 400 || !strings.Contains(msg, "baseline") {
		t.Fatalf("a result that leaves out the baseline its job asked for: HTTP %d %s", code, msg)
	}
	bases := []float64{6.00, 6.02, 5.98, 6.04} // their mean is 6.01
	for i, a := range g[:4] {
		want := "pending"
		if i == 3 {
			want = "settled"
		}
		if code, status := w.send(a, four[i], 5.6, bases[i]); code != 200 || status != want {
			t.Fatalf("%s's grade: HTTP %d %s", a.handle, code, status)
		}
	}
	w.check()
	var rd struct {
		Round struct {
			BaselineLoss float64 `json:"baseline_loss"`
			Verifiers    int     `json:"baseline_verifiers"`
		} `json:"round"`
	}
	anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &rd)
	if math.Abs(rd.Round.BaselineLoss-6.01) > 1e-9 || rd.Round.Verifiers != 4 {
		t.Fatalf("the round's baseline is the mean of the four that measured it: %+v", rd.Round)
	}
	e := w.entry(idx, fifth.Job.Editor.Name)
	if !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.51) > 1e-9 || e.Loss == nil || math.Abs(*e.Loss-5.5) > 1e-9 {
		t.Fatalf("a grade with no baseline of its own is scored against the round's: %+v", e)
	}
	if gr := e.Runs[0].Grades; len(gr) != 1 || gr[0].Handle != "frank" || !gr[0].Counted || gr[0].Gain == nil || math.Abs(*gr[0].Gain-0.51) > 1e-9 {
		t.Fatalf("the grade behind it: %+v", gr)
	}
	// The four are scored against the same mean, not each against its own.
	if e := w.entry(idx, four[1].Job.Editor.Name); e.Gain == nil || math.Abs(*e.Gain-0.41) > 1e-9 {
		t.Fatalf("carol's grade against the round's baseline: %+v", e.Gain)
	}
}

// The first baseline in is no more the round's standard than the next. A machine whose unedited
// student is far out and that reports first is kept, nobody after it is turned away for
// differing from it, and once three have measured it is the odd one out: the round's baseline
// is the median of the others', and every grade is scored against that.
func TestABaselineThatIsFarOutAndFirstInIsNotTheRoundsStandard(t *testing.T) {
	w, anchor, idx := toGrade(t, 4)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin")
	var jobs []*claimed
	for _, a := range g {
		jobs = append(jobs, w.asked(a, true))
	}
	bases := []float64{7.5, 6.00, 6.02, 5.98} // bob's is 1.5 bits per byte out, and first
	for i, a := range g {
		if code, status := w.send(a, jobs[i], 5.9, bases[i]); code != 200 {
			t.Fatalf("%s's grade, baseline %.2f: HTTP %d %s", a.handle, bases[i], code, status)
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			want := map[int]float64{0: 7.5, 1: 6.75, 2: 6.01, 3: 6.0}[i]
			if rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-want) > 1e-9 {
				t.Fatalf("after %d baselines the round's is %v, not %.2f", i+1, rd.BaselineLoss, want)
			}
		})
	}
	// What bob's machine made of its script is no more to be relied on than its baseline: the
	// script wants a grade as if it had none, and the round waits for it.
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].Status != "open" {
			t.Fatalf("the round settled on the grade of a machine whose baseline is the odd one out: %s", s.Rounds[idx].Status)
		}
	})
	again := w.asked(g[1], true)
	if again.Job.Editor.Name != jobs[0].Job.Editor.Name {
		t.Fatalf("carol was handed %s, not the script bob graded (%s)", again.Job.Editor.Name, jobs[0].Job.Editor.Name)
	}
	if code, status := w.send(g[1], again, 5.9, 6.0); code != 200 || status != "settled" {
		t.Fatalf("carol's grade of the script bob graded: HTTP %d %s", code, status)
	}
	w.check()
	var rd struct {
		Round struct {
			BaselineLoss float64 `json:"baseline_loss"`
			Verifiers    int     `json:"baseline_verifiers"`
		} `json:"round"`
	}
	anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &rd)
	if math.Abs(rd.Round.BaselineLoss-6.0) > 1e-9 || rd.Round.Verifiers != 3 {
		t.Fatalf("the round's baseline is the median of the three that agree: %+v", rd.Round)
	}
	for i := range g {
		if e := w.entry(idx, jobs[i].Job.Editor.Name); !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.1) > 1e-9 {
			t.Fatalf("the script %s graded gained 0.1, and settled at %+v", g[i].handle, e.Gain)
		}
	}
	verdict := ""
	for _, gr := range w.entry(idx, jobs[0].Job.Editor.Name).Runs[0].Grades {
		if gr.Handle == "bob" {
			verdict = gr.Verdict
		}
	}
	if b := w.account("bob"); verdict != "baseline differs" || b.Strikes != 0 || b.Agreed != 2*w.cfg.EstablishAfter {
		t.Fatalf("bob's grade is shown as %q, and counted for or against him: %+v", verdict, b)
	}
}

// The server requires train.init_sha256 and script.tokens_sha256 on valid Season 3 grades,
// refuses to let a single bad first grade poison Round.InitSHA, locks consensus once two
// independent established graders agree, refuses subsequent mismatches with HTTP 409, and
// voids earlier bad-init grades as "other data" at settlement.
func TestServerComparesInitSHA256AndTokensSHA256(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.season.Baselines.PerRound = 4
	anchor := w.join("anchor-one", priv)
	owner := w.join("owner-entry", nil)
	owner2 := w.join("owner-entry-2", nil)
	w.submit(owner, "entry-0", "print(0)")
	w.submit(owner2, "entry-1", "print(1)") // left unedited until after 409 checks so round stays open
	idx := w.draw()
	// Confirm entry A with two matching edit scripts while holding entry B's edit replicas in flight
	// so every grade claim goes to entry A and the round stays open through the post-consensus 409 checks.
	g := w.volunteers("bob", "carol", "dave", "frank", "grace", "heidi", "erin")
	erin, heidi := g[6], g[5]
	cEdit0 := w.claim(anchor, true)
	if cEdit0.Job == nil || cEdit0.Job.Kind != "edit" {
		t.Fatalf("anchor did not get edit job 0: %+v", cEdit0.Job)
	}
	if code, msg := w.edit(anchor, cEdit0, "{\"d\": 0}\n"); code != 200 {
		t.Fatalf("anchor edit 0 failed: HTTP %d %s", code, msg)
	}
	entryA := cEdit0.Job.Editor
	var heldJobs []struct {
		a *actor
		c *claimed
	}
	for _, r := range []*actor{anchor, erin, heidi} {
		c := w.claim(r, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("%s did not get edit job: %+v", r.handle, c.Job)
		}
		if c.Job.Editor == entryA {
			if code, msg := w.edit(r, c, "{\"d\": 0}\n"); code != 200 {
				t.Fatalf("%s confirm edit failed: HTTP %d %s", r.handle, code, msg)
			}
			break
		}
		heldJobs = append(heldJobs, struct {
			a *actor
			c *claimed
		}{r, c})
	}

	initA := strings.Repeat("a", 64)
	initB := strings.Repeat("b", 64)
	tokA := strings.Repeat("1", 64)
	tokB := strings.Repeat("2", 64)

	sendWith := func(a *actor, c *claimed, initSHA, tokSHA string) (int, string) {
		t.Helper()
		p := w.result(c, 5.5, 6.0)
		tr := map[string]any{"precision": "bf16", "seconds": 10.0}
		if initSHA != "" {
			tr["init_sha256"] = initSHA
			p["baseline"].(map[string]any)["init_sha256"] = initSHA
		} else {
			delete(p["baseline"].(map[string]any), "init_sha256")
		}
		sc := map[string]any{"script_sha256": c.Job.Script.SHA256}
		if tokSHA != "" {
			sc["tokens_sha256"] = tokSHA
		}
		p["train"], p["script"] = tr, sc
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		if code != 200 {
			return code, ack.Error
		}
		return code, ack.Status
	}

	// Established bob & carol claim the two initial grade replicas; unestablished ivan
	// claims a verification grade on the same script.
	ivan, judy := w.join("ivan", nil), w.join("judy", nil)
	cBad := w.asked(g[0], true)
	c1 := w.asked(g[1], true)
	c3 := w.asked(ivan, true)

	// Missing either integrity digest on a valid Season 3 grade is refused with HTTP 400.
	if code, msg := sendWith(g[0], cBad, "", tokA); code != 400 || !strings.Contains(msg, "init_sha256") {
		t.Fatalf("missing init_sha256: HTTP %d %q, want 400", code, msg)
	}
	if code, msg := sendWith(g[0], cBad, initA, ""); code != 400 || !strings.Contains(msg, "tokens_sha256") {
		t.Fatalf("missing tokens_sha256: HTTP %d %q, want 400", code, msg)
	}

	// First grader (bob) reports a bad initB: accepted initially without locking rd.InitSHA,
	// so honest peers reporting initA are NOT poisoned!
	if code, st := sendWith(g[0], cBad, initB, tokA); code != 200 {
		t.Fatalf("first grade with initB failed: HTTP %d %s", code, st)
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].InitSHA != "" {
			t.Fatalf("single grader prematurely locked Round.InitSHA to %q", s.Rounds[idx].InitSHA)
		}
	})

	// Second grader (carol) reports honest initA -> init_sha256 differs from bob's initB, so
	// judge.same() treats them as a tie and automatically asks a third established grader (dave)!
	if code, st := sendWith(g[1], c1, initA, tokA); code != 200 {
		t.Fatalf("first honest grade failed: HTTP %d %s", code, st)
	}
	c2 := w.asked(g[2], true)
	if code, st := sendWith(g[2], c2, initA, tokA); code != 200 {
		t.Fatalf("second honest grade failed: HTTP %d %s", code, st)
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].InitSHA != initA {
			t.Fatalf("two agreeing graders did not lock Round.InitSHA: got %q, want %q", s.Rounds[idx].InitSHA, initA)
		}
	})

	// Subsequent mismatches on init_sha256 or tokens_sha256 are refused with HTTP 409.
	if code, msg := sendWith(ivan, c3, initB, tokA); code != 409 || !strings.Contains(msg, "initial weights") {
		t.Fatalf("mismatched init_sha256: HTTP %d %q, want 409 mentioning initial weights", code, msg)
	}
	c4 := w.asked(judy, true)
	if code, msg := sendWith(judy, c4, initA, tokB); code != 409 || !strings.Contains(msg, "tokenized training stream") {
		t.Fatalf("mismatched tokens_sha256: HTTP %d %q, want 409 mentioning tokenized training stream", code, msg)
	}

	// Finish the second entry honestly so the round settles cleanly.
	for _, h := range heldJobs {
		if code, msg := w.edit(h.a, h.c, "{\"d\": 1}\n"); code != 200 {
			t.Fatalf("%s held edit failed: HTTP %d %s", h.a.handle, code, msg)
		}
	}
	if len(heldJobs) == 1 {
		c := w.claim(heidi, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("heidi did not get remaining edit job: %+v", c.Job)
		}
		if code, msg := w.edit(heidi, c, "{\"d\": 1}\n"); code != 200 {
			t.Fatalf("heidi remaining edit failed: HTTP %d %s", code, msg)
		}
	}
	if code, st := sendWith(g[3], w.claim(g[3], false), initA, tokA); code != 200 {
		t.Fatalf("frank grade on second entry failed: HTTP %d %s", code, st)
	}
	if code, st := sendWith(g[4], w.claim(g[4], false), initA, tokA); code != 200 {
		t.Fatalf("grace grade on second entry failed: HTTP %d %s", code, st)
	}

	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 2 {
		t.Fatalf("round settled %d entries, want 2", w.settled(idx))
	}

	var differs []Differ
	var bobVerdict string
	w.app.st.view(func(s *State) {
		differs = s.Rounds[idx].Differs
		for _, r := range s.Results {
			if r.RoundIdx == idx && s.Accounts[r.AccountID].Handle == "bob" {
				bobVerdict = r.Verdict
			}
		}
	})
	if bobVerdict != "void" {
		t.Fatalf("bob's bad-init first grade should be voided at settlement, got verdict=%q", bobVerdict)
	}
	if len(differs) != 2 || differs[0].What != "init" || differs[1].What != "tokens" {
		t.Fatalf("expected [init, tokens] in Round.Differs, got %+v", differs)
	}
	if code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); code != 200 ||
		!strings.Contains(page, "initial weights differed") || !strings.Contains(page, "tokenized script differed") {
		t.Fatalf("round page (HTTP %d) did not render init/tokens differ labels", code)
	}
}

// Whichever order they come in, the one that is far out is left out: last in, with two or more
// to say so, it is turned away as it arrives.
func TestABaselineThatIsFarOutAndLastInIsTurnedAway(t *testing.T) {
	w, _, idx := toGrade(t, 4)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin")
	var jobs []*claimed
	for _, a := range g {
		jobs = append(jobs, w.asked(a, true))
	}
	// With one other in, a baseline that differs from it is kept: nothing says which is wrong.
	for i, base := range []float64{6.0, 6.3, 6.02} {
		if code, status := w.send(g[i], jobs[i], 5.9, base); code != 200 {
			t.Fatalf("%s's grade, baseline %.2f: HTTP %d %s", g[i].handle, base, code, status)
		}
	}
	if code, msg := w.send(g[3], jobs[3], 5.9, 7.5); code != 409 || !strings.Contains(msg, "baseline") {
		t.Fatalf("a baseline 1.5 bits per byte from three others': HTTP %d %s", code, msg)
	}
	w.app.st.view(func(s *State) {
		// Carol's was kept as it came, and is left out now that three are in: 6.0 and 6.02 remain.
		if rd := s.Rounds[idx]; rd.BaselineN != 2 || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.01) > 1e-9 {
			t.Fatalf("the round's baseline: %d machines, %v", rd.BaselineN, rd.BaselineLoss)
		}
	})
	w.check()
}

func TestMedianItems(t *testing.T) {
	for _, c := range []struct {
		rows [][]float64
		want []float64
	}{
		{nil, nil},
		{[][]float64{{1, 5}}, []float64{1, 5}},
		{[][]float64{{1, 5}, {3, 1}}, []float64{2, 3}},
		{[][]float64{{9, 5}, {1, 1}, {2, 7}}, []float64{2, 5}},
		{[][]float64{{9, 5}, {1, 1}, {2, 7}, {4, 100}}, []float64{3, 6}},
	} {
		got := medianItems(c.rows)
		if len(got) != len(c.want) {
			t.Fatalf("medianItems(%v) = %v", c.rows, got)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("medianItems(%v) = %v, not %v", c.rows, got, c.want)
			}
		}
	}
}

// A promise is a job's: handed back or run out, it is gone, and the next grader is asked. A
// machine that was ever asked is asked again, since it may have the student trained.
func TestABaselineSomebodyWasAskedForAndDidNotSendGoesToTheNextGrader(t *testing.T) {
	w, anchor, _ := toGrade(t, 8)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin", "frank", "gina", "harry")
	bob, carol, dave, erin, frank, gina, harry := g[0], g[1], g[2], g[3], g[4], g[5], g[6]
	w.asked(bob, true)
	w.asked(carol, true)
	cd, ce := w.asked(dave, true), w.asked(erin, true)
	cf := w.asked(frank, false)

	// Erin hands her job back, and the baseline she was asked for is the next grader's to give.
	w.release(erin, ce)
	if n := w.wanted(anchor); n != 1 {
		t.Fatalf("with a promise handed back the round wants one more: %d", n)
	}
	w.asked(gina, true)
	if n := w.wanted(anchor); n != 0 {
		t.Fatalf("the promise was taken up: %d", n)
	}
	// Dave's job runs out with no word from him.
	w.lapse(cd)
	if n := w.wanted(anchor); n != 1 {
		t.Fatalf("with a promise run out the round wants one more: %d", n)
	}
	w.asked(harry, true)
	// Frank's job is the one he was handed: it does not start asking for a baseline halfway.
	if again := w.asked(frank, false); again.Job.ID != cf.Job.ID {
		t.Fatalf("frank was handed another job: %+v", again.Job)
	}
	// Erin was asked once, and may have the unedited student trained: she is asked again,
	// though four others are measuring it.
	if again := w.asked(erin, true); again.Job.ID == ce.Job.ID {
		t.Fatal("erin was handed the job she handed back")
	}
	// Bob has sent his, and his later grades of the round keep carrying it.
	if code, status := w.send(bob, w.asked(bob, true), 5.5, 6.0); code != 200 {
		t.Fatalf("bob's grade: HTTP %d %s", code, status)
	}
	w.asked(bob, true)
	// A machine that was not asked, and has the round's baseline on its disk all the same,
	// may send it: it is checked like any other, and is one more for the round's.
	if code, status := w.send(frank, cf, 5.5, 6.02); code != 200 {
		t.Fatalf("a baseline nobody asked for: HTTP %d %s", code, status)
	}
	w.app.st.view(func(s *State) {
		if rd := openRounds(s)[0]; rd.BaselineN != 2 || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.01) > 1e-9 {
			t.Fatalf("the round's baseline: %d machines, %v", rd.BaselineN, rd.BaselineLoss)
		}
	})
	w.check()
}

// An account that is not established is always asked: its baseline is how its machine is
// checked against the others', and is no part of the round's, so it is not one of the four.
func TestAVerifierNotYetEstablishedAlwaysMeasuresTheBaselineAndIsNotOneOfTheFour(t *testing.T) {
	w, anchor, idx := toGrade(t, 6)
	w.app.season.Baselines.PerRound = 4
	nina, olga := w.join("nina", nil), w.join("olga", nil)
	w.standing("olga") // one grade that agreed: no newcomer, and not established
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	cn := w.asked(nina, true)
	if n := w.wanted(anchor); n != 4 {
		t.Fatalf("a newcomer's baseline is not one of the round's four: %d", n)
	}
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(g[4], false)
	co := w.asked(olga, true) // four are measuring it, and she is asked all the same
	if code, msg := w.send(olga, co, 5.5, 0); code != 400 {
		t.Fatalf("a grade without the baseline that checks its machine: HTTP %d %s", code, msg)
	}
	// Nina's unedited student is a little off the others': inside the tolerance, and no part
	// of the round's baseline.
	if code, status := w.send(nina, cn, 5.5, 6.1); code != 200 {
		t.Fatalf("nina's grade: HTTP %d %s", code, status)
	}
	if n := w.wanted(anchor); n != 0 {
		t.Fatalf("four established machines are measuring it: %d", n)
	}
	for i, a := range g[:4] {
		if code, status := w.send(a, four[i], 5.5, 6.0); code != 200 {
			t.Fatalf("%s's grade: HTTP %d %s", a.handle, code, status)
		}
	}
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.0) > 1e-9 {
			t.Fatalf("the round's baseline is the established graders': %v", rd.BaselineLoss)
		}
	})
	if code, status := w.send(g[4], cf, 5.5, 0); code != 200 {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	if code, status := w.send(olga, co, 5.5, 6.0); code != 200 {
		t.Fatalf("olga's grade: HTTP %d %s", code, status)
	}
	w.check()
}

// One machine, alone, is the first to be asked in every round, and is asked with every grade
// after that: its rounds settle as they always did.
func TestARoundWithOneGraderStillSettles(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(w.join("alice", nil), "tidy", "print(1)")
	w.submit(w.join("amir", nil), "neat", "print(2)")
	idx := w.draw()
	if w.editAll(anchor) != 2 {
		t.Fatal("the organisers' machine should have run both entries")
	}
	for i, want := range []string{"pending", "settled"} {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "grade" || c.Job.Baseline == nil || !*c.Job.Baseline {
			t.Fatalf("grade %d of a machine alone asks for the baseline: %+v", i, c.Job)
		}
		if code, status := w.send(anchor, c, 5.5, 6.0); code != 200 || status != want {
			t.Fatalf("grade %d: HTTP %d %s", i, code, status)
		}
	}
	for _, name := range []string{"tidy", "neat"} {
		if e := w.entry(idx, name); !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.5) > 1e-9 {
			t.Fatalf("%s: %+v", name, e)
		}
	}
	w.check()
}

// The four machines asked for a round's baseline all vanish, and a fifth, which was not, has
// reported. Its grade has nothing to be scored against: the round does not settle on it, and
// the next grade job handed out, whoever takes it, asks for the baseline.
func TestARoundWhoseBaselineNeverCameAsksTheNextGraderForIt(t *testing.T) {
	w, anchor, idx := toGrade(t, 5)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	frank := g[4]
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(frank, false)
	if code, status := w.send(frank, cf, 5.5, 0); code != 200 || status != "pending" {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	w.lapse(four...)
	if n := w.wanted(anchor); n != 4 {
		t.Fatalf("with every promise gone the round wants its four again: %d", n)
	}
	if w.settled(idx) != 0 {
		t.Fatal("an entry settled on a grade with no baseline to score it against")
	}
	// Frank is still there. He was not asked before, and is now: his next grade measures the
	// baseline, and the grade he sent without one is scored against it.
	for n := 0; n < 4; n++ {
		c := w.asked(frank, true)
		want := "pending"
		if n == 3 {
			want = "settled"
		}
		if code, status := w.send(frank, c, 5.7, 6.0); code != 200 || status != want {
			t.Fatalf("frank's grade %d: HTTP %d %s", n+2, code, status)
		}
	}
	if e := w.entry(idx, cf.Job.Editor.Name); !e.Valid || e.Void || e.Gain == nil || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("the grade sent before the round had a baseline: %+v", e)
	}
	w.check()
}

// A grade that came with no baseline is not left as the only word on its script while the
// round has none: with nobody measuring it, the script wants a grade again, as if it had
// none, and whoever comes next is asked for the baseline whichever script they are handed.
func TestAGradeWithNoBaselineIsNotLeftAsTheRoundsOnlyWord(t *testing.T) {
	w, anchor, idx := toGrade(t, 5)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin", "frank", "gina")
	frank, gina := g[4], g[5]
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(frank, false)
	if code, status := w.send(frank, cf, 5.5, 0); code != 200 {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	// While somebody is measuring the baseline, frank's script has the grade it waits for.
	if n := w.work(anchor).Rounds[0].GradesWanted; n != 0 {
		t.Fatalf("every script has its grade, done or under way: %d wanted", n)
	}
	w.idle(gina, false, "every script has its grade, done or under way")
	w.lapse(four...)
	// Now nobody is: frank's script wants a grade like the four that lost theirs.
	if n := w.work(anchor).Rounds[0].GradesWanted; n != 5 {
		t.Fatalf("with no baseline coming, every script wants a grade: %d wanted", n)
	}
	// The first grade gina sends brings the round its baseline, frank's grade is scored
	// against it, and his script wants nothing more unless she already holds it.
	graded := 0
	for {
		c := w.claim(gina, false)
		if c.Job == nil {
			break
		}
		if c.Job.Baseline == nil || !*c.Job.Baseline {
			t.Fatalf("gina's grade %d does not ask for the baseline: %+v", graded+1, c.Job)
		}
		if code, status := w.send(gina, c, 5.52, 6.0); code != 200 {
			t.Fatalf("gina's grade %d: HTTP %d %s", graded+1, code, status)
		}
		graded++
	}
	if graded < 4 || w.settled(idx) != 5 {
		t.Fatalf("the round did not settle once it had a baseline: %d grades, %d entries settled", graded, w.settled(idx))
	}
	e := w.entry(idx, cf.Job.Editor.Name)
	if !e.Valid || e.Gain == nil || len(e.Runs) != 1 || len(e.Runs[0].Grades) != graded-3 {
		t.Fatalf("frank's entry after %d grades by gina: %+v", graded, e)
	}
	for _, gr := range e.Runs[0].Grades {
		if gr.Gain == nil || gr.Verdict != "agreed" {
			t.Fatalf("a grade of frank's script: %+v", gr)
		}
	}
	w.check()
}

// If nobody at all is left to measure it, the round ends as one that nobody was there to
// grade: its entries are void, which is held against none of them.
func TestARoundNobodyIsLeftToMeasureIsGivenUpNotScoredAgainstNothing(t *testing.T) {
	w, _, idx := toGrade(t, 5)
	w.app.season.Baselines.PerRound = 4
	g := w.volunteers("bob", "carol", "dave", "erin", "frank")
	var four []*claimed
	for _, a := range g[:4] {
		four = append(four, w.asked(a, true))
	}
	cf := w.asked(g[4], false)
	if code, status := w.send(g[4], cf, 5.5, 0); code != 200 {
		t.Fatalf("frank's grade: HTTP %d %s", code, status)
	}
	w.lapse(four...)
	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 5 {
		t.Fatalf("the round stayed open with nobody there: %d entries settled", w.settled(idx))
	}
	e := w.entry(idx, cf.Job.Editor.Name)
	if !e.Void || e.Valid || len(e.Dropped) != 1 || !strings.Contains(e.Dropped[0].Reason, "baseline") {
		t.Fatalf("an entry whose only grade had no baseline to be scored against: %+v", e)
	}
	// The grade is published as what it was: an established one, with nothing to score it by.
	if gr := e.Dropped[0].Grades; len(gr) != 1 || gr[0].Handle != "frank" || gr[0].Verdict != "no baseline" || gr[0].Gain != nil || gr[0].Counted {
		t.Fatalf("the grade that had no baseline: %+v", gr)
	}
	if code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); code != 200 || !strings.Contains(page, "no baseline") {
		t.Fatalf("the round's page: HTTP %d", code)
	}
	w.app.st.view(func(s *State) {
		if rd := s.Rounds[idx]; rd.Status != "settled" || rd.BaselineLoss != nil {
			t.Fatalf("the round: %s, baseline %v", rd.Status, rd.BaselineLoss)
		}
		for _, en := range s.inRound[idx] {
			if failing(s, en.EditorID) || roundsRun(s, en.EditorID) != 0 {
				t.Fatalf("a round nobody could score is held against %s", en.EditorID[:12])
			}
		}
	})
	w.check()
}

// A machine the organisers rent is handed grades before an exam only once they would keep it
// busy until near it (pace.go). One that is not asked for the baseline has a pass less to do
// for its first grade, and is kept waiting that much longer.
func TestARentedMachineNotAskedForTheBaselineHasAPassLessToDo(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.season.Baselines.PerRound = 4
	w.app.cfg.EarlyWindow = 10 * time.Minute
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	w.submit(alice, "one", "print('one')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	w.examAfter(1, 2*time.Hour)
	for _, name := range []string{"two", "three", "four", "five"} {
		w.submit(w.join("owner-"+name, nil), name, "print('"+name+"')")
	}
	w.drawWithoutExam()
	if n := w.editAll(anchor); n != 5 {
		t.Fatalf("runs: %d", n)
	}
	// The exam is two hours off and the window is ten minutes. At 30 minutes a pass, the three
	// grades of a machine asked for the baseline are 122 minutes' work, and it is handed one
	// now. Without the unedited student they are 92, and the machine would only wait.
	var first *actor
	var held *claimed
	for _, handle := range []string{"rented-1", "rented-2", "rented-3", "rented-4"} {
		m := w.rented(anchor, handle)
		c := w.waiting(m, "RTX 3060", 1800)
		if c.Job == nil || c.Job.Kind != "grade" || c.Job.Baseline == nil || !*c.Job.Baseline {
			t.Fatalf("%s is asked for the baseline, and has work enough to start: %+v", handle, c.Job)
		}
		if first == nil {
			first, held = m, c
		}
	}
	fifth := w.rented(anchor, "rented-5")
	if c := w.waiting(fifth, "RTX 3060", 1800); c.Job != nil {
		t.Fatalf("a rented machine with one student to train, not two, was started too soon: %+v", c.Job)
	}
	// One of the four hands its job back. The fifth would be asked now, and so has the work.
	w.release(first, held)
	if c := w.waiting(fifth, "RTX 3060", 1800); c.Job == nil || c.Job.Baseline == nil || !*c.Job.Baseline {
		t.Fatalf("the fifth machine, asked for the baseline in the first's place: %+v", c.Job)
	}
	w.check()
}

// Review 7 Finding 1: when established graders disagree on script.tokens_sha256, the server must
// resolve the token digest BEFORE computing the median gain, picking the representative summary,
// and crediting verifiers — across arrival orders (A,B,A; B,A,A; A,A,B) and 3 distinct digests (A,B,C).
func TestTokensDigestResolvedBeforeMedianAndCredit(t *testing.T) {
	initA := strings.Repeat("a", 64)
	tokA := strings.Repeat("1", 64)
	tokB := strings.Repeat("2", 64)
	tokC := strings.Repeat("3", 64)

	setupTwoRun := func(t *testing.T, postExam bool) (*world, *actor, int) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		anchor := w.join("anchor-one", priv)
		runner := w.join("runner-two", nil)
		w.established("runner-two")
		w.submit(w.join("alice", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		cEdit1 := w.claim(anchor, true)
		if code, msg := w.edit(anchor, cEdit1, "{\"d\": 0}\n"); code != 200 {
			t.Fatalf("anchor edit: HTTP %d %s", code, msg)
		}
		cEdit2 := w.claim(runner, true)
		if code, msg := w.edit(runner, cEdit2, "{\"d\": 0}\n"); code != 200 {
			t.Fatalf("runner edit: HTTP %d %s", code, msg)
		}
		if postExam {
			if code, msg := w.postExam(anchor, idx, testTable()); code != 200 {
				t.Fatalf("postExam: HTTP %d %s", code, msg)
			}
		}
		return w, anchor, idx
	}

	runOrder := func(t *testing.T, name string, seq []struct {
		tok  string
		gain float64
	}, wantDisputed bool) {
		t.Helper()
		t.Run(name, func(t *testing.T) {
			w, _, idx := setupTwoRun(t, true)
			g := w.volunteers("bob", "carol", "dave")
			beforeAgreed := map[string]int{}
			for _, a := range g {
				beforeAgreed[a.handle] = w.account(a.handle).Agreed
			}

			sendTok := func(a *actor, c *claimed, tok string, gain float64) (int, string) {
				t.Helper()
				base := 6.0
				p := w.result(c, base-gain, base)
				if c.Job.Baseline != nil && !*c.Job.Baseline {
					delete(p, "baseline")
				} else {
					p["baseline"].(map[string]any)["init_sha256"] = initA
				}
				p["train"].(map[string]any)["init_sha256"] = initA
				p["script"].(map[string]any)["tokens_sha256"] = tok
				var ack struct {
					Status string `json:"status"`
					Error  string `json:"error"`
				}
				code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
				if code != 200 {
					return code, ack.Error
				}
				return code, ack.Status
			}

			// Claim first two grades so the script does not settle before the second arrives.
			c0 := w.claim(g[0], false)
			c1 := w.claim(g[1], false)
			if c0.Job == nil || c1.Job == nil {
				t.Fatalf("expected two initial grade claims, got %+v %+v", c0.Job, c1.Job)
			}
			if code, msg := sendTok(g[0], c0, seq[0].tok, seq[0].gain); code != 200 {
				t.Fatalf("grade 0 failed: HTTP %d %s", code, msg)
			}
			if code, msg := sendTok(g[1], c1, seq[1].tok, seq[1].gain); code != 200 {
				t.Fatalf("grade 1 failed: HTTP %d %s", code, msg)
			}

			// For A,B,A / B,A,A / A,B,C the first two disagree on TokensSHA -> tie triggers 3rd claim!
			c2 := w.claim(g[2], false)
			if c2.Job == nil {
				t.Fatalf("expected third grade claim to break tokens_sha256 disagreement")
			}
			if code, msg := sendTok(g[2], c2, seq[2].tok, seq[2].gain); code != 200 {
				t.Fatalf("grade 2 failed: HTTP %d %s", code, msg)
			}

			if wantDisputed {
				// With 3 distinct digests (A,B,C), verdict must remain tied and request a 4th grader!
				w.app.st.view(func(s *State) {
					j := w.app.newJudge(s)
					st := j.states(s.Rounds[idx])[0]
					sc := st.Scripts[0]
					v := j.weigh(sc)
					if !v.tie || !v.tokensTie || j.short(sc) != 1 {
						t.Fatalf("3 distinct digests (A,B,C): tie=%v tokensTie=%v short=%d, want true/true/1", v.tie, v.tokensTie, j.short(sc))
					}
				})
				w.pass(w.cfg.ReplicaWait + time.Minute)
				if w.settled(idx) != 1 {
					t.Fatal("disputed A,B,C round did not settle after ReplicaWait")
				}
				e := w.entry(idx, "entry-0")
				if !e.Void || e.Valid || (e.Gain != nil && *e.Gain != 0) {
					t.Fatalf("unresolved A,B,C digest must close unrated/void with zero gain: void=%v valid=%v gain=%v", e.Void, e.Valid, e.Gain)
				}
				if e.Detail == nil || !e.Detail.DisputedTogether {
					t.Fatalf("A,B,C entry should be marked DisputedTogether: %+v", e.Detail)
				}
				if len(e.Dropped) != 2 || len(e.Dropped[0].Grades) != 3 || len(e.Dropped[1].Grades) != 3 {
					t.Fatalf("A,B,C dropped runs should preserve all 3 disputed grades: %+v", e.Dropped)
				}
				for _, gr := range e.Dropped[0].Grades {
					if gr.Verdict != "disputed" {
						t.Fatalf("A,B,C grade %s verdict = %q, want disputed", gr.Handle, gr.Verdict)
					}
				}
				for _, a := range g {
					if got := w.account(a.handle).Agreed; got != beforeAgreed[a.handle] {
						t.Fatalf("A,B,C grader %s earned Agreed credit (%d -> %d)", a.handle, beforeAgreed[a.handle], got)
					}
				}
				// Verify the unresolved digest entry is NOT rated, NOT penalized, and NOT record-eligible.
				w.app.st.view(func(s *State) {
					en := s.inRound[idx][0]
					if s.rated(en) {
						t.Fatalf("unresolved A,B,C entry must not be rated: %+v", en)
					}
					if !en.Void || !en.Unattended || en.Valid || en.Gain != 0 {
						t.Fatalf("unresolved A,B,C state entry should be void+unattended: %+v", en)
					}
					if failing(s, en.EditorID) || roundsRun(s, en.EditorID) != 0 {
						t.Fatalf("unresolved A,B,C entry held against curator: failing=%v roundsRun=%d", failing(s, en.EditorID), roundsRun(s, en.EditorID))
					}
				})
				if b := w.app.board(); b.RecordID != "" || len(b.Entries) != 0 {
					t.Fatalf("unresolved A,B,C entry leaked into board/record: record=%q entries=%d", b.RecordID, len(b.Entries))
				}
				w.check()
				return
			}

			w.pass(w.cfg.ReplicaWait + time.Minute)
			if w.settled(idx) != 1 {
				t.Fatal("round did not settle")
			}
			e := w.entry(idx, "entry-0")
			if e.Gain == nil || math.Abs(*e.Gain-0.020) > 1e-4 {
				t.Fatalf("%s: entry gain = %v, want 0.02000 (stream A median, NOT 0.029 from minority B)", name, e.Gain)
			}
			if e.Detail == nil || e.Detail.DisputedTogether {
				t.Fatalf("%s: unexpected detail: %+v", name, e.Detail)
			}
			var repScript struct {
				TokensSHA256 string `json:"tokens_sha256"`
			}
			json.Unmarshal(e.Detail.Script, &repScript)
			if repScript.TokensSHA256 != tokA {
				t.Fatalf("%s: representative script tokens_sha256 = %q, want winning %q", name, repScript.TokensSHA256, tokA)
			}
			byHandle := map[string]Grade{}
			for _, gr := range e.Runs[0].Grades {
				byHandle[gr.Handle] = gr
			}
			for i, a := range g {
				gr := byHandle[a.handle]
				if seq[i].tok == tokA {
					if !gr.Counted || gr.Verdict != "agreed" {
						t.Fatalf("%s: majority grader %s got counted=%v verdict=%q, want true/agreed", name, a.handle, gr.Counted, gr.Verdict)
					}
					if got := w.account(a.handle).Agreed; got != beforeAgreed[a.handle]+1 {
						t.Fatalf("%s: majority grader %s Agreed=%d, want %d", name, a.handle, got, beforeAgreed[a.handle]+1)
					}
				} else {
					if gr.Counted || gr.Verdict != "other data" {
						t.Fatalf("%s: minority grader %s got counted=%v verdict=%q, want false/other data", name, a.handle, gr.Counted, gr.Verdict)
					}
					if got := w.account(a.handle).Agreed; got != beforeAgreed[a.handle] {
						t.Fatalf("%s: minority grader %s earned Agreed credit (%d -> %d)", name, a.handle, beforeAgreed[a.handle], got)
					}
				}
			}
			w.check()
		})
	}

	runOrder(t, "A_B_A", []struct {
		tok  string
		gain float64
	}{{tokA, 0.010}, {tokB, 0.029}, {tokA, 0.030}}, false)

	runOrder(t, "B_A_A", []struct {
		tok  string
		gain float64
	}{{tokB, 0.029}, {tokA, 0.010}, {tokA, 0.030}}, false)

	runOrder(t, "A_B_C_three_distinct", []struct {
		tok  string
		gain float64
	}{{tokA, 0.010}, {tokB, 0.029}, {tokC, 0.030}}, true)

	// Arrival order A, A, B: admission check refuses B with HTTP 409, AND judge.weigh on an
	// A, A, B fixture directly resolves to stream A (gain 0.020) excluding B.
	t.Run("A_A_B", func(t *testing.T) {
		w, anchor, idx := setupTwoRun(t, false)
		g := w.volunteers("bob", "carol", "dave")
		// All three claim early before the exam is posted so all three hold live grade claims.
		c0, c1, c2 := w.claimEarly(g[0]), w.claimEarly(g[1]), w.claimEarly(g[2])
		if c0.Job == nil || c1.Job == nil || c2.Job == nil {
			t.Fatalf("expected three early grade claims: %+v %+v %+v", c0.Job, c1.Job, c2.Job)
		}
		if code, msg := w.postExam(anchor, idx, testTable()); code != 200 {
			t.Fatalf("postExam: HTTP %d %s", code, msg)
		}
		send := func(a *actor, c *claimed, tok string, gain float64) int {
			p := w.result(c, 6.0-gain, 6.0)
			if c.Job.Baseline != nil && !*c.Job.Baseline {
				delete(p, "baseline")
			} else {
				p["baseline"].(map[string]any)["init_sha256"] = initA
			}
			p["train"].(map[string]any)["init_sha256"] = initA
			p["script"].(map[string]any)["tokens_sha256"] = tok
			return a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, nil)
		}
		if code := send(g[0], c0, tokA, 0.010); code != 200 {
			t.Fatalf("A1: HTTP %d", code)
		}
		if code := send(g[1], c1, tokA, 0.030); code != 200 {
			t.Fatalf("A2: HTTP %d", code)
		}
		// Direct judge.weigh check with A, A, B already in state BEFORE dave's 409 settles the round:
		w.app.st.view(func(s *State) {
			j := w.app.newJudge(s)
			st := j.states(s.Rounds[idx])[0]
			sc := st.Scripts[0]
			orig := sc.Results
			sc.Results = append(append([]*Result(nil), orig...), &Result{
				ID: 999, RoundIdx: idx, EditorID: st.Entry.EditorID, AccountID: w.account("dave").ID,
				Script: sc.SHA, Valid: true, InitSHA: initA, TokensSHA: tokB, Verdict: "pending",
				Score: func() *float64 { x := 0.029; return &x }(),
			})
			sc.verdict = nil
			v := j.weigh(sc)
			if v.tokensSHA != tokA || len(v.counted) != 2 || math.Abs(v.gain-0.020) > 1e-9 || v.tie {
				t.Fatalf("A,A,B direct weigh: tokens=%q counted=%d gain=%.5f tie=%v, want A/2/0.020/false", v.tokensSHA, len(v.counted), v.gain, v.tie)
			}
		})
		if code := send(g[2], c2, tokB, 0.029); code != 409 {
			t.Fatalf("late B after A,A: HTTP %d, want 409", code)
		}
	})
}

// Review 7 Finding 2: when the examiner omits init_sha256, an established grader with the wrong
// initialization cannot poison the round baseline. Its baseline carries its own InitSHA, is
// excluded once rd.InitSHA locks, triggers a replacement baseline request, and every entry's
// gain is scored purely against baselines matching the agreed InitSHA.
func TestBadInitBaselineIsFilteredAndReplacedWhenExaminerOmitsInit(t *testing.T) {
	initA := strings.Repeat("a", 64)
	initB := strings.Repeat("b", 64)
	tokA := strings.Repeat("1", 64)

	setupNoInitExam := func(t *testing.T, perRound int) (*world, *actor, int) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		if perRound > 0 {
			w.app.season.Baselines.PerRound = perRound
		}
		anchor := w.join("anchor-one", priv)
		runner := w.join("runner-two", nil)
		w.established("runner-two")
		w.submit(w.join("alice", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		c1 := w.claim(anchor, true)
		if code, msg := w.edit(anchor, c1, "{\"d\": 0}\n"); code != 200 {
			t.Fatalf("anchor edit: HTTP %d %s", code, msg)
		}
		c2 := w.claim(runner, true)
		if code, msg := w.edit(runner, c2, "{\"d\": 0}\n"); code != 200 {
			t.Fatalf("runner edit: HTTP %d %s", code, msg)
		}
		// Examiner omits init_sha256 (fallback path when examiner CPU check fails).
		post := testExam(testTable())
		delete(post, "init_sha256")
		if code := anchor.do("POST", fmt.Sprintf("/api/v1/admin/rounds/%s/exam", roundID(w.app.season.ID, idx)), post, nil); code != 200 {
			t.Fatalf("post exam without init_sha256: HTTP %d", code)
		}
		return w, anchor, idx
	}

	t.Run("conflicting_baselines_request_replacement_and_exclude_wrong_init", func(t *testing.T) {
		w, _, idx := setupNoInitExam(t, 2) // Season 3 default: Baselines.PerRound == 2

		g := w.volunteers("bob", "carol", "dave", "erin")
		sendGrade := func(a *actor, c *claimed, initSHA, baseInitSHA string, loss, base float64) (int, string) {
			t.Helper()
			p := w.result(c, loss, base)
			p["train"].(map[string]any)["init_sha256"] = initSHA
			p["script"].(map[string]any)["tokens_sha256"] = tokA
			if base == 0 {
				delete(p, "baseline")
			} else if baseInitSHA != "" {
				p["baseline"].(map[string]any)["init_sha256"] = baseInitSHA
			}
			var ack struct {
				Status string `json:"status"`
				Error  string `json:"error"`
			}
			code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
			if code != 200 {
				return code, ack.Error
			}
			return code, ack.Status
		}

		// Bob (wrong initB) submits baseline = 5.0 (loss), exam = 4.9 -> gain +0.10 on initB.
		cb := w.asked(g[0], true)
		// Inconsistent baseline.init_sha256 vs train.init_sha256 is refused with HTTP 400.
		if code, msg := sendGrade(g[0], cb, initB, initA, 4.9, 5.0); code != 400 {
			t.Fatalf("mismatched baseline vs train init_sha256: HTTP %d %q, want 400", code, msg)
		}
		if code, st := sendGrade(g[0], cb, initB, initB, 4.9, 5.0); code != 200 {
			t.Fatalf("bob bad-init grade: HTTP %d %s", code, st)
		}

		// Carol (honest initA) submits numerically different baseline = 6.0, exam = 5.8 -> gain +0.20 on initA.
		cc := w.asked(g[1], true)
		if code, st := sendGrade(g[1], cc, initA, initA, 5.8, 6.0); code != 200 {
			t.Fatalf("carol honest grade: HTTP %d %s", code, st)
		}

		// Both PerRound=2 slots have stored baselines, YET they conflict on InitSHA (initB vs initA)!
		// So the 3rd established grader (dave) breaking the tie MUST be asked for a replacement baseline!
		cd := w.asked(g[2], true)
		if code, st := sendGrade(g[2], cd, initA, initA, 5.8, 6.0); code != 200 {
			t.Fatalf("dave replacement baseline + tie-break grade: HTTP %d %s", code, st)
		}

		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatal("round did not settle")
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.InitSHA != initA {
				t.Fatalf("rd.InitSHA = %q, want %q", rd.InitSHA, initA)
			}
			if rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.0) > 1e-4 {
				t.Fatalf("rd.BaselineLoss = %v, want 6.0 (bob's 5.0 initB baseline must be excluded)", rd.BaselineLoss)
			}
			if rd.BaselineN != 2 {
				t.Fatalf("rd.BaselineN = %d, want 2 (carol + dave)", rd.BaselineN)
			}
		})
		e := w.entry(idx, "entry-0")
		if e.Gain == nil || math.Abs(*e.Gain-0.20) > 1e-4 {
			t.Fatalf("entry gain = %v, want +0.20 (NOT shifted by bob's 5.0 baseline)", e.Gain)
		}
		for _, gr := range e.Runs[0].Grades {
			if gr.Handle == "bob" && (gr.Counted || gr.Verdict != "other data") {
				t.Fatalf("bob's bad-init grade should be set aside as other data: %+v", gr)
			}
		}
		w.check()
	})

	t.Run("sole_bad_init_baseline_is_wiped_and_replaced_after_init_locks", func(t *testing.T) {
		w, _, idx := setupNoInitExam(t, 1)
		g := w.volunteers("bob", "carol", "dave", "erin", "frank")
		send := func(a *actor, c *claimed, initSHA string, loss, base float64) int {
			p := w.result(c, loss, base)
			p["train"].(map[string]any)["init_sha256"] = initSHA
			p["script"].(map[string]any)["tokens_sha256"] = tokA
			if base == 0 {
				delete(p, "baseline")
			} else {
				p["baseline"].(map[string]any)["init_sha256"] = initSHA
			}
			return a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, nil)
		}
		// Bob submits the ONLY baseline (initB, loss 5.0).
		cb := w.asked(g[0], true)
		if code := send(g[0], cb, initB, 4.9, 5.0); code != 200 {
			t.Fatalf("bob: HTTP %d", code)
		}
		// Carol & Dave are NOT asked for baseline initially (PerRound=1 filled by bob's sole baseline),
		// and lock rd.InitSHA = initA without sending a baseline!
		cc := w.asked(g[1], false)
		if code := send(g[1], cc, initA, 5.8, 0); code != 200 {
			t.Fatalf("carol: HTTP %d", code)
		}
		cd := w.asked(g[2], false)
		if code := send(g[2], cd, initA, 5.8, 0); code != 200 {
			t.Fatalf("dave: HTTP %d", code)
		}
		// Now rd.InitSHA == initA, bob's sole initB baseline MUST be wiped (no stale BaseItems or scores)!
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.InitSHA != initA || rd.BaselineLoss != nil || len(rd.BaseItems) != 0 {
				t.Fatalf("after sole bad-init baseline eviction: init=%q baseLoss=%v len(BaseItems)=%d", rd.InitSHA, rd.BaselineLoss, len(rd.BaseItems))
			}
		})
		// Erin is asked for a replacement baseline (Baseline=true), sends wrong initB -> refused 409,
		// releasing the baseline promise so Frank is asked next!
		ce := w.asked(g[3], true)
		if code := send(g[3], ce, initB, 5.8, 5.0); code != 409 {
			t.Fatalf("late bad-init baseline after lock: HTTP %d, want 409", code)
		}
		cf := w.asked(g[4], true)
		if code := send(g[4], cf, initA, 5.8, 6.0); code != 200 {
			t.Fatalf("frank replacement baseline: HTTP %d", code)
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatal("round did not settle after replacement baseline")
		}
		e := w.entry(idx, "entry-0")
		if e.Gain == nil || math.Abs(*e.Gain-0.20) > 1e-4 {
			t.Fatalf("entry gain = %v, want +0.20", e.Gain)
		}
		w.check()
	})
}

// Review 10 P1 #4: two baselines 0.016–0.030 apart (e.g. 6.00 vs 6.02 against tol 0.015) MUST be
// treated as disputed (not settled rated against their 6.01 midpoint), must request a third
// independent baseline, must leave ExamScores.BaselineOf empty if closed unrated, and must resolve
// cleanly if a third baseline arrives within ReplicaWait.
func TestTwoBaselinesBeyondToleranceButWithinTwiceSettleOnMidpoint(t *testing.T) {
	t.Run("unresolved_closes_unrated_with_empty_baseline_of", func(t *testing.T) {
		w, anchor, idx := toGrade(t, 2)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		g := w.volunteers("bob", "carol")
		c0 := w.asked(g[0], true)
		c1 := w.asked(g[1], true)
		if code, status := w.send(g[0], c0, 5.6, 6.00); code != 200 || status != "pending" {
			t.Fatalf("bob: HTTP %d %s", code, status)
		}
		if code, status := w.send(g[1], c1, 5.6, 6.02); code != 200 || status != "pending" {
			t.Fatalf("carol (0.020 > 0.015 apart) must NOT settle round: HTTP %d %s", code, status)
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "open" || rd.BaselineN != 0 || !w.app.baselineDisputed(s, rd) {
				t.Fatalf("expected open disputed round with BaselineN=0, got status=%s baseN=%d disputed=%v",
					rd.Status, rd.BaselineN, w.app.baselineDisputed(s, rd))
			}
		})
		if wk := w.work(anchor).Rounds[0]; !wk.BaselineDisputed || wk.BaselinesWanted == 0 || wk.GradesWanted != 1 || wk.FreshGradersWanted == nil || *wk.FreshGradersWanted == 0 {
			t.Fatalf("work must flag disputed baseline and request fresh tie-break grader: %+v", wk)
		}
		// After ReplicaWait expires unresolved, round closes unrated and archived ExamScores.BaselineOf is empty.
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.archiveExam(t.Context())
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "settled" || rd.BaselineLoss != nil || rd.BaselineN != 0 {
				t.Fatalf("unresolved round: status=%s loss=%v n=%d", rd.Status, rd.BaselineLoss, rd.BaselineN)
			}
			for _, en := range s.inRound[idx] {
				if s.rated(en) || !en.Void || !en.Unattended {
					t.Fatalf("entry must close void+unattended+unrated: %+v", en)
				}
			}
		})
		var scores ExamScores
		if code := anchor.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx)+"/scores", nil, &scores); code != 200 {
			t.Fatalf("GET archived scores: HTTP %d", code)
		}
		if len(scores.Baseline) != 0 || len(scores.BaselineOf) != 0 {
			t.Fatalf("unrated archived ExamScores must have empty Baseline and BaselineOf, got len(Baseline)=%d BaselineOf=%v",
				len(scores.Baseline), scores.BaselineOf)
		}
		w.check()
	})

	t.Run("third_baseline_resolves_0.020_spread", func(t *testing.T) {
		w, _, idx := toGrade(t, 2)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		g := w.volunteers("bob", "carol", "erin")
		c0 := w.asked(g[0], true)
		c1 := w.asked(g[1], true)
		if code, _ := w.send(g[0], c0, 5.6, 6.00); code != 200 {
			t.Fatalf("bob: HTTP %d", code)
		}
		if code, status := w.send(g[1], c1, 5.6, 6.02); code != 200 || status != "pending" {
			t.Fatalf("carol: HTTP %d %s", code, status)
		}
		// Erin reports 6.004 -> median(6.00, 6.004, 6.02) = 6.004; bob (0.004) and erin (0) agree,
		// while carol (0.016 > 0.015) is oddOut!
		c2 := w.asked(g[2], true)
		if code, _ := w.send(g[2], c2, 5.6, 6.004); code != 200 {
			t.Fatalf("erin: HTTP %d", code)
		}
		if c2.Job.Editor.Name != c1.Job.Editor.Name {
			cRe := w.claim(g[0], false)
			if cRe.Job == nil {
				t.Fatalf("expected bob to re-grade carol's oddOut script")
			}
			if code, status := w.send(g[0], cRe, 5.6, 6.00); code != 200 || status != "settled" {
				t.Fatalf("bob re-grade: HTTP %d %s", code, status)
			}
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "settled" || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.002) > 1e-6 || rd.BaselineN != 2 {
				t.Fatalf("resolved 0.020 spread: status=%s loss=%v n=%d", rd.Status, rd.BaselineLoss, rd.BaselineN)
			}
		})
		w.check()
	})
}

// Review 10 P2 #5: when two graders disagree on script.tokens_sha256 with no 3rd grader active yet,
// the entry must stay open for ReplicaWait so a deciding 3rd grader can arrive (via stet-cloud or
// volunteer), resolving cleanly if one arrives and voiding unattended only after ReplicaWait expires.
func TestTokensTieVoidsAtOnceWithoutWaitingForDecidingGrade(t *testing.T) {
	setupTokensTie := func(t *testing.T) (*world, *actor, int, []*actor) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("bob", "carol", "dave")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		c0, c1 := w.claimEarly(g[0]), w.claimEarly(g[1])
		w.exam(idx)
		p0 := w.result(c0, 5.5, 6.0)
		p0["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("a", 64)
		p1 := w.result(c1, 5.5, 6.0)
		p1["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("b", 64)
		if code := g[0].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c0.Job.ID), p0, nil); code != 200 {
			t.Fatalf("bob: HTTP %d", code)
		}
		if code := g[1].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c1.Job.ID), p1, nil); code != 200 {
			t.Fatalf("carol: HTTP %d", code)
		}
		// Immediately after the 2nd disagreeing grade, the entry MUST stay open (not voided!).
		w.app.st.view(func(s *State) {
			if en := s.inRound[idx][0]; en.Status != "open" {
				t.Fatalf("tokens tie settled immediately (%s, void=%v) instead of waiting ReplicaWait for a 3rd grader",
					en.Status, en.Void)
			}
		})
		if wk := w.work(anchor).Rounds[0]; wk.GradesWanted != 1 {
			t.Fatalf("work must request 1 deciding grade during ReplicaWait: %+v", wk)
		}
		return w, anchor, idx, g
	}

	t.Run("third_grader_resolves_within_replica_wait", func(t *testing.T) {
		w, _, idx, g := setupTokensTie(t)
		w.pass(w.cfg.ReplicaWait / 2)
		c2 := w.claim(g[2], false)
		if c2.Job == nil || c2.Job.Kind != "grade" {
			t.Fatalf("expected 3rd grader to receive deciding grade during ReplicaWait, got %+v", c2.Job)
		}
		p2 := w.result(c2, 5.5, 6.0)
		p2["script"].(map[string]any)["tokens_sha256"] = strings.Repeat("a", 64)
		if code := g[2].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c2.Job.ID), p2, nil); code != 200 {
			t.Fatalf("dave: HTTP %d", code)
		}
		if w.settled(idx) != 1 {
			t.Fatalf("expected round to settle once 3rd grader broke tokens tie")
		}
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid {
			t.Fatalf("resolved tokens tie should settle valid & rated: %+v", e)
		}
		w.check()
	})

	t.Run("unresolved_voids_after_replica_wait", func(t *testing.T) {
		w, _, idx, _ := setupTokensTie(t)
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected unresolved tokens tie to settle after ReplicaWait")
		}
		e := w.entry(idx, "entry-0")
		if !e.Void || e.Valid {
			t.Fatalf("unresolved tokens tie after ReplicaWait should void unattended: %+v", e)
		}
		w.check()
	})
}

// Gate 4 P2 #6: when 2 baselines disagree beyond tolerance on a PerRound=2 round, baselinesWanted
// must ask for only ONE tie-break baseline (3rd), NOT two simultaneous tie-breakers (3rd + 4th).
func TestDisputedBaselinesOnlyAskForOneSimultaneousTieBreaker(t *testing.T) {
	w, anchor, idx := toGrade(t, 4)
	w.app.season.Baselines.PerRound = 2
	w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
	g := w.volunteers("bob", "carol", "dave", "erin")

	c0 := w.asked(g[0], true)
	c1 := w.asked(g[1], true)
	if code, _ := w.send(g[0], c0, 5.6, 6.000); code != 200 {
		t.Fatalf("bob: HTTP %d", code)
	}
	if code, _ := w.send(g[1], c1, 5.6, 6.020); code != 200 {
		t.Fatalf("carol: HTTP %d", code)
	}
	// Disputed 2-baseline round with 0 tie-breakers live -> BaselinesWanted MUST be 1 (not 2!).
	if wk := w.work(anchor).Rounds[0]; !wk.BaselineDisputed || wk.BaselinesWanted != 1 {
		t.Fatalf("disputed 2-baseline round before tie-breaker claim: want BaselineDisputed=true BaselinesWanted=1, got %+v", wk)
	}
	// 3rd grader claims -> asks for baseline, and BaselinesWanted drops to 0 so 4th grader does NOT
	// also train a redundant 4th baseline on another open script!
	c2 := w.asked(g[2], true)
	if c2.Job.Baseline == nil || !*c2.Job.Baseline {
		t.Fatalf("3rd grader must be asked for the tie-break baseline")
	}
	if wk := w.work(anchor).Rounds[0]; wk.BaselinesWanted != 0 {
		t.Fatalf("while 3rd tie-break baseline is live, want BaselinesWanted=0, got %+v", wk)
	}
	c3 := w.asked(g[3], false)
	if c3.Job.Baseline != nil && *c3.Job.Baseline {
		t.Fatalf("4th grader must NOT be asked for a redundant 4th baseline while 3rd tie-breaker is live")
	}
	_ = idx
	w.check()
}

// Gate 4 P1 #2: releasing a pre-exam grade due to ExamWaitTimeout must NOT burn maxGradeAttempts
// (sc.Tries) to abandon innocent curators' scripts, and must NOT immediately re-offer pre-exam
// grades to the same Anchor account until the round's exam is actually posted.
func TestExamWaitTimeoutDoesNotBurnAttemptsOrReofferPreExamGrade(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
	idx := w.drawWithoutExam()
	ce := w.claim(anchor, true)
	if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
		t.Fatalf("edit: HTTP %d", code)
	}

	// Simulate 6 Anchor machines timing out waiting for the stalled exam (> maxGradeAttempts=5).
	anchors := make([]*actor, 6)
	for i := range anchors {
		a := w.rented(anchor, fmt.Sprintf("fleet-%d", i))
		anchors[i] = a

		c := w.claimEarly(a)
		if c.Job == nil || c.Job.Kind != "grade" {
			t.Fatalf("fleet-%d expected pre-exam grade, got %+v", i, c.Job)
		}
		if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID),
			map[string]any{"reason": examWaitTimeoutReason}, nil); code != 200 {
			t.Fatalf("fleet-%d release: HTTP %d", i, code)
		}
		// Immediately re-asking with early=true before exam posts MUST NOT hand the grade back!
		if again := w.claimEarly(a); again.Job != nil {
			t.Fatalf("fleet-%d was immediately re-handed pre-exam grade %d after exam wait timeout (infinite loop!)", i, again.Job.ID)
		}
	}

	// Review 16 P2 #3: an established volunteer who times out waiting for the exam must ALSO
	// respect the pre-exam cooldown (no immediate pre-exam reoffer), and once the exam posts must
	// regain eligibility to grade that exact script (not be permanently excluded by sc.Offered).
	bob := w.join("bob", nil)
	carol := w.join("carol", nil)
	w.established("bob", "carol")
	cBobEarly := w.claimEarly(bob)
	if cBobEarly.Job == nil || cBobEarly.Job.Kind != "grade" {
		t.Fatalf("bob expected pre-exam grade, got %+v", cBobEarly.Job)
	}
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cBobEarly.Job.ID),
		map[string]any{"reason": examWaitTimeoutReason}, nil); code != 200 {
		t.Fatalf("bob release: HTTP %d", code)
	}
	if again := w.claimEarly(bob); again.Job != nil {
		t.Fatalf("bob was immediately re-handed pre-exam grade %d after exam wait timeout", again.Job.ID)
	}

	// Gate 4 security check: forging examEndedBeforePostReason pre-exam on an open round OR
	// forging examWaitTimeoutReason post-exam MUST NOT bypass sc.Offered / j.quit.
	forger := w.join("forger", nil)
	w.established("forger")
	cForgePre := w.claimEarly(forger)
	if cForgePre.Job == nil || cForgePre.Job.Kind != "grade" {
		t.Fatalf("forger expected pre-exam grade, got %+v", cForgePre.Job)
	}
	if code := forger.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cForgePre.Job.ID),
		map[string]any{"reason": examEndedBeforePostReason}, nil); code != 200 {
		t.Fatalf("forger pre-exam release: HTTP %d", code)
	}
	if again := w.claimEarly(forger); again.Job != nil {
		t.Fatalf("forging examEndedBeforePostReason on open round must not allow immediate re-claim, got %+v", again.Job)
	}

	// Even after 6 anchor timeouts + 1 volunteer timeout (> maxGradeAttempts), the script MUST still be open!
	w.app.st.view(func(s *State) {
		en := s.inRound[idx][0]
		if en.Status != "open" || en.Void {
			t.Fatalf("entry abandoned after exam wait timeouts: status=%s void=%v", en.Status, en.Void)
		}
	})

	// Once the exam posts, the legitimate timed-out volunteer (bob) regains eligibility, while
	// forger (who forged examEndedBeforePostReason on an open round) remains excluded, AND a fresh
	// volunteer (carol) who claims post-exam and forges examWaitTimeoutReason cannot re-claim!
	w.exam(idx)
	if forgePost := w.claim(forger, false); forgePost.Job != nil {
		t.Fatalf("forger who faked examEndedBeforePostReason on open round must stay excluded post-exam, got %+v", forgePost.Job)
	}
	cCarolPost := w.claim(carol, false)
	if cCarolPost.Job == nil || cCarolPost.Job.Kind != "grade" {
		t.Fatalf("carol expected post-exam grade, got %+v", cCarolPost.Job)
	}
	if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cCarolPost.Job.ID),
		map[string]any{"reason": examWaitTimeoutReason}, nil); code != 200 {
		t.Fatalf("carol forged post-exam release: HTTP %d", code)
	}
	if carolAgain := w.claim(carol, false); carolAgain.Job != nil {
		t.Fatalf("forging examWaitTimeoutReason post-exam must not allow re-claim, got %+v", carolAgain.Job)
	}
	cBobPost := w.claim(bob, false)
	if cBobPost.Job == nil || cBobPost.Job.Kind != "grade" {
		t.Fatalf("after exam posts, timed-out established volunteer bob must regain eligibility to grade, got %+v", cBobPost.Job)
	}
	if code, status := w.send(bob, cBobPost, 5.5, 6.0); code != 200 || (status != "pending" && status != "settled") {
		t.Fatalf("bob post-exam grade: HTTP %d %s", code, status)
	}
	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 1 {
		t.Fatalf("expected entry settled after bob's post-exam grade, settled=%d", w.settled(idx))
	}
	// Verify timed-out anchor is also eligible in a fresh round.
	idx2 := w.drawWithoutExam()
	ce2 := w.claim(anchor, true)
	if code, _ := w.edit(anchor, ce2, "0\n2\n"); code != 200 {
		t.Fatalf("round 2 edit: HTTP %d", code)
	}
	cA0 := w.claimEarly(anchors[0])
	if code := anchors[0].do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cA0.Job.ID),
		map[string]any{"reason": examWaitTimeoutReason}, nil); code != 200 {
		t.Fatalf("anchor round 2 release: HTTP %d", code)
	}
	w.exam(idx2)
	cPost := w.claim(anchors[0], false)
	if cPost.Job == nil || cPost.Job.Kind != "grade" {
		t.Fatalf("after exam posts, timed-out anchor must be eligible to grade again, got %+v", cPost.Job)
	}
	if code, status := w.send(anchors[0], cPost, 5.5, 6.0); code != 200 || status != "settled" {
		t.Fatalf("post-exam grade: HTTP %d %s", code, status)
	}
	w.check()
}

// Review 11 P1 #1: when rd.InitSHA is unset (e.g. examiner CPU init failed) and two established
// graders report identical baseline loss (6.000) and script gain (+0.050) but DIFFERENT init_sha256,
// they MUST NOT be pooled into a rated result. The server must treat the round as disputed, leave
// BaselineN=0 and rd.InitSHA empty, request an independent measurement, and either resolve once a
// 3rd grader establishes consensus OR close unrated with zero curator penalty past ReplicaWait.
func TestConflictingInitSHAWIthEqualBaselinesDoesNotSettleRated(t *testing.T) {
	initA := strings.Repeat("a", 64)
	initB := strings.Repeat("b", 64)
	tokA := strings.Repeat("1", 64)

	setupConflict := func(t *testing.T) (*world, *actor, int, []*actor) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
		w.app.season.Baselines.PerRound = 2
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("bob", "carol", "dave")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		c0, c1 := w.claimEarly(g[0]), w.claimEarly(g[1])
		w.exam(idx)
		sendWithInit := func(a *actor, c *claimed, initSHA string) (int, string) {
			p := w.result(c, 5.950, 6.000)
			p["train"] = map[string]any{"init_sha256": initSHA}
			p["baseline"].(map[string]any)["init_sha256"] = initSHA
			p["script"].(map[string]any)["tokens_sha256"] = tokA
			var ack struct {
				Status string `json:"status"`
				Error  string `json:"error"`
			}
			code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
			if ack.Error != "" {
				return code, ack.Error
			}
			return code, ack.Status
		}

		if code, st := sendWithInit(g[0], c0, initA); code != 200 || st != "pending" {
			t.Fatalf("bob (initA): HTTP %d %s", code, st)
		}
		if code, st := sendWithInit(g[1], c1, initB); code != 200 || st != "pending" {
			t.Fatalf("carol (initB, equal baseline 6.000) must NOT settle immediately: HTTP %d %s", code, st)
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.Status != "open" || rd.InitSHA != "" || rd.BaselineN != 0 || !w.app.baselineDisputed(s, rd) {
				t.Fatalf("conflicting InitSHA with equal baselines: status=%s init=%q baseN=%d disputed=%v",
					rd.Status, rd.InitSHA, rd.BaselineN, w.app.baselineDisputed(s, rd))
			}
		})
		if wk := w.work(anchor).Rounds[0]; !wk.BaselineDisputed || wk.BaselinesWanted != 1 || wk.GradesWanted != 1 {
			t.Fatalf("work must request independent measurement for conflicting InitSHA: %+v", wk)
		}
		return w, anchor, idx, g
	}

	t.Run("unresolved_closes_unrated_after_replica_wait", func(t *testing.T) {
		w, _, idx, _ := setupConflict(t)
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected unresolved conflicting-InitSHA round to close after ReplicaWait")
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.BaselineLoss != nil || rd.BaselineN != 0 || len(rd.BaseItems) != 0 || rd.InitSHA != "" {
				t.Fatalf("closed unresolved InitSHA round must have empty baseline & init: loss=%v n=%d items=%d init=%q",
					rd.BaselineLoss, rd.BaselineN, len(rd.BaseItems), rd.InitSHA)
			}
		})
		e := w.entry(idx, "entry-0")
		if !e.Void || e.Valid {
			t.Fatalf("entry must close void/unrated (not valid +0.050!): %+v", e)
		}
		w.check()
	})

	t.Run("third_grader_resolves_init_sha_consensus", func(t *testing.T) {
		w, _, idx, g := setupConflict(t)
		c2 := w.asked(g[2], true)
		p2 := w.result(c2, 5.950, 6.000)
		p2["train"] = map[string]any{"init_sha256": initA}
		p2["baseline"].(map[string]any)["init_sha256"] = initA
		p2["script"].(map[string]any)["tokens_sha256"] = tokA
		if code := g[2].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c2.Job.ID), p2, nil); code != 200 {
			t.Fatalf("dave (initA): HTTP %d", code)
		}
		if w.settled(idx) != 1 {
			t.Fatalf("expected round to settle once 3rd grader locked initA consensus")
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.InitSHA != initA || rd.BaselineN != 2 || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.0) > 1e-6 {
				t.Fatalf("resolved initA round: init=%q baseN=%d loss=%v", rd.InitSHA, rd.BaselineN, rd.BaselineLoss)
			}
		})
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.050) > 1e-6 {
			t.Fatalf("resolved initA entry: %+v", e)
		}
		w.check()
	})
}

// Review 11 P2 #2: a numerical grade dispute (+0.010 vs +0.090 against tol 0.012) with no 3rd
// grader online yet MUST remain open and claimable through ReplicaWait (exposing GradesWanted=1
// and FreshGradersWanted=1 on /admin/work so stet-cloud can supply a deciding grader), then apply
// the documented midpoint fallback only after ReplicaWait expires.
func TestNumericalGradeDisputeWaitsThroughReplicaWaitForDecidingGrade(t *testing.T) {
	setupNumTie := func(t *testing.T) (*world, *actor, int, []*actor) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("bob", "carol", "dave")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		c0, c1 := w.claimEarly(g[0]), w.claimEarly(g[1])
		w.exam(idx)
		// Both report identical baseline 6.000, but gains +0.010 (loss 5.990) vs +0.090 (loss 5.910) -> diff 0.080 > 0.012!
		if code, st := w.send(g[0], c0, 5.990, 6.000); code != 200 || st != "pending" {
			t.Fatalf("bob (+0.010): HTTP %d %s", code, st)
		}
		if code, st := w.send(g[1], c1, 5.910, 6.000); code != 200 || st != "pending" {
			t.Fatalf("carol (+0.090) must NOT immediately settle midpoint: HTTP %d %s", code, st)
		}
		w.app.st.view(func(s *State) {
			if en := s.inRound[idx][0]; en.Status != "open" {
				t.Fatalf("numerical grade dispute settled immediately (%s) instead of waiting ReplicaWait", en.Status)
			}
		})
		if wk := w.work(anchor).Rounds[0]; wk.GradesWanted != 1 {
			t.Fatalf("work must expose GradesWanted=1 for numerical grade dispute: %+v", wk)
		}
		return w, anchor, idx, g
	}

	t.Run("third_grader_breaks_numerical_tie_within_replica_wait", func(t *testing.T) {
		w, _, idx, g := setupNumTie(t)
		w.pass(w.cfg.ReplicaWait / 2)
		c2 := w.claim(g[2], false)
		if c2.Job == nil || c2.Job.Kind != "grade" {
			t.Fatalf("expected 3rd grader to receive deciding grade, got %+v", c2.Job)
		}
		// 3rd grader agrees with bob (+0.012, loss 5.988) -> settles immediately at median +0.012 (not +0.050!).
		if code, st := w.send(g[2], c2, 5.988, 6.000); code != 200 || st != "settled" {
			t.Fatalf("dave deciding grade: HTTP %d %s", code, st)
		}
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.012) > 1e-6 {
			t.Fatalf("expected deciding grade median +0.012, got %+v", e)
		}
		w.check()
	})

	t.Run("unresolved_applies_documented_midpoint_fallback_after_replica_wait", func(t *testing.T) {
		w, _, idx, _ := setupNumTie(t)
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected unresolved numerical dispute to settle on fallback after ReplicaWait")
		}
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.050) > 1e-6 {
			t.Fatalf("expected midpoint fallback +0.050, got %+v", e)
		}
		if e.Detail == nil || !e.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on unresolved midpoint fallback: %+v", e.Detail)
		}
		w.check()
	})
}

// Review 12 P1 #1: in a 3-entry round with rd.InitSHA unset, if Grader 1 returns baseline A +
// grade A, Grader 2 releases its baseline job (leaving BaselineN=1, rd.InitSHA=""), and Grader 3
// returns a script-only grade with init B for Entry 1, Entry 1 MUST NOT settle rated against
// baseline A. It must request corroboration and either settle once consensus locks OR void
// unattended past ReplicaWait while preserving the matching 1-grader case.
func TestScriptGradeWithDifferentInitSHADoesNotSettleAgainstMismatchedBaseline(t *testing.T) {
	initA := strings.Repeat("a", 64)
	initB := strings.Repeat("b", 64)
	tokA := strings.Repeat("1", 64)

	sendWithInit := func(w *world, a *actor, c *claimed, examLoss, baseLoss float64, initSHA string) (int, string) {
		p := w.result(c, examLoss, baseLoss)
		p["train"] = map[string]any{"init_sha256": initSHA}
		if baseLoss > 0 {
			p["baseline"].(map[string]any)["init_sha256"] = initSHA
		} else {
			delete(p, "baseline")
		}
		p["script"].(map[string]any)["tokens_sha256"] = tokA
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		if ack.Error != "" {
			return code, ack.Error
		}
		return code, ack.Status
	}

	setupMismatch := func(t *testing.T) (*world, *actor, int, []*actor, string, string) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
		w.app.season.Baselines.PerRound = 2
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("bob", "carol", "dave", "erin")
		for i := 0; i < 3; i++ {
			w.submit(w.join(fmt.Sprintf("owner-%d", i), nil), fmt.Sprintf("entry-%d", i), fmt.Sprintf("print(%d)", i))
		}
		idx := w.drawWithoutExam()
		if w.editAll(anchor) != 3 {
			t.Fatalf("expected anchor to edit all 3 entries")
		}
		// Bob and Carol claim the 2 baseline jobs; Dave claims a script-only job on another entry.
		cBob := w.claimEarly(g[0])
		cCarol := w.claimEarly(g[1])
		cDave := w.claimEarly(g[2])
		if cBob.Job.Baseline == nil || !*cBob.Job.Baseline || cCarol.Job.Baseline == nil || !*cCarol.Job.Baseline {
			t.Fatalf("expected bob and carol to be asked for baselines")
		}
		if cDave.Job.Baseline == nil || *cDave.Job.Baseline {
			t.Fatalf("expected dave to be a script-only grade (baseline=false), got %+v", cDave.Job.Baseline)
		}
		// Examiner omits optional round init_sha256.
		w.exam(idx)
		// Bob submits baseline A + valid grade A (gain +0.050).
		if code, st := sendWithInit(w, g[0], cBob, 5.950, 6.000, initA); code != 200 || st != "pending" {
			t.Fatalf("bob (initA): HTTP %d %s", code, st)
		}
		// Carol releases her baseline job, leaving BaselineN=1 and rd.InitSHA="".
		w.release(g[1], cCarol)
		// Dave submits script-only valid grade with init B (loss 5.950 -> would look like +0.050 against baseline A!).
		if code, st := sendWithInit(w, g[2], cDave, 5.950, 0, initB); code != 200 || st != "pending" {
			t.Fatalf("dave (initB script-only): HTTP %d %s", code, st)
		}
		// Verify Dave's mismatched grade was NOT scored against baseline A.
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if rd.InitSHA != "" || rd.BaselineN != 1 {
				t.Fatalf("expected unlocked round with BaselineN=1, got init=%q baseN=%d", rd.InitSHA, rd.BaselineN)
			}
			for _, r := range s.Results {
				if r.RoundIdx == idx && r.InitSHA == initB && r.Score != nil {
					t.Fatalf("grade with initB must NOT be scored against baseline initA: score=%v", *r.Score)
				}
			}
		})
		// /admin/work must still want grades to corroborate Dave's entry.
		if wk := w.work(anchor).Rounds[0]; wk.GradesWanted == 0 {
			t.Fatalf("expected GradesWanted > 0 for baseline-vs-grade init mismatch: %+v", wk)
		}
		return w, anchor, idx, g, cBob.Job.Editor.Name, cDave.Job.Editor.Name
	}

	t.Run("unresolved_mismatched_entry_voids_unrated_after_replica_wait", func(t *testing.T) {
		w, _, idx, _, bobEntry, daveEntry := setupMismatch(t)
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 3 {
			t.Fatalf("expected all 3 entries to settle after ReplicaWait, got %d", w.settled(idx))
		}
		eb := w.entry(idx, bobEntry)
		if eb.Void || !eb.Valid || eb.Gain == nil || math.Abs(*eb.Gain-0.050) > 1e-6 {
			t.Fatalf("matching initA entry (1-grader case) must stay valid +0.050: %+v", eb)
		}
		ed := w.entry(idx, daveEntry)
		if !ed.Void || ed.Valid || (ed.Gain != nil && *ed.Gain != 0) {
			t.Fatalf("mismatched initB entry must settle void/unrated (not valid +0.050!): %+v", ed)
		}
		w.app.st.view(func(s *State) {
			for _, r := range s.Results {
				if r.RoundIdx == idx && r.InitSHA == initB && r.Verdict != "disputed" {
					t.Fatalf("expected mismatched initB grade verdict=disputed, got %q", r.Verdict)
				}
			}
		})
		w.check()
	})

	t.Run("corroboration_locks_initA_and_settles_valid", func(t *testing.T) {
		w, _, idx, g, _, daveEntry := setupMismatch(t)
		// Erin claims the remaining/mismatched scripts with initA -> locks rd.InitSHA=initA!
		for i := 0; i < 2; i++ {
			c := w.claim(g[3], false)
			if c.Job == nil {
				break
			}
			baseLoss := 0.0
			if c.Job.Baseline != nil && *c.Job.Baseline {
				baseLoss = 6.000
			}
			if code, st := sendWithInit(w, g[3], c, 5.960, baseLoss, initA); code != 200 {
				t.Fatalf("erin grade %d: HTTP %d %s", i, code, st)
			}
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			if rd := s.Rounds[idx]; rd.InitSHA != initA {
				t.Fatalf("expected rd.InitSHA locked to initA, got %q", rd.InitSHA)
			}
		})
		ed := w.entry(idx, daveEntry)
		if ed.Void || !ed.Valid || ed.Gain == nil || math.Abs(*ed.Gain-0.040) > 1e-6 {
			t.Fatalf("corroborated entry should settle at initA gain +0.040, got %+v", ed)
		}
		w.check()
	})

	t.Run("valid_initB_tied_with_invalid_and_majority_invalid_never_deadlock", func(t *testing.T) {
		// Gate 4 P1 #1 Case B: [Valid(initB, base=false), Invalid] must NOT deadlock after ReplicaWait.
		w, _, idx, g, _, daveEntry := setupMismatch(t)
		cBob2 := w.claim(g[0], false)
		for cBob2.Job != nil && cBob2.Job.Editor.Name != daveEntry {
			w.release(g[0], cBob2)
			cBob2 = w.claim(g[0], false)
		}
		if cBob2.Job == nil {
			t.Fatalf("expected bob to be able to claim dave's initB entry")
		}
		pInv := w.result(cBob2, 0, 6.000)
		pInv["valid"] = false
		pInv["invalid_reason"] = "bad script"
		pInv["baseline"].(map[string]any)["init_sha256"] = initA
		delete(pInv, "exam")
		delete(pInv, "train")
		if code := g[0].do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cBob2.Job.ID), pInv, nil); code != 200 {
			t.Fatalf("bob invalid grade on daveEntry: HTTP %d", code)
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 3 {
			t.Fatalf("expected [Valid(initB), Invalid] round to settle without deadlock, settled=%d", w.settled(idx))
		}
		ed := w.entry(idx, daveEntry)
		if !ed.Void || ed.Valid {
			t.Fatalf("expected [Valid(initB), Invalid] to void unrated, got %+v", ed)
		}
		w.check()

		// Gate 4 P1 #1 Case C: [Valid(initB, base=false), Invalid, Invalid] settles invalid cleanly
		// without deadlocking AND marks Dave's initB grade "void" (zero false outlier strike).
		w2, _, idx2, g2, _, daveEntry2 := setupMismatch(t)
		w2.app.season.Baselines.PerRound = 1 // keep Erin script-only so rd.InitSHA stays ""
		sendInvalid := func(a *actor) {
			c := w2.claim(a, false)
			for c.Job != nil && c.Job.Editor.Name != daveEntry2 {
				w2.release(a, c)
				c = w2.claim(a, false)
			}
			if c.Job == nil {
				t.Fatalf("expected %s to claim %s", a.handle, daveEntry2)
			}
			p := w2.result(c, 0, 6.000)
			p["valid"] = false
			p["invalid_reason"] = "bad script"
			if c.Job.Baseline != nil && *c.Job.Baseline {
				p["baseline"].(map[string]any)["init_sha256"] = initA
			} else {
				delete(p, "baseline")
			}
			delete(p, "exam")
			delete(p, "train")
			if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, nil); code != 200 {
				t.Fatalf("%s invalid grade: HTTP %d", a.handle, code)
			}
		}
		sendInvalid(g2[0])
		sendInvalid(g2[3])
		w2.pass(w2.cfg.ReplicaWait + time.Minute)
		if w2.settled(idx2) != 3 {
			t.Fatalf("expected [Valid(initB), Invalid, Invalid] round to settle without deadlock, settled=%d", w2.settled(idx2))
		}
		ed2 := w2.entry(idx2, daveEntry2)
		if ed2.Void || ed2.Valid {
			t.Fatalf("expected 2-to-1 invalid majority to settle invalid (not void/valid), got %+v", ed2)
		}
		w2.app.st.view(func(s *State) {
			if s.Rounds[idx2].InitSHA != "" {
				t.Fatalf("expected rd.InitSHA to remain empty in Case C, got %q", s.Rounds[idx2].InitSHA)
			}
			for _, r := range s.Results {
				if r.RoundIdx == idx2 && r.InitSHA == initB && r.Verdict != "void" {
					t.Fatalf("expected minority initB grade verdict=void (not outlier), got %q", r.Verdict)
				}
			}
			for _, acct := range s.Accounts {
				if acct.Handle == "dave" && acct.Strikes != 0 {
					t.Fatalf("dave must not receive an outlier strike for initB mismatch: strikes=%d", acct.Strikes)
				}
			}
		})
		w2.check()
	})

	t.Run("non_established_mismatched_init_gets_void_not_agreed_or_outlier", func(t *testing.T) {
		// Gate 4 P2 #2: while rd.InitSHA == "" and BaselineN == 1 (initA), a non-established
		// volunteer submitting initB must be marked "void" ("other data") — neither earning false
		// standing credit when |v.gain| <= tol.Grade nor taking a false strike when |v.gain| > tol.Grade.
		for _, tc := range []struct {
			name     string
			examLoss float64 // against base 6.000 -> gain 0.005 (<= 0.012) or 0.050 (> 0.012)
		}{
			{"small_gain_no_false_agreed", 5.995},
			{"large_gain_no_false_strike", 5.950},
		} {
			t.Run(tc.name, func(t *testing.T) {
				pub, priv := newKey()
				w := newWorld(t, pub)
				w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
				w.app.season.Baselines.PerRound = 2
				w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
				anchor := w.join("anchor-one", priv)
				newbie := w.join("newbie", nil)
				w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
				idx := w.drawWithoutExam()
				ce := w.claim(anchor, true)
				if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
					t.Fatalf("edit: HTTP %d", code)
				}
				cAnchor := w.claimEarly(anchor)
				cNewbie := w.claimEarly(newbie)
				w.exam(idx)
				// Non-established volunteer submits baseline B (loss 6.000, within 0.015) + grade B first.
				if code, _ := sendWithInit(w, newbie, cNewbie, tc.examLoss, 6.000, initB); code != 200 {
					t.Fatalf("newbie (initB): HTTP %d", code)
				}
				// Anchor submits baseline A + grade A (rd.InitSHA stays "" with 1 established voter) -> settles!
				if code, _ := sendWithInit(w, anchor, cAnchor, tc.examLoss, 6.000, initA); code != 200 {
					t.Fatalf("anchor (initA): HTTP %d", code)
				}
				w.pass(w.cfg.ReplicaWait + time.Minute)
				if w.settled(idx) != 1 {
					t.Fatalf("expected round to settle, got %d", w.settled(idx))
				}
				w.app.st.view(func(s *State) {
					if s.Rounds[idx].InitSHA != "" {
						t.Fatalf("%s: expected rd.InitSHA empty, got %q", tc.name, s.Rounds[idx].InitSHA)
					}
					for _, r := range s.Results {
						if r.RoundIdx == idx && r.InitSHA == initB && r.Verdict != "void" {
							t.Fatalf("%s: expected newbie initB verdict=void, got %q", tc.name, r.Verdict)
						}
					}
					for _, acct := range s.Accounts {
						if acct.Handle == "newbie" && (acct.Agreed != 0 || acct.Strikes != 0) {
							t.Fatalf("%s: newbie got Agreed=%d Strikes=%d, want 0/0", tc.name, acct.Agreed, acct.Strikes)
						}
					}
				})
				w.check()
			})
		}
	})
}

// Review 12 P2 #2 & #3 (server-side): on a pinned-student season, a valid grade MUST NOT copy
// train.init_sha256 into a missing or malformed baseline.init_sha256, and baseline.hardware from a
// disallowed backend (e.g. AMD ROCm on a CUDA-only season) is rejected with HTTP 400.
func TestPinnedStudentRejectsMissingMalformedBaselineInitAndDisallowedBaselineHardware(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
	w.app.season.GradeBackends = []string{"cuda"}
	anchor := w.join("anchor-one", priv)
	bob := w.volunteers("bob")[0]
	w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
	idx := w.draw()
	ce := w.claim(anchor, true)
	if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
		t.Fatalf("edit: HTTP %d", code)
	}
	c := w.asked(bob, true)
	w.app.cfg.Dev = false
	initA := strings.Repeat("a", 64)
	tokA := strings.Repeat("1", 64)
	nvHW := map[string]any{"device": "cuda", "backend": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 32.0, "cuda_capability": "12.0", "bf16_native": true}

	// 1. Absent baseline.init_sha256 alongside valid train.init_sha256 -> HTTP 400.
	pMissing := w.result(c, 5.95, 6.00)
	pMissing["hardware"] = nvHW
	pMissing["train"] = map[string]any{"init_sha256": initA}
	pMissing["script"].(map[string]any)["tokens_sha256"] = tokA
	delete(pMissing["baseline"].(map[string]any), "init_sha256")
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), pMissing, nil); code != 400 {
		t.Fatalf("absent baseline.init_sha256 must return HTTP 400, got %d", code)
	}

	// 2. Malformed baseline.init_sha256 alongside valid train.init_sha256 -> HTTP 400.
	pBad := w.result(c, 5.95, 6.00)
	pBad["hardware"] = nvHW
	pBad["train"] = map[string]any{"init_sha256": initA}
	pBad["script"].(map[string]any)["tokens_sha256"] = tokA
	pBad["baseline"].(map[string]any)["init_sha256"] = "not-a-valid-sha256"
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), pBad, nil); code != 400 {
		t.Fatalf("malformed baseline.init_sha256 must return HTTP 400, got %d", code)
	}

	// 3. Disallowed ROCm baseline.hardware on NVIDIA grade -> HTTP 400.
	pROCm := w.result(c, 5.95, 6.00)
	pROCm["hardware"] = nvHW
	pROCm["train"] = map[string]any{"init_sha256": initA}
	pROCm["script"].(map[string]any)["tokens_sha256"] = tokA
	pROCm["baseline"].(map[string]any)["init_sha256"] = initA
	pROCm["baseline"].(map[string]any)["hardware"] = map[string]any{"device": "cuda", "backend": "rocm", "gpu": "AMD Radeon 8060S"}
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), pROCm, nil); code != 400 {
		t.Fatalf("ROCm baseline.hardware on CUDA-only season must return HTTP 400, got %d", code)
	}

	// 4. Review 13 P2 #4: omitted baseline.hardware on hardware-restricted production season -> HTTP 400.
	pNoHW := w.result(c, 5.95, 6.00)
	pNoHW["hardware"] = nvHW
	pNoHW["train"] = map[string]any{"init_sha256": initA}
	pNoHW["script"].(map[string]any)["tokens_sha256"] = tokA
	pNoHW["baseline"].(map[string]any)["init_sha256"] = initA
	delete(pNoHW["baseline"].(map[string]any), "hardware")
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), pNoHW, nil); code != 400 {
		t.Fatalf("omitted baseline.hardware on CUDA-only production season must return HTTP 400, got %d", code)
	}

	// 5. Untagged CUDA baseline.hardware (missing backend) on CUDA-only production season -> HTTP 400.
	pUntagged := w.result(c, 5.95, 6.00)
	pUntagged["hardware"] = nvHW
	pUntagged["train"] = map[string]any{"init_sha256": initA}
	pUntagged["script"].(map[string]any)["tokens_sha256"] = tokA
	pUntagged["baseline"].(map[string]any)["init_sha256"] = initA
	pUntagged["baseline"].(map[string]any)["hardware"] = map[string]any{"device": "cuda", "gpu": "NVIDIA GeForce RTX 5090"}
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), pUntagged, nil); code != 400 {
		t.Fatalf("untagged CUDA baseline.hardware on CUDA-only production season must return HTTP 400, got %d", code)
	}

	// Verify NO baseline was stored by any of the rejected submissions.
	w.app.st.view(func(s *State) {
		if len(s.Baselines) != 0 || s.Rounds[idx].BaselineN != 0 {
			t.Fatalf("rejected submissions must not store baselines: count=%d baseN=%d", len(s.Baselines), s.Rounds[idx].BaselineN)
		}
	})

	// 6. Valid NVIDIA CUDA baseline.hardware succeeds with HTTP 200.
	pOK := w.result(c, 5.95, 6.00)
	pOK["hardware"] = nvHW
	pOK["train"] = map[string]any{"init_sha256": initA}
	pOK["script"].(map[string]any)["tokens_sha256"] = tokA
	pOK["baseline"].(map[string]any)["init_sha256"] = initA
	pOK["baseline"].(map[string]any)["hardware"] = nvHW
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), pOK, nil); code != 200 {
		t.Fatalf("valid NVIDIA CUDA baseline.hardware must return HTTP 200, got %d", code)
	}
}

// Review 13 P1 #2: three mutually inconsistent established volunteer grades (+0.010, +0.090, +0.050,
// every pair > 0.012 tolerance) have no agreeing pair and no agreeing organiser's grade. The third
// grade must keep the script disputed through ReplicaWait and settle with ZERO strikes on all three
// volunteers when no fourth grader agrees; once a fourth grader agrees with +0.050 (+0.051), the
// agreeing pair resolves the dispute and strikes the two outliers.
func TestThreeMutuallyInconsistentGradesCauseZeroStrikes(t *testing.T) {
	runThreeWay := func(t *testing.T, thirdIsAnchor bool, addFourthAgreeing bool) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		bob := w.join("bob", nil)
		carol := w.join("carol", nil)
		dave := w.join("dave", nil)
		erin := w.join("erin", nil)
		for _, h := range []string{"bob", "carol", "dave", "erin"} {
			w.established(h)
		}
		var thirdAnchor *actor
		if thirdIsAnchor {
			pk, sk := newKey()
			if code := w.vouch(anchor, pk); code != 201 {
				t.Fatalf("vouch anchor-two: HTTP %d", code)
			}
			thirdAnchor = w.join("anchor-two", sk)
		}
		w.submit(alice, "tidy", "print(1)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, scriptOne); code != 200 {
			t.Fatalf("anchor edit: HTTP %d", code)
		}
		// Bob and Carol both claim the script before the exam posts so two grades arrive.
		cBob := w.claimEarly(bob)
		cCarol := w.claimEarly(carol)
		w.exam(idx)

		// Submit matching baselines (6.000), matching init/tokens, and gains +0.010, +0.090, +0.050.
		initA := strings.Repeat("a", 64)
		tokA := strings.Repeat("1", 64)
		submitAt := func(a *actor, c *claimed, loss float64) string {
			p := w.result(c, loss, 6.000)
			p["train"] = map[string]any{"init_sha256": initA}
			p["script"].(map[string]any)["tokens_sha256"] = tokA
			if c.Job.Baseline != nil && *c.Job.Baseline {
				p["baseline"].(map[string]any)["init_sha256"] = initA
			} else {
				delete(p, "baseline")
			}
			var ack struct {
				Status string `json:"status"`
			}
			if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack); code != 200 {
				t.Fatalf("%s submit: HTTP %d", a.handle, code)
			}
			return ack.Status
		}
		gradeAt := func(a *actor, loss float64) string {
			c := w.claim(a, false)
			if c.Job == nil || c.Job.Kind != "grade" {
				t.Fatalf("%s expected grade job, got %+v", a.handle, c.Job)
			}
			return submitAt(a, c, loss)
		}

		// Bob (+0.010), Carol (+0.090), then third (+0.050).
		if st := submitAt(bob, cBob, 5.990); st != "pending" {
			t.Fatalf("after bob: status=%q, want pending", st)
		}
		if st := submitAt(carol, cCarol, 5.910); st != "pending" {
			t.Fatalf("after carol: status=%q, want pending", st)
		}
		third := dave
		if thirdIsAnchor {
			third = thirdAnchor
		}
		st3 := gradeAt(third, 5.950)

		if thirdIsAnchor {
			// Organiser's machine agreed with median +0.050 -> solid by published standing rule!
			if st3 != "settled" {
				t.Fatalf("anchor median (+0.050) should settle solid, got status=%q", st3)
			}
			if w.account("bob").Strikes != 1 || w.account("carol").Strikes != 1 {
				t.Fatalf("against anchor median, bob/carol should each have 1 strike: bob=%d carol=%d",
					w.account("bob").Strikes, w.account("carol").Strikes)
			}
			w.check()
			return
		}

		// Three mutually inconsistent volunteers MUST NOT immediately settle and strike!
		if st3 != "pending" || w.settled(idx) != 0 {
			t.Fatalf("three mutually inconsistent volunteer grades must NOT settle immediately: status=%q settled=%d", st3, w.settled(idx))
		}
		if addFourthAgreeing {
			// Fourth volunteer agrees with Dave (+0.051 within 0.012 of +0.050) -> resolves dispute!
			if st4 := gradeAt(erin, 5.949); st4 != "settled" {
				t.Fatalf("fourth agreeing grader should settle dispute, got status=%q", st4)
			}
			if w.account("bob").Strikes != 1 || w.account("carol").Strikes != 1 {
				t.Fatalf("expected 1 strike each for outliers bob/carol after agreeing pair, got bob=%d carol=%d",
					w.account("bob").Strikes, w.account("carol").Strikes)
			}
			if w.account("dave").Strikes != 0 || w.account("erin").Strikes != 0 {
				t.Fatalf("agreeing pair dave/erin must have 0 strikes, got dave=%d erin=%d",
					w.account("dave").Strikes, w.account("erin").Strikes)
			}
			w.check()
			return
		}

		// No fourth grader arrives -> wait out ReplicaWait -> settles disputed with ZERO strikes!
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected 3-way dispute to settle after ReplicaWait, settled=%d", w.settled(idx))
		}
		for _, h := range []string{"bob", "carol", "dave"} {
			if acct := w.account(h); acct.Strikes != 0 {
				t.Fatalf("three mutually inconsistent volunteer grades struck %s (strikes=%d, want 0)", h, acct.Strikes)
			}
		}
		en := w.entries(alice, idx)[0]
		if !en.Valid || en.Gain == nil || math.Abs(*en.Gain-0.050) > 1e-4 {
			t.Fatalf("expected settled median gain +0.050, got %+v", en)
		}
		if en.Detail == nil || !en.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on unresolved 3-way dispute, got %+v", en.Detail)
		}
		for _, gr := range en.Runs[0].Grades {
			if gr.Verdict != "disputed" {
				t.Fatalf("expected all 3 mutually inconsistent grades published as disputed, got %+v", en.Runs[0].Grades)
			}
		}
		w.check()
	}

	t.Run("three_volunteers_no_fourth_zero_strikes", func(t *testing.T) { runThreeWay(t, false, false) })
	t.Run("three_volunteers_fourth_agrees_strikes_outliers", func(t *testing.T) { runThreeWay(t, false, true) })
	t.Run("anchor_as_median_strikes_outliers", func(t *testing.T) { runThreeWay(t, true, false) })

	// Gate 4 P2 #4: an exact 2-2 Valid-vs-Invalid split among 4 established volunteers must
	// remain disputed (zero strikes on both Valid volunteers) instead of settling solid invalid.
	t.Run("two_valid_vs_two_invalid_split_zero_strikes", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Baselines.PerRound = 2
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		g := w.volunteers("bob", "carol", "dave", "erin")
		w.submit(alice, "tidy", "print(1)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, scriptOne); code != 200 {
			t.Fatalf("anchor edit: HTTP %d", code)
		}
		cs := make([]*claimed, len(g))
		for i, a := range g {
			cs[i] = w.claimEarly(a)
		}
		w.exam(idx)
		initA, tokA := strings.Repeat("a", 64), strings.Repeat("1", 64)
		for i, a := range g {
			p := w.result(cs[i], 5.950, 6.000)
			if i < 2 {
				p["train"] = map[string]any{"init_sha256": initA}
				p["script"].(map[string]any)["tokens_sha256"] = tokA
				if cs[i].Job.Baseline != nil && *cs[i].Job.Baseline {
					p["baseline"].(map[string]any)["init_sha256"] = initA
				} else {
					delete(p, "baseline")
				}
			} else {
				p["valid"], p["invalid_reason"] = false, "line 1: bad script"
				delete(p, "exam")
				delete(p, "baseline")
			}
			var ack struct {
				Status string `json:"status"`
			}
			if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cs[i].Job.ID), p, &ack); code != 200 {
				t.Fatalf("%s submit: HTTP %d", a.handle, code)
			}
			if i == 3 && ack.Status != "pending" {
				t.Fatalf("2-2 valid-vs-invalid split must stay pending through ReplicaWait, got %q", ack.Status)
			}
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected 2-2 split to settle after ReplicaWait, settled=%d", w.settled(idx))
		}
		for _, a := range g {
			if acct := w.account(a.handle); acct.Strikes != 0 {
				t.Fatalf("2-2 valid-vs-invalid split struck %s (strikes=%d, want 0)", a.handle, acct.Strikes)
			}
		}
		en := w.entries(alice, idx)[0]
		if !en.Void || en.Valid {
			t.Fatalf("2-2 validity split must void unattended (not count as invalid round!), got Void=%v Valid=%v", en.Void, en.Valid)
		}
		if en.Detail == nil || !en.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on 2-2 split, got %+v", en.Detail)
		}
		w.app.st.view(func(s *State) {
			raw := s.entry(idx, en.Editor)
			if raw == nil || !raw.Unattended {
				t.Fatalf("2-2 validity split must set raw.Unattended=true, got %+v", raw)
			}
			if failing(s, en.Editor) {
				t.Fatalf("2-2 validity tie must not mark curator as failing")
			}
		})
		w.check()
	})
}

// Review 12 P2 #8 (server-side): when a round's baselines are disputed but an idle organiser
// machine already online has NOT measured the baseline and can take the tie-breaking job,
// /admin/work explicitly emits fresh_graders_wanted: 0 (non-nil pointer) so stet-cloud honors
// the server's eligibility decision instead of evicting an idle worker.
func TestDisputedBaselineWithEligibleIdleAnchorEmitsExplicitZeroFreshGradersWanted(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.season.Baselines.PerRound = 2
	w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
	anchor := w.join("anchor-one", priv)
	newAnchor := func(name string) *actor {
		pk, sk := newKey()
		if code := w.vouch(anchor, pk); code != 201 {
			t.Fatalf("vouch %s: HTTP %d", name, code)
		}
		return w.join(name, sk)
	}
	a1 := newAnchor("stet-w1")
	a2 := newAnchor("stet-w2")
	a3 := newAnchor("stet-w3")
	w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
	idx := w.drawWithoutExam()
	ce := w.claim(anchor, true)
	if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
		t.Fatalf("edit: HTTP %d", code)
	}
	c1, c2 := w.claimEarly(a1), w.claimEarly(a2)
	w.exam(idx)
	// a1 and a2 submit conflicting baselines (6.000 vs 6.080 > 0.015), while a3 checks in idle.
	if code, _ := w.send(a1, c1, 5.950, 6.000); code != 200 {
		t.Fatalf("a1: HTTP %d", code)
	}
	if code, _ := w.send(a2, c2, 6.030, 6.080); code != 200 {
		t.Fatalf("a2: HTTP %d", code)
	}
	// Mark a3 recently active as an idle anchor who has NOT measured the baseline.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, acct := range tx.s.Accounts {
			if acct.Handle == a3.handle {
				acct.LastSeen = now()
				tx.saveAccount(acct)
			}
		}
		return nil
	})
	wk := w.work(anchor).Rounds[0]
	if !wk.BaselineDisputed || wk.GradesWanted != 1 {
		t.Fatalf("expected BaselineDisputed=true GradesWanted=1, got %+v", wk)
	}
	if wk.FreshGradersWanted == nil || *wk.FreshGradersWanted != 0 {
		t.Fatalf("eligible idle anchor a3 already suffices: want explicit non-nil FreshGradersWanted=0, got %v", wk.FreshGradersWanted)
	}
}

// Review 14 P1 #1: four mutually inconsistent established volunteer grades (+0.010, +0.090,
// +0.040, +0.060; every pair > 0.012 tolerance) have synthetic midpoint +0.050 within 0.010 of
// both +0.040 and +0.060 even though no actual pair agrees. Must remain pending through
// ReplicaWait and settle DisputedTogether=true with ZERO strikes on all four volunteers; once
// a fifth grader agrees with +0.040 (+0.042), or when the inner pair mutually agrees (+0.045,
// +0.055), the agreeing pair resolves the dispute and strikes only the outliers.
func TestFourMutuallyInconsistentGradesCauseZeroStrikes(t *testing.T) {
	setupFour := func(t *testing.T, gains []float64) (*world, int, *actor, []*actor, string) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		g := w.volunteers("bob", "carol", "dave", "erin", "frank")
		w.submit(alice, "tidy", "print(1)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, scriptOne); code != 200 {
			t.Fatalf("anchor edit: HTTP %d", code)
		}
		cBob := w.claimEarly(g[0])
		cCarol := w.claimEarly(g[1])
		w.exam(idx)

		initA, tokA := strings.Repeat("a", 64), strings.Repeat("1", 64)
		submitAt := func(a *actor, c *claimed, gain float64) string {
			p := w.result(c, 6.000-gain, 6.000)
			p["train"] = map[string]any{"init_sha256": initA}
			p["script"].(map[string]any)["tokens_sha256"] = tokA
			if c.Job.Baseline != nil && *c.Job.Baseline {
				p["baseline"].(map[string]any)["init_sha256"] = initA
			} else {
				delete(p, "baseline")
			}
			var ack struct {
				Status string `json:"status"`
			}
			if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack); code != 200 {
				t.Fatalf("%s submit: HTTP %d", a.handle, code)
			}
			return ack.Status
		}
		if st := submitAt(g[0], cBob, gains[0]); st != "pending" {
			t.Fatalf("after bob: status=%q, want pending", st)
		}
		if st := submitAt(g[1], cCarol, gains[1]); st != "pending" {
			t.Fatalf("after carol: status=%q, want pending", st)
		}
		var lastStatus string
		for i := 2; i < len(gains); i++ {
			c := w.claim(g[i], false)
			if c.Job == nil || c.Job.Kind != "grade" {
				t.Fatalf("%s expected grade job, got %+v", g[i].handle, c.Job)
			}
			lastStatus = submitAt(g[i], c, gains[i])
		}
		return w, idx, alice, g, lastStatus
	}

	t.Run("four_volunteers_no_agreeing_pair_zero_strikes", func(t *testing.T) {
		w, idx, alice, g, st4 := setupFour(t, []float64{0.010, 0.090, 0.040, 0.060})
		if st4 != "pending" || w.settled(idx) != 0 {
			t.Fatalf("four mutually inconsistent volunteer grades must stay pending: status=%q settled=%d", st4, w.settled(idx))
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected 4-way dispute to settle after ReplicaWait, settled=%d", w.settled(idx))
		}
		for _, a := range g[:4] {
			if acct := w.account(a.handle); acct.Strikes != 0 {
				t.Fatalf("four mutually inconsistent volunteer grades struck %s (strikes=%d, want 0)", a.handle, acct.Strikes)
			}
		}
		en := w.entries(alice, idx)[0]
		if en.Detail == nil || !en.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on unresolved 4-way dispute, got %+v", en.Detail)
		}
		for _, gr := range en.Runs[0].Grades {
			if gr.Verdict != "disputed" {
				t.Fatalf("expected all 4 mutually inconsistent grades published as disputed, got %+v", en.Runs[0].Grades)
			}
		}
		w.check()
	})

	t.Run("fifth_agreeing_grader_resolves_four_way_dispute", func(t *testing.T) {
		w, idx, _, g, st5 := setupFour(t, []float64{0.010, 0.090, 0.040, 0.060, 0.042})
		if st5 != "settled" || w.settled(idx) != 1 {
			t.Fatalf("fifth agreeing grader (+0.042 matching +0.040) should settle dispute: status=%q settled=%d", st5, w.settled(idx))
		}
		for _, h := range []string{"dave", "frank"} {
			if acct := w.account(h); acct.Strikes != 0 {
				t.Fatalf("agreeing pair member %s got %d strikes, want 0", h, acct.Strikes)
			}
		}
		for _, h := range []string{"bob", "carol", "erin"} {
			if acct := w.account(h); acct.Strikes != 1 {
				t.Fatalf("outlier %s got %d strikes, want 1", h, acct.Strikes)
			}
		}
		_ = g
		w.check()
	})

	t.Run("four_volunteers_inner_pair_mutually_agrees_settles_solid", func(t *testing.T) {
		w, idx, _, _, st4 := setupFour(t, []float64{0.010, 0.090, 0.045, 0.055})
		if st4 != "settled" || w.settled(idx) != 1 {
			t.Fatalf("agreeing inner pair (+0.045/+0.055 within 0.012) should settle solid: status=%q settled=%d", st4, w.settled(idx))
		}
		if w.account("bob").Strikes != 1 || w.account("carol").Strikes != 1 {
			t.Fatalf("expected 1 strike each on outer outliers, got bob=%d carol=%d", w.account("bob").Strikes, w.account("carol").Strikes)
		}
		if w.account("dave").Strikes != 0 || w.account("erin").Strikes != 0 {
			t.Fatalf("agreeing inner pair must have 0 strikes, got dave=%d erin=%d", w.account("dave").Strikes, w.account("erin").Strikes)
		}
		w.check()
	})
}

// Review 14 P2 #2: during a three-way baseline dispute (6.000, 6.080, 6.040; tolerance 0.015), a
// fourth independent grader repeating 6.000 must be accepted with HTTP 200 (not rejected with HTTP
// 409 against the disputed median 6.020) and resolve the round's baseline to 6.000 so subsequent
// outlier baselines (6.080) are rejected with 409. Four mutually inconsistent baselines (6.000,
// 6.080, 6.040, 6.060) and a 2-vs-2 split (6.000, 6.080, 6.000, 6.080) must both remain disputed.
func TestThreeWayBaselineDisputeAcceptsFourthAgreeingBaselineAndResolves(t *testing.T) {
	setupDispute := func(t *testing.T) (*world, int, *actor, []*actor) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("gr1", "gr2", "gr3", "gr4", "gr5")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		c1, c2 := w.claimEarly(g[0]), w.claimEarly(g[1])
		w.exam(idx)
		if code, _ := w.send(g[0], c1, 5.950, 6.000); code != 200 {
			t.Fatalf("g1 (6.000): HTTP %d", code)
		}
		if code, _ := w.send(g[1], c2, 6.030, 6.080); code != 200 {
			t.Fatalf("g2 (6.080): HTTP %d", code)
		}
		c3 := w.claim(g[2], false)
		if c3.Job == nil || c3.Job.Baseline == nil || !*c3.Job.Baseline {
			t.Fatalf("expected tie-breaking baseline job for g3, got %+v", c3.Job)
		}
		if code, _ := w.send(g[2], c3, 5.990, 6.040); code != 200 {
			t.Fatalf("g3 (6.040): HTTP %d", code)
		}
		w.app.st.view(func(s *State) {
			if !w.app.baselineDisputed(s, s.Rounds[idx]) {
				t.Fatalf("expected 3-way split (6.000/6.080/6.040) to be disputed")
			}
		})
		return w, idx, anchor, g
	}

	t.Run("fourth_grader_repeating_6000_resolves_dispute_and_subsequent_outlier_409s", func(t *testing.T) {
		w, idx, _, g := setupDispute(t)
		c4 := w.claim(g[3], false)
		if c4.Job == nil || c4.Job.Baseline == nil || !*c4.Job.Baseline {
			t.Fatalf("expected tie-breaking baseline job for g4, got %+v", c4.Job)
		}
		if code, st := w.send(g[3], c4, 5.950, 6.000); code != 200 {
			t.Fatalf("g4 repeating 6.000 must succeed with HTTP 200 (not 409), got HTTP %d (%s)", code, st)
		}
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if w.app.baselineDisputed(s, rd) {
				t.Fatalf("expected 6.000 pair (g1, g4) to resolve 3-way baseline dispute")
			}
			if rd.BaselineN != 2 || rd.BaselineLoss == nil || math.Abs(*rd.BaselineLoss-6.000) > 1e-4 {
				t.Fatalf("expected resolved baseline 6.000 (n=2), got %v (n=%d)", rd.BaselineLoss, rd.BaselineN)
			}
			left := w.app.baselinesLeftOut(s, rd)
			if !left[w.account("gr2").ID] || !left[w.account("gr3").ID] || left[w.account("gr1").ID] || left[w.account("gr4").ID] {
				t.Fatalf("expected gr2/gr3 oddOut and gr1/gr4 kept, got %+v", left)
			}
			// Now that baseline consensus is established at 6.000, baselineDev MUST reject 6.080!
			outlier := make([]float64, len(rd.ExamTable))
			for i := range outlier {
				outlier[i] = -6.080
			}
			dev, ok := w.app.baselineDev(s, rd, w.account("gr5").ID, outlier)
			if !ok || within(dev, w.app.tolerance().Baseline) {
				t.Fatalf("expected 6.080 rejected against resolved 6.000 baseline: dev=%.4f ok=%v", dev, ok)
			}
		})
		if w.settled(idx) != 1 {
			t.Fatalf("expected round to settle once baseline resolved and g1/g4 agreed, settled=%d", w.settled(idx))
		}
		w.check()
	})

	t.Run("four_mutually_inconsistent_baselines_stay_disputed", func(t *testing.T) {
		w, idx, _, g := setupDispute(t)
		c4 := w.claim(g[3], false)
		if code, _ := w.send(g[3], c4, 6.010, 6.060); code != 200 {
			t.Fatalf("g4 (6.060) during dispute should be stored with HTTP 200, got %d", code)
		}
		w.app.st.view(func(s *State) {
			if !w.app.baselineDisputed(s, s.Rounds[idx]) {
				t.Fatalf("4 mutually inconsistent baselines (6.000/6.040/6.060/6.080) must stay disputed")
			}
		})
		w.check()
	})

	t.Run("two_vs_two_baseline_tie_stays_disputed", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("gr1", "gr2", "gr3", "gr4")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		c1, c2 := w.claimEarly(g[0]), w.claimEarly(g[1])
		w.exam(idx)
		if code, _ := w.send(g[0], c1, 5.950, 6.000); code != 200 {
			t.Fatalf("gr1 (6.000): HTTP %d", code)
		}
		if code, _ := w.send(g[1], c2, 6.030, 6.080); code != 200 {
			t.Fatalf("gr2 (6.080): HTTP %d", code)
		}
		// Inject simultaneous second 6.000 and second 6.080 baselines (exact 2-vs-2 split).
		id1, id3, id4 := w.account("gr1").ID, w.account("gr3").ID, w.account("gr4").ID
		w.app.st.update(context.Background(), func(tx *Tx) error {
			rd := tx.s.Rounds[idx]
			initSHA := tx.s.Baselines[baselineKey(idx, id1)].InitSHA
			for _, pair := range []struct {
				id   int64
				loss float64
			}{{id3, 6.000}, {id4, 6.080}} {
				items := make([]float64, len(rd.ExamTable))
				for k := range items {
					items[k] = -pair.loss
				}
				tx.saveBaseline(&Baseline{
					RoundIdx: idx, AccountID: pair.id, Loss: pair.loss, Items: items,
					CorpusSHA: rd.CorpusSHA, ExamSHA: rd.ExamSHA, InitSHA: initSHA, Established: true, ReceivedAt: now(),
				})
			}
			w.app.refreshScores(tx, rd)
			return nil
		})
		w.app.st.view(func(s *State) {
			rd := s.Rounds[idx]
			if !w.app.baselineDisputed(s, rd) {
				t.Fatalf("exact 2-vs-2 baseline split (6.000/6.000 vs 6.080/6.080) must stay disputed")
			}
			if kept := w.app.baselinesKept(s, rd); len(kept) != 4 {
				t.Fatalf("2-vs-2 baseline tie must keep all 4 baselines (no arbitrary winner), got %d", len(kept))
			}
		})
		w.check()
	})
}

// Review 15 P1 #1: a 1-vs-1 (or 2-vs-2) validity tie among established graders must drop the
// disputed run at ReplicaWait expiry and void the entry as unattended (`Void=true, Unattended=true`)
// rather than counting as an invalid round toward curator retirement (`failing()`).
func TestValidityTieVoidsUnattendedWithoutRetiringCurator(t *testing.T) {
	t.Run("one_valid_one_invalid_across_two_rounds_never_retires_curator", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Baselines.PerRound = 2
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		bob := w.join("bob", nil)
		carol := w.join("carol", nil)
		newbie := w.join("newbie", nil) // non-established grader: should also be marked disputed
		w.established("bob", "carol")
		w.submit(alice, "tidy", "print(1)")

		initA, tokA := strings.Repeat("a", 64), strings.Repeat("1", 64)
		var editorID string
		for roundNum := 1; roundNum <= retireAfterInvalid; roundNum++ {
			idx := w.drawWithoutExam()
			ce := w.claim(anchor, true)
			if code, _ := w.edit(anchor, ce, scriptOne); code != 200 {
				t.Fatalf("round %d anchor edit: HTTP %d", roundNum, code)
			}
			cBob := w.claimEarly(bob)
			cCarol := w.claimEarly(carol)
			cNewbie := w.claimEarly(newbie)
			w.exam(idx)

			// Bob submits valid (+0.050), Carol submits invalid ("bad script") with matching baselines.
			pBob := w.result(cBob, 5.950, 6.000)
			pBob["train"] = map[string]any{"init_sha256": initA}
			pBob["script"].(map[string]any)["tokens_sha256"] = tokA
			pBob["baseline"].(map[string]any)["init_sha256"] = initA
			if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cBob.Job.ID), pBob, nil); code != 200 {
				t.Fatalf("round %d bob valid: HTTP %d", roundNum, code)
			}

			pCarol := w.result(cCarol, 0, 6.000)
			pCarol["valid"], pCarol["invalid_reason"] = false, "bad script"
			pCarol["baseline"].(map[string]any)["init_sha256"] = initA
			delete(pCarol, "exam")
			delete(pCarol, "train")
			var ack struct {
				Status string `json:"status"`
			}
			if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cCarol.Job.ID), pCarol, &ack); code != 200 {
				t.Fatalf("round %d carol invalid: HTTP %d", roundNum, code)
			}
			if ack.Status != "pending" {
				t.Fatalf("round %d 1-1 validity tie must stay pending through ReplicaWait, got %q", roundNum, ack.Status)
			}

			// Non-established newbie also submits valid; on a validity tie it must settle disputed (0 strikes/outliers).
			pNewbie := w.result(cNewbie, 5.950, 6.000)
			pNewbie["train"] = map[string]any{"init_sha256": initA}
			pNewbie["script"].(map[string]any)["tokens_sha256"] = tokA
			pNewbie["baseline"].(map[string]any)["init_sha256"] = initA
			if code := newbie.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cNewbie.Job.ID), pNewbie, nil); code != 200 {
				t.Fatalf("round %d newbie valid: HTTP %d", roundNum, code)
			}

			w.pass(w.cfg.ReplicaWait + time.Minute)
			if w.settled(idx) != 1 {
				t.Fatalf("round %d expected entry to settle after ReplicaWait, got %d", roundNum, w.settled(idx))
			}
			en := w.entries(alice, idx)[0]
			editorID = en.Editor
			if !en.Void || en.Valid {
				t.Fatalf("round %d 1-1 validity tie must settle Void=true Valid=false, got %+v", roundNum, en)
			}
			w.app.st.view(func(s *State) {
				raw := s.entry(idx, editorID)
				if raw == nil || !raw.Unattended {
					t.Fatalf("round %d 1-1 validity tie must set raw.Unattended=true, got %+v", roundNum, raw)
				}
			})
			if en.Detail == nil || !en.Detail.DisputedTogether {
				t.Fatalf("round %d expected DisputedTogether=true, got %+v", roundNum, en.Detail)
			}
			if len(en.Runs) != 0 || len(en.Dropped) != 1 {
				t.Fatalf("round %d expected tied run in Dropped (0 counted, 1 dropped), got runs=%+v dropped=%+v", roundNum, en.Runs, en.Dropped)
			}
			for _, gr := range en.Dropped[0].Grades {
				if gr.Verdict != "disputed" {
					t.Fatalf("round %d expected every grade verdict=disputed on 1-1 validity tie, got %+v", roundNum, en.Dropped[0].Grades)
				}
			}
			for _, h := range []string{"bob", "carol", "newbie"} {
				if acct := w.account(h); acct.Strikes != 0 {
					t.Fatalf("round %d 1-1 validity tie struck %s (%d strikes)", roundNum, h, acct.Strikes)
				}
			}
		}

		// After retireAfterInvalid (2) consecutive 1-1 validity ties, curator must NOT be retired!
		w.app.st.view(func(s *State) {
			if failing(s, editorID) {
				t.Fatalf("curator %s must NOT be retired by %d validity-tied rounds", editorID, retireAfterInvalid)
			}
		})
		// Confirm a 3rd round still draws this curator cleanly.
		idx3 := w.drawWithoutExam()
		if len(w.entries(alice, idx3)) != 1 {
			t.Fatalf("non-retired curator should still be scheduled in round 3")
		}
		w.check()
	})

	t.Run("two_runs_drops_tied_run_and_scores_remaining_valid_run", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.cfg.RunsPerEntry = 2
		w.app.season.Baselines.PerRound = 2
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		bob := w.join("bob", nil)
		carol := w.join("carol", nil)
		dave := w.join("dave", nil)
		w.established("bob", "carol", "dave")
		w.submit(alice, "tidy", "print(1)")
		idx := w.drawWithoutExam()

		ce1 := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce1, scriptOne); code != 200 {
			t.Fatalf("Run 1 edit: HTTP %d", code)
		}
		// Claim Bob and Carol onto Run 1 before Run 2's script is uploaded so assignment is deterministic.
		cBob := w.claimEarly(bob)
		cCarol := w.claimEarly(carol)
		ce2 := w.claim(dave, true)
		if code, _ := w.edit(dave, ce2, scriptTwo); code != 200 {
			t.Fatalf("Run 2 edit: HTTP %d", code)
		}
		cAnchor := w.claimEarly(anchor)
		w.exam(idx)

		initA, tokA := strings.Repeat("a", 64), strings.Repeat("1", 64)
		// Run 1 gets 1-1 validity tie (Bob valid +0.050 vs Carol invalid); Run 2 gets anchor valid (+0.040).
		sendValid := func(a *actor, c *claimed, loss float64) {
			p := w.result(c, loss, 6.000)
			p["train"] = map[string]any{"init_sha256": initA}
			p["script"].(map[string]any)["tokens_sha256"] = tokA
			if c.Job.Baseline != nil && *c.Job.Baseline {
				p["baseline"].(map[string]any)["init_sha256"] = initA
			} else {
				delete(p, "baseline")
			}
			if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, nil); code != 200 {
				t.Fatalf("%s valid: HTTP %d", a.handle, code)
			}
		}
		if cBob.Job.Run != 1 || cCarol.Job.Run != 1 {
			// Ensure whichever claim landed on Run 1 gets the tie and Run 2 gets valid.
			t.Fatalf("expected bob/carol on Run 1, got bob=%d carol=%d anchor=%d", cBob.Job.Run, cCarol.Job.Run, cAnchor.Job.Run)
		}
		sendValid(bob, cBob, 5.950)
		pCarol := w.result(cCarol, 0, 6.000)
		pCarol["valid"], pCarol["invalid_reason"] = false, "bad script"
		pCarol["baseline"].(map[string]any)["init_sha256"] = initA
		delete(pCarol, "exam")
		delete(pCarol, "train")
		if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cCarol.Job.ID), pCarol, nil); code != 200 {
			t.Fatalf("carol invalid: HTTP %d", code)
		}
		// Anchor grades Run 2 valid (+0.040).
		if cAnchor.Job.Run != 2 {
			w.release(anchor, cAnchor)
			cAnchor = w.claim(anchor, false)
		}
		sendValid(anchor, cAnchor, 5.960)
		w.pass(w.cfg.ReplicaWait + time.Minute)
		en := w.entries(alice, idx)[0]
		if en.Void || !en.Valid || en.Gain == nil || math.Abs(*en.Gain-0.040) > 1e-4 {
			t.Fatalf("expected entry to drop tied Run 1 and settle valid from Run 2 (+0.040), got %+v", en)
		}
		w.check()
	})
}

// Review 29 P1 #1: two incompatible pairs of established graders (+0.015/+0.015 vs -0.005/-0.005
// against tol 0.012, diff 0.020 > 0.012) MUST NOT settle as unanimous agreement at synthetic
// midpoint +0.005 in any submission order. The script MUST stay open for a 5th deciding grade
// (resolving cleanly when a 5th grade arrives) and publish disputed grades with
// DisputedTogether=true and zero strikes/credit if ReplicaWait expires unresolved.
func TestReview29IncompatiblePairsOfGradersRemainOpenForDecidingGrade(t *testing.T) {
	initA := strings.Repeat("a", 64)
	tokA := strings.Repeat("1", 64)

	sendGrade := func(t *testing.T, w *world, a *actor, c *claimed, gain float64) (int, string) {
		t.Helper()
		examLoss := 6.000 - gain
		p := w.result(c, examLoss, 6.000)
		p["train"] = map[string]any{"init_sha256": initA}
		p["script"].(map[string]any)["tokens_sha256"] = tokA
		if c.Job.Baseline != nil && *c.Job.Baseline {
			p["baseline"].(map[string]any)["init_sha256"] = initA
		} else {
			delete(p, "baseline")
		}
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		if ack.Error != "" {
			return code, ack.Error
		}
		return code, ack.Status
	}

	setupSplit := func(t *testing.T, order [4]float64) (*world, *actor, int, []*actor) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("bob", "carol", "dave", "erin", "frank", "grace", "heidi")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		claims := make([]*claimed, 4)
		for i := 0; i < 4; i++ {
			claims[i] = w.claimEarly(g[i])
		}
		w.exam(idx)
		for i := 0; i < 4; i++ {
			code, st := sendGrade(t, w, g[i], claims[i], order[i])
			if code != 200 || st != "pending" {
				t.Fatalf("order %v grader %d (%+.3f): HTTP %d status=%q, want 200 pending", order, i, order[i], code, st)
			}
		}
		w.app.st.view(func(s *State) {
			if en := s.inRound[idx][0]; en.Status != "open" {
				t.Fatalf("order %v settled prematurely (%s) instead of staying open for deciding grade", order, en.Status)
			}
		})
		if wk := w.work(anchor).Rounds[0]; wk.GradesWanted != 1 {
			t.Fatalf("order %v work must expose GradesWanted=1, got %+v", order, wk)
		}
		return w, anchor, idx, g
	}

	orders := [][4]float64{
		{+0.015, +0.015, -0.005, -0.005},
		{+0.015, -0.005, +0.015, -0.005},
		{+0.015, -0.005, -0.005, +0.015},
		{-0.005, +0.015, +0.015, -0.005},
		{-0.005, +0.015, -0.005, +0.015},
		{-0.005, -0.005, +0.015, +0.015},
	}

	t.Run("all_six_submission_orders_stay_open_and_fifth_grader_resolves", func(t *testing.T) {
		for _, ord := range orders {
			w, _, idx, g := setupSplit(t, ord)
			c5 := w.claim(g[4], false)
			if c5.Job == nil || c5.Job.Kind != "grade" {
				t.Fatalf("order %v: expected 5th verifier to claim deciding grade, got %+v", ord, c5.Job)
			}
			// 5th grader sides with +0.015 -> immediately settles at +0.015 with 3 agreed and 2 outliers.
			code, st := sendGrade(t, w, g[4], c5, +0.015)
			if code != 200 || st != "settled" {
				t.Fatalf("order %v: 5th grader (+0.015): HTTP %d status=%q, want 200 settled", ord, code, st)
			}
			e := w.entry(idx, "entry-0")
			if e.Void || !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.015) > 1e-6 {
				t.Fatalf("order %v: want settled gain +0.015, got %+v", ord, e)
			}
			if e.Detail == nil || e.Detail.DisputedTogether || e.Detail.Disputed != 2 {
				t.Fatalf("order %v: want DisputedTogether=false Disputed=2, got %+v", ord, e.Detail)
			}
			agreedN, outlierN := 0, 0
			for _, gr := range e.Runs[0].Grades {
				switch gr.Verdict {
				case "agreed":
					agreedN++
				case "outlier":
					outlierN++
				default:
					t.Fatalf("order %v: unexpected grade verdict %+v", ord, gr)
				}
			}
			if agreedN != 3 || outlierN != 2 {
				t.Fatalf("order %v: want 3 agreed + 2 outlier, got agreed=%d outlier=%d", ord, agreedN, outlierN)
			}
			w.check()
		}
	})

	t.Run("replica_wait_expiry_publishes_disputed_without_strikes_or_credit", func(t *testing.T) {
		w, _, idx, g := setupSplit(t, orders[0])
		preAgreed := make([]int, 4)
		w.app.st.view(func(s *State) {
			for i := 0; i < 4; i++ {
				acct := s.byHandle[g[i].handle]
				preAgreed[i] = acct.Agreed
			}
		})
		// Ensure the 5th/6th volunteers are not active so ReplicaWait expiry settles the dispute.
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatal("expected unresolved 2-vs-2 split to settle after ReplicaWait")
		}
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid || e.Gain == nil || math.Abs(*e.Gain-0.005) > 1e-6 {
			t.Fatalf("expected midpoint fallback +0.005 after ReplicaWait, got %+v", e)
		}
		if e.Detail == nil || !e.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on unresolved 2-vs-2 split, got %+v", e.Detail)
		}
		if len(e.Runs) != 1 || len(e.Runs[0].Grades) != 4 {
			t.Fatalf("expected 4 published grades, got %+v", e.Runs)
		}
		for _, gr := range e.Runs[0].Grades {
			if gr.Verdict != "disputed" {
				t.Fatalf("unresolved 2-vs-2 split grade %s verdict = %q, want disputed", gr.Handle, gr.Verdict)
			}
		}
		w.app.st.view(func(s *State) {
			for i := 0; i < 4; i++ {
				acct := s.byHandle[g[i].handle]
				if acct.Strikes != 0 || acct.Agreed != preAgreed[i] {
					t.Fatalf("grader %s must get zero strikes and zero agreed credit on unresolved split: strikes=%d agreed=%d (was %d)",
						acct.Handle, acct.Strikes, acct.Agreed, preAgreed[i])
				}
			}
		})
		w.check()
	})

	t.Run("five_graders_two_vs_two_plus_outlier_stays_open_until_true_majority", func(t *testing.T) {
		// Case A: 5th grader submits high outlier (+0.035) shifting median onto +0.015 while
		// -0.005 (2) and +0.015 (2) remain tied -> MUST stay open; 6th grader at +0.015 gives
		// 3-vs-2-vs-1 with matching middle pair (+0.015, +0.015) -> settles cleanly at +0.015.
		wA, _, idxA, gA := setupSplit(t, orders[0])
		c5A := wA.claim(gA[4], false)
		if code, st := sendGrade(t, wA, gA[4], c5A, +0.035); code != 200 || st != "pending" {
			t.Fatalf("5th grader outlier (+0.035) must NOT settle 2-vs-2 split: HTTP %d status=%q", code, st)
		}
		c6A := wA.claim(gA[5], false)
		if c6A.Job == nil || c6A.Job.Kind != "grade" {
			t.Fatalf("expected 6th grader to claim deciding grade, got %+v", c6A.Job)
		}
		if code, st := sendGrade(t, wA, gA[5], c6A, +0.015); code != 200 || st != "settled" {
			t.Fatalf("6th grader (+0.015) should resolve 3-vs-2-vs-1: HTTP %d status=%q", code, st)
		}
		eA := wA.entry(idxA, "entry-0")
		if eA.Gain == nil || math.Abs(*eA.Gain-0.015) > 1e-6 || eA.Detail == nil || eA.Detail.DisputedTogether {
			t.Fatalf("expected resolved gain +0.015 without DisputedTogether, got %+v", eA)
		}
		wA.check()

		// Case B: 5th grader submits +0.035 (stays open), then 6th grader submits -0.005:
		// sorted grades [-0.005, -0.005, -0.005, +0.015, +0.015, +0.035] STILL split 3 low vs
		// 3 high with synthetic midpoint +0.005, so MUST stay open for a 7th grader; 7th grader
		// at -0.005 gives 4-vs-2-vs-1 and settles cleanly at -0.005!
		wB, _, idxB, gB := setupSplit(t, orders[0])
		c5B := wB.claim(gB[4], false)
		if code, st := sendGrade(t, wB, gB[4], c5B, +0.035); code != 200 || st != "pending" {
			t.Fatalf("5th grader outlier (+0.035) must NOT settle 2-vs-2 split: HTTP %d status=%q", code, st)
		}
		c6B := wB.claim(gB[5], false)
		if code, st := sendGrade(t, wB, gB[5], c6B, -0.005); code != 200 || st != "pending" {
			t.Fatalf("6th grader (-0.005) on 3-low vs 3-high split must NOT settle on synthetic midpoint +0.005: HTTP %d status=%q", code, st)
		}
		c7B := wB.claim(gB[6], false)
		if c7B.Job == nil || c7B.Job.Kind != "grade" {
			t.Fatalf("expected 7th grader to claim deciding grade, got %+v", c7B.Job)
		}
		if code, st := sendGrade(t, wB, gB[6], c7B, -0.005); code != 200 || st != "settled" {
			t.Fatalf("7th grader (-0.005) should resolve 4-vs-2-vs-1: HTTP %d status=%q", code, st)
		}
		eB := wB.entry(idxB, "entry-0")
		if eB.Gain == nil || math.Abs(*eB.Gain-(-0.005)) > 1e-6 || eB.Detail == nil || eB.Detail.DisputedTogether {
			t.Fatalf("expected resolved gain -0.005 without DisputedTogether, got %+v", eB)
		}
		wB.check()
	})

	t.Run("maxAgreeingCluster_and_numericalSplit_order_independent_with_mixed_init_digests", func(t *testing.T) {
		pub, _ := newKey()
		w := newWorld(t, pub)
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		j := w.app.newJudge(w.app.st.s)
		mk := func(gain float64, initSHA string) *gradeView {
			return &gradeView{
				r:    &Result{Valid: true, InitSHA: initSHA},
				acct: &Account{},
				gain: gain,
				base: true,
			}
		}
		// Agreeing cluster has {"", B, B} (size 3) plus A (size 2 with "") vs rival {A, A} (size 2):
		// exact max cluster MUST be 3 regardless of whether A arrives before or after B!
		for _, order := range [][]*gradeView{
			{mk(-0.020, "A"), mk(-0.020, "A"), mk(+0.015, ""), mk(+0.015, "A"), mk(+0.015, "B"), mk(+0.015, "B"), mk(+0.050, "B")},
			{mk(-0.020, "A"), mk(-0.020, "A"), mk(+0.015, "B"), mk(+0.015, "B"), mk(+0.015, ""), mk(+0.015, "A"), mk(+0.050, "B")},
		} {
			v := &verdict{known: true, valid: true, scored: true, gain: +0.015, counted: order}
			if j.numericalSplit(v) {
				t.Fatalf("exact agreeing cluster size 3 (\"\", B, B) beats rival size 2 (A, A) regardless of order; got numericalSplit=true")
			}
		}
		// Equal-gain 2-vs-2 InitSHA split must return numericalSplit=true in every arrival order.
		for _, order := range [][]*gradeView{
			{mk(0.010, "A"), mk(0.010, "A"), mk(0.010, "B"), mk(0.010, "B")},
			{mk(0.010, "B"), mk(0.010, "A"), mk(0.010, "A"), mk(0.010, "B")},
		} {
			v := &verdict{known: true, valid: true, scored: true, gain: 0.010, counted: order}
			if !j.numericalSplit(v) {
				t.Fatalf("equal-gain 2-vs-2 InitSHA split must stay open in every arrival order; got numericalSplit=false")
			}
		}
	})
}

// Review 30 P1 #1: numericalSplit MUST evaluate mutually compatible groups across the FULL set of
// valid grades rather than partitioning around the median's tolerance band first, so the judge
// never strikes a member of the strongest agreeing group (e.g. 6-grader [-0.017, +0.019, -0.011,
// +0.040, +0.005, -0.006] where {-0.017, -0.011, -0.006} is the only 3-grader consensus) or
// strikes members of equal-support overlapping pairs (e.g. 4-grader [-0.015, -0.005, +0.005, +0.015]).
func TestReview30ConsensusDoesNotStrikeStrongestAgreeingGroup(t *testing.T) {
	initA := strings.Repeat("a", 64)
	tokA := strings.Repeat("1", 64)

	sendGrade := func(t *testing.T, w *world, a *actor, c *claimed, gain float64) (int, string) {
		t.Helper()
		examLoss := 6.000 - gain
		p := w.result(c, examLoss, 6.000)
		p["train"] = map[string]any{"init_sha256": initA}
		p["script"].(map[string]any)["tokens_sha256"] = tokA
		if c.Job.Baseline != nil && *c.Job.Baseline {
			p["baseline"].(map[string]any)["init_sha256"] = initA
		} else {
			delete(p, "baseline")
		}
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		if ack.Error != "" {
			return code, ack.Error
		}
		return code, ack.Status
	}

	setup := func(t *testing.T, gains []float64) (*world, *actor, int, []*actor) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("grd1", "grd2", "grd3", "grd4", "grd5", "grd6", "grd7", "grd8")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		earlyN := min(4, len(gains))
		claims := make([]*claimed, len(gains))
		for i := 0; i < earlyN; i++ {
			claims[i] = w.claimEarly(g[i])
			if claims[i].Job == nil {
				t.Fatalf("expected early claim for grader %d", i)
			}
		}
		w.exam(idx)
		for i, gain := range gains {
			if claims[i] == nil {
				claims[i] = w.claim(g[i], false)
				if claims[i].Job == nil || claims[i].Job.Kind != "grade" {
					t.Fatalf("expected post-exam grade claim for grader %d, got %+v", i, claims[i].Job)
				}
			}
			code, st := sendGrade(t, w, g[i], claims[i], gain)
			if code != 200 || st != "pending" {
				t.Fatalf("grader %d (%+.4f): HTTP %d status=%q, want 200 pending", i, gain, code, st)
			}
		}
		w.app.st.view(func(s *State) {
			if en := s.inRound[idx][0]; en.Status != "open" {
				t.Fatalf("settled prematurely (%s) instead of staying open for deciding grade", en.Status)
			}
		})
		if wk := w.work(anchor).Rounds[0]; wk.GradesWanted != 1 {
			t.Fatalf("expected GradesWanted=1 while unresolved, got %+v", wk)
		}
		return w, anchor, idx, g
	}

	sixGains := []float64{-0.017, +0.019, -0.011, +0.040, +0.005, -0.006}

	t.Run("six_graders_three_member_consensus_stays_open_and_seventh_resolves_without_striking_consensus", func(t *testing.T) {
		w, _, idx, g := setup(t, sixGains)
		c7 := w.claim(g[6], false)
		if c7.Job == nil || c7.Job.Kind != "grade" {
			t.Fatalf("expected 7th grader to claim deciding grade, got %+v", c7.Job)
		}
		// 7th grader at -0.011 confirms the 3-member consensus -> settles cleanly at -0.006!
		if code, st := sendGrade(t, w, g[6], c7, -0.011); code != 200 || st != "settled" {
			t.Fatalf("7th grader (-0.011) should resolve cleanly: HTTP %d status=%q", code, st)
		}
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid || e.Gain == nil || math.Abs(*e.Gain-(-0.006)) > 1e-6 {
			t.Fatalf("expected resolved gain -0.006, got %+v", e)
		}
		if e.Detail == nil || e.Detail.DisputedTogether {
			t.Fatalf("expected clean resolution (DisputedTogether=false), got %+v", e.Detail)
		}
		// Consensus members (-0.017, -0.011, -0.006, +0.005, -0.011) MUST have zero strikes;
		// only genuine outliers (+0.019, +0.040) receive strikes.
		w.app.st.view(func(s *State) {
			for i, wantStrike := range []int{0, 1, 0, 1, 0, 0, 0} {
				acct := s.byHandle[g[i].handle]
				if acct.Strikes != wantStrike {
					t.Fatalf("grader %d (%s): strikes=%d, want %d", i, acct.Handle, acct.Strikes, wantStrike)
				}
			}
		})
		w.check()
	})

	t.Run("six_graders_replica_wait_expiry_preserves_no_strike_disputed_outcome", func(t *testing.T) {
		w, _, idx, g := setup(t, sixGains)
		preAgreed := make([]int, 6)
		w.app.st.view(func(s *State) {
			for i := 0; i < 6; i++ {
				preAgreed[i] = s.byHandle[g[i].handle].Agreed
			}
		})
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatal("expected unresolved 6-grader dispute to settle after ReplicaWait")
		}
		e := w.entry(idx, "entry-0")
		if e.Void || !e.Valid || e.Detail == nil || !e.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on ReplicaWait expiry, got %+v", e)
		}
		for _, gr := range e.Runs[0].Grades {
			if gr.Verdict != "disputed" {
				t.Fatalf("grade %s verdict = %q, want disputed", gr.Handle, gr.Verdict)
			}
		}
		w.app.st.view(func(s *State) {
			for i := 0; i < 6; i++ {
				acct := s.byHandle[g[i].handle]
				if acct.Strikes != 0 || acct.Agreed != preAgreed[i] {
					t.Fatalf("grader %s must get zero strikes and zero agreed credit on expired dispute: strikes=%d agreed=%d (was %d)",
						acct.Handle, acct.Strikes, acct.Agreed, preAgreed[i])
				}
			}
		})
		w.check()
	})

	t.Run("seven_graders_overlapping_3_vs_3_tie_stays_open_for_eighth", func(t *testing.T) {
		w, _, idx, g := setup(t, sixGains)
		c7 := w.claim(g[6], false)
		// 7th grader at +0.005 creates overlapping 3-vs-3 tie: {-0.017,-0.011,-0.006} vs {-0.006,+0.005,+0.005}.
		if code, st := sendGrade(t, w, g[6], c7, +0.005); code != 200 || st != "pending" {
			t.Fatalf("7th grader (+0.005) overlapping 3-vs-3 tie must stay open: HTTP %d status=%q", code, st)
		}
		c8 := w.claim(g[7], false)
		if c8.Job == nil || c8.Job.Kind != "grade" {
			t.Fatalf("expected 8th grader to claim deciding grade, got %+v", c8.Job)
		}
		if code, st := sendGrade(t, w, g[7], c8, +0.005); code != 200 || st != "settled" {
			t.Fatalf("8th grader (+0.005) should resolve 4-vs-3 cleanly: HTTP %d status=%q", code, st)
		}
		e := w.entry(idx, "entry-0")
		if e.Gain == nil || math.Abs(*e.Gain-0.005) > 1e-6 || e.Detail == nil || e.Detail.DisputedTogether {
			t.Fatalf("expected clean +0.005 settlement after 8th grader, got %+v", e)
		}
		w.check()
	})

	t.Run("four_graders_three_equal_adjacent_pairs_stays_open_without_strikes", func(t *testing.T) {
		fourGains := []float64{-0.015, -0.005, +0.005, +0.015}
		// Branch A: 5th grader at +0.005 breaks the 3-way tie cleanly at +0.005.
		wA, _, idxA, gA := setup(t, fourGains)
		c5 := wA.claim(gA[4], false)
		if code, st := sendGrade(t, wA, gA[4], c5, +0.005); code != 200 || st != "settled" {
			t.Fatalf("5th grader (+0.005) should resolve 4-grader adjacent-pairs tie: HTTP %d status=%q", code, st)
		}
		eA := wA.entry(idxA, "entry-0")
		if eA.Gain == nil || math.Abs(*eA.Gain-0.005) > 1e-6 || eA.Detail == nil || eA.Detail.DisputedTogether {
			t.Fatalf("expected clean +0.005 settlement after 5th grader, got %+v", eA)
		}
		wA.check()

		// Branch B: ReplicaWait expires -> DisputedTogether=true with ZERO strikes!
		wB, _, idxB, gB := setup(t, fourGains)
		wB.pass(wB.cfg.ReplicaWait + time.Minute)
		eB := wB.entry(idxB, "entry-0")
		if eB.Detail == nil || !eB.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true after ReplicaWait on 4-grader tie, got %+v", eB)
		}
		wB.app.st.view(func(s *State) {
			for i := 0; i < 4; i++ {
				if acct := s.byHandle[gB[i].handle]; acct.Strikes != 0 {
					t.Fatalf("grader %s got %d strike(s) on unresolved 4-grader tie, want 0", acct.Handle, acct.Strikes)
				}
			}
		})
		wB.check()
	})

	t.Run("mixed_valid_and_invalid_does_not_let_weaker_or_tied_valid_cluster_strike_invalid_group", func(t *testing.T) {
		sendInvalid := func(t *testing.T, w *world, a *actor, c *claimed) (int, string) {
			t.Helper()
			p := w.result(c, 0, 6.000)
			p["valid"] = false
			delete(p, "exam")
			if c.Job.Baseline != nil && *c.Job.Baseline {
				p["baseline"].(map[string]any)["init_sha256"] = initA
			} else {
				delete(p, "baseline")
			}
			var ack struct {
				Status string `json:"status"`
				Error  string `json:"error"`
			}
			code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
			if ack.Error != "" {
				return code, ack.Error
			}
			return code, ack.Status
		}

		// Case A (7 graders): [Invalid, Invalid, Valid(-0.050), Valid(+0.050), Valid(+0.010), Invalid, Valid(+0.010)]
		// 2-grader Valid(+0.010) cluster MUST NOT settle and strike 3 mutually agreeing Invalid graders!
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.season.Student = map[string]any{"kind": "hf_topreset", "repo": "baidu/ERNIE-4.5-0.3B-Base-PT"}
		w.app.season.Baselines.PerRound = 2
		w.app.season.Tolerance[w.cfg.Scale] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		anchor := w.join("anchor-one", priv)
		g := w.volunteers("inv1", "inv2", "val1", "val2", "val3", "inv3", "val4")
		w.submit(w.join("owner-0", nil), "entry-0", "print(0)")
		idx := w.drawWithoutExam()
		ce := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce, "0\n1\n"); code != 200 {
			t.Fatalf("edit: HTTP %d", code)
		}
		c0, c1, c2, c3 := w.claimEarly(g[0]), w.claimEarly(g[1]), w.claimEarly(g[2]), w.claimEarly(g[3])
		w.exam(idx)
		for i, res := range []struct {
			code int
			st   string
		}{
			func() struct {
				code int
				st   string
			} {
				c, s := sendInvalid(t, w, g[0], c0)
				return struct {
					code int
					st   string
				}{c, s}
			}(),
			func() struct {
				code int
				st   string
			} {
				c, s := sendInvalid(t, w, g[1], c1)
				return struct {
					code int
					st   string
				}{c, s}
			}(),
			func() struct {
				code int
				st   string
			} {
				c, s := sendGrade(t, w, g[2], c2, -0.050)
				return struct {
					code int
					st   string
				}{c, s}
			}(),
			func() struct {
				code int
				st   string
			} {
				c, s := sendGrade(t, w, g[3], c3, +0.050)
				return struct {
					code int
					st   string
				}{c, s}
			}(),
		} {
			if res.code != 200 || res.st != "pending" {
				t.Fatalf("early grader %d: HTTP %d status=%q, want 200 pending", i, res.code, res.st)
			}
		}
		c4 := w.claim(g[4], false)
		if code, st := sendGrade(t, w, g[4], c4, +0.010); code != 200 || st != "pending" {
			t.Fatalf("5th grader Valid(+0.010): HTTP %d status=%q, want 200 pending", code, st)
		}
		c5 := w.claim(g[5], false)
		if code, st := sendInvalid(t, w, g[5], c5); code != 200 || st != "pending" {
			t.Fatalf("6th grader Invalid: HTTP %d status=%q, want 200 pending", code, st)
		}
		c6 := w.claim(g[6], false)
		if code, st := sendGrade(t, w, g[6], c6, +0.010); code != 200 || st != "pending" {
			t.Fatalf("7th grader Valid(+0.010) must NOT strike 3-grader Invalid consensus: HTTP %d status=%q", code, st)
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		e := w.entry(idx, "entry-0")
		if e.Detail == nil || !e.Detail.DisputedTogether {
			t.Fatalf("expected DisputedTogether=true on unresolved 2-valid vs 3-invalid split, got %+v", e)
		}
		w.app.st.view(func(s *State) {
			for i := range g {
				if acct := s.byHandle[g[i].handle]; acct.Strikes != 0 {
					t.Fatalf("grader %s received %d strike(s) on unresolved valid/invalid split, want 0", acct.Handle, acct.Strikes)
				}
			}
		})
		w.check()

		// Case B (5 graders): [Valid(-0.005), Valid(+0.005), Valid(+0.015), Invalid, Invalid] where all 3
		// valid grades agree with +0.005 (zero valid disagreeing) yet maxAgreeingCluster(valid)=2 == invalidCount=2!
		j := w.app.newJudge(w.app.st.s)
		mkV := func(gain float64) *gradeView {
			return &gradeView{r: &Result{Valid: true, InitSHA: initA, TokensSHA: tokA}, acct: &Account{}, gain: gain, base: true}
		}
		mkInv := func() *gradeView {
			return &gradeView{r: &Result{Valid: false}, acct: &Account{}, base: true}
		}
		vTie := &verdict{
			known: true, valid: true, scored: true, gain: +0.005,
			counted: []*gradeView{mkV(-0.005), mkV(+0.005), mkV(+0.015), mkInv(), mkInv()},
		}
		if !j.numericalSplit(vTie) {
			t.Fatal("2-member valid clique tied with 2 invalid grades (zero valid disagreeing) must set numericalSplit=true")
		}
		// Clean 3-agreeing-valid vs 2-invalid MUST still settle (numericalSplit=false).
		vClean := &verdict{
			known: true, valid: true, scored: true, gain: +0.005,
			counted: []*gradeView{mkV(+0.005), mkV(+0.005), mkV(+0.005), mkInv(), mkInv()},
		}
		if j.numericalSplit(vClean) {
			t.Fatal("3 mutually agreeing valid grades vs 2 invalid grades should settle (numericalSplit=false)")
		}
	})
}

// Review 32 P1 #1: Restarting the server with a different default scale (STET_SCALE) on the same
// store MUST judge every already-drawn round against its persisted Round.Scale across baseline
// agreement, grade agreement, and run agreement — exercised in BOTH directions (standard -> nano
// and nano -> standard) all the way through settlement.
func TestReview32RoundJudgedAgainstPersistedRoundScaleAcrossServerRestart(t *testing.T) {
	initSHA := strings.Repeat("b", 64)
	tokSHA := strings.Repeat("c", 64)

	roundState := func(w *world, idx int) Round {
		var rd Round
		w.app.st.view(func(s *State) {
			if r := s.Rounds[idx]; r != nil {
				rd = *r
			}
		})
		return rd
	}

	submitGradeWithBase := func(t *testing.T, w *world, idx int, a *actor, c *claimed, baseLoss, gain float64) (int, string) {
		t.Helper()
		rd := roundState(w, idx)
		n := len(rd.ExamTable)
		baseItems := make([]float64, n)
		examItems := make([]float64, n)
		for i := range n {
			baseItems[i] = -baseLoss
			examItems[i] = -(baseLoss - gain)
		}
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		p := map[string]any{
			"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
			"round": c.Job.Round["id"], "scale": c.Job.Scale,
			"corpus_sha256": rd.CorpusSHA, "exam_sha256": rd.ExamSHA,
			"script_sha256": c.Job.Script.SHA256, "valid": true,
			"hardware": map[string]any{"gpu": "NVIDIA RTX 4090", "vram_gb": 24, "backend": "cuda", "cc": "8.9", "bf16_native": true},
			"train":    map[string]any{"seconds": 120.0, "init_sha256": initSHA},
			"script":   map[string]any{"tokens_sha256": tokSHA},
			"exam":     map[string]any{"mean": -(baseLoss - gain), "items": n, "item_scores": examItems},
			"baseline": map[string]any{
				"mean": -baseLoss, "items": n, "item_scores": baseItems, "init_sha256": initSHA,
				"hardware": map[string]any{"gpu": "NVIDIA RTX 4090", "vram_gb": 24, "backend": "cuda", "cc": "8.9", "bf16_native": true},
			},
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		if ack.Error != "" {
			return code, ack.Error
		}
		return code, ack.Status
	}

	repoint := func(w *world, actors ...*actor) {
		for _, a := range actors {
			a.srv = w.srv
		}
	}

	t.Run("standard_round_after_restart_to_nano_enforces_standard_baseline_grade_and_run_tolerances", func(t *testing.T) {
		pub, priv := newKey()
		kv := newMemKV()
		cfg := testConfig(t, pub)
		cfg.Scale = "standard"
		cfg.RunsPerEntry = 2
		blobs := dirBlobs{filepath.Join(cfg.DataDir, "blobs")}
		w1 := startWorld(t, cfg, kv, blobs)
		w1.app.season.Baselines.PerRound = 2
		w1.app.season.Tolerance["standard"] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		w1.app.season.Tolerance["nano"] = Tolerance{Grade: 0.15, Baseline: 0.20, Run: 0.20}

		anchor := w1.join("r32-s2n-anchor", priv)
		g := w1.volunteers("r32-s2n-g1", "r32-s2n-g2", "r32-s2n-g3", "r32-s2n-g4", "r32-s2n-g5", "r32-s2n-g6", "r32-s2n-r3")
		w1.submit(w1.join("r32-s2n-owner", nil), "entry-std", "print(1)")
		idx := w1.drawWithoutExam()
		// Submit run 1 and hold run 2 open while baseline + grade agreement run on script 1.
		ce1 := w1.claim(anchor, true)
		if code, _ := w1.edit(anchor, ce1, "0\n1\n"); code != 200 {
			t.Fatalf("edit1: HTTP %d", code)
		}
		ce2 := w1.claim(g[4], true)
		if ce2.Job == nil || ce2.Job.Kind != "edit" {
			t.Fatalf("expected run 2 edit claim, got %+v", ce2.Job)
		}
		c1 := w1.claimEarly(g[0])
		c2 := w1.claimEarly(g[1])
		c3 := w1.claimEarly(g[2])
		if c1.Job.Scale != "standard" || c2.Job.Scale != "standard" || c3.Job.Scale != "standard" {
			t.Fatalf("expected standard claims before restart, got %q/%q/%q", c1.Job.Scale, c2.Job.Scale, c3.Job.Scale)
		}
		w1.exam(idx)
		// Verifier 1 submits baseline 6.00 (gain +0.100) BEFORE restart.
		if code, st := submitGradeWithBase(t, w1, idx, g[0], c1, 6.00, 0.100); code != 200 || st != "pending" {
			t.Fatalf("g1 submit before restart: HTTP %d %q", code, st)
		}

		// Restart server on the SAME store with STET_SCALE=nano!
		cfgNano := cfg
		cfgNano.Scale = "nano"
		w2 := startWorld(t, cfgNano, kv, blobs)
		w2.app.season.Baselines.PerRound = 2
		w2.app.season.Tolerance["standard"] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		w2.app.season.Tolerance["nano"] = Tolerance{Grade: 0.15, Baseline: 0.20, Run: 0.20}
		repoint(w2, anchor, g[0], g[1], g[2], g[3], g[4], g[5], g[6])

		// 1. Baseline comparison: Verifier 2 submits baseline 6.02 (|0.02| > 0.015 standard tol, < 0.20 nano tol) ->
		//    MUST remain disputed (NOT settle with 2 baselines)!
		if code, st := submitGradeWithBase(t, w2, idx, g[1], c2, 6.02, 0.100); code != 200 {
			t.Fatalf("g2 submit after restart: HTTP %d %q", code, st)
		}
		var disputed bool
		w2.app.st.view(func(s *State) {
			disputed = w2.app.baselineDisputed(s, s.Rounds[idx])
		})
		if !disputed {
			t.Fatal("standard round with baselines 6.00 and 6.02 MUST be disputed after restart to STET_SCALE=nano (0.02 > 0.015)")
		}
		if rd := roundState(w2, idx); rd.Status == "settled" {
			t.Fatalf("standard round MUST NOT settle with conflicting baselines 6.00 vs 6.02 after restart to nano")
		}

		// 2. Verifier 3 breaks the baseline tie with 6.00, AND submits gain +0.115 on script 1
		//    (|0.115 - 0.100| = 0.015 > 0.012 standard grade tol, < 0.15 nano grade tol) ->
		//    grade agreement MUST still use standard 0.012 tol so script 1 needs a tie-breaker!
		if code, st := submitGradeWithBase(t, w2, idx, g[2], c3, 6.00, 0.115); code != 200 || st != "pending" {
			t.Fatalf("g3 tie-break baseline + disagreeing grade (+0.115 vs +0.100): HTTP %d %q, want 200 pending", code, st)
		}
		c4 := w2.claim(g[3], false)
		if c4.Job == nil || c4.Job.Kind != "grade" || c4.Job.Scale != "standard" || c4.Job.Script.SHA256 != c1.Job.Script.SHA256 {
			t.Fatalf("standard round grade split (+0.100 vs +0.115) MUST request standard tie-break grade on script 1 after restart to nano, got %+v", c4.Job)
		}
		if code, st := submitGradeWithBase(t, w2, idx, g[3], c4, 6.00, 0.100); code != 200 {
			t.Fatalf("g4 agree on script 1: HTTP %d %q", code, st)
		}

		// 3. Run agreement: submit run 2 (distinct script) and grade it at +0.120 (|0.120 - 0.100| = 0.020 >
		//    0.015 standard run tol, < 0.20 nano run tol) -> with a 3rd fresh runner (g[6]) online,
		//    j.runsWanted MUST request a 3rd run instead of settling!
		_ = w2.claim(g[6], true) // mark g[6] as an active fresh runner (RunnerSeen)
		if code, _ := w2.edit(g[4], ce2, "0\n1\n2\n"); code != 200 {
			t.Fatalf("edit2: HTTP %d", code)
		}
		c5 := w2.claim(g[5], false)
		if c5.Job == nil || c5.Job.Scale != "standard" {
			t.Fatalf("expected standard g6 claim for script 2, got %+v", c5.Job)
		}
		if code, st := submitGradeWithBase(t, w2, idx, g[5], c5, 6.00, 0.120); code != 200 {
			t.Fatalf("g6 grade script 2: HTTP %d %q", code, st)
		}
		var wantRuns int
		w2.app.st.view(func(s *State) {
			j := w2.app.newJudge(s)
			for _, st := range j.states(s.Rounds[idx]) {
				wantRuns = j.runsWanted(st)
			}
		})
		if wantRuns != 3 {
			t.Fatalf("standard round with runs +0.100 and +0.120 after restart to nano: runsWanted=%d, want 3 (0.020 > 0.015)", wantRuns)
		}
		// Complete 3rd run matching script 1 (+0.100) -> round settles cleanly!
		ce3 := w2.claim(g[6], true)
		if ce3.Job == nil || ce3.Job.Kind != "edit" {
			t.Fatalf("expected 3rd edit claim, got %+v", ce3.Job)
		}
		if code, _ := w2.edit(g[6], ce3, "0\n1\n"); code != 200 {
			t.Fatalf("edit3: HTTP %d", code)
		}
		if rd := roundState(w2, idx); rd.Status != "settled" {
			t.Fatalf("expected standard round to settle after 3rd confirming run, got status=%q", rd.Status)
		}
		w2.check()
	})

	t.Run("nano_round_after_restart_to_standard_enforces_nano_baseline_grade_and_run_tolerances", func(t *testing.T) {
		pub, priv := newKey()
		kv := newMemKV()
		cfg := testConfig(t, pub)
		cfg.Scale = "nano"
		cfg.RunsPerEntry = 2
		blobs := dirBlobs{filepath.Join(cfg.DataDir, "blobs")}
		w1 := startWorld(t, cfg, kv, blobs)
		w1.app.season.Baselines.PerRound = 2
		w1.app.season.Tolerance["standard"] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		w1.app.season.Tolerance["nano"] = Tolerance{Grade: 0.15, Baseline: 0.20, Run: 0.20}

		anchor := w1.join("r32-n2s-anchor", priv)
		g := w1.volunteers("r32-n2s-g1", "r32-n2s-g2", "r32-n2s-g3", "r32-n2s-g4")
		w1.submit(w1.join("r32-n2s-owner", nil), "entry-nano", "print(2)")
		idx := w1.drawWithoutExam()
		// Submit run 1 and hold run 2 open so script 1 is graded deterministically.
		ce1 := w1.claim(anchor, true)
		if code, _ := w1.edit(anchor, ce1, "0\n1\n"); code != 200 {
			t.Fatalf("edit1: HTTP %d", code)
		}
		ce2 := w1.claim(g[2], true)
		if ce2.Job == nil || ce2.Job.Kind != "edit" {
			t.Fatalf("expected run 2 edit claim, got %+v", ce2.Job)
		}
		c1 := w1.claimEarly(g[0])
		c2 := w1.claimEarly(g[1])
		if c1.Job.Scale != "nano" || c2.Job.Scale != "nano" {
			t.Fatalf("expected nano claims before restart, got %q/%q", c1.Job.Scale, c2.Job.Scale)
		}
		w1.exam(idx)
		// Verifier 1 submits baseline 6.00 (gain +0.100) on script 1 BEFORE restart.
		if code, st := submitGradeWithBase(t, w1, idx, g[0], c1, 6.00, 0.100); code != 200 || st != "pending" {
			t.Fatalf("g1 submit before restart: HTTP %d %q", code, st)
		}

		// Restart server on the SAME store with STET_SCALE=standard!
		cfgStd := cfg
		cfgStd.Scale = "standard"
		w2 := startWorld(t, cfgStd, kv, blobs)
		w2.app.season.Baselines.PerRound = 2
		w2.app.season.Tolerance["standard"] = Tolerance{Grade: 0.012, Baseline: 0.015, Run: 0.015}
		w2.app.season.Tolerance["nano"] = Tolerance{Grade: 0.15, Baseline: 0.20, Run: 0.20}
		repoint(w2, anchor, g[0], g[1], g[2], g[3])

		// 1. Baseline (6.10 vs 6.00 -> diff 0.10 <= 0.20 nano tol, > 0.015 standard tol) AND
		//    Grade (+0.200 vs +0.100 -> diff 0.10 <= 0.15 nano grade tol, > 0.012 standard tol) MUST agree!
		if code, st := submitGradeWithBase(t, w2, idx, g[1], c2, 6.10, 0.200); code != 200 {
			t.Fatalf("g2 submit after restart to standard: HTTP %d %q", code, st)
		}
		var disputed bool
		w2.app.st.view(func(s *State) {
			disputed = w2.app.baselineDisputed(s, s.Rounds[idx])
		})
		if disputed {
			t.Fatal("nano round with baselines 6.00 and 6.10 MUST NOT be disputed after restart to STET_SCALE=standard (0.10 <= 0.20)")
		}
		// Script 1 MUST already be decided (no extra tie-break grade needed under nano 0.15 grade tol)!
		if extra := w2.claim(g[3], false); extra.Job != nil {
			t.Fatalf("nano round grades +0.100 and +0.200 MUST agree without requesting extra grade after restart to standard, got %+v", extra.Job)
		}

		// 2. Run agreement: submit run 2 (distinct script) and grade it at +0.270 (diff from script 1's +0.150
		//    is 0.12 <= 0.20 nano run tol, > 0.015 standard run tol) -> MUST settle with ONLY 2 runs!
		if code, _ := w2.edit(g[2], ce2, "0\n1\n2\n"); code != 200 {
			t.Fatalf("edit2: HTTP %d", code)
		}
		c3 := w2.claim(g[3], false)
		if c3.Job == nil || c3.Job.Scale != "nano" {
			t.Fatalf("expected persisted round scale 'nano' for g4 after restart to standard, got %+v", c3.Job)
		}
		if code, st := submitGradeWithBase(t, w2, idx, g[3], c3, 6.00, 0.270); code != 200 {
			t.Fatalf("g4 grade script 2: HTTP %d %q", code, st)
		}
		rd := roundState(w2, idx)
		if rd.Status != "settled" {
			t.Fatalf("nano round MUST settle with 2 agreeing baselines/grades/runs after restart to standard, got status=%q", rd.Status)
		}
		e := w2.entry(idx, "entry-nano")
		if e.RunsDiffer || !e.Confirmed {
			t.Fatalf("nano round runs within 0.20 run tolerance must settle Confirmed=true RunsDiffer=false, got %+v", e)
		}
		w2.check()
	})
}
