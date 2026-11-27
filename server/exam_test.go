package main

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

// examResult is a grade whose items score differently: each item scores base + gain less a
// little that depends on the item (the baseline, base less the same).
func (w *world) examResult(c *claimed, n int, gain, base float64) map[string]any {
	p := w.result(c, 5, 6)
	items := func(x float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = x - float64(i%3)*0.01
		}
		return out
	}
	mean := func(v []float64) float64 {
		s := 0.0
		for _, x := range v {
			s += x
		}
		return s / float64(len(v))
	}
	b, e := items(base), items(base+gain)
	p["baseline"] = map[string]any{"mean": mean(b), "item_scores": b, "init_sha256": p["train"].(map[string]any)["init_sha256"]}
	p["exam"] = map[string]any{"mean": mean(e), "item_scores": e}
	return p
}

func TestTheExamIsGradedOnceItIsPostedAndScoredAgainstTheRoundsBaseline(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.drawWithoutExam()
	if w.editAll(anchor) != 1 {
		t.Fatal("the anchor should have run the entry")
	}
	w.established("bob")
	// Until the exam is made, there is nothing to grade, and nothing is given up.
	if c := w.claim(bob, false); c.Job != nil {
		t.Fatalf("a grade before the round's exam exists: %+v", c.Job)
	}
	w.pass(w.cfg.ReplicaWait * 4)
	if w.settled(idx) != 0 {
		t.Fatal("the round settled while waiting for its exam")
	}
	// Only the organisers post it, and only a table the season scores.
	table := testTable()
	if code, _ := w.postExam(bob, idx, table); code != 403 {
		t.Fatalf("a volunteer posted the exam: HTTP %d", code)
	}
	bad := append([]ItemInfo(nil), table...)
	bad[0].Pillar = "astrology"
	if code, _ := w.postExam(anchor, idx, bad); code != 400 {
		t.Fatalf("an exam with a pillar the season does not have: HTTP %d", code)
	}
	// Nor one with an item of a part the season does not name: nothing is sat that is not scored.
	bad = append([]ItemInfo(nil), table...)
	bad[0].Part = "riddles"
	if code, _ := w.postExam(anchor, idx, bad); code != 400 {
		t.Fatalf("an exam with a part the season does not name: HTTP %d", code)
	}
	// Nor one where any pillar falls below the season's minimum item support (min_pillar_items=20).
	sparse := []ItemInfo{
		{Pillar: "reference", Part: "reference", Lang: "en", Stratum: "prose/context"},
		{Pillar: "explain", Part: "explain", Lang: "en", Stratum: "prose/answer"},
	}
	if code, msg := w.postExam(anchor, idx, sparse); code != 400 || !strings.Contains(msg, "minimum") {
		t.Fatalf("a sparse exam with 1 reasoning item: HTTP %d %q, want 400 mentioning minimum", code, msg)
	}
	// A verifier that trains before the exam takes the script while it is made, and carol
	// takes it too, as a replica.
	early := w.claimEarly(bob)
	if early.Job == nil || early.Job.Kind != "grade" || early.Job.Round["exam_digest"] != nil {
		t.Fatalf("bob should be grading, with no exam yet: %+v", early.Job)
	}
	carol := w.join("carol", nil)
	w.established("carol")
	ca := w.claimEarly(carol)
	if ca.Job == nil || ca.Job.Kind != "grade" {
		t.Fatalf("carol should be grading: %+v", ca.Job)
	}
	if code, msg := w.postExam(anchor, idx, table); code != 200 {
		t.Fatalf("posting the exam: HTTP %d %s", code, msg)
	}
	// Grades go on: the job now carries the exam's digest, and the grader may fetch the exam.
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.ID != early.Job.ID || cb.Job.Kind != "grade" || cb.Job.Round["exam_digest"] != examSHA {
		t.Fatalf("no grade job with the exam's digest: %+v", cb.Job)
	}
	if code := bob.do("GET", fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)), nil, nil); code != 200 {
		t.Fatalf("the grader could not fetch the exam: HTTP %d", code)
	}
	dave := w.join("dave", nil)
	if code := dave.do("GET", fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)), nil, nil); code != 403 {
		t.Fatalf("an account with no grade job fetched the exam: HTTP %d", code)
	}
	// A report that is not one score for each item is refused.
	short := w.examResult(cb, len(table)-1, 0.1, -3)
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), short, nil); code != 400 {
		t.Fatalf("a report one item short: HTTP %d", code)
	}
	// Bob's baseline scores -3 on every item, his student -2.9: a gain of 0.1 against the round's baseline.
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), w.examResult(cb, len(table), 0.1, -3), nil); code != 200 {
		t.Fatal("bob's grade was refused")
	}
	// Carol's baseline is a little better: the round's is the mean of the two.
	if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", ca.Job.ID), w.examResult(ca, len(table), 0.08, -2.98), nil); code != 200 {
		t.Fatal("carol's grade was refused")
	}
	w.pass(w.cfg.ReplicaWait + time.Minute)
	if w.settled(idx) != 1 {
		t.Fatal("the round did not settle")
	}
	e := w.entry(idx, "ref-a")
	// Each grade's score is against the round's baseline (-2.99 on each item): bob's student and
	// carol's both score -2.90, +0.09 on each item.
	var want float64
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		base := make([]float64, len(table))
		for i := range base {
			base[i] = -2.99 - float64(i%3)*0.01
		}
		gains := make([]float64, len(table))
		for i := range gains {
			gains[i] = (-2.9 - float64(i%3)*0.01) - base[i]
		}
		want, _, _ = w.app.season.Exam.ScoreItems(table, gains, rd.Shares)
	})
	if e.Gain == nil || math.Abs(*e.Gain-places5(want)) > 1e-4 {
		t.Fatalf("the entry's gain is %v, want %.5f", e.Gain, want)
	}
	// Settled, the round's published exam is anyone's, unsigned.
	resp, err := http.Get(w.srv.URL + fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)))
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("a settled round's published exam: %v %v", err, resp)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	dec, _ := zstd.NewReader(nil)
	if plain, _ := dec.DecodeAll(got, nil); !strings.Contains(string(plain), "withheld") {
		t.Fatalf("the published exam is not the public one: %q", plain)
	}
	// Once settled, what the round held of its exam is in the file store and out of the database.
	w.app.archiveExam(context.Background())
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if len(rd.ExamTable) != 0 || len(rd.BaseItems) != 0 {
			t.Fatal("a settled round still holds its exam's table in the database")
		}
		for _, r := range s.results(idx, e.Editor) {
			if len(r.Items) != 0 {
				t.Fatal("a settled round's grade still holds its item scores")
			}
		}
	})
	if _, _, err := w.blobs.Open(context.Background(), w.app.examBlob(idx, examScores)); err != nil {
		t.Fatalf("the round's item scores were not kept: %v", err)
	}
	// Under the season's own name: a later season's round of the same number is another round.
	if key := w.app.examBlob(idx, examScores); !strings.HasPrefix(key, "exams/"+w.app.season.ID+"/") {
		t.Fatalf("the round's exam is kept under %s, not under its season", key)
	}
	w.check()
}

// postingBlobs is a file store that does something the first time a file is kept in it.
type postingBlobs struct {
	Blobs
	first func()
}

func (b *postingBlobs) Put(ctx context.Context, key string, data []byte, kind string) error {
	if do := b.first; do != nil {
		b.first = nil
		do()
	}
	return b.Blobs.Put(ctx, key, data, kind)
}

// The examiner's post is retried, and two of the same exam can be under way at once: both find
// the round without one, and both keep its files. The second to finish changes nothing: the
// round's clock and its grades' time started with the first, and one exam is announced.
func TestTheSameExamPostedTwiceAtOnceIsPostedOnce(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(w.join("alice", nil), "tidy", "print('v1')")
	idx := w.drawWithoutExam()
	w.run(anchor, scriptOne)
	ctx := context.Background()
	raw, _ := json.Marshal(testExam(testTable()))
	var p, q ExamPost
	json.Unmarshal(raw, &p)
	json.Unmarshal(raw, &q)
	var first bool
	var at int64
	cache := w.app.st.blobs
	inner := cache.Blobs
	// While the one post keeps its files, the other is made and finishes, a while ago.
	cache.Blobs = &postingBlobs{Blobs: inner, first: func() {
		var err error
		if first, err = w.app.postExam(ctx, idx, &q); err != nil {
			t.Errorf("the post that finished first: %v", err)
		}
		w.app.st.update(ctx, func(tx *Tx) error {
			rd := tx.s.Rounds[idx]
			rd.ExamAt -= 600
			at = rd.ExamAt
			tx.saveRound(rd)
			return nil
		})
	}}
	second, err := w.app.postExam(ctx, idx, &p)
	cache.Blobs = inner
	if err != nil || !first || second {
		t.Fatalf("two posts of one exam: the first new: %v, the second new: %v, %v", first, second, err)
	}
	w.app.st.view(func(s *State) {
		if rd := s.Rounds[idx]; rd.ExamAt != at || rd.ExamSHA != examSHA {
			t.Fatalf("the second post moved the round's exam: at %d, not %d", rd.ExamAt, at)
		}
		n := 0
		for _, ev := range s.Events {
			if ev.Kind == "exam" {
				n++
			}
		}
		if n != 1 {
			t.Fatalf("one exam was announced %d times", n)
		}
	})
}

// Runners' uploads take a place before their job is looked at, and there are few places. The
// organisers' exam has its own: with every runner's place taken, a round is still given its exam.
func TestRunnersUploadsDoNotKeepARoundFromItsExam(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(w.join("alice", nil), "tidy", "print('v1')")
	idx := w.drawWithoutExam()
	for i := 0; i < cap(w.app.runnerUploads); i++ {
		w.app.runnerUploads <- struct{}{}
	}
	if code, msg := w.postExam(anchor, idx, testTable()); code != 200 {
		t.Fatalf("the exam, with every runner's place for an upload taken: HTTP %d %s", code, msg)
	}
}

// When the examiner posts an authoritative init_sha256 with the round's exam, rd.InitSHA is
// locked immediately so even the very first grader is verified against it (409 on mismatch).
func TestAuthoritativeExamInitSHALocksRound(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	bob := w.join("bob", nil)
	w.established("bob")
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.drawWithoutExam()
	if w.editAll(anchor) != 1 {
		t.Fatal("anchor should have run entry")
	}
	wantInit := strings.Repeat("a", 64)
	post := testExam(testTable())
	post["init_sha256"] = wantInit
	var body map[string]any
	if code := anchor.do("POST", fmt.Sprintf("/api/v1/admin/rounds/%s/exam", roundID(w.app.season.ID, idx)), post, &body); code != 200 {
		t.Fatalf("post exam with init_sha256: HTTP %d %v", code, body)
	}
	w.app.st.view(func(s *State) {
		if got := s.Rounds[idx].InitSHA; got != wantInit {
			t.Fatalf("rd.InitSHA = %q, want %q", got, wantInit)
		}
	})
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" {
		t.Fatalf("expected grade job, got %+v", cb.Job)
	}
	bad := w.examResult(cb, len(testTable()), 0.1, -3)
	badInit := strings.Repeat("b", 64)
	bad["train"].(map[string]any)["init_sha256"] = badInit
	bad["baseline"].(map[string]any)["init_sha256"] = badInit
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), bad, nil); code != 409 {
		t.Fatalf("bad init_sha256 against authoritative exam lock: HTTP %d, want 409", code)
	}
	carol := w.join("carol", nil)
	w.established("carol")
	cc := w.claim(carol, false)
	if cc.Job == nil || cc.Job.Kind != "grade" {
		t.Fatalf("expected grade job for carol, got %+v", cc.Job)
	}
	good := w.examResult(cc, len(testTable()), 0.1, -3)
	good["train"].(map[string]any)["init_sha256"] = wantInit
	good["baseline"].(map[string]any)["init_sha256"] = wantInit
	if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cc.Job.ID), good, nil); code != 200 {
		t.Fatalf("matching init_sha256 against authoritative exam lock: HTTP %d, want 200", code)
	}
}

// Per RULES.md, a pillar that falls below min_pillar_items (such as a news=0 draw) is omitted
// cleanly and its share goes to the rest of its group; the round page shows it as omitted.
// Refusing an exam happens only when a present pillar is sparse (<min_pillar_items) or an
// entire declared group has zero items.
func TestOmittedPillarIsAcceptedAndMissingGroupIsRefused(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.drawWithoutExam()

	// Missing an entire declared group ("reasoning") is refused.
	noReasoning := []ItemInfo{}
	for _, it := range testTable() {
		if it.Pillar == "reference" || it.Pillar == "news" || it.Pillar == "technical" || it.Pillar == "conversation" {
			noReasoning = append(noReasoning, it)
		}
	}
	if code, msg := w.postExam(anchor, idx, noReasoning); code != 400 || !strings.Contains(msg, "group") {
		t.Fatalf("exam missing entire reasoning group: HTTP %d %q, want 400 mentioning group", code, msg)
	}

	// Sparse present pillar (news has 5 items < 20) is refused.
	sparseNews := []ItemInfo{}
	newsSeen := 0
	for _, it := range testTable() {
		if it.Pillar == "news" {
			newsSeen++
			if newsSeen > 5 {
				continue
			}
		}
		sparseNews = append(sparseNews, it)
	}
	if code, msg := w.postExam(anchor, idx, sparseNews); code != 400 || !strings.Contains(msg, "minimum") {
		t.Fatalf("exam with 5 news items: HTTP %d %q, want 400 mentioning minimum", code, msg)
	}

	// Omitting news completely (0 items) while the rest of comprehension & reasoning are full succeeds!
	noNews := []ItemInfo{}
	for _, it := range testTable() {
		if it.Pillar != "news" {
			noNews = append(noNews, it)
		}
	}
	if code, msg := w.postExam(anchor, idx, noNews); code != 200 {
		t.Fatalf("exam omitting news cleanly: HTTP %d %s, want 200", code, msg)
	}
	var mix *ExamMix
	w.app.st.view(func(s *State) { mix = s.Rounds[idx].ExamMix })
	if mix == nil {
		t.Fatal("missing ExamMix")
	}
	for _, p := range mix.Parts {
		switch p.Pillar {
		case "news":
			if p.Items != 0 || p.Share != 0 {
				t.Fatalf("omitted news pillar should have 0 items and 0 share, got %+v", p)
			}
		case "reference", "technical", "conversation":
			if p.Share != 0 {
				t.Fatalf("diagnostic comprehension pillar %s share = %.4f, want 0.0000", p.Pillar, p.Share)
			}
		case "explain", "infer", "synthesis":
			want := 1.0 / 3.0
			if math.Abs(p.Share-want) > 1e-4 {
				t.Fatalf("scored reasoning pillar %s share = %.4f, want %.4f", p.Pillar, p.Share, want)
			}
		}
	}
	rid := roundID(w.app.season.ID, idx)
	if code, page := w.page("/rounds/" + rid); code != 200 || !strings.Contains(page, `<dt>news</dt><dd>omitted`) {
		t.Fatalf("round page (HTTP %d) did not render omitted news pillar", code)
	}
	w.check()
}

// Organisers can cancel an un-examined round cleanly via POST /api/v1/admin/rounds/{id}/cancel
// with nothing held against any entry, and once an exam is posted cancellation is refused.
func TestAdminCancelRoundAndExamTimeoutAutoCancel(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	alice := w.join("alice", nil)
	edID := w.submit(alice, "tidy", "print(1)")

	// 1) Manual organiser cancel before exam is posted.
	idx := w.drawWithoutExam()
	rid := roundID(w.app.season.ID, idx)
	if w.editAll(anchor) != 1 {
		t.Fatal("anchor should have run entry")
	}
	if code := alice.do("POST", "/api/v1/admin/rounds/"+rid+"/cancel", nil, nil); code != 403 {
		t.Fatalf("non-organiser cancel: HTTP %d, want 403", code)
	}
	var ack map[string]any
	if code := anchor.do("POST", "/api/v1/admin/rounds/"+rid+"/cancel", map[string]string{"reason": "harvest outage"}, &ack); code != 200 {
		t.Fatalf("organiser cancel: HTTP %d %v", code, ack)
	}
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if rd.Status != "settled" || rd.BaselineLoss != nil {
			t.Fatalf("cancelled round state: status=%s baseline=%v", rd.Status, rd.BaselineLoss)
		}
		if failing(s, edID) || roundsRun(s, edID) != 0 {
			t.Fatalf("cancelled round held against entry: failing=%v roundsRun=%d", failing(s, edID), roundsRun(s, edID))
		}
		en := s.inRound[idx][0]
		if !en.Void || !en.Unattended || en.Valid {
			t.Fatalf("cancelled entry should be void+unattended: %+v", en)
		}
	})
	w.check()

	// 2) Automatic safety-valve cancellation when ExamTimeout elapses with no exam posted.
	idx2 := w.drawWithoutExam()
	w.pass(w.cfg.ExamTimeout + time.Minute)
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx2]
		if rd.Status != "settled" || rd.BaselineLoss != nil {
			t.Fatalf("timed-out un-examined round should auto-cancel cleanly: status=%s baseline=%v", rd.Status, rd.BaselineLoss)
		}
		if failing(s, edID) || roundsRun(s, edID) != 0 {
			t.Fatalf("timed-out round held against entry: failing=%v roundsRun=%d", failing(s, edID), roundsRun(s, edID))
		}
	})
	w.check()

	// 3) Once an exam is posted — and even after a baseline and grade have been submitted on
	//    one entry while another entry keeps the round open — admin cancel without force is
	//    refused with 409, while {"force": true} cancels cleanly, dropping all round baselines
	//    and voiding every entry unattended so refreshScores cannot resurrect a phantom baseline.
	bob := w.join("bob", nil)
	carol := w.join("carol", nil)
	w.established("bob")
	edID2 := w.submit(carol, "tidy-two", "print(2)")
	idx3 := w.draw()
	rid3 := roundID(w.app.season.ID, idx3)
	if w.editAll(anchor) != 2 {
		t.Fatal("anchor should have run both entries in round 3")
	}
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" {
		t.Fatalf("bob should get grade job in round 3, got %+v", cb.Job)
	}
	res := w.result(cb, 5.5, 6.0)
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), res, &ack); code != 200 {
		t.Fatalf("bob grade+baseline in round 3: HTTP %d %v", code, ack)
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx3].Status != "open" {
			t.Fatalf("round 3 should still be open with entry 2 ungraded, got %s", s.Rounds[idx3].Status)
		}
		nBase := 0
		for _, b := range s.Baselines {
			if b.RoundIdx == idx3 {
				nBase++
			}
		}
		if nBase == 0 {
			t.Fatal("expected baseline to be recorded before force-cancel")
		}
	})
	if code := anchor.do("POST", "/api/v1/admin/rounds/"+rid3+"/cancel", nil, nil); code != 409 {
		t.Fatalf("cancelling a round that already has its exam without force: HTTP %d, want 409", code)
	}
	if code := anchor.do("POST", "/api/v1/admin/rounds/"+rid3+"/cancel", map[string]any{"reason": "examiner library mismatch", "force": true}, &ack); code != 200 {
		t.Fatalf("force-cancelling an examined round: HTTP %d %v", code, ack)
	}
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx3]
		if rd.Status != "settled" || rd.BaselineLoss != nil {
			t.Fatalf("force-cancelled round state: status=%s baseline=%v", rd.Status, rd.BaselineLoss)
		}
		nBase := 0
		for _, b := range s.Baselines {
			if b.RoundIdx == idx3 {
				nBase++
			}
		}
		if nBase != 0 {
			t.Fatalf("force-cancelled round kept %d baselines", nBase)
		}
		for _, en := range s.inRound[idx3] {
			if !en.Void || !en.Unattended || en.Valid || en.Loss != nil || en.Gain != 0 {
				t.Fatalf("force-cancelled entry should be void+unattended with cleared scores: %+v", en)
			}
		}
		if failing(s, edID) || roundsRun(s, edID) != 0 || failing(s, edID2) || roundsRun(s, edID2) != 0 {
			t.Fatalf("force-cancelled round held against entries: ed1=(%v,%d) ed2=(%v,%d)", failing(s, edID), roundsRun(s, edID), failing(s, edID2), roundsRun(s, edID2))
		}
	})
	w.check()
}

// Review 8 Finding 3: a single invalid grade asks a second established grader before failing the
// entry, and a script that fails training across attempts is voided unattended with "training failed"
// wording instead of claiming the curator wrote no usable edit script.
func TestSingleInvalidGradeSeeksSecondOpinionAndTrainingFailedWording(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	bob := w.join("bob", nil)
	carol := w.join("carol", nil)
	alice := w.join("alice", nil)
	w.established("bob")
	w.established("carol")
	w.submit(alice, "tidy", "print(1)")
	idx := w.draw()
	w.run(anchor, scriptOne)

	// Bob reports invalid (e.g. older client reporting a fault). Because carol is also active
	// and established, the server must NOT settle invalid on bob's word alone!
	carolID := w.account("carol").ID
	w.app.st.update(context.Background(), func(tx *Tx) error {
		tx.s.Accounts[carolID].LastSeen = now()
		return nil
	})
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" {
		t.Fatalf("bob should get grade job, got %+v", cb.Job)
	}
	inv := w.result(cb, 5.5, 6.0)
	inv["valid"], inv["invalid_reason"] = false, "SFT probe diverged at step 2"
	delete(inv, "exam")
	var ack map[string]any
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), inv, &ack); code != 200 {
		t.Fatalf("bob invalid report: HTTP %d %v", code, ack)
	}
	if w.settled(idx) != 0 {
		t.Fatal("round settled invalid on a single grader's report while a second established grader was active")
	}

	// Carol gets offered the script as a required second grade!
	cc := w.claim(carol, false)
	if cc.Job == nil || cc.Job.Kind != "grade" {
		t.Fatalf("carol should be offered second grade after single invalid report, got %+v", cc.Job)
	}

	// Check trustOf wording distinguishes training failure vs script rule violation vs no script.
	if tv := trustOf(EntryView{Settled: true, Void: true, Reason: "the training failed on this entry's script in 5 attempts"}); !strings.Contains(strings.ToLower(tv.Says), "training failed") {
		t.Fatalf("void training failed trustOf = %+v", tv)
	}
	if tv := trustOf(EntryView{Settled: true, Valid: false, NRuns: 1, ScriptSHA: strings.Repeat("a", 64), Reason: "SFT probe diverged at step 2"}); !strings.Contains(strings.ToLower(tv.Says), "training failed") {
		t.Fatalf("diverged trustOf = %+v", tv)
	}
	if tv := trustOf(EntryView{Settled: true, Valid: false, NRuns: 1, ScriptSHA: strings.Repeat("a", 64), Reason: "line 3: bad"}); !strings.Contains(strings.ToLower(tv.Says), "broke a rule") {
		t.Fatalf("script rule error trustOf = %+v", tv)
	}
	if tv := trustOf(EntryView{Settled: true, Valid: false, NRuns: 1, ScriptSHA: "", Reason: "timeout"}); !strings.Contains(strings.ToLower(tv.Says), "no usable edit script") {
		t.Fatalf("no script trustOf = %+v", tv)
	}
}

// Review 8 Finding 4: grade_backends distinguishes NVIDIA CUDA from AMD ROCm when device=="cuda".
func TestGradeBackendsEnforcement(t *testing.T) {
	s := &Season{GradeDevices: []string{"cuda"}, GradeBackends: []string{"cuda"}}
	if !s.gradesOnHardware(hardwareFacts{Device: "cuda", Backend: "cuda"}) {
		t.Fatal("NVIDIA cuda should be accepted")
	}
	if s.gradesOnHardware(hardwareFacts{Device: "cuda", Backend: "rocm"}) {
		t.Fatal("AMD rocm should be rejected when grade_backends is [cuda]")
	}
	if s.gradesOnHardware(hardwareFacts{Device: "cuda", Backend: ""}) {
		t.Fatal("missing backend should be rejected when grade_backends is [cuda]")
	}
	if !strings.Contains(s.gradeHardwareRequirement(), "NVIDIA (CUDA)") || strings.Contains(s.gradeHardwareRequirement(), "AMD") {
		t.Fatalf("unexpected requirement text: %q", s.gradeHardwareRequirement())
	}

	// End-to-end HTTP check with Dev=false: claim rejects AMD ROCm with 400 and accepts NVIDIA CUDA.
	pub, priv := newKey()
	cfg := testConfig(t, pub)
	cfg.Dev = false
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	rocm := w.join("rocm-vol", nil)
	cuda := w.join("cuda-vol", nil)
	_ = priv
	hello := func(backend string) map[string]any {
		return map[string]any{
			"harness":       w.app.season.Harness,
			"season":        w.app.season.ID,
			"season_digest": w.app.season.Digest,
			"hardware":      map[string]any{"device": "cuda", "backend": backend, "gpu": "Test GPU"},
		}
	}
	var errResp map[string]any
	if code := rocm.do("POST", "/api/v1/jobs/claim", hello("rocm"), &errResp); code != 400 || !strings.Contains(fmt.Sprint(errResp["error"]), "NVIDIA (CUDA)") {
		t.Fatalf("Dev=false ROCm claim: HTTP %d %v, want 400 mentioning NVIDIA (CUDA)", code, errResp)
	}
	var okResp map[string]any
	if code := cuda.do("POST", "/api/v1/jobs/claim", hello("cuda"), &okResp); code != 200 {
		t.Fatalf("Dev=false CUDA claim: HTTP %d %v, want 200", code, okResp)
	}
}

// Review 7 Finding 3: two overlapping exam publications with DIFFERENT digests cannot overwrite
// each other's files. Both private (while open) and public (after settlement) downloads must
// serve the winning digest's exact bytes even when the losing publisher writes blobs concurrently.
func TestOverlappingExamPublicationsDoNotOverwriteWinnerFiles(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	bob := w.join("bob", nil)
	w.established("bob")
	w.submit(w.join("alice", nil), "tidy", "print('v1')")
	idx := w.drawWithoutExam()
	if w.editAll(anchor) != 1 {
		t.Fatal("anchor should have run entry")
	}
	ctx := context.Background()
	table := testTable()
	enc, _ := zstd.NewWriter(nil)
	makePost := func(digest, privLine, pubLine string) ExamPost {
		privZst := enc.EncodeAll([]byte(privLine+"\n"), nil)
		pubZst := enc.EncodeAll([]byte(pubLine+"\n"), nil)
		return ExamPost{
			Round:    roundID(w.app.season.ID, idx),
			Digest:   digest,
			File:     base64.StdEncoding.EncodeToString(privZst),
			Public:   base64.StdEncoding.EncodeToString(pubZst),
			Table:    table,
			Shares:   map[string]float64{"en": 60, "de": 40},
			Snapshot: strings.Repeat("5", 64),
			Fresh:    len(table),
		}
	}
	digestA := strings.Repeat("a", 64)
	digestB := strings.Repeat("b", 64)
	loser := makePost(digestA, `{"id":"loser-private"}`, `{"withheld":true,"id":"loser-public"}`)
	winner := makePost(digestB, `{"id":"winner-private"}`, `{"withheld":true,"id":"winner-public"}`)

	cache := w.app.st.blobs
	inner := cache.Blobs
	var winnerPosted bool
	var winnerErr error
	// Start loser's postExam; right before its first blob Put completes, winner races ahead and commits!
	cache.Blobs = &postingBlobs{Blobs: inner, first: func() {
		winnerPosted, winnerErr = w.app.postExam(ctx, idx, &winner)
	}}
	loserPosted, loserErr := w.app.postExam(ctx, idx, &loser)
	cache.Blobs = inner

	if winnerErr != nil || !winnerPosted {
		t.Fatalf("winning exam post failed: posted=%v err=%v", winnerPosted, winnerErr)
	}
	if loserPosted || loserErr == nil {
		t.Fatalf("losing exam post should fail with 409, got posted=%v err=%v", loserPosted, loserErr)
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].ExamSHA != digestB {
			t.Fatalf("round ExamSHA = %q, want winner %q", s.Rounds[idx].ExamSHA, digestB)
		}
	})

	// 1. Private exam download (while open, authenticated grader holding a grade claim) MUST belong to winner!
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" {
		t.Fatalf("bob should get grade job: %+v", cb.Job)
	}
	rid := roundID(w.app.season.ID, idx)
	path := "/api/v1/rounds/" + rid + "/exam"
	req, _ := http.NewRequest("GET", w.srv.URL+path, nil)
	ts := strconv.FormatInt(time.Now().Unix(), 10)
	sum := sha256.Sum256(nil)
	msg := strings.Join([]string{"stet-v1", "GET", path, ts, hex.EncodeToString(sum[:])}, "\n")
	req.Header.Set("X-Stet-Key", bob.pub)
	req.Header.Set("X-Stet-Body", hex.EncodeToString(sum[:]))
	req.Header.Set("X-Stet-Time", ts)
	req.Header.Set("X-Stet-Sig", hex.EncodeToString(ed25519.Sign(bob.priv, []byte(msg))))
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("private exam fetch: err=%v status=%v", err, resp)
	}
	privBytes, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	dec, _ := zstd.NewReader(nil)
	if plain, _ := dec.DecodeAll(privBytes, nil); !strings.Contains(string(plain), "winner-private") {
		t.Fatalf("private exam download served wrong bytes: %q", plain)
	}

	// 2. After settlement, public exam download MUST also belong to winner!
	w.app.st.update(ctx, func(tx *Tx) error {
		rd := tx.s.Rounds[idx]
		rd.Status = "settled"
		tx.saveRound(rd)
		return nil
	})
	resp, err = http.Get(w.srv.URL + "/api/v1/rounds/" + rid + "/exam")
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("public exam fetch: err=%v status=%v", err, resp)
	}
	pubBytes, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if plain, _ := dec.DecodeAll(pubBytes, nil); !strings.Contains(string(plain), "winner-public") {
		t.Fatalf("public exam download served wrong bytes: %q", plain)
	}
}

// Review 13 P2 #3: a lone invalid report from an established volunteer with NO second eligible
// grader online must NOT immediately settle the entry invalid; it stays open through ReplicaWait,
// exposes GradesWanted=1 / FreshGradersWanted=1 on /admin/work, closes unrated (Void=true,
// Unattended=true) if corroboration never arrives, and settles invalid if an on-demand second
// grader arrives and confirms it.
func TestLoneInvalidVolunteerReportWaitsForReplicaWindowAndVoidsIfUncorroborated(t *testing.T) {
	setup := func(t *testing.T) (*world, *actor, *actor, *actor, int) {
		t.Helper()
		pub, priv := newKey()
		w := newWorld(t, pub)
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		bob := w.join("bob", nil)
		w.established("bob")
		w.submit(alice, "tidy", "print(1)")
		idx := w.draw()
		w.run(anchor, scriptOne)
		// No other eligible grader is online (anchor wrote the script alone and cannot self-corroborate).
		cb := w.claim(bob, false)
		if cb.Job == nil || cb.Job.Kind != "grade" {
			t.Fatalf("bob should get grade job, got %+v", cb.Job)
		}
		inv := w.result(cb, 5.5, 6.0)
		inv["valid"], inv["invalid_reason"] = false, "SFT probe diverged at step 2"
		delete(inv, "exam")
		var ack struct {
			Status string `json:"status"`
		}
		if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), inv, &ack); code != 200 {
			t.Fatalf("bob invalid report: HTTP %d", code)
		}
		if ack.Status != "pending" || w.settled(idx) != 0 {
			t.Fatalf("lone volunteer invalid report with nobody else online must NOT settle immediately: status=%q settled=%d", ack.Status, w.settled(idx))
		}
		wk := w.work(anchor).Rounds[0]
		if wk.GradesWanted != 1 {
			t.Fatalf("expected /admin/work GradesWanted=1 for uncorroborated invalid report, got %+v", wk)
		}
		if wk.FreshGradersWanted == nil || *wk.FreshGradersWanted != 1 {
			t.Fatalf("expected FreshGradersWanted=1 when only script-writer anchor is idle, got %v", wk.FreshGradersWanted)
		}
		return w, anchor, alice, bob, idx
	}

	t.Run("closes_unrated_if_uncorroborated", func(t *testing.T) {
		w, _, alice, _, idx := setup(t)
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected round to settle cleanly after ReplicaWait, settled=%d", w.settled(idx))
		}
		en := w.entries(alice, idx)[0]
		if !en.Void || en.Valid {
			t.Fatalf("uncorroborated lone volunteer invalid report must close unrated (Void=true, Valid=false), got %+v", en)
		}
		w.app.st.view(func(s *State) {
			raw := s.inRound[idx][0]
			if !raw.Void || !raw.Unattended || raw.Valid {
				t.Fatalf("expected stored entry Void=true Unattended=true Valid=false, got %+v", raw)
			}
		})
		w.check()
	})

	t.Run("on_demand_second_grader_confirms_invalid", func(t *testing.T) {
		w, _, alice, _, idx := setup(t)
		// An on-demand second established grader arrives AFTER bob's invalid report.
		carol := w.join("carol", nil)
		w.established("carol")
		cc := w.claim(carol, false)
		if cc.Job == nil || cc.Job.Kind != "grade" {
			t.Fatalf("on-demand second grader carol should receive the corroboration grade, got %+v", cc.Job)
		}
		inv2 := w.result(cc, 5.5, 6.0)
		inv2["valid"], inv2["invalid_reason"] = false, "line 1: bad script"
		delete(inv2, "exam")
		var ack struct {
			Status string `json:"status"`
		}
		if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cc.Job.ID), inv2, &ack); code != 200 {
			t.Fatalf("carol confirm invalid: HTTP %d", code)
		}
		if ack.Status != "settled" || w.settled(idx) != 1 {
			t.Fatalf("corroborated invalid report should settle immediately: status=%q settled=%d", ack.Status, w.settled(idx))
		}
		en := w.entries(alice, idx)[0]
		if en.Void || en.Valid {
			t.Fatalf("corroborated invalid entry should settle Valid=false Void=false, got %+v", en)
		}
		w.check()
	})

	// Gate 4 P2 #1: an in-flight second edit run (LiveEdits > 0) must NOT be expired/voided when
	// Run 1 has an uncorroborated invalid report and ReplicaWait elapses mid-edit.
	t.Run("does_not_void_mid_edit_when_second_run_is_live", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.cfg.RunsPerEntry = 2
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		bob := w.join("bob", nil)
		dave := w.join("dave", nil)
		w.established("bob", "dave")
		w.submit(alice, "tidy", "print(1)")
		idx := w.draw()

		// Run 1 by anchor writes scriptOne; Dave holds live edit claim for Run 2.
		ce1 := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce1, scriptOne); code != 200 {
			t.Fatalf("anchor edit: HTTP %d", code)
		}
		ce2 := w.claim(dave, true)
		if ce2.Job == nil || ce2.Job.Kind != "edit" {
			t.Fatalf("dave should get Run 2 edit claim, got %+v", ce2.Job)
		}
		// Bob reports Run 1's script invalid while Dave's edit is still in flight.
		cb := w.claim(bob, false)
		if cb.Job == nil || cb.Job.Kind != "grade" {
			t.Fatalf("bob should get grade job, got %+v", cb.Job)
		}
		inv := w.result(cb, 5.5, 6.0)
		inv["valid"], inv["invalid_reason"] = false, "SFT probe diverged"
		delete(inv, "exam")
		if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cb.Job.ID), inv, nil); code != 200 {
			t.Fatalf("bob invalid report: HTTP %d", code)
		}
		// ReplicaWait passes while Dave's edit lease is still active: entry MUST stay open and Dave's edit MUST succeed!
		w.pass(w.cfg.ReplicaWait + time.Minute)
		if w.settled(idx) != 0 {
			t.Fatalf("entry must NOT be voided while Dave's Run 2 edit is live")
		}
		if code, status := w.edit(dave, ce2, scriptTwo); code != 200 {
			t.Fatalf("dave's in-flight Run 2 edit should still be accepted, got HTTP %d %s", code, status)
		}
		w.check()
	})
}

// Review 13 P2 #7 (server-side): SkipEarlyRounds on Hello excludes only the named round(s) from
// pre-exam grade offers while still handing out pre-exam grades for newer rounds with Early=true.
func TestSkipEarlyRoundsIsRoundScoped(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	alice := w.join("alice", nil)
	bob := w.join("bob", nil)
	w.established("bob")
	w.submit(alice, "tidy", "print(1)")

	// Round 1 (no exam yet) has a script waiting for early grades.
	idx1 := w.drawWithoutExam()
	ce1 := w.claim(anchor, true)
	if code, _ := w.edit(anchor, ce1, scriptOne); code != 200 {
		t.Fatalf("round 1 edit: HTTP %d", code)
	}
	rid1 := roundID(w.app.season.ID, idx1)

	// Claiming with Early=true and SkipEarlyRounds=[rid1] must NOT receive round 1's pre-exam grade!
	claimWithSkip := func(skip []string) *claimed {
		var out claimed
		code := bob.do("POST", "/api/v1/jobs/claim", map[string]any{
			"harness":           w.app.season.Harness,
			"season":            w.app.season.ID,
			"season_digest":     w.app.season.Digest,
			"hardware":          map[string]any{"device": "cuda", "backend": "cuda"},
			"early":             true,
			"skip_early_rounds": skip,
		}, &out)
		if code != 200 {
			t.Fatalf("claim skip=%v: HTTP %d", skip, code)
		}
		return &out
	}
	if got := claimWithSkip([]string{rid1}); got.Job != nil {
		t.Fatalf("expected no pre-exam job for skipped round %s, got %+v", rid1, got.Job)
	}

	// Open a second round (Round 2, also pre-exam) alongside Round 1: bob with SkipEarlyRounds=[rid1]
	// MUST immediately receive Round 2's pre-exam grade!
	anchorID := w.account("anchor-one").ID
	w.app.st.update(context.Background(), func(tx *Tx) error {
		rd2 := &Round{
			Idx: idx1 + 1, Status: "open", Scale: w.cfg.Scale,
			DrawnAt: now(), ActiveAt: now(),
			Seed: strings.Repeat("2", 64), CorpusSHA: strings.Repeat("3", 64),
		}
		tx.saveRound(rd2)
		edID := tx.s.inRound[idx1][0].EditorID
		tx.saveEntry(&Entry{
			RoundIdx: rd2.Idx, EditorID: edID, Status: "open",
			Runs: []*Run{{N: 1, By: anchorID, Script: strings.Repeat("b", 64), ScriptSize: 16, At: now()}},
		})
		return nil
	})
	rid2 := roundID(w.app.season.ID, idx1+1)
	got2 := claimWithSkip([]string{rid1})
	if got2.Job == nil || got2.Job.Kind != "grade" || got2.Job.Round["id"] != rid2 {
		t.Fatalf("expected pre-exam grade for round %s while %s is skipped, got %+v", rid2, rid1, got2.Job)
	}
}

// Review 14 P2 #3: five failed grade attempts on Run 1 must drop Run 1's script WITHOUT voiding
// the entry while Run 2's edit claim is still executing; Run 2's completed edit must be accepted
// with HTTP 200 and settle cleanly. And if Run 2 subsequently fails maxEditAttempts edits with no
// non-dropped runs remaining, abandonStuck must still void the entry.
func TestAbandonStuckDoesNotVoidEntryWhileSecondRunEditIsLive(t *testing.T) {
	t.Run("keeps_entry_open_during_live_second_edit", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.cfg.RunsPerEntry = 2
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		graders := w.volunteers("gr1", "gr2", "gr3", "gr4", "gr5")
		dave := w.join("dave", nil)
		w.established("dave")
		w.submit(alice, "tidy", "print(1)")
		idx := w.draw()

		// Anchor completes Run 1; Dave holds a live edit lease for Run 2.
		ce1 := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce1, scriptOne); code != 200 {
			t.Fatalf("anchor Run 1 edit: HTTP %d", code)
		}
		ce2 := w.claim(dave, true)
		if ce2.Job == nil || ce2.Job.Kind != "edit" {
			t.Fatalf("expected Dave to claim Run 2 edit, got %+v", ce2.Job)
		}

		// Five established graders fail grading Run 1 with "training failed".
		for i, g := range graders {
			cg := w.claim(g, false)
			if cg.Job == nil || cg.Job.Kind != "grade" {
				t.Fatalf("grader %d expected Run 1 grade job, got %+v", i, cg.Job)
			}
			if code := g.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cg.Job.ID), map[string]any{"reason": "training failed: OOM"}, nil); code != 200 {
				t.Fatalf("grader %d release: HTTP %d", i, code)
			}
		}

		// Trigger abandonStuck while Dave's Run 2 edit lease is still active.
		w.pass(time.Minute)
		w.app.st.view(func(s *State) {
			e := s.inRound[idx][0]
			if e.Status != "open" || e.Void {
				t.Fatalf("entry must stay open while Run 2 edit is live: status=%q void=%v", e.Status, e.Void)
			}
			if len(e.Runs) != 1 || !e.Runs[0].Dropped {
				t.Fatalf("Run 1 should be dropped after %d failed grade attempts: %+v", maxGradeAttempts, e.Runs)
			}
		})

		// Dave finishes Run 2 edit -> MUST return HTTP 200!
		if code, st := w.edit(dave, ce2, scriptTwo); code != 200 {
			t.Fatalf("Dave's live Run 2 edit must be accepted with HTTP 200, got HTTP %d (%s)", code, st)
		}
		// Anchor grades Run 2 -> entry settles Valid=true, Void=false!
		cg2 := w.claim(anchor, false)
		if cg2.Job == nil || cg2.Job.Kind != "grade" {
			t.Fatalf("expected anchor to grade Run 2, got %+v", cg2.Job)
		}
		if code, _ := w.send(anchor, cg2, 5.950, 6.000); code != 200 {
			t.Fatalf("anchor Run 2 grade: HTTP %d", code)
		}
		w.pass(w.cfg.ReplicaWait + time.Minute)
		en := w.entries(alice, idx)[0]
		if en.Void || !en.Valid {
			t.Fatalf("expected entry to settle valid from Run 2, got %+v", en)
		}
		w.check()
	})

	t.Run("voids_entry_if_second_run_exhausts_edit_attempts_after_run1_dropped", func(t *testing.T) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		w.app.cfg.RunsPerEntry = 2
		anchor := w.join("anchor-one", priv)
		alice := w.join("alice", nil)
		graders := w.volunteers("gr1", "gr2", "gr3", "gr4", "gr5")
		runners := w.volunteers("rn1", "rn2", "rn3", "rn4", "rn5")
		w.submit(alice, "tidy", "print(1)")
		idx := w.draw()

		ce1 := w.claim(anchor, true)
		if code, _ := w.edit(anchor, ce1, scriptOne); code != 200 {
			t.Fatalf("anchor Run 1 edit: HTTP %d", code)
		}
		// Hold a live edit claim while Run 1 gets dropped so the entry stays open.
		ceLive := w.claim(runners[0], true)
		for _, g := range graders {
			cg := w.claim(g, false)
			if code := g.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cg.Job.ID), map[string]any{"reason": "training failed: OOM"}, nil); code != 200 {
				t.Fatalf("release grade: HTTP %d", code)
			}
		}
		w.pass(time.Minute)
		// Now Run 2's edit fails maxEditAttempts times (no non-dropped runs left!).
		if code := runners[0].do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", ceLive.Job.ID), map[string]any{"reason": "curator crashed"}, nil); code != 200 {
			t.Fatalf("release edit 0: HTTP %d", code)
		}
		for i := 1; i < maxEditAttempts; i++ {
			ce := w.claim(runners[i], true)
			if ce.Job == nil || ce.Job.Kind != "edit" {
				t.Fatalf("expected edit attempt %d, got %+v", i, ce.Job)
			}
			if code := runners[i].do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", ce.Job.ID), map[string]any{"reason": "curator crashed"}, nil); code != 200 {
				t.Fatalf("release edit %d: HTTP %d", i, code)
			}
		}
		w.pass(time.Minute)
		if w.settled(idx) != 1 {
			t.Fatalf("expected entry voided once Run 1 dropped and Run 2 exhausted edit attempts, settled=%d", w.settled(idx))
		}
		en := w.entries(alice, idx)[0]
		if !en.Void || en.Valid {
			t.Fatalf("expected Void=true Valid=false, got %+v", en)
		}
		w.check()
	})
}

// Review 16 P2 #4: two overlapping exam posts with the SAME items_digest but DIFFERENT derived
// public_file or classification table must stage into distinct content-addressed prefixes so the
// losing post (posted=false) can never overwrite the winner's stored public file or table.
func TestOverlappingExamPostsWithDifferentDerivedFilesPreserveWinner(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(w.join("alice", nil), "tidy", "print('v1')")
	idx := w.drawWithoutExam()
	w.run(anchor, scriptOne)
	ctx := context.Background()

	enc, _ := zstd.NewWriter(nil)
	defer enc.Close()

	// Post A (winner): table Lang="en", public_file="winner-public-payload".
	tableA := testTable()
	for i := range tableA {
		tableA[i].Lang = "en"
	}
	rawA, _ := json.Marshal(testExam(tableA))
	var postA ExamPost
	if err := json.Unmarshal(rawA, &postA); err != nil {
		t.Fatalf("unmarshal postA: %v", err)
	}
	publicA := enc.EncodeAll([]byte("{\"id\":\"winner\",\"prompt\":\"clean\"}\n"), nil)
	postA.Public = base64.StdEncoding.EncodeToString(publicA)

	// Post B (loser): SAME Digest (items_digest), DIFFERENT table Lang="fr" and public_file!
	tableB := testTable()
	for i := range tableB {
		tableB[i].Lang = "fr"
	}
	rawB, _ := json.Marshal(testExam(tableB))
	var postB ExamPost
	if err := json.Unmarshal(rawB, &postB); err != nil {
		t.Fatalf("unmarshal postB: %v", err)
	}
	publicB := enc.EncodeAll([]byte("{\"id\":\"loser\",\"prompt\":\"stale-redaction\"}\n"), nil)
	postB.Public = base64.StdEncoding.EncodeToString(publicB)
	if postA.Digest != postB.Digest {
		t.Fatalf("test setup error: expected identical items_digest")
	}

	var firstPosted bool
	cache := w.app.st.blobs
	inner := cache.Blobs
	// Interleave: postB starts putting blobs -> hook runs postA to completion -> postB resumes
	// putting its blobs and enters the DB transaction (losing with posted=false).
	cache.Blobs = &postingBlobs{Blobs: inner, first: func() {
		var err error
		if firstPosted, err = w.app.postExam(ctx, idx, &postA); err != nil {
			t.Errorf("winning postA failed: %v", err)
		}
	}}
	secondPosted, err := w.app.postExam(ctx, idx, &postB)
	cache.Blobs = inner
	if err != nil || !firstPosted || secondPosted {
		t.Fatalf("overlapping posts: first=%v second=%v err=%v, want true/false/nil", firstPosted, secondPosted, err)
	}

	var blobSHA string
	w.app.st.view(func(s *State) {
		rd := s.Rounds[idx]
		blobSHA = rd.ExamBlobSHA
		if blobSHA == "" {
			t.Fatalf("expected non-empty ExamBlobSHA on winning round")
		}
		if len(rd.ExamTable) == 0 || rd.ExamTable[0].Lang != "en" {
			t.Fatalf("winning round ExamTable corrupted: %+v", rd.ExamTable)
		}
	})

	// Stored table blob under winning prefix must match postA ("en"), NOT postB ("fr").
	tf, _, err := w.blobs.Open(ctx, w.app.examDigestBlob(idx, blobSHA, examTable))
	if err != nil {
		t.Fatalf("open winning table blob: %v", err)
	}
	tableRaw, _ := io.ReadAll(tf)
	tf.Close()
	var storedTable []ItemInfo
	if err := json.Unmarshal(tableRaw, &storedTable); err != nil || len(storedTable) == 0 || storedTable[0].Lang != "en" {
		t.Fatalf("stored winning table blob overwritten by loser: %s", string(tableRaw))
	}

	// Settle round and confirm the public download serves postA's exact public file, NOT postB's!
	c := w.claim(anchor, false)
	if code, status := w.send(anchor, c, 5.5, 6.0); code != 200 || status != "settled" {
		t.Fatalf("settle grade: HTTP %d %s", code, status)
	}
	req, _ := http.NewRequest("GET", w.srv.URL+"/api/v1/rounds/"+roundID(w.app.season.ID, idx)+"/exam", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET settled exam: %v", err)
	}
	gotPub, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || string(gotPub) != string(publicA) {
		t.Fatalf("settled public download corrupted by losing post: HTTP %d got=%q want=%q", resp.StatusCode, string(gotPub), string(publicA))
	}
	w.check()
}
