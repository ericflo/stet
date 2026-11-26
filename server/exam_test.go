package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
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
	p["baseline"] = map[string]any{"mean": mean(b), "item_scores": b}
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
	bad["train"].(map[string]any)["init_sha256"] = strings.Repeat("b", 64)
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
	if code := carol.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", cc.Job.ID), good, nil); code != 200 {
		t.Fatalf("matching init_sha256 against authoritative exam lock: HTTP %d, want 200", code)
	}
}
