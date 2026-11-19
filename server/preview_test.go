//go:build preview

package main

// A season in miniature, played in memory and served, for looking at the website with
// something on it: go test -tags preview -run TestPreview -timeout 0 .
//
//   STET_PREVIEW_ADDR   where to serve it (default 127.0.0.1:8099)
//   STET_PREVIEW_STATE  "open" (default): a round is being run and graded now
//                       "scheduled": the last round has settled and the next is announced
//                       "volunteers": as "open", with volunteers established from round 3
//                       "empty": nothing has happened yet
//                       "stress": a long season with a crowd in it: 70 rounds, thirty
//                       people's curators, some of them failing, the record changing hands

import (
	"context"
	"encoding/base64"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"crypto/sha256"
	"encoding/hex"

	"github.com/klauspost/compress/zstd"
)

type previewActor struct {
	*actor
	gpu   string
	edits bool
	skew  float64 // how far this machine's grades sit from the truth
}

func TestPreview(t *testing.T) {
	state := os.Getenv("STET_PREVIEW_STATE")
	if state == "" {
		state = "open"
	}
	addr := os.Getenv("STET_PREVIEW_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8099"
	}
	orgPub, orgPriv := newKey()
	machPub, machPriv := newKey()
	cfg := testConfig(t, orgPub, machPub)
	cfg.Scale, cfg.QualifyRounds, cfg.RatedRounds, cfg.LadderSize, cfg.MaxNewPerRound = "micro", 3, 3, 8, 3
	cfg.BaseURL = "https://www.stetrun.com"
	if state == "stress" {
		cfg.LadderSize, cfg.MaxNewPerRound = 24, 6
	}
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	if state == "empty" {
		w.app.tick(context.Background())
		serve(t, w, addr)
		return
	}

	stet := w.join("stet", orgPriv)
	machine := &previewActor{actor: w.join("ericflo-wsl", machPriv), gpu: "NVIDIA GeForce RTX 4090", edits: true}
	alice := &previewActor{actor: w.join("alice", nil), gpu: "AMD Radeon RX 7900 XTX", edits: true, skew: 0.012}
	bob := &previewActor{actor: w.join("bob", nil), gpu: "NVIDIA GeForce RTX 3080", skew: -0.009}
	maria := w.join("maria", nil)

	// gain is what a curator is truly worth in a round, in bits per byte: its own level (a
	// tenth of it), the round's mood.
	level := map[string]float64{"identity": 0, "english": 0.92, "tidy": 0.95, "compass": 1.12, "boilerplate-be-gone": 0.71}
	gain := func(name string, round int) float64 {
		wobble := 0.04 * math.Sin(float64(round)*1.7+float64(len(name)))
		if name == "identity" {
			wobble /= 2
		}
		return (level[name] + wobble) / 10
	}
	baseline := func(round int) float64 { return 1.84 + 0.01*math.Sin(float64(round)*2.3) }

	for _, n := range []string{"identity", "english", "tidy"} {
		w.submit(stet, n, fmt.Sprintf("print(%q)", n))
	}
	crew := []*previewActor{machine}
	volunteers := state == "volunteers" || state == "stress"
	rounds := 6
	var stressCrew []*previewActor
	// The crowd: one curator a person, entered a few a round, each with a level of its own.
	// Some arrive late and better than whoever holds the record.
	var crowd []string
	if state == "stress" {
		rounds = 70
		crowd = []string{"dedupe-first", "longform", "no-nav", "prose-only", "wiki-heavy", "short-lines-out", "paragraphs", "anti-seo", "readable", "sentence-rank", "low-entropy-out", "textbook-bias", "quotes-kept", "tables-out", "plain-english", "noisy", "half-and-half", "footer-cutter", "perplexity-sort", "curriculum", "ngram-novelty", "boiler-two", "keep-it-all", "minhash", "backwards", "density", "late-bloomer", "the-long-name-of-a-curator-entry", "second-wind", "closer"}
		for i, n := range crowd {
			level[n] = 0.25 + 0.75*math.Abs(math.Sin(float64(i)*2.1+0.4))
		}
		level["backwards"], level["noisy"] = -0.21, 0.31
		level["late-bloomer"], level["second-wind"], level["closer"] = 1.19, 1.27, 1.31
		// Two more machines, so a third can run what two disagree about; one curator whose
		// runs come out different on one of them, and one grader who is sometimes out alone.
		carol := &previewActor{actor: w.join("carol", nil), gpu: "NVIDIA GeForce RTX 4070", edits: true, skew: 0.004}
		dave := &previewActor{actor: w.join("dave", nil), gpu: "AMD Radeon RX 7800 XT", skew: -0.005}
		stressCrew = []*previewActor{carol, dave}
		previewTwist = func(name, script, grader string, round int) float64 {
			switch {
			case name == "ngram-novelty" && round%4 == 1 && previewRanBy[script] == "alice":
				return 0.19
			case grader == "dave" && round > 40 && round%6 == 0 && name == "readable":
				return 0.31
			}
			return 0
		}
		previewFails = func(name string, round int) bool {
			switch name {
			case "noisy":
				return round%3 == 0
			case "half-and-half", "tables-out":
				return round%11 == 4
			}
			return false
		}
	}
	entered := 0
	for r := 1; r <= rounds; r++ {
		if r == 6 {
			crew = append(crew, stressCrew...)
		}
		switch r {
		case 2:
			w.submit(maria, "boilerplate-be-gone", "print('bbg')")
		case 3:
			if volunteers {
				crew = append(crew, alice, bob)
			}
		case 5:
			w.submit(stet, "compass", "print('compass')")
			w.submit(maria, "kitchen-sink", "print('ks')")
		}
		// Three of the crowd a round, the late ones held back until the season is under way.
		for k := 0; k < 3 && entered < len(crowd) && r >= 3; k++ {
			n := crowd[entered]
			if late := map[string]int{"late-bloomer": 22, "second-wind": 38, "closer": 61}[n]; r < late {
				break
			}
			who := []string{"noor", "tomas", "yuki", "priya", "olu", "ines", "mateo", "linh", "sven", "amara"}[entered%10]
			if entered >= 10 {
				who = fmt.Sprintf("%s%d", who, entered/10+1)
			}
			w.submit(w.join(who, nil), n, fmt.Sprintf("print(%q)", n))
			entered++
		}
		playRound(t, w, crew, gain, baseline)
		shift(w, 5*time.Hour+17*time.Minute)
	}
	switch state {
	case "scheduled":
		w.app.tick(context.Background())
	default:
		// A round in progress: drawn, some of it run, nothing yet published.
		idx := w.draw()
		_ = idx
		for i := 0; i < 3; i++ {
			c := claimAs(t, w, machine)
			if c.Job == nil {
				break
			}
			if c.Job.Kind == "edit" {
				editAs(t, w, machine, c, c.Job.Editor.Name)
			} else {
				reportAs(t, w, machine, c, baseline(idx)-gain(c.Job.Editor.Name, idx), baseline(idx))
			}
		}
		shift(w, 19*time.Minute)
	}
	serve(t, w, addr)
}

func serve(t *testing.T, w *world, addr string) {
	w.app.cfg.Dev = false // built in development mode, which waives the runner's bench; shown as production
	srv := &http.Server{Addr: addr, Handler: w.app.routes()}
	fmt.Fprintf(os.Stderr, "\npreview: http://%s/\n", addr)
	if err := srv.ListenAndServe(); err != nil {
		t.Fatal(err)
	}
}

// playRound draws a round and has the crew work it until it settles.
func playRound(t *testing.T, w *world, crew []*previewActor, gain func(string, int) float64, baseline func(int) float64) {
	idx := w.draw()
	for guard := 0; guard < 2000; guard++ {
		busy := false
		for _, a := range crew {
			c := claimAs(t, w, a)
			if c.Job == nil {
				if guard < 3 {
					fmt.Fprintf(os.Stderr, "preview: %s idle: %s\n", a.handle, c.Message)
				}
				continue
			}
			busy = true
			w.app.limits = newLimiter() // a preview's hour is over in a moment
			if c.Job.Kind == "edit" {
				editAs(t, w, a, c, c.Job.Editor.Name)
				continue
			}
			loss := baseline(idx) - gain(c.Job.Editor.Name, idx) + a.skew
			if previewTwist != nil {
				loss += previewTwist(c.Job.Editor.Name, c.Job.Script.SHA256, a.handle, idx)
			}
			reportAs(t, w, a, c, loss, baseline(idx)+a.skew)
		}
		settled := false
		w.app.st.view(func(s *State) { settled = s.Rounds[idx].Status == "settled" })
		if settled {
			return
		}
		if !busy {
			w.pass(50 * time.Minute)
		}
	}
	t.Fatalf("round %d never settled", idx)
}

func claimAs(t *testing.T, w *world, a *previewActor) *claimed {
	h := w.hello(a.edits)
	h["hardware"] = map[string]any{"gpu": a.gpu, "device": "cuda", "backend": backendOf(a.gpu)}
	return w.claimWith(a.actor, h)
}

// previewFails says whether a curator's run fails in a round, in the seasons where some do.
var previewFails func(name string, round int) bool

// previewTwist is what a season adds to one grade of one script: a run that came out worse
// than the other, a grader out on their own. previewRanBy is whose run wrote each script.
var (
	previewTwist func(name, script, grader string, round int) float64
	previewRanBy = map[string]string{}
)

func roundOf(w0 *world, c *claimed) (idx int) {
	fmt.Sscanf(fmt.Sprint(c.Job.Round["id"]), w0.app.season.ID+"-o%d", &idx)
	return idx
}

func editAs(t *testing.T, w *world, a *previewActor, c *claimed, name string) {
	failed := previewFails != nil && previewFails(name, roundOf(w, c))
	if name == "kitchen-sink" || failed {
		w.edit(a.actor, c, "")
		return
	}
	script := fmt.Sprintf("{\"doc\": 1, \"round\": %v, \"by\": %q}\n", c.Job.Round["id"], name)
	if name == "compass" || name == "ngram-novelty" { // it uses the GPU: no two runs write the same bytes
		script += a.handle + "\n"
	}
	p := map[string]any{
		"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"round": c.Job.Round["id"], "scale": c.Job.Scale, "corpus_sha256": corpusSHA,
		"edit": map[string]any{"status": "ok", "seconds": 20 + float64(len(name))*2.7, "time_limit": 600.0, "sandbox": "bwrap"},
	}
	enc, _ := zstd.NewWriter(nil)
	sum := sha256.Sum256([]byte(script))
	p["script_sha256"], p["script_size"] = hex.EncodeToString(sum[:]), len(script)
	previewRanBy[hex.EncodeToString(sum[:])] = a.handle
	p["script_zst"] = base64.StdEncoding.EncodeToString(enc.EncodeAll([]byte(script), nil))
	var ack struct {
		Error string `json:"error"`
	}
	if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/script", c.Job.ID), p, &ack); code != 200 {
		t.Fatalf("%s's run of %s: HTTP %d %s", a.handle, name, code, ack.Error)
	}
}

func reportAs(t *testing.T, w *world, a *previewActor, c *claimed, loss, base float64) {
	// Each item of the exam is harder or easier, and the student gains more on some pillars
	// than on others; loss and base are its mean bits per byte and the unedited student's.
	table := nextTable()
	ob, oe := make([]float64, len(table)), make([]float64, len(table))
	var mb, me float64
	for i := range table {
		ob[i] = 0.3*math.Sin(float64(i)*1.9) + 0.1*float64(i%3)
		oe[i] = ob[i] - 0.06*float64(i%2)
		mb, me = mb+ob[i]/float64(len(table)), me+oe[i]/float64(len(table))
	}
	exam, baseline := make([]float64, len(table)), make([]float64, len(table))
	for i := range table {
		exam[i], baseline[i] = -(loss + oe[i] - me), -(base + ob[i] - mb)
	}
	kept, ok := map[string]float64{"identity": 1, "english": 0.103, "tidy": 0.096, "compass": 0.088, "boilerplate-be-gone": 0.41}[c.Job.Editor.Name]
	if !ok {
		kept = 0.05 + 0.6*math.Abs(math.Sin(float64(len(c.Job.Editor.Name))))
	}
	var curve [][]float64
	for i := 1; i <= 60; i++ {
		step := float64(i) * 80
		curve = append(curve, []float64{step, 3.4 + loss - base + 4.2*math.Exp(-float64(i)/6) + 0.05*math.Sin(float64(i))})
	}
	p := map[string]any{
		"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"round": c.Job.Round["id"], "scale": c.Job.Scale, "corpus_sha256": corpusSHA, "exam_sha256": examSHA,
		"script_sha256": c.Job.Script.SHA256, "valid": true, "hardware": map[string]any{"gpu": a.gpu, "device": "cuda", "backend": backendOf(a.gpu)},
		"script":     map[string]any{"script_sha256": c.Job.Script.SHA256, "kept_doc_frac": kept, "kept_byte_frac": kept * 0.8, "passes": 1.0, "out_docs": 1200, "ink_frac": 0.0},
		"baseline":   map[string]any{"kind": "next", "mean": -base, "item_scores": baseline},
		"exam":       map[string]any{"kind": "next", "mean": -loss, "item_scores": exam},
		"train":      map[string]any{"seconds": 410.0, "tokens_per_second": 100000.0, "final_train_loss": 3.5 + loss - base, "curve": curve},
		"started_at": time.Now().Unix() - 600, "finished_at": time.Now().Unix(),
	}
	var ack struct {
		Error string `json:"error"`
	}
	if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack); code != 200 {
		t.Fatalf("%s's grade: HTTP %d %s", a.handle, code, ack.Error)
	}
}

// shift moves everything that has happened back in time, as if that long had gone by.
func shift(w *world, d time.Duration) {
	n := int64(d.Seconds())
	back := func(p *int64) {
		if *p > 0 {
			*p -= n
		}
	}
	w.app.st.update(context.Background(), func(tx *Tx) error {
		s := tx.s
		for _, r := range s.Rounds {
			if r.Status == "scheduled" {
				continue
			}
			back(&r.DrawAt)
			back(&r.DrawnAt)
			back(&r.SettledAt)
			back(&r.ActiveAt)
		}
		for _, e := range s.Editors {
			back(&e.CreatedAt)
		}
		for _, e := range s.Entries {
			back(&e.SettledAt)
			back(&e.EditedAt)
		}
		for _, ev := range s.Events {
			back(&ev.At)
		}
		for _, r := range s.Records {
			back(&r.At)
		}
		for _, a := range s.Accounts {
			back(&a.CreatedAt)
			back(&a.LastSeen)
			back(&a.RunnerSeen)
		}
		for _, r := range s.Results {
			back(&r.ReceivedAt)
		}
		for _, c := range s.Claims {
			back(&c.ClaimedAt)
			back(&c.ExpiresAt)
		}
		return nil
	})
}

func backendOf(gpu string) string {
	if len(gpu) > 3 && gpu[:3] == "AMD" {
		return "rocm"
	}
	return "cuda"
}
