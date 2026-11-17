package main

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/klauspost/compress/zstd"
)

// --- parity with the Python harness ---------------------------------------------------------

func TestSeedMatchesTheHarness(t *testing.T) {
	s := loadSeason()
	var p struct {
		Index  int `json:"index"`
		Beacon struct {
			Round      int64  `json:"round"`
			Randomness string `json:"randomness"`
		} `json:"beacon"`
		Crawl  Crawl  `json:"crawl"`
		Seed   string `json:"seed"`
		Shards []int  `json:"shards"`
		Order  []int  `json:"order"`
		PRF    string `json:"prf"`
		Grid   struct {
			Round int64  `json:"round"`
			Slot  int64  `json:"slot"`
			Off   *int64 `json:"off"`
		} `json:"grid"`
	}
	if err := json.Unmarshal(s.Parity, &p); err != nil {
		t.Fatal(err)
	}
	seed := officialSeed(s.ID, p.Index, p.Beacon.Round, p.Beacon.Randomness, p.Crawl)
	if hexOf(seed) != p.Seed {
		t.Fatalf("seed: got %s, harness says %s", hexOf(seed), p.Seed)
	}
	if got := strconv.FormatUint(prfU64([]byte("seed"), "shard", 0), 10); got != p.PRF {
		t.Fatalf("prf: got %s, harness says %s", got, p.PRF)
	}
	sc := s.Scales[s.OfficialScale]
	got := sampleWithoutReplacement(seed, "shard", p.Crawl.NShards, sc.CorpusUnits*s.Pools["official"].ShardsPerUnit)
	if fmt.Sprint(got) != fmt.Sprint(p.Shards) {
		t.Fatalf("shards: got %v, harness says %v", got, p.Shards)
	}
	if got := docOrder(seed, len(p.Order)); fmt.Sprint(got) != fmt.Sprint(p.Order) {
		t.Fatalf("order: got %v, harness says %v", got, p.Order)
	}
	if slot, ok := s.drawSlot(p.Grid.Round); !ok || slot != p.Grid.Slot {
		t.Fatalf("grid: round %d is slot %d (%v), harness says %d", p.Grid.Round, slot, ok, p.Grid.Slot)
	}
	if _, ok := s.drawSlot(p.Grid.Round + 1); ok || p.Grid.Off != nil {
		t.Fatal("grid: the round after a grid round must not be one")
	}
}

func TestDrandRoundArithmetic(t *testing.T) {
	s := loadSeason()
	b := s.Beacon
	r := b.drandRoundAt(b.Genesis + 10)
	if b.timeOf(r) < b.Genesis+10 || b.timeOf(r-1) >= b.Genesis+10 {
		t.Fatalf("round %d is not the first at or after the time", r)
	}
	// Whatever the scheduler picks is a round every verifier accepts.
	for _, at := range []int64{1800000000, 1800000600, s.nextGrid(1800000123)} {
		if slot, ok := s.drawSlot(b.drandRoundAt(at)); !ok || slot != at {
			t.Fatalf("a draw at %d is not on the grid (slot %d, %v)", at, slot, ok)
		}
	}
}

// --- statistics -----------------------------------------------------------------------------

func TestPairedP(t *testing.T) {
	// t = 4.2426 on 4 degrees of freedom; one-sided p = 0.00662 (any t table).
	if p := pairedP([]float64{1, 2, 3, 4, 5}); math.Abs(p-0.00662) > 2e-4 {
		t.Fatalf("p = %v", p)
	}
	if p := pairedP([]float64{-1, -2, -3, -4, -5}); math.Abs(p-(1-0.00662)) > 2e-4 {
		t.Fatalf("p = %v", p)
	}
	if p := pairedP([]float64{0.1, -0.1, 0.1, -0.1}); math.Abs(p-0.5) > 1e-9 {
		t.Fatalf("p = %v", p)
	}
}

func TestRatingsAdjustForRoundDifficulty(t *testing.T) {
	// Rounds 1-2 are easy (+0.2), rounds 3-4 hard (-0.2). "old" ran all four; "late" ran only
	// the hard ones and is in truth 0.1 better. A plain mean would rank it far below.
	var es []entryRow
	diff := map[int]float64{1: 0.2, 2: 0.2, 3: -0.2, 4: -0.2}
	for r, d := range diff {
		es = append(es, entryRow{RoundIdx: r, EditorID: "old", Gain: 0.5 + d, Valid: true})
		es = append(es, entryRow{RoundIdx: r, EditorID: "anchor", Gain: 0.3 + d, Valid: true})
	}
	for _, r := range []int{3, 4} {
		es = append(es, entryRow{RoundIdx: r, EditorID: "late", Gain: 0.6 + diff[r], Valid: true})
	}
	rt, _ := fitRatings(es)
	if rt["late"].Mean >= rt["old"].Mean {
		t.Fatal("test is not testing anything")
	}
	if d := rt["late"].Adjusted - rt["old"].Adjusted; math.Abs(d-0.1) > 1e-6 {
		t.Fatalf("adjusted difference = %v, want 0.1", d)
	}
	if math.Abs(rt["old"].Adjusted-0.5) > 1e-6 {
		t.Fatalf("old = %v, want 0.5 (its mean over the whole season)", rt["old"].Adjusted)
	}
}

func TestMultiplierNeverExtrapolates(t *testing.T) {
	s := &Season{Curves: map[string][][2]float64{"k": {{0.5, 6.4}, {1, 6.0}, {2, 5.6}, {4, 5.2}}}}
	if m, capped, ok := s.multiplier("k", 0.4); !ok || capped || math.Abs(m-2) > 1e-9 {
		t.Fatalf("got %v %v %v", m, capped, ok)
	}
	if m, capped, ok := s.multiplier("k", 0.6); !ok || capped || math.Abs(m-math.Pow(2, 1.5)) > 1e-9 {
		t.Fatalf("got %v %v %v", m, capped, ok)
	}
	if m, capped, ok := s.multiplier("k", 1.2); !ok || !capped || m != 4 {
		t.Fatalf("beyond the curve: got %v %v %v, want the last measured point and capped", m, capped, ok)
	}
	if m, capped, ok := s.multiplier("k", -0.9); !ok || !capped || m != 0.5 {
		t.Fatalf("below the curve: got %v %v %v", m, capped, ok)
	}
	if _, _, ok := s.multiplier("missing", 0.1); ok {
		t.Fatal("no curve, no multiplier")
	}
	if got := multText(ptr(4), true); got != ">4×" {
		t.Fatalf("a capped multiplier reads %q", got)
	}
}

func ptr(v float64) *float64 { return &v }

// --- a whole season in miniature ------------------------------------------------------------

type actor struct {
	t      *testing.T
	srv    *httptest.Server
	priv   ed25519.PrivateKey
	pub    string
	handle string
}

func (a *actor) do(method, path string, body any, into any) int {
	a.t.Helper()
	var raw []byte
	switch b := body.(type) {
	case nil:
	case []byte:
		raw = b
	default:
		raw, _ = json.Marshal(b)
	}
	req, _ := http.NewRequest(method, a.srv.URL+path, bytes.NewReader(raw))
	if a.priv != nil {
		ts := strconv.FormatInt(time.Now().Unix(), 10)
		sum := sha256.Sum256(raw)
		msg := strings.Join([]string{"stet-v1", method, path, ts, hex.EncodeToString(sum[:])}, "\n")
		req.Header.Set("X-Stet-Key", a.pub)
		req.Header.Set("X-Stet-Time", ts)
		req.Header.Set("X-Stet-Sig", hex.EncodeToString(ed25519.Sign(a.priv, []byte(msg))))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		a.t.Fatal(err)
	}
	defer resp.Body.Close()
	var buf bytes.Buffer
	buf.ReadFrom(resp.Body)
	if into != nil {
		if err := json.Unmarshal(buf.Bytes(), into); err != nil && resp.StatusCode < 300 {
			a.t.Fatalf("%s %s: %v: %s", method, path, err, buf.String())
		}
	}
	return resp.StatusCode
}

type world struct {
	t     *testing.T
	app   *App
	srv   *httptest.Server
	kv    KV
	blobs Blobs
	cfg   Config
}

func testConfig(t *testing.T, anchors ...ed25519.PublicKey) Config {
	cfg := Config{
		BaseURL: "http://site.test", APIURL: "http://api.test", RedirectHosts: map[string]bool{"bare.test": true},
		DataDir: t.TempDir(), FilesDir: t.TempDir(), DocsDir: "../docs", AgentsFile: "../src/stet/templates/AGENTS.md", AnchorKeys: map[string]bool{}, Scale: "nano",
		RoundInterval: 6 * time.Hour, MaxNewPerRound: 8, LadderSize: 10, QualifyRounds: 5, RatedRounds: 3, SpotRate: 0,
		Lease: 3 * time.Hour, FixedCrawl: &Crawl{Crawl: "CC-MAIN-TEST", PathsSHA256: strings.Repeat("ab", 32), NShards: 1000},
		GithubMinAgeDays: 30, Dev: true,
	}
	for _, k := range anchors {
		cfg.AnchorKeys[hex.EncodeToString(k)] = true
	}
	return cfg
}

func fakeBeacon(t *testing.T) {
	beaconSource = func(ctx context.Context, b Beacon, round int64) (string, error) {
		sum := sha256.Sum256([]byte(fmt.Sprint("test-beacon-", round)))
		return hex.EncodeToString(sum[:]), nil
	}
	beaconIsFake = true
	t.Cleanup(func() { beaconSource, beaconIsFake = fetchBeacon, false })
}

func startWorld(t *testing.T, cfg Config, kv KV, blobs Blobs) *world {
	t.Helper()
	app, err := newApp(context.Background(), cfg, loadSeason(), kv, blobs)
	if err != nil {
		t.Fatal(err)
	}
	fakeBeacon(t)
	srv := httptest.NewServer(app.routes())
	t.Cleanup(srv.Close)
	return &world{t: t, app: app, srv: srv, kv: kv, blobs: blobs, cfg: cfg}
}

func newWorld(t *testing.T, anchors ...ed25519.PublicKey) *world {
	t.Helper()
	cfg := testConfig(t, anchors...)
	return startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
}

// check fails the test if what the server holds in memory is not exactly what is stored.
func (w *world) check() {
	w.t.Helper()
	if err := w.app.st.audit(context.Background()); err != nil {
		w.t.Fatalf("memory and storage disagree: %v", err)
	}
}

func newKey() (ed25519.PublicKey, ed25519.PrivateKey) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	return pub, priv
}

func (w *world) join(handle string, priv ed25519.PrivateKey) *actor {
	w.t.Helper()
	if priv == nil {
		_, priv = newKey()
	}
	a := &actor{t: w.t, srv: w.srv, priv: priv, pub: hex.EncodeToString(priv.Public().(ed25519.PublicKey)), handle: handle}
	var out struct {
		Error string `json:"error"`
	}
	if code := a.do("POST", "/api/v1/accounts", map[string]string{"handle": handle}, &out); code != 201 {
		w.t.Fatalf("creating %s: HTTP %d %s", handle, code, out.Error)
	}
	return a
}

func bundle(t *testing.T, name, body string) []byte {
	t.Helper()
	var tb bytes.Buffer
	tw := tar.NewWriter(&tb)
	files := map[string]string{
		"stet.toml": fmt.Sprintf("name = %q\nentry = \"python editor.py\"\ntechnique = \"test\"\n", name),
		"editor.py": body,
	}
	for _, n := range []string{"editor.py", "stet.toml"} {
		tw.WriteHeader(&tar.Header{Name: n, Mode: 0o644, Size: int64(len(files[n])), Typeflag: tar.TypeReg})
		tw.Write([]byte(files[n]))
	}
	tw.Close()
	enc, _ := zstd.NewWriter(nil)
	return enc.EncodeAll(tb.Bytes(), nil)
}

func (w *world) submit(a *actor, name, body string) string {
	w.t.Helper()
	var out struct {
		Editor struct {
			ID string `json:"id"`
		} `json:"editor"`
		Error string `json:"error"`
	}
	if code := a.do("POST", "/api/v1/editors", bundle(w.t, name, body), &out); code != 201 {
		w.t.Fatalf("submit %s: HTTP %d %s", name, code, out.Error)
	}
	return out.Editor.ID
}

// draw schedules and draws the next round, and returns its index.
func (w *world) draw() int {
	w.t.Helper()
	ctx := context.Background()
	w.app.tick(ctx)
	// Entries must be committed well before the draw.
	w.app.st.update(ctx, func(tx *Tx) error {
		for _, e := range tx.s.Editors {
			if e.Status == "queued" {
				e.CreatedAt -= 3600
				tx.saveEditor(e)
			}
		}
		return nil
	})
	if err := w.app.maybeDraw(ctx, true); err != nil {
		w.t.Fatalf("draw: %v", err)
	}
	idx := 0
	w.app.st.view(func(s *State) {
		if r := s.roundWithStatus("open"); r != nil {
			idx = r.Idx
		}
	})
	if idx == 0 {
		w.t.Fatal("no open round after the draw")
	}
	return idx
}

type claimed struct {
	Job *struct {
		ID     int64          `json:"id"`
		Kind   string         `json:"kind"`
		Round  map[string]any `json:"round"`
		Scale  string         `json:"scale"`
		Editor struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"editor"`
		Script *ScriptRef `json:"script"`
	} `json:"job"`
	Announced map[string]any `json:"announced"`
	Error     string         `json:"error"`
}

func (w *world) hello(edits bool) map[string]any {
	return map[string]any{"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"hardware": map[string]any{"gpu": "Test GPU", "device": "cuda"}, "edits": edits, "sandbox": "bwrap"}
}

func (w *world) claim(a *actor, edits bool) *claimed {
	w.t.Helper()
	var c claimed
	if code := a.do("POST", "/api/v1/jobs/claim", w.hello(edits), &c); code != 200 {
		w.t.Fatalf("claim by %s: HTTP %d %s", a.handle, code, c.Error)
	}
	return &c
}

var corpusSHA, examSHA = strings.Repeat("c0", 32), strings.Repeat("e0", 32)

// edit reports an edit job. An empty script means the editor failed.
func (w *world) edit(a *actor, c *claimed, script string) (int, string) {
	w.t.Helper()
	p := map[string]any{
		"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"round": c.Job.Round["id"], "scale": c.Job.Scale, "corpus_sha256": corpusSHA,
		"edit": map[string]any{"status": "ok", "seconds": 3.0, "time_limit": 60.0, "sandbox": "bwrap"},
	}
	if script == "" {
		p["edit"] = map[string]any{"status": "timeout", "seconds": 60.0, "time_limit": 60.0, "reason": "the editor ran out of time"}
	} else {
		enc, _ := zstd.NewWriter(nil)
		sum := sha256.Sum256([]byte(script))
		p["script_sha256"], p["script_size"] = hex.EncodeToString(sum[:]), len(script)
		p["script_zst"] = base64.StdEncoding.EncodeToString(enc.EncodeAll([]byte(script), nil))
	}
	var ack struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/script", c.Job.ID), p, &ack)
	if code != 200 {
		return code, ack.Error
	}
	return code, ack.Status
}

func (w *world) result(c *claimed, loss, baseline float64) map[string]any {
	by, base := map[string]float64{}, map[string]float64{}
	for _, s := range w.app.season.ExamSources {
		by[s], base[s] = loss, baseline
	}
	return map[string]any{
		"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"round": c.Job.Round["id"], "scale": c.Job.Scale, "corpus_sha256": corpusSHA, "exam_sha256": examSHA,
		"script_sha256": c.Job.Script.SHA256, "valid": true, "hardware": map[string]any{"gpu": "Test GPU", "device": "cuda"},
		"script":   map[string]any{"script_sha256": c.Job.Script.SHA256, "kept_doc_frac": 0.1, "kept_byte_frac": 0.08, "passes": 1.4, "out_docs": 1200},
		"baseline": map[string]any{"loss": baseline, "by_source": base}, "exam": map[string]any{"loss": loss, "by_source": by},
		"train":      map[string]any{"seconds": 30.0, "tokens_per_second": 100000.0, "final_train_loss": 5.0, "curve": [][]float64{{1, 9}, {50, 6}, {100, 5.5}, {150, 5.2}}},
		"started_at": time.Now().Unix() - 60, "finished_at": time.Now().Unix(),
	}
}

func (w *world) report(a *actor, c *claimed, loss float64) (int, string) {
	w.t.Helper()
	var ack struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), w.result(c, loss, 6.0), &ack)
	if code != 200 {
		return code, ack.Error
	}
	return code, ack.Status
}

type entryJ struct {
	Editor  string         `json:"editor"`
	Status  string         `json:"status"`
	Stage   string         `json:"stage"`
	Valid   bool           `json:"valid"`
	Loss    *float64       `json:"loss"`
	Gain    *float64       `json:"gain"`
	Results int            `json:"results"`
	Reason  string         `json:"reason"`
	Script  map[string]any `json:"script"`
	Detail  *EntryDetail   `json:"detail"`
}

func (w *world) entries(a *actor, idx int) []entryJ {
	w.t.Helper()
	var out struct {
		Round   map[string]any `json:"round"`
		Entries []entryJ       `json:"entries"`
	}
	if code := a.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &out); code != 200 {
		w.t.Fatalf("round %d: HTTP %d", idx, code)
	}
	return out.Entries
}

func (w *world) account(handle string) Account {
	var out Account
	w.app.st.view(func(s *State) { out = *s.byHandle[handle] })
	return out
}

func TestAnEntryIsEditedByAnAnchorThenGradedByAnyone(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	alice := w.join("alice", nil)
	bob := w.join("bob", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	w.check()

	// Not public, and not downloadable by strangers, until it is drawn.
	stranger := &actor{t: t, srv: w.srv}
	if code := stranger.do("GET", "/api/v1/editors/"+id, nil, nil); code != 404 {
		t.Fatalf("an undrawn editor is visible: HTTP %d", code)
	}
	if code := stranger.do("GET", "/api/v1/editors/"+id+"/bundle", nil, nil); code != 404 {
		t.Fatalf("an undrawn editor can be downloaded: HTTP %d", code)
	}
	if code := alice.do("GET", "/api/v1/editors/"+id, nil, nil); code != 200 {
		t.Fatalf("its owner cannot see it: HTTP %d", code)
	}

	idx := w.draw()
	w.check()
	if code := stranger.do("GET", "/api/v1/editors/"+id+"/bundle", nil, nil); code != 200 {
		t.Fatalf("a drawn editor cannot be downloaded: HTTP %d", code)
	}

	// Nobody but an anchor is ever handed an entrant's code to run.
	if c := w.claim(bob, true); c.Job != nil {
		t.Fatalf("a volunteer was given %s job before any script existed", c.Job.Kind)
	}
	c := w.claim(anchor, true)
	if c.Job == nil || c.Job.Kind != "edit" || c.Job.Script != nil || c.Job.Editor.ID != id {
		t.Fatalf("the anchor should be editing: %+v", c.Job)
	}
	if c.Announced == nil {
		t.Fatal("a claim does not say what the next draw is committed to")
	}
	if again := w.claim(anchor, true); again.Job == nil || again.Job.ID != c.Job.ID {
		t.Fatal("a restarted anchor does not get its unfinished job back")
	}
	w.check()
	if code, msg := w.edit(bob, c, "{}\n"); code != 403 && code != 404 {
		t.Fatalf("a volunteer reported an edit: HTTP %d %s", code, msg)
	}
	script := `{"doc": 3}` + "\n" + `{"doc": 1, "cut": [[0, 40]]}` + "\n"
	if code, status := w.edit(anchor, c, script); code != 200 || status != "edited" {
		t.Fatalf("edit: HTTP %d %s", code, status)
	}
	w.check()
	if code, _ := w.edit(anchor, c, script); code != 409 {
		t.Fatalf("an edit was reported twice: HTTP %d", code)
	}

	// The script is public the moment it exists, and is exactly what was uploaded.
	es := w.entries(stranger, idx)
	if len(es) != 1 || es[0].Stage != "grading" || es[0].Loss != nil || es[0].Script == nil {
		t.Fatalf("after the edit: %+v", es)
	}
	sha, _ := es[0].Script["sha256"].(string)
	resp, err := http.Get(w.srv.URL + "/api/v1/scripts/" + sha)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("fetching the script: %v", err)
	}
	blob, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if raw, err := unzstd(blob, 1<<20); err != nil || string(raw) != script {
		t.Fatalf("the published script is not the one uploaded: %q %v", raw, err)
	}

	// A new verifier's grade is held, and says nothing to anyone.
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" || cb.Job.Script == nil || cb.Job.Script.SHA256 != sha {
		t.Fatalf("bob should be grading that script: %+v", cb.Job)
	}
	if code, status := w.report(bob, cb, 5.50); code != 200 || status != "pending" {
		t.Fatalf("bob's grade: HTTP %d %s", code, status)
	}
	w.check()
	if es := w.entries(stranger, idx); es[0].Status != "open" || es[0].Loss != nil || es[0].Gain != nil || es[0].Detail != nil {
		t.Fatalf("an unsettled entry leaks its grade: %+v", es[0])
	}
	if code, _ := w.report(bob, cb, 5.40); code != 409 {
		t.Fatalf("a grade was reported twice: HTTP %d", code)
	}

	// The anchor's grade settles it, and bob's agreement counts for him.
	ca := w.claim(anchor, true)
	if ca.Job == nil || ca.Job.Kind != "grade" {
		t.Fatalf("the anchor should be grading: %+v", ca.Job)
	}
	wrong := w.result(ca, 5.52, 6.0)
	wrong["script_sha256"] = strings.Repeat("9", 64)
	if code := anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", ca.Job.ID), wrong, nil); code != 400 {
		t.Fatalf("a grade of some other script was accepted: HTTP %d", code)
	}
	if code, status := w.report(anchor, ca, 5.52); code != 200 || status != "settled" {
		t.Fatalf("the anchor's grade: HTTP %d %s", code, status)
	}
	w.check()
	es = w.entries(stranger, idx)
	if es[0].Status != "settled" || !es[0].Valid || es[0].Loss == nil || math.Abs(*es[0].Loss-5.51) > 1e-9 || math.Abs(*es[0].Gain-0.49) > 1e-9 {
		t.Fatalf("settled entry: %+v", es[0])
	}
	if es[0].Detail == nil || !es[0].Detail.Audited || len(es[0].Detail.Verifiers) != 2 || es[0].Detail.Train == nil {
		t.Fatalf("settled entry's detail: %+v", es[0].Detail)
	}
	if b := w.account("bob"); b.Agreed != 1 || b.Streak != 1 || b.Trusted || b.Strikes != 0 {
		t.Fatalf("bob after agreeing once: %+v", b)
	}
	w.app.st.view(func(s *State) {
		if r := s.Rounds[idx]; r.Status != "settled" || r.BaselineLoss == nil || *r.BaselineLoss != 6.0 || r.CorpusSHA != corpusSHA {
			t.Fatalf("the round did not settle: %+v", r)
		}
		for _, r := range s.Results {
			if r.Summary != nil || r.Verdict != "agreed" {
				t.Fatalf("a settled entry's results: %+v", r)
			}
		}
	})

	var lb struct {
		Editors     []BoardRow `json:"editors"`
		Provisional []BoardRow `json:"provisional"`
	}
	stranger.do("GET", "/api/v1/leaderboard", nil, &lb)
	if len(lb.Editors) != 0 || len(lb.Provisional) != 1 || lb.Provisional[0].ID != id || math.Abs(*lb.Provisional[0].Rating-0.49) > 1e-9 {
		t.Fatalf("leaderboard: %+v", lb)
	}

	// A server that restarts has lost nothing.
	again := startWorld(t, w.cfg, w.kv, w.blobs)
	var lb2 struct {
		Provisional []BoardRow `json:"provisional"`
	}
	(&actor{t: t, srv: again.srv}).do("GET", "/api/v1/leaderboard", nil, &lb2)
	if len(lb2.Provisional) != 1 || *lb2.Provisional[0].Rating != *lb.Provisional[0].Rating {
		t.Fatalf("after a restart: %+v", lb2)
	}
	again.check()
	if a := again.account("anchor-one"); !a.Anchor {
		t.Fatal("the anchor is no longer one after a restart")
	}
}

func TestADisagreeingGradeIsAStrike(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	if code, status := w.edit(anchor, w.claim(anchor, true), "{\"doc\": 1}\n"); code != 200 {
		t.Fatal(status)
	}
	cb := w.claim(bob, false)
	if code, status := w.report(bob, cb, 5.0); code != 200 || status != "pending" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	if code, status := w.report(anchor, w.claim(anchor, true), 5.5); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	es := w.entries(bob, idx)
	if math.Abs(*es[0].Loss-5.5) > 1e-9 || es[0].Results != 1 {
		t.Fatalf("the outlier moved the score: %+v", es[0])
	}
	if b := w.account("bob"); b.Strikes != 1 || b.Agreed != 0 || b.Trusted {
		t.Fatalf("bob after an outlier: %+v", b)
	}
}

func TestAFailedEditScoresAsUneditedAndRetiresTheEditor(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	id := w.submit(alice, "slow", "import time; time.sleep(1e9)")
	for n := 1; n <= retireAfterInvalid; n++ {
		idx := w.draw()
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("round %d: %+v", n, c.Job)
		}
		if code, status := w.edit(anchor, c, ""); code != 200 || status != "settled" {
			t.Fatalf("HTTP %d %s", code, status)
		}
		w.check()
		es := w.entries(alice, idx)
		if es[0].Status != "settled" || es[0].Valid || es[0].Gain == nil || *es[0].Gain != 0 || !strings.Contains(es[0].Reason, "out of time") {
			t.Fatalf("a failed edit: %+v", es[0])
		}
		if c := w.claim(anchor, true); c.Job != nil {
			t.Fatalf("there is nothing to grade, but the anchor got %+v", c.Job)
		}
	}
	w.app.st.view(func(s *State) {
		if e := s.Editors[id]; e.Status != "retired" {
			t.Fatalf("an editor that failed %d times is still %s", retireAfterInvalid, e.Status)
		}
		if r := s.roundWithStatus("open"); r != nil {
			t.Fatal("a round whose only entry failed is still open")
		}
	})
}

func TestOnlyAnAnchorPublishesFiles(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, eve := w.join("anchor-one", priv), w.join("eve", nil)
	pack := []byte("the exam pack, which clients check against the hash in their season file")
	var out struct {
		Error  string `json:"error"`
		SHA256 string `json:"sha256"`
		Bytes  int    `json:"bytes"`
	}
	if code := eve.do("PUT", "/api/v1/admin/files/exam-s9.jsonl.zst", pack, &out); code != 403 {
		t.Fatalf("an entrant publishing a file: HTTP %d %s", code, out.Error)
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/..%2Fmeta", pack, &out); code == 201 {
		t.Fatal("a file was published under a name that is not a plain file name")
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/exam-s9.jsonl.zst", pack, &out); code != 201 || out.Bytes != len(pack) {
		t.Fatalf("an anchor publishing a file: HTTP %d %+v", code, out)
	}
	if sum := sha256.Sum256(pack); out.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("the server reports hash %s", out.SHA256)
	}
	resp, err := http.Get(w.srv.URL + "/files/exam-s9.jsonl.zst")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got, _ := io.ReadAll(resp.Body); resp.StatusCode != 200 || !bytes.Equal(got, pack) {
		t.Fatalf("downloading it back: HTTP %d, %d bytes", resp.StatusCode, len(got))
	}
}

func TestAnAnchorGivesVolunteersFirstGo(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.cfg.AnchorPatience = time.Hour
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.submit(anchor, "ref-a", "print(1)")
	w.submit(anchor, "ref-b", "print(2)")
	idx := w.draw()
	for i := 0; i < 2; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("expected an edit job, got %+v", c.Job)
		}
		w.edit(anchor, c, fmt.Sprintf("{\"doc\": %d}\n", i))
	}
	// Nobody else has ever asked for work, so the anchor grades rather than leave the board waiting.
	alone := w.claim(anchor, true)
	if alone.Job == nil || alone.Job.Kind != "grade" {
		t.Fatalf("an anchor with nobody around was given %+v", alone.Job)
	}
	anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", alone.Job.ID), map[string]string{"reason": "test"}, nil)

	// A volunteer turns up and starts on one entry. Now the anchor holds back on both: the
	// one being graded, and the one a volunteer could still take.
	cb := w.claim(bob, false)
	if cb.Job == nil {
		t.Fatal("the volunteer was given nothing")
	}
	if c := w.claim(anchor, true); c.Job != nil {
		t.Fatalf("the anchor took %s of %s from under a volunteer", c.Job.Kind, c.Job.Editor.Name)
	}
	// The volunteer's grade arrives and waits for someone to vouch for it: that is the anchor's cue.
	if _, status := w.report(bob, cb, 5.5); status != "pending" {
		t.Fatal(status)
	}
	ca := w.claim(anchor, true)
	if ca.Job == nil || ca.Job.Editor.ID != cb.Job.Editor.ID {
		t.Fatalf("the anchor should vouch for the waiting grade, got %+v", ca.Job)
	}
	if _, status := w.report(anchor, ca, 5.5); status != "settled" {
		t.Fatal(status)
	}
	if b := w.account("bob"); b.Agreed != 1 {
		t.Fatalf("the volunteer's agreed grade was not counted: %+v", b)
	}
	// The other entry is still inside its patience...
	if c := w.claim(anchor, true); c.Job != nil {
		t.Fatalf("the anchor did not wait for volunteers: %+v", c.Job)
	}
	// ...and once that has run out, the anchor does it, volunteers or no volunteers.
	if err := w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, e := range tx.s.inRound[idx] {
			if e.Status == "open" {
				e.EditedAt -= 2 * 3600
				tx.saveEntry(e)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	last := w.claim(anchor, true)
	if last.Job == nil || last.Job.Kind != "grade" {
		t.Fatalf("an entry nobody took was never graded: %+v", last.Job)
	}
	if _, status := w.report(anchor, last, 5.4); status != "settled" {
		t.Fatal(status)
	}
	w.check()
}

func TestTrustIsEarnedAndThenSettlesAlone(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	for i := 0; i < trustAfterAgreements; i++ {
		w.submit(anchor, fmt.Sprintf("ref-%d", i), fmt.Sprintf("print(%d)", i))
	}
	w.draw()
	release := func(c *claimed) {
		if c.Job != nil {
			anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "test"}, nil)
		}
	}
	for {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			release(c) // the anchor steps away before grading anything
			break
		}
		if code, status := w.edit(anchor, c, "{\"doc\": "+c.Job.Editor.Name[4:]+"}\n"); code != 200 {
			t.Fatal(status)
		}
	}
	for i := 0; i < trustAfterAgreements; i++ {
		cb := w.claim(bob, false)
		if cb.Job == nil {
			t.Fatalf("bob ran out of work after %d grades", i)
		}
		if _, status := w.report(bob, cb, 5.5); status != "pending" {
			t.Fatalf("an untrusted grade settled something: %s", status)
		}
	}
	if b := w.account("bob"); b.Trusted || b.Agreed != 0 {
		t.Fatalf("bob earned trust from nobody: %+v", b)
	}
	for {
		c := w.claim(anchor, true)
		if c.Job == nil {
			break
		}
		if _, status := w.report(anchor, c, 5.5); status != "settled" {
			t.Fatal(status)
		}
	}
	w.check()
	if b := w.account("bob"); !b.Trusted || b.Agreed != trustAfterAgreements {
		t.Fatalf("bob after %d agreements: %+v", trustAfterAgreements, b)
	}

	// Next round, with the anchor away, bob's grade is enough.
	idx := w.draw()
	for {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			release(c)
			break
		}
		w.edit(anchor, c, "{\"doc\": 7"+c.Job.Editor.Name[4:]+"}\n")
	}
	cb := w.claim(bob, false)
	if cb.Job == nil {
		t.Fatal("a trusted verifier was given nothing")
	}
	if code, status := w.report(bob, cb, 5.4); code != 200 || status != "settled" {
		t.Fatalf("a trusted grade did not settle: HTTP %d %s", code, status)
	}
	w.check()
	for _, e := range w.entries(bob, idx) {
		if e.Status == "settled" && (e.Detail == nil || e.Detail.Audited) {
			t.Fatalf("an entry no anchor graded is marked audited: %+v", e)
		}
	}
}

// season plays whole rounds with an anchor doing all the work. loss gives each editor's
// exam loss in a round, by editor name.
func (w *world) play(anchor *actor, rounds int, loss func(name string, round int) float64) {
	w.t.Helper()
	for n := 0; n < rounds; n++ {
		idx := w.draw()
		for {
			c := w.claim(anchor, true)
			if c.Job == nil {
				break
			}
			if c.Job.Kind == "edit" {
				if code, status := w.edit(anchor, c, fmt.Sprintf("{\"doc\": 1, \"round\": %d, \"by\": %q}\n", idx, c.Job.Editor.Name)); code != 200 {
					w.t.Fatal(status)
				}
				continue
			}
			if code, status := w.report(anchor, c, loss(c.Job.Editor.Name, idx)); code != 200 || status != "settled" {
				w.t.Fatalf("HTTP %d %s", code, status)
			}
		}
		w.check()
		w.app.st.view(func(s *State) {
			if s.Rounds[idx].Status != "settled" {
				w.t.Fatalf("round %d did not settle", idx)
			}
		})
	}
}

func TestTheRecordChangesHandsOnlyAtALookAndOnlyByAMargin(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	margin := w.app.season.Record.Margin["nano"]
	wobble := func(round int) float64 { return 0.004 * float64(round%3-1) }
	loss := func(name string, round int) float64 {
		switch name {
		case "first":
			return 5.5 + wobble(round)
		case "copy": // the holder again, give or take nothing that matters
			return 5.5 - margin/3 + wobble(round+1)
		case "better":
			return 5.5 - 3*margin + wobble(round+2)
		}
		return 6.0
	}
	holder := func() string {
		id := ""
		w.app.st.view(func(s *State) {
			if h := s.holder(); h != nil {
				id = s.Editors[h.EditorID].Name
			}
		})
		return id
	}
	w.submit(anchor, "first", "print(1)")
	w.play(anchor, 4, loss)
	if holder() != "" {
		t.Fatal("a record was set before the first look")
	}
	w.play(anchor, 1, loss)
	if holder() != "first" {
		t.Fatalf("after five rounds well clear of the baseline the record is %q", holder())
	}

	// "first" has finished qualifying, but the record holder keeps running: a challenger
	// can only take the record by sharing rounds with it.
	copyLooks := func() (n int) {
		w.app.st.view(func(s *State) {
			for _, l := range s.Looks {
				if s.Editors[l.Challenger].Name == "copy" {
					n++
					if l.P < 0.05 {
						t.Fatalf("copy's look: %+v", l)
					}
				}
			}
		})
		return
	}
	w.submit(anchor, "copy", "print(2)")
	w.play(anchor, 4, loss)
	if copyLooks() != 0 {
		t.Fatal("a challenger was looked at before it had shared five rounds with the holder")
	}
	w.play(anchor, 3, loss)
	if holder() != "first" || copyLooks() != 1 {
		t.Fatalf("a copy of the holder: record %q after %d looks, want the record unmoved after exactly one", holder(), copyLooks())
	}
	w.submit(anchor, "better", "print(3)")
	w.play(anchor, 4, loss)
	if holder() != "first" {
		t.Fatalf("the record moved to %q before the challenger's look", holder())
	}
	w.play(anchor, 1, loss)
	if holder() != "better" {
		t.Fatalf("after five shared rounds three margins ahead, the record is %q", holder())
	}
	var recs struct {
		Records []RecordRow `json:"records"`
	}
	anchor.do("GET", "/api/v1/records", nil, &recs)
	if len(recs.Records) != 2 || recs.Records[1].Name != "better" || recs.Records[1].Rounds != 5 || math.Abs(recs.Records[1].Advantage-3*margin) > 0.01 {
		t.Fatalf("records: %+v", recs.Records)
	}
}

func TestEnteringNeedsAnEstablishedGithubAccount(t *testing.T) {
	cfg := testConfig(t)
	cfg.Dev, cfg.RequireGithub = false, true
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	alice := w.join("alice", nil)
	var out struct {
		Error string `json:"error"`
	}
	if code := alice.do("POST", "/api/v1/editors", bundle(t, "tidy", "print(1)"), &out); code != 403 || !strings.Contains(out.Error, "GitHub") {
		t.Fatalf("an account with no GitHub identity entered: HTTP %d %s", code, out.Error)
	}
	created := time.Now().Add(-48 * time.Hour).Unix()
	realLookup := githubLookup
	t.Cleanup(func() { githubLookup = realLookup })
	githubLookup = func(ctx context.Context, gist string) (string, int64, int64, string, error) {
		return "alice-gh", 4242, created, "stet-key:" + alice.pub + "\n", nil
	}
	link := map[string]string{"gist": "https://gist.github.com/alice-gh/0123456789abcdef0123456789abcdef"}
	if code := alice.do("POST", "/api/v1/accounts/github", link, &out); code != 200 {
		t.Fatalf("linking: HTTP %d %s", code, out.Error)
	}
	if code := alice.do("POST", "/api/v1/editors", bundle(t, "tidy", "print(1)"), &out); code != 403 || !strings.Contains(out.Error, "younger") {
		t.Fatalf("a two-day-old GitHub account entered: HTTP %d %s", code, out.Error)
	}
	created = time.Now().Add(-400 * 24 * time.Hour).Unix()
	alice.do("POST", "/api/v1/accounts/github", link, &out)
	if code := alice.do("POST", "/api/v1/editors", bundle(t, "tidy", "print(1)"), &out); code != 201 {
		t.Fatalf("an established account could not enter: HTTP %d %s", code, out.Error)
	}
	// The same GitHub identity proves itself for a second key: it moves, it is not shared.
	mallory := w.join("mallory", nil)
	githubLookup = func(ctx context.Context, gist string) (string, int64, int64, string, error) {
		return "alice-gh", 4242, created, "stet-key:" + mallory.pub + "\n", nil
	}
	if code := mallory.do("POST", "/api/v1/accounts/github", link, &out); code != 200 {
		t.Fatalf("HTTP %d %s", code, out.Error)
	}
	if a := w.account("alice"); a.Github != "" {
		t.Fatalf("one GitHub identity is linked to two accounts: %+v", a)
	}
	w.check()

	// Outside development, grades must come from the hardware the season names.
	bob := w.join("bob", nil)
	hello := w.hello(false)
	hello["hardware"] = map[string]any{"gpu": "Apple M4", "device": "mps"}
	if code := bob.do("POST", "/api/v1/jobs/claim", hello, &out); code != 400 {
		t.Fatalf("a verifier on other hardware was given work: HTTP %d", code)
	}
}

func TestStorageFailuresNeverLeaveMemoryAhead(t *testing.T) {
	w := newWorld(t)
	kv := w.kv.(*memKV)
	alice := w.join("alice", nil)

	// The commit never happened: the server says so, and the entry does not exist.
	kv.failNext = "before"
	var out struct {
		Editor struct {
			ID string `json:"id"`
		} `json:"editor"`
		Error string `json:"error"`
	}
	blob := bundle(t, "tidy", "print(1)")
	if code := alice.do("POST", "/api/v1/editors", blob, &out); code != 503 {
		t.Fatalf("a failed commit answered HTTP %d", code)
	}
	w.check()
	w.app.st.view(func(s *State) {
		if len(s.Editors) != 0 {
			t.Fatal("an entry exists in memory that was never stored")
		}
	})
	if code := alice.do("POST", "/api/v1/editors", blob, &out); code != 201 {
		t.Fatalf("the retry: HTTP %d %s", code, out.Error)
	}

	// The commit happened but the answer was lost: the server cannot know, says try again,
	// and on looking finds the entry there.
	kv.failNext = "after"
	blob2 := bundle(t, "tidy", "print(2)")
	if code := alice.do("POST", "/api/v1/editors", blob2, &out); code != 503 {
		t.Fatalf("a lost answer was reported as HTTP %d", code)
	}
	w.check()
	if code := alice.do("POST", "/api/v1/editors", blob2, &out); code != 409 || !strings.Contains(out.Error, "already submitted") {
		t.Fatalf("the retry after a lost answer: HTTP %d %s", code, out.Error)
	}
	w.app.st.view(func(s *State) {
		if len(s.Editors) != 2 {
			t.Fatalf("%d entries, want 2", len(s.Editors))
		}
	})

	// Someone else wrote to the database: this server notices and reads it again.
	kv.Commit(context.Background(), kv.rev, []KVWrite{{Key: "meta/crawl", Value: json.RawMessage(`{"crawl":"CC-MAIN-OTHER","paths_sha256":"","n_shards":5,"checked_at":1}`)}})
	if code := alice.do("POST", "/api/v1/editors", bundle(t, "tidy", "print(3)"), &out); code != 503 {
		t.Fatalf("a conflicting write answered HTTP %d", code)
	}
	w.check()
	w.app.st.view(func(s *State) {
		if s.Crawl == nil || s.Crawl.Crawl.Crawl != "CC-MAIN-OTHER" {
			t.Fatal("the other writer's change was not picked up")
		}
	})
}

func TestOneProcessServesTheSiteTheAPIAndTheRedirect(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "tidy", "print(1)")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })

	get := func(host, path string) (int, string, http.Header) {
		req, _ := http.NewRequest("GET", w.srv.URL+path, nil)
		req.Host = host
		client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(body), resp.Header
	}
	if code, _, h := get("bare.test", "/leaderboard?x=1"); code != 308 || h.Get("Location") != "http://site.test/leaderboard?x=1" {
		t.Fatalf("the bare domain: HTTP %d to %q", code, h.Get("Location"))
	}
	if code, body, _ := get("api.test", "/"); code != 200 || !strings.Contains(body, `"service"`) {
		t.Fatalf("the API's root: HTTP %d %s", code, body)
	}
	if code, body, _ := get("api.test", "/leaderboard"); code != 404 || !strings.HasPrefix(body, "{") {
		t.Fatalf("the API host served a page: HTTP %d", code)
	}
	if code, _, _ := get("api.test", "/api/v1/season"); code != 200 {
		t.Fatalf("the API host: HTTP %d", code)
	}
	if code, _, _ := get("api.test", "/readyz"); code != 200 {
		t.Fatalf("readyz: HTTP %d", code)
	}

	// Every page renders, with something on it.
	var id, round string
	w.app.st.view(func(s *State) {
		for k := range s.Editors {
			id = k
		}
		for _, r := range s.Rounds {
			if r.Status == "settled" {
				round = roundID(w.app.season.ID, r.Idx)
			}
		}
	})
	for path, want := range map[string]string{
		"/": "Stet", "/leaderboard": "tidy", "/editors/" + id[:12]: "editor.py", "/rounds": round, "/rounds/" + round: "tidy",
		"/verifiers": "anchor-one", "/u/anchor-one": "tidy", "/static/site.css": "--paper", "/nothing-here": "nothing at this address",
	} {
		code, body, _ := get("site.test", path)
		wantCode := 200
		if path == "/nothing-here" {
			wantCode = 404
		}
		if code != wantCode || !strings.Contains(body, want) {
			t.Errorf("%s: HTTP %d, and %q is missing from the page", path, code, want)
		}
		if strings.Contains(body, "something went wrong") || strings.Contains(body, "<no value>") || strings.Contains(body, "%!") {
			t.Errorf("%s: the page rendered with an error in it", path)
		}
	}
}

// --- against a real durable-db --------------------------------------------------------------

func freePort(t *testing.T) int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// startDurableDB runs the real database, if this machine has one built.
func startDurableDB(t *testing.T) *ddbKV {
	bin := os.Getenv("DURABLE_DB_BIN")
	if bin == "" {
		home, _ := os.UserHomeDir()
		bin = filepath.Join(home, "Development", "durable-db", "target", "release", "durable-db")
	}
	if _, err := os.Stat(bin); err != nil {
		t.Skip("no durable-db binary (set DURABLE_DB_BIN to run this)")
	}
	dir := t.TempDir()
	addr := fmt.Sprintf("127.0.0.1:%d", freePort(t))
	cmd := exec.Command(bin, "serve", "--listen", addr, "--data", filepath.Join(dir, "data"), "--key-file", filepath.Join(dir, "admin.key"), "--no-fsync")
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
	var key []byte
	for i := 0; i < 100; i++ {
		if raw, err := os.ReadFile(filepath.Join(dir, "admin.key")); err == nil {
			if key, err = parseDBKey(string(raw)); err == nil {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	if key == nil {
		t.Fatal("durable-db did not write its key file")
	}
	db := newDurableDB("http://"+addr, "stet-test", key)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := db.ensure(ctx); err != nil {
		t.Fatalf("creating the database: %v", err)
	}
	if err := db.ensure(ctx); err != nil {
		t.Fatalf("creating the database a second time: %v", err)
	}
	return db
}

func TestTheWholeFlowOnDurableDB(t *testing.T) {
	db := startDurableDB(t)
	pub, priv := newKey()
	cfg := testConfig(t, pub)
	blobs := dirBlobs{filepath.Join(cfg.DataDir, "blobs")}
	w := startWorld(t, cfg, db, blobs)
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "first", "print(1)")
	w.submit(anchor, "second", "print(2)")
	w.check()
	w.play(anchor, 5, func(name string, round int) float64 {
		if name == "first" {
			return 5.4 + 0.003*float64(round%2)
		}
		return 5.7 + 0.003*float64(round%3)
	})
	var before, after struct {
		Editors []BoardRow `json:"editors"`
	}
	anchor.do("GET", "/api/v1/leaderboard", nil, &before)
	if len(before.Editors) != 2 || before.Editors[0].Name != "first" || !before.Editors[0].Record {
		t.Fatalf("leaderboard: %+v", before.Editors)
	}

	// A second server on the same database sees exactly the same season...
	other := startWorld(t, cfg, db, blobs)
	(&actor{t: t, srv: other.srv}).do("GET", "/api/v1/leaderboard", nil, &after)
	if fmt.Sprint(mustJSON(before)) != fmt.Sprint(mustJSON(after)) {
		t.Fatalf("a second server disagrees:\n%s\n%s", mustJSON(before), mustJSON(after))
	}
	other.check()
	// ...and when it writes, the first notices instead of overwriting it.
	bob := other.join("bob", nil)
	_ = bob
	var out struct {
		Error string `json:"error"`
	}
	if code := anchor.do("POST", "/api/v1/editors", bundle(t, "third", "print(3)"), &out); code != 503 {
		t.Fatalf("a write from a stale server answered HTTP %d %s", code, out.Error)
	}
	if code := anchor.do("POST", "/api/v1/editors", bundle(t, "third", "print(3)"), &out); code != 201 {
		t.Fatalf("the retry: HTTP %d %s", code, out.Error)
	}
	w.check()
	if a := w.account("bob"); a.Handle != "bob" {
		t.Fatal("the first server never learned of the account made on the second")
	}

	// A server that has nothing to write still catches up: its scheduler asks the database
	// for its revision, so the old server of a deploy does not answer from memory.
	editors := func() (n int) {
		other.app.st.view(func(s *State) { n = len(s.Editors) })
		return n
	}
	if n := editors(); n != 2 {
		t.Fatalf("the second server holds %d editors before it has been told of the third", n)
	}
	other.app.st.sync(context.Background())
	if n := editors(); n != 3 {
		t.Fatalf("the second server holds %d editors after asking the database", n)
	}
	other.check()
}
