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
		req.Header.Set("X-Stet-Body", hex.EncodeToString(sum[:]))
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
		RoundInterval: 6 * time.Hour, MaxNewPerRound: 8, LadderSize: 10, QualifyRounds: 5, RatedRounds: 3, ReplicaWait: 45 * time.Minute,
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

	// A volunteer's grade is sealed, and says nothing to anyone.
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" || cb.Job.Script == nil || cb.Job.Script.SHA256 != sha {
		t.Fatalf("bob should be grading that script: %+v", cb.Job)
	}
	w.app.st.update(context.Background(), func(tx *Tx) error {
		c := tx.s.Claims[cb.Job.ID]
		c.ClaimedAt -= 120 // bob has had the job for two minutes
		tx.saveClaim(c)
		return nil
	})
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

	// The anchor's grade is the score, to the digit, and bob's agreement counts for him.
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
	if es[0].Status != "settled" || !es[0].Valid || es[0].Loss == nil || *es[0].Loss != 5.52 || math.Abs(*es[0].Gain-0.48) > 1e-9 {
		t.Fatalf("settled entry: %+v", es[0])
	}
	if es[0].Detail == nil || !es[0].Detail.Audited || len(es[0].Detail.Verifiers) != 2 || es[0].Detail.Train == nil {
		t.Fatalf("settled entry's detail: %+v", es[0].Detail)
	}
	if b := w.account("bob"); b.Agreed != 1 || b.Streak != 1 || b.Strikes != 0 || b.GPUSeconds != 60 {
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
	if len(lb.Editors) != 0 || len(lb.Provisional) != 1 || lb.Provisional[0].ID != id || math.Abs(*lb.Provisional[0].Rating-0.48) > 1e-9 {
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
	if b := w.account("bob"); b.Strikes != 1 || b.Agreed != 0 || b.GPUSeconds != 0 || b.LastStrikeAt == 0 || muted(&b, now()) {
		t.Fatalf("bob after an outlier: %+v", b)
	}
	// Every volunteer's grade is on show beside the anchor's: this one differed, and it came
	// from an account nobody has reason to believe yet. Both are said.
	d := es[0].Detail
	if d.Disputed != 1 || d.DisputedTogether || len(d.Verifiers) != 1 || len(d.Replications) != 1 {
		t.Fatalf("the entry's detail: %+v", d)
	}
	if r := d.Replications[0]; r.Handle != "bob" || r.Verdict != "differs" || !r.Valid || r.Loss == nil || *r.Loss != 5.0 ||
		r.Diff == nil || math.Abs(*r.Diff+0.5) > 1e-9 || r.BaseDiff == nil || *r.BaseDiff != 0 || !r.First {
		t.Fatalf("bob's grade as published: %+v", r)
	}
	// And on the round's page, for people.
	req, _ := http.NewRequest("GET", w.srv.URL+"/rounds/"+roundID(w.app.season.ID, idx), nil)
	req.Host = "site.test"
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	page, _ := io.ReadAll(resp.Body)
	for _, want := range []string{`id="replications"`, `<a href="/u/bob">bob</a>`, `5.000`, `−0.500`, `differs`, `first grade`, `1 differed`} {
		if !strings.Contains(string(page), want) {
			t.Fatalf("the round's page does not show %q", want)
		}
	}
}

// established gives a volunteer a grade that agreed, some time ago.
func (w *world) established(handles ...string) {
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, h := range handles {
			c := tx.s.byHandle[h]
			c.Agreed++
			tx.saveAccount(c)
		}
		return nil
	})
}

func TestVolunteersWhoFindTheSameThingAreNotStruck(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, carol := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol")
	idx := w.draw()
	if code, status := w.edit(anchor, w.claim(anchor, true), "{\"doc\": 1}\n"); code != 200 {
		t.Fatal(status)
	}
	for who, loss := range map[*actor]float64{bob: 5.0, carol: 5.01} {
		if code, status := w.report(who, w.claim(who, false), loss); code != 200 || status != "pending" {
			t.Fatalf("%s: HTTP %d %s", who.handle, code, status)
		}
	}
	if code, status := w.report(anchor, w.claim(anchor, true), 5.5); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	// The score is still the anchor's. But two volunteers found the same thing against it,
	// so neither is struck, and the entry says the anchor's grade is in question.
	es := w.entries(alice, idx)
	d := es[0].Detail
	if math.Abs(*es[0].Loss-5.5) > 1e-9 || d.Disputed != 2 || !d.DisputedTogether || len(d.Replications) != 2 {
		t.Fatalf("the entry: %+v %+v", es[0], d)
	}
	for _, h := range []string{"bob", "carol"} {
		if c := w.account(h); c.Strikes != 0 || c.LastStrikeAt != 0 {
			t.Fatalf("%s was struck for a grade another volunteer also found: %+v", h, c)
		}
	}
}

func TestAgreeingAboutTheUneditedCorpusEarnsNoStanding(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "identity", "print('everything')")
	idx := w.draw()
	if code, status := w.edit(anchor, w.claim(anchor, true), "{\"doc\": 1}\n"); code != 200 {
		t.Fatal(status)
	}
	// The unedited corpus's loss is the round's baseline (6.0 here), which every grader
	// reports with every grade and which hardly moves from round to round. A volunteer who
	// "agrees" about it may not have trained anything.
	if code, status := w.report(bob, w.claim(bob, false), 6.0); code != 200 {
		t.Fatal(status)
	}
	if code, status := w.report(anchor, w.claim(anchor, true), 6.0); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	d := w.entries(alice, idx)[0].Detail
	if len(d.Replications) != 1 || d.Replications[0].Verdict != "agreed" || len(d.Verifiers) != 2 {
		t.Fatalf("the entry's detail: %+v", d)
	}
	if b := w.account("bob"); b.Agreed != 0 || b.Streak != 0 || b.Strikes != 0 {
		t.Fatalf("bob after agreeing about the unedited corpus: %+v", b)
	}
	w.app.st.view(func(s *State) {
		if recentAgreed(s)[s.byHandle["bob"].ID] != 0 {
			t.Fatal("it moved him up the queue")
		}
	})
}

func TestAMuteLiftsAndAJobLeftToRunOutCostsADay(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	ctx := context.Background()
	asks := func() int {
		var out claimed
		return bob.do("POST", "/api/v1/jobs/claim", w.hello(false), &out)
	}
	set := func(f func(c *Account)) {
		w.app.st.update(ctx, func(tx *Tx) error {
			c := tx.s.byHandle["bob"]
			f(c)
			tx.saveAccount(c)
			return nil
		})
	}
	set(func(c *Account) { c.Strikes, c.LastStrikeAt = muteAfterStrikes, now() })
	if code := asks(); code != 403 {
		t.Fatalf("a muted account asked for work: HTTP %d", code)
	}
	// The machine may be honest and merely different. After a while it may try again.
	set(func(c *Account) { c.LastStrikeAt = now() - muteFor - 1 })
	if code := asks(); code != 200 {
		t.Fatalf("a mute never lifted: HTTP %d", code)
	}
	set(func(c *Account) { c.Strikes = 0 })

	w.submit(alice, "tidy", "print('v1')")
	w.draw()
	w.editAll(anchor)
	c := w.claim(bob, false)
	if c.Job == nil {
		t.Fatal("no work for a volunteer")
	}
	lapse := func(ago int64) {
		w.app.st.update(ctx, func(tx *Tx) error {
			cl := tx.s.Claims[c.Job.ID]
			cl.ExpiresAt = now() - ago
			tx.saveClaim(cl)
			return nil
		})
		w.app.tick(ctx)
	}
	lapse(10)
	w.app.st.view(func(s *State) {
		if s.Claims[c.Job.ID].Status != "lapsed" {
			t.Fatalf("a job whose time ran out: %+v", s.Claims[c.Job.ID])
		}
	})
	if code := asks(); code != 403 {
		t.Fatalf("a verifier that let a job run out was handed another at once: HTTP %d", code)
	}
	lapse(lapseRest + 10)
	if code := asks(); code != 200 {
		t.Fatalf("a day later: HTTP %d", code)
	}
	w.check()
}

func TestAnEntryTheAnchorCannotGradeIsGivenUp(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	if code, status := w.edit(anchor, w.claim(anchor, true), "{\"doc\": 1}\n"); code != 200 {
		t.Fatal(status)
	}
	// The script is one the harness falls over on, every time.
	for i := 0; i < maxGradeAttempts; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "grade" {
			t.Fatalf("attempt %d: %+v", i, c.Job)
		}
		anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "the harness fell over"}, nil)
	}
	if c := w.claim(anchor, true); c.Job != nil {
		t.Fatalf("a sixth attempt was handed out: %+v", c.Job)
	}
	w.app.tick(context.Background())
	es := w.entries(alice, idx)
	if es[0].Status != "settled" || es[0].Valid || !strings.Contains(es[0].Reason, "could not grade") {
		t.Fatalf("the entry: %+v", es[0])
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].Status != "settled" || !s.ofEditor[id][0].Void {
			t.Fatalf("the round is held open by an entry nobody can grade: %+v", s.Rounds[idx])
		}
	})
	w.check()
}

func TestAnInvalidEntrySaysNothingAboutItsRound(t *testing.T) {
	var rows []entryRow
	gains := map[string]float64{"a": 0.90, "b": 0.95, "c": 0.97}
	hard := map[int]float64{1: 0.05, 2: -0.03, 3: 0.0, 4: -0.02}
	for r, d := range hard {
		for id, g := range gains {
			if id == "c" && r > 2 {
				continue // c ran only the first two rounds
			}
			rows = append(rows, entryRow{RoundIdx: r, EditorID: id, Gain: g + d, Valid: true})
		}
	}
	before, _ := fitRatings(rows)
	// One more editor times out in round 2. Its zero is its own business.
	after, _ := fitRatings(append(rows, entryRow{RoundIdx: 2, EditorID: "z", Gain: 0, Valid: false}))
	for id := range gains {
		if math.Abs(before[id].Adjusted-after[id].Adjusted) > 1e-9 {
			t.Fatalf("%s moved from %.5f to %.5f because someone else's entry was invalid", id, before[id].Adjusted, after[id].Adjusted)
		}
	}
	if z := after["z"]; z.Adjusted != 0 || z.Invalid != 1 {
		t.Fatalf("the invalid entry's own rating: %+v", z)
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
	// The season names the one file an organiser publishes, and its hash.
	pack := []byte("the exam pack, which clients check against the hash in their season file")
	sum := sha256.Sum256(pack)
	name := w.app.season.ExamFile
	w.app.season.ExamSHA256 = hex.EncodeToString(sum[:])
	var out struct {
		Error  string `json:"error"`
		SHA256 string `json:"sha256"`
		Bytes  int    `json:"bytes"`
	}
	if code := eve.do("PUT", "/api/v1/admin/files/"+name, pack, &out); code != 403 {
		t.Fatalf("an entrant publishing a file: HTTP %d %s", code, out.Error)
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/..%2Fmeta", pack, &out); code == 201 {
		t.Fatal("a file was published under a name that is not a plain file name")
	}
	// Not even an anchor's key publishes anything else: another name, or other bytes.
	if code := anchor.do("PUT", "/api/v1/admin/files/install.sh", pack, &out); code != 400 {
		t.Fatalf("an anchor published a file the season does not name: HTTP %d %s", code, out.Error)
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/"+name, []byte("some other exam"), &out); code != 400 {
		t.Fatalf("an anchor published an exam pack with the wrong hash: HTTP %d %s", code, out.Error)
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/"+name, pack, &out); code != 201 || out.Bytes != len(pack) {
		t.Fatalf("an anchor publishing a file: HTTP %d %+v", code, out)
	}
	if out.SHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("the server reports hash %s", out.SHA256)
	}
	resp, err := http.Get(w.srv.URL + "/files/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got, _ := io.ReadAll(resp.Body); resp.StatusCode != 200 || !bytes.Equal(got, pack) {
		t.Fatalf("downloading it back: HTTP %d, %d bytes", resp.StatusCode, len(got))
	}
	// The held-out exam is the one other file, published when the season closes: only the
	// file the season committed to, under its own name.
	held := []byte("the held-out exam, which nobody has seen until the season closes")
	hsum := sha256.Sum256(held)
	heldName := w.app.season.heldoutFile()
	w.app.season.HeldoutSHA256 = ""
	if code := anchor.do("PUT", "/api/v1/admin/files/"+heldName, held, &out); code != 400 {
		t.Fatalf("a held-out exam was published in a season that committed to none: HTTP %d", code)
	}
	w.app.season.HeldoutSHA256 = hex.EncodeToString(hsum[:])
	if code := anchor.do("PUT", "/api/v1/admin/files/"+heldName, pack, &out); code != 400 {
		t.Fatalf("something else was published as the held-out exam: HTTP %d", code)
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/"+name, held, &out); code != 400 {
		t.Fatalf("the held-out exam was published under the exam's name: HTTP %d", code)
	}
	// It cannot be taken back. One command sent a month early must not be enough.
	if code := anchor.do("PUT", "/api/v1/admin/files/"+heldName, held, &out); code != 403 {
		t.Fatalf("the held-out exam was published while the season was open: HTTP %d", code)
	}
	if _, _, err := w.blobs.Open(context.Background(), "files/"+heldName); err == nil {
		t.Fatal("the held-out exam is in the bucket")
	}
	seedName := "heldout-" + w.app.season.ID + ".seed"
	if code := anchor.do("PUT", "/api/v1/admin/files/"+seedName, []byte("abc"), &out); code != 400 {
		t.Fatalf("the seed was published while the season was open: HTTP %d", code)
	}
	w.app.cfg.SeasonClosed = true
	if code := anchor.do("PUT", "/api/v1/admin/files/"+heldName, held, &out); code != 201 || out.SHA256 != hex.EncodeToString(hsum[:]) {
		t.Fatalf("publishing the held-out exam: HTTP %d %+v", code, out)
	}
	// With it go the seed, the results and the table, and still nothing else.
	if code := anchor.do("PUT", "/api/v1/admin/files/"+seedName, []byte("abc"), &out); code != 201 {
		t.Fatalf("publishing the seed after the close: HTTP %d %+v", code, out)
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/anything.txt", []byte("abc"), &out); code != 400 {
		t.Fatalf("something else was published after the close: HTTP %d", code)
	}
}

func TestTheBoardsCountWholeRoundsAndRealGraders(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(anchor, "good", "print(1)")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	rating := func() float64 {
		for _, r := range append(w.app.board().Rated, w.app.board().Provisional...) {
			if r.Name == "good" {
				return *r.Rating
			}
		}
		t.Fatal("the editor is not on the board")
		return 0
	}
	before := rating()
	// A second round: alice's editor writes no script, which is known at once, while the
	// rest of the round is still sealed. Alone, that zero would stand for the whole round
	// and move everybody else's rating.
	w.submit(alice, "nothing", "print(2)")
	w.draw()
	for {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			break
		}
		script := "{\"doc\": 1}\n"
		if c.Job.Editor.Name == "nothing" {
			script = ""
		}
		if code, status := w.edit(anchor, c, script); code != 200 {
			t.Fatal(status)
		}
	}
	if got := rating(); math.Abs(got-before) > 1e-9 {
		t.Fatalf("an open round moved a rating from %.5f to %.5f", before, got)
	}
	// Asking for work puts nobody on the verifier board; a grade that agreed does.
	bob.do("POST", "/api/v1/jobs/claim", w.hello(false), nil)
	var vs struct {
		Verifiers []VerifierRow `json:"verifiers"`
	}
	bob.do("GET", "/api/v1/verifiers", nil, &vs)
	if len(vs.Verifiers) != 1 || !vs.Verifiers[0].Anchor || vs.Verifiers[0].Validated != 0 {
		t.Fatalf("the verifier board: %+v", vs.Verifiers)
	}
	w.established("bob")
	bob.do("GET", "/api/v1/verifiers", nil, &vs)
	if len(vs.Verifiers) != 2 {
		t.Fatalf("the verifier board after a grade agreed: %+v", vs.Verifiers)
	}
}

// editAll has the anchor run every edit job of the open round, and returns how many.
func (w *world) editAll(anchor *actor) int {
	w.t.Helper()
	n := 0
	for {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			if c.Job != nil {
				anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "test"}, nil)
			}
			return n
		}
		if code, status := w.edit(anchor, c, fmt.Sprintf("{\"doc\": %d, \"by\": %q}\n", n, c.Job.Editor.Name)); code != 200 {
			w.t.Fatal(status)
		}
		n++
	}
}

func (w *world) settled(idx int) (n int) {
	w.app.st.view(func(s *State) {
		for _, e := range s.inRound[idx] {
			if e.Status == "settled" {
				n++
			}
		}
	})
	return n
}

func TestARoundSettlesAsAWholeAndWaitsForAVolunteerWhoHasStarted(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob, carol := w.join("anchor-one", priv), w.join("bob", nil), w.join("carol", nil)
	w.submit(anchor, "ref-a", "print(1)")
	w.submit(anchor, "ref-b", "print(2)")
	idx := w.draw()
	if w.editAll(anchor) != 2 {
		t.Fatal("the anchor should have edited both entries")
	}
	w.established("bob") // the round waits only for a volunteer who has finished a job before
	cb := w.claim(bob, false)
	if cb.Job == nil {
		t.Fatal("the volunteer was given nothing")
	}
	// The anchor does not wait for anyone: it grades both entries.
	for i := 0; i < 2; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "grade" {
			t.Fatalf("the anchor should be grading, got %+v", c.Job)
		}
		if code, status := w.report(anchor, c, 5.5); code != 200 || status != "pending" {
			t.Fatalf("an anchor's grade with a volunteer still at work: HTTP %d %s", code, status)
		}
	}
	// Nothing is public while a volunteer who started in time is still training...
	if w.settled(idx) != 0 {
		t.Fatal("the round settled under a volunteer who had started")
	}
	for _, e := range w.entries(bob, idx) {
		if e.Loss != nil || e.Detail != nil {
			t.Fatalf("a sealed entry shows its grade: %+v", e)
		}
	}
	var rd struct {
		Round map[string]any `json:"round"`
	}
	bob.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &rd)
	if _, ok := rd.Round["baseline_loss"]; ok {
		t.Fatal("an open round publishes the anchor's baseline, which is half of every grade")
	}
	// ...and nobody new is started on a round that is only waiting to close.
	if c := w.claim(carol, false); c.Job != nil {
		t.Fatalf("a volunteer was started on a round that is about to settle: %+v", c.Job)
	}
	// The volunteer reports, and the whole round is published at once.
	if code, status := w.report(bob, cb, 5.52); code != 200 || status != "settled" {
		t.Fatalf("the last grade of the round: HTTP %d %s", code, status)
	}
	if w.settled(idx) != 2 {
		t.Fatal("the round did not settle once everyone had reported")
	}
	if b := w.account("bob"); b.Agreed != 2 || b.Streak != 1 {
		t.Fatalf("the volunteer's agreed grade was not counted: %+v", b)
	}
	w.check()
}

func TestAGradeThatArrivesAfterTheRoundSettlesCountsForNothing(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, mallory := w.join("anchor-one", priv), w.join("mallory", nil)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.draw()
	w.editAll(anchor)
	w.established("mallory")
	// Mallory takes the job and trains nothing.
	cm := w.claim(mallory, false)
	if cm.Job == nil {
		t.Fatal("no job")
	}
	if _, status := w.report(anchor, w.claim(anchor, true), 5.5); status != "pending" {
		t.Fatalf("the anchor's grade should wait for the volunteer: %s", status)
	}
	// The wait runs out. The round settles without her.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, r := range tx.s.Results {
			r.ReceivedAt -= 3600
			tx.saveResult(r)
		}
		return nil
	})
	w.app.tick(context.Background())
	if w.settled(idx) != 1 {
		t.Fatal("the round waited for ever for a volunteer who never reported")
	}
	// Holding a round up like that is not free: no more work for a day.
	var refused claimed
	if code := mallory.do("POST", "/api/v1/jobs/claim", w.hello(false), &refused); code != 403 || !strings.Contains(refused.Error, "a day") {
		t.Fatalf("a volunteer the round had to give up on asked for more work: HTTP %d %s", code, refused.Error)
	}
	// Now the score is public, and she reports it as her own.
	es := w.entries(mallory, idx)
	if es[0].Loss == nil {
		t.Fatal("a settled entry has no loss")
	}
	if code, msg := w.report(mallory, cm, *es[0].Loss); code != 409 {
		t.Fatalf("a grade copied from the published score was accepted: HTTP %d %s", code, msg)
	}
	if m := w.account("mallory"); m.Agreed != 1 || m.Streak != 0 || m.GPUSeconds != 0 {
		t.Fatalf("a copied grade earned credit: %+v", m)
	}
	es = w.entries(mallory, idx)
	if *es[0].Loss != 5.5 || len(es[0].Detail.Verifiers) != 1 {
		t.Fatalf("a late grade changed a settled entry: %+v", es[0])
	}
	w.check()
}

func TestOnlyAnAnchorSetsAScore(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, m1, m2 := w.join("anchor-one", priv), w.join("mallory", nil), w.join("mallet", nil)
	// Mallory has agreed with the anchor before, so her account is not a newcomer's.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		acct := tx.s.byHandle["mallory"]
		acct.Agreed = 4
		tx.saveAccount(acct)
		return nil
	})
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.draw()
	w.editAll(anchor)
	// Two accounts report the same invented number, 0.3 nats better than the truth.
	for _, m := range []*actor{m1, m2} {
		c := w.claim(m, false)
		if c.Job == nil {
			t.Fatalf("%s was given nothing", m.handle)
		}
		if code, status := w.report(m, c, 5.2); code != 200 || status != "pending" {
			t.Fatalf("HTTP %d %s", code, status)
		}
	}
	if w.settled(idx) != 0 {
		t.Fatal("two volunteers settled an entry between them")
	}
	if code, status := w.report(anchor, w.claim(anchor, true), 5.5); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	es := w.entries(m1, idx)
	if *es[0].Loss != 5.5 {
		t.Fatalf("volunteers moved the score: %v", *es[0].Loss)
	}
	// Both are on record as differing, with their numbers. One of them has never agreed
	// with an anchor, so neither can stand behind the other: both are struck.
	if d := es[0].Detail; d.Disputed != 2 || d.DisputedTogether || len(d.Verifiers) != 1 || len(d.Replications) != 2 {
		t.Fatalf("the entry's detail: %+v", d)
	}
	for _, h := range []string{"mallory", "mallet"} {
		if m := w.account(h); m.Strikes != 1 || m.Streak != 0 {
			t.Fatalf("%s after disagreeing with the anchor: %+v", h, m)
		}
	}
	w.check()
}

func TestAVolunteerIsOfferedAnEntryOnceAndNewcomersShareOneSeat(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob, carol, dave := w.join("anchor-one", priv), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.app.st.update(context.Background(), func(tx *Tx) error {
		acct := tx.s.byHandle["dave"]
		acct.Agreed = 1
		tx.saveAccount(acct)
		return nil
	})
	w.submit(anchor, "ref-a", "print(1)")
	w.draw()
	w.editAll(anchor)
	cb := w.claim(bob, false)
	if cb.Job == nil {
		t.Fatal("no job")
	}
	// One newcomer per entry: carol waits, dave (who has agreed before) does not.
	if c := w.claim(carol, false); c.Job != nil {
		t.Fatalf("a second newcomer was seated on the same entry: %+v", c.Job)
	}
	if c := w.claim(dave, false); c.Job == nil {
		t.Fatal("an established volunteer was kept out by a newcomer")
	}
	// Handing a job back does not buy another go at it, or a way to pick and choose.
	bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", cb.Job.ID), map[string]string{"reason": "test"}, nil)
	if c := w.claim(bob, false); c.Job != nil {
		t.Fatalf("a released job was handed back: %+v", c.Job)
	}
	if code, _ := w.report(bob, cb, 5.5); code != 409 {
		t.Fatalf("a released job was reported: HTTP %d", code)
	}
	// The anchor was never held up by any of it.
	if c := w.claim(anchor, true); c.Job == nil || c.Job.Kind != "grade" {
		t.Fatalf("the anchor should be grading: %+v", c.Job)
	}
	w.check()
}

func TestGradesThatKeepDisagreeingEndTheWorkButNotTheEntries(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	for i := 0; i < muteAfterStrikes; i++ {
		w.submit(anchor, fmt.Sprintf("ref-%d", i), fmt.Sprintf("print(%d)", i))
	}
	w.draw()
	w.editAll(anchor)
	for i := 0; i < muteAfterStrikes; i++ {
		c := w.claim(bob, false)
		if c.Job == nil {
			t.Fatalf("bob ran out of work after %d grades", i)
		}
		w.report(bob, c, 4.0)
	}
	for {
		c := w.claim(anchor, true)
		if c.Job == nil {
			break
		}
		w.report(anchor, c, 5.5)
	}
	if b := w.account("bob"); b.Strikes != muteAfterStrikes || !muted(&b, now()) {
		t.Fatalf("bob after %d disagreements: %+v", muteAfterStrikes, b)
	}
	var out claimed
	if code := bob.do("POST", "/api/v1/jobs/claim", w.hello(false), &out); code != 403 {
		t.Fatalf("a muted account asked for work: HTTP %d", code)
	}
	// It says nothing about his editors.
	w.submit(bob, "mine", "print('bob')")
	w.check()
}

func TestAnEditorTheAnchorCannotRunDoesNotHoldTheRoundOpen(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	id := w.submit(alice, "trouble", "print(1)")
	idx := w.draw()
	for i := 0; i < maxEditAttempts; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("attempt %d: %+v", i, c.Job)
		}
		anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "the harness fell over"}, nil)
	}
	if c := w.claim(anchor, true); c.Job != nil {
		t.Fatalf("a fourth attempt was handed out: %+v", c.Job)
	}
	w.app.tick(context.Background())
	es := w.entries(alice, idx)
	if es[0].Status != "settled" || es[0].Valid || !strings.Contains(es[0].Reason, "could not run") {
		t.Fatalf("the entry: %+v", es[0])
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].Status != "settled" {
			t.Fatal("the round is still open")
		}
		if e := s.Editors[id]; e.Status == "retired" {
			t.Fatal("the editor was retired for something that may be the anchor's fault")
		}
		// The round does not count for the editor: not as one of its qualifying rounds,
		// and not as a gain of nothing in its rating.
		if e := s.ofEditor[id][0]; !e.Void || roundsRun(s, id) != 0 {
			t.Fatalf("a round the anchor could not run counts for the editor: %+v", e)
		}
	})
	for _, r := range append(w.app.board().Rated, w.app.board().Provisional...) {
		if r.ID == id {
			t.Fatalf("the editor has a rating from a round that was never run: %+v", r)
		}
	}
	// It is drawn again, however often that happens: a round in which the anchor ran
	// nothing at all says the anchor was in trouble. Three rounds in a row in which it ran
	// someone else's entry and not this one say it is this entry.
	w.submit(anchor, "fine", "print(2)")
	for n := 0; n < maxEditAttempts; n++ {
		w.app.st.view(func(s *State) {
			if failing(s, id) {
				t.Fatalf("stopped after %d rounds in which the anchor ran others and not this", n)
			}
		})
		w.draw()
		tries := 0
		for {
			c := w.claim(anchor, true)
			if c.Job == nil {
				break
			}
			switch {
			case c.Job.Editor.ID == id:
				tries++
				anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "the harness fell over"}, nil)
			case c.Job.Kind == "edit":
				if code, status := w.edit(anchor, c, "{\"doc\": 1}\n"); code != 200 {
					t.Fatal(status)
				}
			default:
				if code, status := w.report(anchor, c, 5.5); code != 200 {
					t.Fatal(status)
				}
			}
		}
		if tries != maxEditAttempts {
			t.Fatalf("round %d: the editor was tried %d times", n, tries)
		}
		w.app.tick(context.Background())
	}
	w.app.st.view(func(s *State) {
		if !failing(s, id) {
			t.Fatal("an editor the anchors could not run in three rounds, while running others, is still being scheduled")
		}
	})
	w.check()
}

func TestARoundDoesNotWaitForAnAccountThatHasNeverAgreed(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, eve := w.join("anchor-one", priv), w.join("eve", nil)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.draw()
	w.editAll(anchor)
	// A key costs nothing. If taking a job were enough to hold a round back, one would do.
	if c := w.claim(eve, false); c.Job == nil {
		t.Fatal("no job")
	}
	if _, status := w.report(anchor, w.claim(anchor, true), 5.5); status != "settled" {
		t.Fatalf("the round waited for an account nobody has reason to wait for: %s", status)
	}
	if w.settled(idx) != 1 {
		t.Fatal("the round did not settle")
	}
	// Nothing is held against a newcomer who was merely too late.
	var again claimed
	if code := eve.do("POST", "/api/v1/jobs/claim", w.hello(false), &again); code != 200 {
		t.Fatalf("HTTP %d %s", code, again.Error)
	}
	w.check()
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
			if code, status := w.report(anchor, c, loss(c.Job.Editor.Name, idx)); code != 200 {
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
	// ...and the entry it was vouching for leaves play, so one person cannot keep an entry
	// running under every key they make.
	w.app.st.view(func(s *State) {
		for _, e := range s.ownedBy[s.byHandle["alice"].ID] {
			if e.Status == "queued" || e.Status == "running" {
				t.Fatalf("an entry stayed in play after its GitHub identity moved on: %+v", e)
			}
		}
	})
	// The day's submissions are counted for the person, whichever key they use.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for i := 0; i < maxSubmissionsPerDay; i++ {
			tx.saveEditor(&Editor{ID: fmt.Sprintf("%064x", i+1), AccountID: tx.s.byHandle["alice"].ID, Name: "old", Version: i + 1,
				CreatedAt: now() - 60, Status: "withdrawn", GithubID: 4242})
		}
		return nil
	})
	if code := mallory.do("POST", "/api/v1/editors", bundle(t, "again", "print(2)"), &out); code != 429 {
		t.Fatalf("a second key got a fresh day of submissions: HTTP %d %s", code, out.Error)
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

func TestAChartOfAnyNumbersIsDrawnAndEnds(t *testing.T) {
	// One rounding error apart: adding the step to the lower value used to change nothing.
	for _, curve := range [][][]float64{
		{{0, 9}, {1, 3.0}, {2, 3.0000000000000004}},
		{{0, 1e17}, {1, 1e17 + 16}},
		{{0, 5}, {1, 5}},
	} {
		done := make(chan bool)
		go func() { curveChart(curve); done <- true }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("drawing %v did not finish", curve)
		}
	}
	if n := len(ticks(3.0, 3.0000000000000004, 2e-16)); n > 41 {
		t.Fatalf("%d gridlines", n)
	}
	// What a grader sends is kept only if it is a curve.
	bad := &TrainInfo{Seconds: math.Inf(1), Curve: [][]float64{{1, 9}, {2, math.NaN()}}}
	if c := cleanTrain(bad); c.Curve != nil || c.Seconds != 0 {
		t.Fatalf("a curve with a NaN in it was kept: %+v", c)
	}
	if c := cleanTrain(&TrainInfo{Curve: [][]float64{{1, 9, 4}}}); c.Curve != nil {
		t.Fatal("a curve point with three numbers was kept")
	}
	if c := cleanTrain(&TrainInfo{Curve: [][]float64{{1, 9}, {50, 6}}}); len(c.Curve) != 2 {
		t.Fatal("a good curve was dropped")
	}
}

func TestAnEditorReferenceCannotBeTakenOver(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, mallory := w.join("anchor-one", priv), w.join("alice", nil), w.join("mallory", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	w.play(anchor, 1, func(string, int) float64 { return 5.5 })
	stranger := &actor{t: t, srv: w.srv}
	// A later editor whose id starts the same way, public or not, does not answer to the short id.
	clash := id[:12] + strings.Repeat("f", 52)
	w.app.st.update(context.Background(), func(tx *Tx) error {
		tx.saveEditor(&Editor{ID: clash, AccountID: tx.s.byHandle["mallory"].ID, Name: "tidy", Version: 1, CreatedAt: now() + 10, Status: "queued", Public: true})
		return nil
	})
	var out struct {
		Editor struct {
			ID string `json:"id"`
		} `json:"editor"`
	}
	if code := stranger.do("GET", "/api/v1/editors/"+id[:12], nil, &out); code != 200 || out.Editor.ID != id {
		t.Fatalf("the short id now means %s (HTTP %d)", out.Editor.ID, code)
	}
	// A new version that has not run does not hide the public one behind handle/name.
	v2 := w.submit(alice, "tidy", "print('v2')")
	if code := stranger.do("GET", "/api/v1/editors/alice%2Ftidy", nil, &out); code != 200 || out.Editor.ID != id {
		t.Fatalf("alice/tidy for a stranger: HTTP %d %s", code, out.Editor.ID)
	}
	if code := alice.do("GET", "/api/v1/editors/alice%2Ftidy", nil, &out); code != 200 || out.Editor.ID != v2 {
		t.Fatalf("alice/tidy for alice: HTTP %d %s", code, out.Editor.ID)
	}
	if code := stranger.do("GET", "/api/v1/editors/"+v2, nil, nil); code != 404 {
		t.Fatalf("an editor that has not run is visible by its id: HTTP %d", code)
	}
	_ = mallory
}

func TestABundleIsBoundedInEveryDimension(t *testing.T) {
	pack := func(headers []*tar.Header, tail []byte) []byte {
		var tb bytes.Buffer
		tw := tar.NewWriter(&tb)
		for _, h := range headers {
			body := []byte("name = \"tidy\"\nentry = \"python editor.py\"\n")
			h.Size, h.Mode, h.Typeflag = int64(len(body)), 0o644, tar.TypeReg
			tw.WriteHeader(h)
			tw.Write(body)
		}
		tw.Close()
		tb.Write(tail)
		enc, _ := zstd.NewWriter(nil)
		return enc.EncodeAll(tb.Bytes(), nil)
	}
	if _, _, err := parseBundle(pack([]*tar.Header{{Name: "stet.toml"}}, nil)); err != nil {
		t.Fatalf("a plain bundle: %v", err)
	}
	if tr, _, done, err := untar(sparseBundle()); err == nil {
		tr.Next()
		if h, err := tr.Next(); err != nil || h.Name != "big" || h.Size != 1<<40 {
			t.Fatalf("the test's sparse file is not read as one: %+v %v", h, err)
		}
		done()
	}
	for what, blob := range map[string][]byte{
		"a path of a megabyte":  pack([]*tar.Header{{Name: "stet.toml"}, {Name: strings.Repeat("a/", 300) + "x"}}, nil),
		"a sparse file":         sparseBundle(),
		"zeros after the tar":   pack([]*tar.Header{{Name: "stet.toml"}}, make([]byte, maxUnpackedBytes+1)),
		"a hidden file":         pack([]*tar.Header{{Name: "stet.toml"}, {Name: "src/.env"}}, nil),
		"a virtual environment": pack([]*tar.Header{{Name: "stet.toml"}, {Name: "venv/lib/site.py"}}, nil),
		"compiled Python":       pack([]*tar.Header{{Name: "stet.toml"}, {Name: "editor.pyc"}}, nil),
		// What the site shows and what an anchor runs must be one file.
		"the same file twice":         pack([]*tar.Header{{Name: "stet.toml"}, {Name: "editor.py"}, {Name: "editor.py"}}, nil),
		"a file by a roundabout path": pack([]*tar.Header{{Name: "stet.toml"}, {Name: "x/../editor.py"}}, nil),
		"a file as ./file":            pack([]*tar.Header{{Name: "stet.toml"}, {Name: "./editor.py"}}, nil),
		"a book of file names":        pack(manyNames(400, 200), nil),
		"a manifest of a megabyte":    bigManifest(),
	} {
		if _, _, err := parseBundle(blob); err == nil {
			t.Fatalf("a bundle with %s was accepted", what)
		}
	}
}

func manyNames(n, length int) []*tar.Header {
	hs := []*tar.Header{{Name: "stet.toml"}}
	for i := 0; i < n; i++ {
		hs = append(hs, &tar.Header{Name: fmt.Sprintf("%0*d", length, i)})
	}
	return hs
}

func bigManifest() []byte {
	var tb bytes.Buffer
	tw := tar.NewWriter(&tb)
	body := []byte("name = \"tidy\"\nentry = \"python editor.py\"\nx = " + strings.Repeat("[", 1<<20) + "\n")
	tw.WriteHeader(&tar.Header{Name: "stet.toml", Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg})
	tw.Write(body)
	tw.Close()
	enc, _ := zstd.NewWriter(nil)
	return enc.EncodeAll(tb.Bytes(), nil)
}

// sparseBundle is a bundle holding a file that says it is a terabyte of nothing. Go's tar
// writer will not make one, so the blocks are written by hand.
func sparseBundle() []byte {
	block := func(name string, size int, typeflag byte) []byte {
		h := make([]byte, 512)
		copy(h, name)
		copy(h[100:], "0000644\x00")
		copy(h[108:], "0000000\x000000000\x00")
		copy(h[124:], fmt.Sprintf("%011o\x00", size))
		copy(h[136:], "00000000000\x00")
		h[156] = typeflag
		copy(h[257:], "ustar\x0000")
		copy(h[148:], "        ")
		sum := 0
		for _, c := range h {
			sum += int(c)
		}
		copy(h[148:], fmt.Sprintf("%06o\x00 ", sum))
		return h
	}
	pad := func(b []byte) []byte { return append(b, make([]byte, (512-len(b)%512)%512)...) }
	record := func(k, v string) string {
		n := len(k) + len(v) + 3 // the space, the equals sign and the newline
		n += len(fmt.Sprint(n + len(fmt.Sprint(n))))
		return fmt.Sprintf("%d %s=%s\n", n, k, v)
	}
	manifest := []byte("name = \"tidy\"\nentry = \"python editor.py\"\n")
	pax := []byte(record("GNU.sparse.size", "1099511627776") + record("GNU.sparse.numblocks", "1") + record("GNU.sparse.map", "0,0"))
	var tb bytes.Buffer
	tb.Write(block("stet.toml", len(manifest), '0'))
	tb.Write(pad(manifest))
	tb.Write(block("PaxHeaders/big", len(pax), 'x'))
	tb.Write(pad(pax))
	tb.Write(block("big", 0, '0'))
	tb.Write(make([]byte, 1024))
	enc, _ := zstd.NewWriter(nil)
	return enc.EncodeAll(tb.Bytes(), nil)
}

func TestTheClientAddressIsTheOneTheProxySaw(t *testing.T) {
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "10.0.0.7:4000"
	r.Header.Set("X-Forwarded-For", "6.6.6.6, 203.0.113.9")
	if ip := clientIP(r); ip != "203.0.113.9" {
		t.Fatalf("the client chose its own address: %s", ip)
	}
	r.Header.Set("X-Forwarded-For", "2001:db8:1:2:aaaa:bbbb:cccc:dddd")
	if ip := clientIP(r); ip != "2001:db8:1:2::/64" {
		t.Fatalf("an IPv6 subscriber is not counted as its /64: %s", ip)
	}
}

func TestAStrangerCannotMakeTheServerHoldAnUpload(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.join("anchor-one", priv)
	bob := w.join("bob", nil)
	// The body is never read: the refusal comes from the headers alone, so a body that
	// would fail the size limit is refused for who sent it, not for its size.
	huge := bytes.Repeat([]byte("x"), 3<<20)
	for path, want := range map[string]int{"/api/v1/admin/files/exam.zst": 403, "/api/v1/jobs/1/script": 403} {
		method := "POST"
		if strings.Contains(path, "admin") {
			method = "PUT"
		}
		if code := bob.do(method, path, huge, nil); code != want {
			t.Fatalf("%s from a volunteer: HTTP %d, want %d", path, code, want)
		}
	}
	_, key := newKey()
	nobody := &actor{t: t, srv: w.srv, priv: key, pub: hex.EncodeToString(key.Public().(ed25519.PublicKey))}
	if code := nobody.do("POST", "/api/v1/editors", huge, nil); code != 401 {
		t.Fatalf("an upload from a key with no account: HTTP %d", code)
	}
}

func TestNamingSomeoneElsesKeyCostsThemNothing(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.join("anchor-one", priv)
	alice := w.join("alice", nil)
	// A key is public: it sits in a gist. Anyone can send uploads that claim to be hers.
	forged := func(headers map[string]string) int {
		req, _ := http.NewRequest("POST", w.srv.URL+"/api/v1/editors", bytes.NewReader([]byte("x")))
		req.Header.Set("X-Stet-Key", alice.pub)
		req.Header.Set("X-Stet-Time", strconv.FormatInt(time.Now().Unix(), 10))
		req.Header.Set("X-Stet-Sig", strings.Repeat("ab", 64))
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	for i := 0; i < 60; i++ {
		if code := forged(map[string]string{"X-Stet-Body": strings.Repeat("cd", 32)}); code != 401 {
			t.Fatalf("a forged upload: HTTP %d", code)
		}
	}
	// None of it was counted against her, and none of it held a place.
	if len(w.app.uploads) != 0 {
		t.Fatal("a forged upload holds a place")
	}
	w.submit(alice, "tidy", "print('v1')")
	// A client too old to say what it signed is told to upgrade, not that its bundle is wrong.
	if code := forged(nil); code != 426 {
		t.Fatalf("an upload with no body hash: HTTP %d", code)
	}
	// The hash sent must be the hash of what arrives.
	body := bundle(t, "other", "print(2)")
	ts := strconv.FormatInt(time.Now().Unix(), 10)
	lie := sha256.Sum256([]byte("something else"))
	msg := strings.Join([]string{"stet-v1", "POST", "/api/v1/editors", ts, hex.EncodeToString(lie[:])}, "\n")
	req, _ := http.NewRequest("POST", w.srv.URL+"/api/v1/editors", bytes.NewReader(body))
	req.Header.Set("X-Stet-Key", alice.pub)
	req.Header.Set("X-Stet-Time", ts)
	req.Header.Set("X-Stet-Body", hex.EncodeToString(lie[:]))
	req.Header.Set("X-Stet-Sig", hex.EncodeToString(ed25519.Sign(alice.priv, []byte(msg))))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("a body other than the one signed: HTTP %d", resp.StatusCode)
	}
	// The place is given back once the handler is done.
	for i := 0; i < 50 && len(w.app.uploads) != 0; i++ {
		time.Sleep(10 * time.Millisecond)
	}
	if len(w.app.uploads) != 0 {
		t.Fatal("an upload's place was never given back")
	}
}

func TestAResultCarriesTheSeasonsSourcesAndNothingElse(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.submit(anchor, "ref-a", "print(1)")
	w.draw()
	w.editAll(anchor)
	c := w.claim(bob, false)
	res := w.result(c, 5.5, 6.0)
	// Whatever is in a result is stored. A map with fifty thousand keys would be too.
	res["exam"].(map[string]any)["by_source"].(map[string]float64)["padding"] = 5.5
	var out struct {
		Error string `json:"error"`
	}
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), res, &out); code != 400 || !strings.Contains(out.Error, "sources") {
		t.Fatalf("a result with a source the season does not have: HTTP %d %s", code, out.Error)
	}
}

func TestADrawNeverReusesASlotAnnouncedForAnotherRound(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	ctx := context.Background()
	w.submit(anchor, "ref-a", "print(1)")
	w.app.tick(ctx)
	scheduled := func() (r Round) {
		w.app.st.view(func(s *State) { r = *s.roundWithStatus("scheduled") })
		return r
	}
	first := scheduled()
	// An anchor asks for the draw sooner. Verifiers have already been told the old slot was this round's.
	if _, err := w.app.bringForward(ctx); err != nil {
		t.Fatal(err)
	}
	if moved := scheduled(); moved.DrandRound == first.DrandRound || len(moved.Was) != 1 || moved.Was[0] != first.DrandRound {
		t.Fatalf("the round was not moved, or forgot where it had been: %+v", moved)
	}
	idx := w.draw()
	if next := scheduled(); next.Idx == idx || next.DrandRound == first.DrandRound {
		t.Fatalf("the next round was announced for a slot that was announced for another: %+v", next)
	} else if grid := w.app.season.DrawGrid; next.DrawAt != first.DrawAt+grid {
		t.Fatalf("a slot that was taken cost the next round %d s; the grid is %d s", next.DrawAt-first.DrawAt, grid)
	}

	// The open round runs past the next draw. That draw is put off, and then made at the
	// first grid mark after the round settles, not a whole interval later.
	if err := w.app.maybeDraw(ctx, true); err == nil {
		t.Fatal("a round was drawn while another was open")
	}
	if next := scheduled(); !next.Postponed {
		t.Fatalf("the draw was not marked as put off: %+v", next)
	}
	w.editAll(anchor)
	if _, status := w.report(anchor, w.claim(anchor, true), 5.5); status != "settled" {
		t.Fatal(status)
	}
	next := scheduled()
	if grid := w.app.season.DrawGrid; next.Postponed || next.DrawAt > now()+drawNotice+grid || next.DrawAt < now()+drawNotice {
		t.Fatalf("a put-off draw waits %d s after the round settled; the grid is %d s", next.DrawAt-now(), grid)
	}
	w.check()
}

func TestAnEditorThatKeepsFailingLeavesTheLadder(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	id := w.submit(alice, "flaky", "print(1)")
	// Good enough to be ranked, not far enough from the baseline to set a record.
	w.play(anchor, w.app.cfg.QualifyRounds, func(string, int) float64 { return 5.99 })
	w.app.st.view(func(s *State) {
		if s.holder() != nil {
			t.Fatal("this test is about an editor that does not hold the record")
		}
	})
	onLadder := func() bool {
		for _, r := range w.app.board().Ladder {
			if r.ID == id {
				return true
			}
		}
		return false
	}
	if !onLadder() {
		t.Fatal("a rated editor with nobody above it is not on the ladder")
	}
	// On the ladder it runs every round. Now it times out twice.
	for n := 0; n < retireAfterInvalid; n++ {
		w.draw()
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" || c.Job.Editor.ID != id {
			t.Fatalf("the ladder editor was not drawn: %+v", c.Job)
		}
		if code, status := w.edit(anchor, c, ""); code != 200 {
			t.Fatal(status)
		}
	}
	if onLadder() {
		t.Fatal("an editor that failed twice running is still on the ladder, costing an anchor an edit every round")
	}
	w.check()
}

func TestTheRecordHolderKeepsRunningWhateverItDoes(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	id := w.submit(alice, "champion", "print(1)")
	w.play(anchor, w.app.cfg.QualifyRounds, func(string, int) float64 { return 5.0 })
	w.app.st.view(func(s *State) {
		if h := s.holder(); h == nil || h.EditorID != id {
			t.Fatalf("the record: %+v", h)
		}
	})
	// A record changes hands only in rounds the holder and a challenger share. If a holder
	// that began to fail stopped being drawn, nobody could ever take the record from it.
	for n := 0; n <= retireAfterInvalid; n++ {
		w.draw()
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" || c.Job.Editor.ID != id {
			t.Fatalf("round %d: the record holder was not drawn: %+v", n, c.Job)
		}
		if code, status := w.edit(anchor, c, ""); code != 200 {
			t.Fatal(status)
		}
	}
	w.check()
}

func TestWhileTheDatabaseIsDownNothingUnstoredIsShown(t *testing.T) {
	w := newWorld(t)
	kv := w.kv.(*memKV)
	alice := w.join("alice", nil)
	get := func(path string) (int, string) {
		resp, err := http.Get(w.srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(body)
	}
	// Reads go on while nothing has gone wrong: the database being away is not, by itself,
	// a reason to stop showing what was stored.
	kv.down = true
	if code, _ := get("/leaderboard"); code != 200 {
		t.Fatalf("a page that needs no write: HTTP %d", code)
	}
	// A write fails and cannot be read back. What is in memory may now hold it.
	if code := alice.do("POST", "/api/v1/editors", bundle(t, "tidy", "print(1)"), nil); code != 503 {
		t.Fatalf("a submission with the database down: HTTP %d", code)
	}
	if code, body := get("/"); code != 503 || !strings.Contains(body, "Back in a moment") {
		t.Fatalf("the site with records it cannot vouch for: HTTP %d", code)
	}
	if code, body := get("/api/v1/leaderboard"); code != 503 || !strings.Contains(body, "error") {
		t.Fatalf("the API with records it cannot vouch for: HTTP %d %s", code, body)
	}
	if code, _ := get("/healthz"); code != 200 {
		t.Fatalf("the process is alive, and says HTTP %d", code)
	}
	if code, _ := get("/readyz"); code != 503 {
		t.Fatalf("/readyz with the database down: HTTP %d", code)
	}
	if code, _ := get("/static/site.css"); code != 200 {
		t.Fatalf("a static file: HTTP %d", code)
	}
	// The database comes back; the next tick reads it and the site with it.
	kv.down = false
	w.app.tick(context.Background())
	if code, _ := get("/"); code != 200 {
		t.Fatalf("the site after the database came back: HTTP %d", code)
	}
	w.app.st.view(func(s *State) {
		if len(s.Editors) != 0 {
			t.Fatal("the submission that failed exists")
		}
	})
	w.check()
}

func TestAMachineThatBuildsTheRoundDifferentlyIsNotGivenMoreOfIt(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	for _, name := range []string{"ref-a", "ref-b", "ref-c", "ref-d"} {
		w.submit(anchor, name, "print('"+name+"')")
	}
	w.draw()
	w.editAll(anchor)
	// The anchor grades one entry, which fixes what the round's corpus is.
	if code, status := w.report(anchor, w.claim(anchor, true), 5.5); code != 200 {
		t.Fatal(status)
	}
	// Bob's corpus is some other corpus: a stale cache, a cut-short download.
	for n := 0; n < 2; n++ {
		c := w.claim(bob, false)
		if c.Job == nil {
			t.Fatalf("grade %d: no work for a volunteer", n)
		}
		res := w.result(c, 5.5, 6.0)
		res["corpus_sha256"] = strings.Repeat("ab", 32)
		var out struct {
			Error string `json:"error"`
		}
		if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), res, &out); code != 409 {
			t.Fatalf("a grade on another corpus: HTTP %d %s", code, out.Error)
		}
	}
	// A third would cost Bob a training run and end the same way.
	if c := w.claim(bob, false); c.Job != nil {
		t.Fatalf("after two refusals in a round the server still hands out its entries: %+v", c.Job)
	}
	if a := w.account("bob"); a.Strikes != 0 || muted(&a, now()) {
		t.Fatalf("a broken cache was held against the volunteer: %+v", a)
	}
	// Text that would be printed in other people's terminals is plain text.
	var tb bytes.Buffer
	tw := tar.NewWriter(&tb)
	manifest := "name = \"tidy\"\nentry = \"python editor.py\"\ndescription = \"nice \\u001b]0;owned\\u0007 editor\"\n"
	tw.WriteHeader(&tar.Header{Name: "stet.toml", Mode: 0o644, Size: int64(len(manifest)), Typeflag: tar.TypeReg})
	tw.Write([]byte(manifest))
	tw.Close()
	enc, _ := zstd.NewWriter(nil)
	if _, _, err := parseBundle(enc.EncodeAll(tb.Bytes(), nil)); err == nil || !strings.Contains(err.Error(), "ordinary text") {
		t.Fatalf("a description with terminal escapes in it: %v", err)
	}
	w.check()
}
