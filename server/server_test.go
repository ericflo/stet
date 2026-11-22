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
	rt, _ := fitRatings(es, noiseFloor, 0)
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

// fetch downloads a script as this actor: in a signed request if it has a key.
func (a *actor) fetch(sha string) (int, []byte) {
	a.t.Helper()
	path := "/api/v1/scripts/" + sha
	req, _ := http.NewRequest("GET", a.srv.URL+path, nil)
	if a.priv != nil {
		ts := strconv.FormatInt(time.Now().Unix(), 10)
		sum := sha256.Sum256(nil)
		msg := strings.Join([]string{"stet-v1", "GET", path, ts, hex.EncodeToString(sum[:])}, "\n")
		req.Header.Set("X-Stet-Key", a.pub)
		req.Header.Set("X-Stet-Time", ts)
		req.Header.Set("X-Stet-Sig", hex.EncodeToString(ed25519.Sign(a.priv, []byte(msg))))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		a.t.Fatal(err)
	}
	defer resp.Body.Close()
	blob, err := io.ReadAll(resp.Body)
	if err != nil {
		a.t.Fatal(err)
	}
	return resp.StatusCode, blob
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
		RoundInterval: 6 * time.Hour, LadderSize: 10, QualifyRounds: 5, RatedRounds: 3, ReplicaWait: 45 * time.Minute,
		Lease: 3 * time.Hour, FixedCrawl: &Crawl{Crawl: "CC-MAIN-TEST", PathsSHA256: strings.Repeat("ab", 32), NShards: 1000},
		GithubMinAgeDays: 30, Dev: true, EstablishAfter: 5, RunsPerEntry: 2, ActiveWindow: 10 * time.Minute,
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
	return bundleWith(t, name, body, "")
}

// bundleWith is a bundle whose manifest ends with more TOML: its [[assets]] tables.
func bundleWith(t *testing.T, name, body, more string) []byte {
	t.Helper()
	var tb bytes.Buffer
	tw := tar.NewWriter(&tb)
	files := map[string]string{
		"stet.toml": fmt.Sprintf("name = %q\nentry = \"python editor.py\"\ntechnique = \"test\"\n", name) + more,
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

// draw schedules and draws the next round, gives it its exam as the organisers' examiner
// would, and returns its index.
func (w *world) draw() int {
	w.t.Helper()
	idx := w.drawWithoutExam()
	w.exam(idx)
	return idx
}

// drawWithoutExam draws the next round and leaves it waiting for its exam.
func (w *world) drawWithoutExam() int {
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
		Script   *ScriptRef `json:"script"`
		Run      int        `json:"run"`
		Baseline *bool      `json:"baseline"`
	} `json:"job"`
	Runner    map[string]any `json:"runner"`
	Announced map[string]any `json:"announced"`
	Message   string         `json:"message"`
	Error     string         `json:"error"`
}

func (w *world) hello(edits bool) map[string]any {
	return map[string]any{"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"hardware": map[string]any{"gpu": "Test GPU", "device": "cuda"}, "edits": edits, "sandbox": "bwrap"}
}

func (w *world) claim(a *actor, edits bool) *claimed {
	w.t.Helper()
	return w.claimWith(a, w.hello(edits))
}

// claimWith asks for work with a hello of the caller's own making.
func (w *world) claimWith(a *actor, hello map[string]any) *claimed {
	w.t.Helper()
	var c claimed
	if code := a.do("POST", "/api/v1/jobs/claim", hello, &c); code != 200 {
		w.t.Fatalf("claim by %s: HTTP %d %s", a.handle, code, c.Error)
	}
	return &c
}

// claimEarly asks for work as a verifier that trains before the exam (stet 0.4.3 and later).
func (w *world) claimEarly(a *actor) *claimed {
	w.t.Helper()
	h := w.hello(false)
	h["early"] = true
	return w.claimWith(a, h)
}

// corpusSHA is the round every test machine builds, and examSHA the digest of the exam every
// round is given (testExam).
var corpusSHA, examSHA = strings.Repeat("c0", 32), strings.Repeat("d1", 32)

// testTable is a small exam: items in two pillars, a skill and a kind of writing, and two languages.
func testTable() []ItemInfo {
	var out []ItemInfo
	for i := 0; i < 12; i++ {
		out = append(out, ItemInfo{Pillar: "logic", Part: "logic", Lang: "en", Stratum: "prose/answer"})
	}
	for i := 0; i < 6; i++ {
		out = append(out, ItemInfo{Pillar: "reference", Part: "reference", Lang: []string{"en", "de"}[i%2], Stratum: "prose/passage"})
	}
	return out
}

// testExam is what the organisers' examiner posts for a round: its items for graders, what
// is published when it settles, and what each item counts toward.
func testExam(table []ItemInfo) map[string]any {
	enc, _ := zstd.NewWriter(nil)
	file := enc.EncodeAll([]byte("{\"kind\": \"passage\"}\n"), nil)
	public := enc.EncodeAll([]byte("{\"withheld\": true}\n"), nil)
	return map[string]any{
		"items_digest": examSHA, "items_file": base64.StdEncoding.EncodeToString(file), "table": table,
		"public_file": base64.StdEncoding.EncodeToString(public),
		"languages":   map[string]float64{"en": 60, "de": 40}, "fresh": 6,
	}
}

// postExam has an account post a round's exam, as the organisers' examiner does.
func (w *world) postExam(by *actor, idx int, table []ItemInfo) (int, string) {
	w.t.Helper()
	var out struct {
		Error string `json:"error"`
	}
	code := by.do("POST", fmt.Sprintf("/api/v1/admin/rounds/%s/exam", roundID(w.app.season.ID, idx)), testExam(table), &out)
	return code, out.Error
}

// exam gives a round its exam (testTable), as the organisers' examiner would.
func (w *world) exam(idx int) {
	w.t.Helper()
	raw, _ := json.Marshal(testExam(testTable()))
	var p ExamPost
	json.Unmarshal(raw, &p)
	if _, err := w.app.postExam(context.Background(), idx, &p); err != nil {
		w.t.Fatalf("posting round %d's exam: %v", idx, err)
	}
}

// itemScores is a report on the exam with every item scoring the same.
func itemScores(score float64) map[string]any {
	scores := make([]float64, len(testTable()))
	for i := range scores {
		scores[i] = score
	}
	return map[string]any{"mean": score, "item_scores": scores}
}

// edit reports an edit job. An empty script means the editor failed.
func (w *world) edit(a *actor, c *claimed, script string) (int, string) {
	w.t.Helper()
	p := map[string]any{
		"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"round": c.Job.Round["id"], "scale": c.Job.Scale, "corpus_sha256": corpusSHA,
		"edit": map[string]any{"status": "ok", "seconds": 3.0, "time_limit": 60.0, "sandbox": "bwrap"},
	}
	if script == "" {
		p["edit"] = map[string]any{"status": "timeout", "seconds": 60.0, "time_limit": 60.0, "reason": "the curator ran out of time"}
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

// result is a grade whose student scores -loss on every item of the exam, and its baseline
// -baseline: the exam's aggregate of a gain that is the same on every item is that gain, so
// against a round whose baseline is baseline, its gain is baseline - loss.
func (w *world) result(c *claimed, loss, baseline float64) map[string]any {
	return map[string]any{
		"harness": w.app.season.Harness, "season": w.app.season.ID, "season_digest": w.app.season.Digest,
		"round": c.Job.Round["id"], "scale": c.Job.Scale, "corpus_sha256": corpusSHA, "exam_sha256": examSHA,
		"script_sha256": c.Job.Script.SHA256, "valid": true, "hardware": map[string]any{"gpu": "Test GPU", "device": "cuda"},
		"script":   map[string]any{"script_sha256": c.Job.Script.SHA256, "kept_doc_frac": 0.1, "kept_byte_frac": 0.08, "passes": 1.4, "out_docs": 1200},
		"baseline": itemScores(-baseline), "exam": itemScores(-loss),
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
	Editor     string             `json:"editor"`
	Name       string             `json:"name"`
	Status     string             `json:"status"`
	Stage      string             `json:"stage"`
	Valid      bool               `json:"valid"`
	Void       bool               `json:"void"`
	Loss       *float64           `json:"loss"`
	Gain       *float64           `json:"gain"`
	CountsAs   *float64           `json:"counts_as"`
	BySource   map[string]float64 `json:"by_source"`
	Results    int                `json:"results"`
	Reason     string             `json:"reason"`
	Script     map[string]any     `json:"script"`
	Edit       map[string]any     `json:"edit"`
	Detail     *EntryDetail       `json:"detail"`
	Runs       []RunView          `json:"runs"`
	Dropped    []RunView          `json:"dropped_runs"`
	NRuns      int                `json:"n_runs"`
	Confirmed  bool               `json:"confirmed"`
	OneGrader  bool               `json:"one_grader"`
	OneRunner  bool               `json:"one_runner"`
	RunsDiffer bool               `json:"runs_differ"`
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

// entry is one entry of a round as the API shows it, by its editor's name.
func (w *world) entry(idx int, name string) entryJ {
	w.t.Helper()
	for _, e := range w.entries(&actor{t: w.t, srv: w.srv}, idx) {
		if e.Name == name {
			return e
		}
	}
	w.t.Fatalf("round %d has no entry called %s", idx, name)
	return entryJ{}
}

// page fetches a page of the website.
func (w *world) page(path string) (int, string) {
	w.t.Helper()
	req, _ := http.NewRequest("GET", w.srv.URL+path, nil)
	req.Host = "site.test"
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		w.t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body)
}

// age makes everything that has happened in the open round, and every verifier's last
// word, that much older: the same as that much time going by with nobody doing anything.
func (w *world) age(d time.Duration) {
	n := int64(d.Seconds())
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, r := range tx.s.Rounds {
			if r.Status == "open" {
				r.DrawnAt, r.ActiveAt = r.DrawnAt-n, r.ActiveAt-n
				tx.saveRound(r)
			}
		}
		for _, c := range tx.s.Accounts {
			if c.LastSeen > 0 {
				c.LastSeen -= n
			}
			if c.RunnerSeen > 0 {
				c.RunnerSeen -= n
			}
			tx.saveAccount(c)
		}
		for _, c := range tx.s.Claims {
			c.ClaimedAt, c.ExpiresAt = c.ClaimedAt-n, c.ExpiresAt-n
			tx.saveClaim(c)
		}
		for _, r := range tx.s.Results {
			r.ReceivedAt -= n
			tx.saveResult(r)
		}
		return nil
	})
}

// pass lets that much time go by, and the server's clock tick at the end of it.
func (w *world) pass(d time.Duration) {
	w.age(d)
	w.app.tick(context.Background())
}

// run has a verifier take an edit job and report it. An empty script means the editor failed.
func (w *world) run(a *actor, script string) *claimed {
	w.t.Helper()
	c := w.claim(a, true)
	if c.Job == nil || c.Job.Kind != "edit" {
		w.t.Fatalf("%s should be running a curator: %+v", a.handle, c.Job)
	}
	if code, status := w.edit(a, c, script); code != 200 {
		w.t.Fatalf("%s's run: HTTP %d %s", a.handle, code, status)
	}
	return c
}

// grade has a verifier take a grade job and report it, and returns the job and what the
// server said. edits says whether the verifier is also asking for editors to run.
func (w *world) grade(a *actor, edits bool, loss float64) (*claimed, string) {
	w.t.Helper()
	c := w.claim(a, edits)
	if c.Job == nil || c.Job.Kind != "grade" {
		w.t.Fatalf("%s should be grading: %+v", a.handle, c.Job)
	}
	code, status := w.report(a, c, loss)
	if code != 200 {
		w.t.Fatalf("%s's grade: HTTP %d %s", a.handle, code, status)
	}
	return c, status
}

// idle fails the test if a verifier that asks for work is handed any.
func (w *world) idle(a *actor, edits bool, why string) {
	w.t.Helper()
	if c := w.claim(a, edits); c.Job != nil {
		w.t.Fatalf("%s: %s was handed %s job on %s", why, a.handle, article(c.Job.Kind), c.Job.Editor.Name)
	}
}

// as is somebody of another world in this one: the same key, talking to this server. (A
// world started on another's storage is the same server after a restart.)
func (w *world) as(a *actor) *actor {
	c := *a
	c.srv = w.srv
	return &c
}

func (w *world) release(a *actor, c *claimed) {
	w.t.Helper()
	if code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "test"}, nil); code != 200 {
		w.t.Fatalf("%s handing job %d back: HTTP %d", a.handle, c.Job.ID, code)
	}
}

func TestAnEntryIsRunThenGradedAndSettles(t *testing.T) {
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
		t.Fatalf("an undrawn curator is visible: HTTP %d", code)
	}
	if code := stranger.do("GET", "/api/v1/editors/"+id+"/bundle", nil, nil); code != 404 {
		t.Fatalf("an undrawn curator can be downloaded: HTTP %d", code)
	}
	if code := alice.do("GET", "/api/v1/editors/"+id, nil, nil); code != 200 {
		t.Fatalf("its owner cannot see it: HTTP %d", code)
	}

	idx := w.draw()
	w.check()
	if code := stranger.do("GET", "/api/v1/editors/"+id+"/bundle", nil, nil); code != 200 {
		t.Fatalf("a drawn curator cannot be downloaded: HTTP %d", code)
	}

	// Nobody but an established verifier is ever handed an entrant's code to run. One who
	// asks and is not is told why, and is given grading when there is some.
	nb := w.claim(bob, true)
	if nb.Job != nil {
		t.Fatalf("a volunteer was given %s job before any script existed", nb.Job.Kind)
	}
	if why, _ := nb.Runner["reason"].(string); nb.Runner["eligible"] != false || !strings.Contains(why, "5 grades that agreed") {
		t.Fatalf("a volunteer who asked for curators was not told why it gets none: %+v", nb.Runner)
	}
	c := w.claim(anchor, true)
	if c.Job == nil || c.Job.Kind != "edit" || c.Job.Script != nil || c.Job.Editor.ID != id || c.Runner["eligible"] != true {
		t.Fatalf("the organisers' machine should be running the curator: %+v %+v", c.Job, c.Runner)
	}
	if c.Announced == nil {
		t.Fatal("a claim does not say what the next draw is committed to")
	}
	if again := w.claim(anchor, true); again.Job == nil || again.Job.ID != c.Job.ID {
		t.Fatal("a restarted verifier does not get its unfinished job back")
	}
	w.check()
	if code, msg := w.edit(bob, c, "{}\n"); code != 403 {
		t.Fatalf("a volunteer with no standing reported an edit: HTTP %d %s", code, msg)
	}
	script := `{"doc": 3}` + "\n" + `{"doc": 1, "cut": [[0, 40]]}` + "\n"
	if code, status := w.edit(anchor, c, script); code != 200 || status != "edited" {
		t.Fatalf("edit: HTTP %d %s", code, status)
	}
	w.check()
	if code, _ := w.edit(anchor, c, script); code != 409 {
		t.Fatalf("an edit was reported twice: HTTP %d", code)
	}

	// The run is sealed: the entry says it is being graded, and nothing about by whom, how
	// long it took or what it wrote.
	es := w.entries(stranger, idx)
	if len(es) != 1 || es[0].Stage != "grading" || es[0].Loss != nil || es[0].Script != nil || es[0].Edit != nil || len(es[0].Runs) != 0 {
		t.Fatalf("after the edit: %+v", es)
	}
	// The script is sealed too: its name gets nobody its bytes, with a key or without.
	sha := shaOf(script)
	if code, _ := stranger.fetch(sha); code != 403 {
		t.Fatalf("a sealed script was served to somebody with no key: HTTP %d", code)
	}
	if code, _ := bob.fetch(sha); code != 403 {
		t.Fatalf("a sealed script was served to an account that does not hold it: HTTP %d", code)
	}

	// A volunteer's grade is sealed, and says nothing to anyone.
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" || cb.Job.Script == nil || cb.Job.Script.SHA256 != sha || cb.Job.Run != 1 || cb.Runner != nil {
		t.Fatalf("bob should be grading that script: %+v", cb.Job)
	}
	// The script is stored exactly as uploaded, for whoever is handed it to grade.
	code, blob := bob.fetch(sha)
	if code != 200 {
		t.Fatalf("fetching the script: HTTP %d", code)
	}
	if raw, err := unzstd(blob, 1<<20); err != nil || string(raw) != script {
		t.Fatalf("the stored script is not the one uploaded: %q %v", raw, err)
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
	if es := w.entries(stranger, idx); es[0].Status != "open" || es[0].Loss != nil || es[0].Gain != nil || es[0].Detail != nil || len(es[0].Runs) != 0 {
		t.Fatalf("an unsettled entry leaks its grade: %+v", es[0])
	}
	if code, _ := w.report(bob, cb, 5.40); code != 409 {
		t.Fatalf("a grade was reported twice: HTTP %d", code)
	}
	if b := w.account("bob"); b.Agreed != 0 || b.GPUSeconds != 0 {
		t.Fatalf("a sealed grade has already earned something: %+v", b)
	}

	// Nobody established is there but the machine that ran the editor, so it grades its own
	// script. That is the score, and the score says so. Bob's agreement counts for him.
	ca := w.claim(anchor, true)
	if ca.Job == nil || ca.Job.Kind != "grade" || ca.Job.Script.SHA256 != sha {
		t.Fatalf("the organisers' machine should be grading: %+v", ca.Job)
	}
	wrong := w.result(ca, 5.52, 6.0)
	wrong["script_sha256"] = strings.Repeat("9", 64)
	if code := anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", ca.Job.ID), wrong, nil); code != 400 {
		t.Fatalf("a grade of some other script was accepted: HTTP %d", code)
	}
	if code, status := w.report(anchor, ca, 5.52); code != 200 || status != "settled" {
		t.Fatalf("the deciding grade: HTTP %d %s", code, status)
	}
	w.check()
	es = w.entries(stranger, idx)
	if es[0].Status != "settled" || !es[0].Valid || es[0].Loss == nil || *es[0].Loss != 5.52 || math.Abs(*es[0].Gain-0.48) > 1e-9 {
		t.Fatalf("settled entry: %+v", es[0])
	}
	if es[0].Detail == nil || !es[0].Detail.Audited || len(es[0].Detail.Verifiers) != 2 || es[0].Detail.Verifiers[0] != "anchor-one" || es[0].Detail.Train == nil {
		t.Fatalf("settled entry's detail: %+v", es[0].Detail)
	}
	if es[0].Script == nil || es[0].Script["sha256"] != sha || es[0].Edit == nil {
		t.Fatalf("a settled entry does not name its script: %+v", es[0])
	}
	// And the script is anybody's to read and grade again.
	if code, again := stranger.fetch(sha); code != 200 || !bytes.Equal(again, blob) {
		t.Fatalf("a settled round's script, asked for with no key: HTTP %d", code)
	}
	if code, _ := stranger.fetch(strings.Repeat("9", 64)); code != 404 {
		t.Fatalf("a script nobody wrote: HTTP %d", code)
	}
	// One run, graded by its own runner: a score, and not a confirmed one.
	if es[0].NRuns != 1 || es[0].Confirmed || es[0].RunsDiffer || len(es[0].Runs) != 1 || len(es[0].Dropped) != 0 {
		t.Fatalf("settled entry's runs: %+v", es[0])
	}
	run := es[0].Runs[0]
	if run.N != 1 || run.By != "anchor-one" || run.Status != "ok" || run.Script != sha || run.ScriptSize != int64(len(script)) ||
		math.Abs(run.Gain-0.48) > 1e-9 || run.Loss == nil || *run.Loss != 5.52 || !run.Self || run.GPU != "Test GPU" || run.EditSeconds != 3 || len(run.Grades) != 2 {
		t.Fatalf("the run as published: %+v", run)
	}
	if g := run.Grades[0]; g.Handle != "anchor-one" || g.Verdict != "agreed" || !g.Counted || !g.Self || !g.Established || g.First || g.Loss == nil || *g.Loss != 5.52 || math.Abs(*g.Gain-0.48) > 1e-9 {
		t.Fatalf("the deciding grade as published: %+v", g)
	}
	if g := run.Grades[1]; g.Handle != "bob" || g.Verdict != "agreed" || g.Counted || g.Self || g.Established || !g.First || *g.Loss != 5.5 || math.Abs(*g.Gain-0.5) > 1e-9 {
		t.Fatalf("bob's grade as published: %+v", g)
	}
	if b := w.account("bob"); b.Agreed != 1 || b.Streak != 1 || b.Strikes != 0 || b.GPUSeconds != 60 {
		t.Fatalf("bob after agreeing once: %+v", b)
	}
	if a := w.account("anchor-one"); a.Runs != 1 || a.Agreed != 0 || a.Streak != 0 {
		t.Fatalf("the organisers' account after a run and a grade: %+v", a)
	}
	w.app.st.view(func(s *State) {
		if r := s.Rounds[idx]; r.Status != "settled" || r.BaselineLoss == nil || *r.BaselineLoss != 6.0 || r.CorpusSHA != corpusSHA {
			t.Fatalf("the round did not settle: %+v", r)
		}
		for _, r := range s.Results {
			// The grade that stood alone is the gain; the other agreed with it.
			want := map[bool]string{true: "counted", false: "agreed"}[r.AccountID == s.byHandle["anchor-one"].ID]
			if r.Summary != nil || r.Verdict != want || r.Script != sha {
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
		t.Fatal("the organisers' account is no longer one after a restart")
	}
	if es2 := again.entries(&actor{t: t, srv: again.srv}, idx); len(es2) != 1 || len(es2[0].Runs) != 1 || len(es2[0].Runs[0].Grades) != 2 {
		t.Fatalf("the runs after a restart: %+v", es2)
	}
}

func TestADisagreeingGradeIsAStrikeAndCannotMoveTheScore(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	if _, status := w.grade(bob, false, 5.0); status != "pending" {
		t.Fatalf("a grade by an account that is not established: %s", status)
	}
	if _, status := w.grade(anchor, true, 5.5); status != "settled" {
		t.Fatalf("the established grade: %s", status)
	}
	w.check()
	es := w.entries(bob, idx)
	if math.Abs(*es[0].Loss-5.5) > 1e-9 || es[0].Results != 1 {
		t.Fatalf("the outlier moved the score: %+v", es[0])
	}
	if b := w.account("bob"); b.Strikes != 1 || b.Agreed != 0 || b.GPUSeconds != 0 || b.LastStrikeAt == 0 || muted(&b, now()) {
		t.Fatalf("bob after an outlier: %+v", b)
	}
	// Every grade is on show beside the script's gain: this one differed, and it came from
	// an account nobody has reason to believe yet. Both are said.
	d := es[0].Detail
	if d.Disputed != 1 || d.DisputedTogether || len(d.Verifiers) != 1 || len(es[0].Runs) != 1 || len(es[0].Runs[0].Grades) != 2 {
		t.Fatalf("the entry's detail: %+v %+v", d, es[0].Runs)
	}
	if g := es[0].Runs[0].Grades[1]; g.Handle != "bob" || g.Verdict != "outlier" || !g.Valid || g.Loss == nil || *g.Loss != 5.0 ||
		g.Gain == nil || math.Abs(*g.Gain-1.0) > 1e-9 || !g.First || g.Counted || g.Established {
		t.Fatalf("bob's grade as published: %+v", g)
	}
	// And on the round's page, for people.
	_, page := w.page("/rounds/" + roundID(w.app.season.ID, idx))
	for _, want := range []string{`id="runs"`, `id="grades"`, `<a href="/u/bob">bob</a>`, `5.000`, `&#43;1.000`, `outlier`, `first grade`, `1 differed`, `>run once<`, `>graded by its own runner<`} {
		if !strings.Contains(page, want) {
			t.Fatalf("the round's page does not show %q", want)
		}
	}
}

// established gives volunteers the record of an established verifier: enough grades that
// agreed, some time ago, for their grades to count and for them to be handed editors.
func (w *world) established(handles ...string) {
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, h := range handles {
			c := tx.s.byHandle[h]
			c.Agreed += 2 * w.cfg.EstablishAfter
			tx.saveAccount(c)
		}
		return nil
	})
}

// standing gives volunteers one grade that agreed, some time ago: no longer newcomers, and
// not yet established.
func (w *world) standing(handles ...string) {
	w.app.st.update(context.Background(), func(tx *Tx) error {
		for _, h := range handles {
			c := tx.s.byHandle[h]
			c.Agreed++
			tx.saveAccount(c)
		}
		return nil
	})
}

func TestAnIndependentGradeReplacesTheRunnersOwn(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, zed, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("zed", nil), w.join("bob", nil)
	w.submit(alice, "one", "print(1)")
	w.submit(zed, "two", "print(2)")
	w.established("bob")
	idx := w.draw()
	// The organisers' machine is alone. It runs both entries, and grades the first of its
	// own scripts, generously.
	for i := 0; i < 2; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" {
			t.Fatalf("run %d: %+v", i, c.Job)
		}
		w.edit(anchor, c, fmt.Sprintf("{\"doc\": 1, \"by\": %q}\n", c.Job.Editor.Name))
	}
	own, status := w.grade(anchor, true, 5.0)
	if status != "pending" {
		t.Fatalf("a round with an ungraded script settled: %s", status)
	}
	// Before the round settles an established volunteer arrives. Both scripts want a grade
	// from somebody who did not write them: the one with none first.
	first, status := w.grade(bob, false, 5.5)
	if status != "pending" || first.Job.Editor.Name == own.Job.Editor.Name {
		t.Fatalf("bob's first grade: %s of %s", status, first.Job.Editor.Name)
	}
	// With him around, the machine that wrote the scripts is not asked to grade another.
	w.idle(anchor, true, "somebody else is there to grade it")
	second, status := w.grade(bob, false, 5.5)
	if status != "settled" || second.Job.Editor.Name != own.Job.Editor.Name {
		t.Fatalf("bob's second grade: %s of %s", status, second.Job.Editor.Name)
	}
	w.check()
	// His grade is the gain. The runner's own counts for nothing now, and is seen to differ.
	e := w.entry(idx, own.Job.Editor.Name)
	if !e.Valid || math.Abs(*e.Gain-0.5) > 1e-9 || e.NRuns != 1 || e.Confirmed || e.Detail.Disputed != 1 || e.Runs[0].Self || len(e.Runs[0].Grades) != 2 {
		t.Fatalf("the entry its runner graded first: %+v %+v", e, e.Runs)
	}
	if g := e.Runs[0].Grades[0]; g.Handle != "bob" || !g.Counted || g.Self || g.Verdict != "agreed" {
		t.Fatalf("the grade that counts: %+v", g)
	}
	// One volunteer's grade against the runner's own: they differ, and nothing says which
	// of the two is wrong.
	if g := e.Runs[0].Grades[1]; g.Handle != "anchor-one" || g.Counted || !g.Self || g.Verdict != "unresolved" || math.Abs(*g.Gain-1.0) > 1e-9 {
		t.Fatalf("the runner's own grade: %+v", g)
	}
	if o := w.entry(idx, first.Job.Editor.Name); math.Abs(*o.Gain-0.5) > 1e-9 || o.Runs[0].Self || len(o.Runs[0].Grades) != 1 {
		t.Fatalf("the entry only he graded: %+v %+v", o, o.Runs)
	}
	// The organisers' accounts are not struck, and earn nothing.
	if a := w.account("anchor-one"); a.Strikes != 0 || a.Agreed != 0 {
		t.Fatalf("the organisers' account: %+v", a)
	}
	// A grade that stands alone has nothing to agree with, so it earns no standing either.
	if b := w.account("bob"); b.Agreed != 2*w.cfg.EstablishAfter || b.Strikes != 0 {
		t.Fatalf("bob: %+v", b)
	}
}

func TestAgreeingAboutTheUneditedCorpusEarnsNoStanding(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "identity", "print('everything')")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	// The unedited corpus's loss is the round's baseline (6.0 here), which every grader
	// reports with every grade and which hardly moves from round to round. A volunteer who
	// "agrees" about it may not have trained anything.
	w.grade(bob, false, 6.0)
	if _, status := w.grade(anchor, true, 6.0); status != "settled" {
		t.Fatal(status)
	}
	w.check()
	e := w.entries(alice, idx)[0]
	if g := e.Runs[0].Grades; len(g) != 2 || g[1].Handle != "bob" || g[1].Verdict != "agreed" || len(e.Detail.Verifiers) != 2 {
		t.Fatalf("the entry's grades: %+v %+v", g, e.Detail)
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

func TestAScriptNoEstablishedVerifierCanGradeIsGivenUp(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	// A volunteer's machine manages what the established one will not.
	if _, status := w.grade(bob, false, 5.2); status != "pending" {
		t.Fatal(status)
	}
	// The script is one its harness falls over on, every time.
	for i := 0; i < maxGradeAttempts; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "grade" {
			t.Fatalf("attempt %d: %+v", i, c.Job)
		}
		w.release(anchor, c)
	}
	w.idle(anchor, true, "a sixth attempt")
	w.app.tick(context.Background())
	es := w.entries(alice, idx)
	if es[0].Status != "settled" || es[0].Valid || !es[0].Void || !strings.Contains(es[0].Reason, "could not grade") {
		t.Fatalf("the entry: %+v", es[0])
	}
	w.app.st.view(func(s *State) {
		if s.Rounds[idx].Status != "settled" || !s.ofEditor[id][0].Void || s.ofEditor[id][0].Unattended {
			t.Fatalf("the round is held open by an entry nobody can grade: %+v", s.Rounds[idx])
		}
		// The volunteer's grade is not left pending for ever.
		for _, r := range s.results(idx, id) {
			if r.Verdict != "void" || r.Summary != nil {
				t.Fatalf("a grade of an entry that was given up: %+v", r)
			}
		}
	})
	// It is published, with nothing to compare it with, and counts for and against nobody.
	if len(es[0].Runs) != 0 || len(es[0].Dropped) != 1 || es[0].NRuns != 0 {
		t.Fatalf("the run nobody could grade: %+v %+v", es[0].Runs, es[0].Dropped)
	}
	d := es[0].Dropped[0]
	if d.Status != "dropped" || d.By != "anchor-one" || !strings.Contains(d.Reason, "could not grade") || len(d.Grades) != 1 ||
		d.Grades[0].Handle != "bob" || d.Grades[0].Verdict != "no established grade" || d.Grades[0].Loss == nil {
		t.Fatalf("the volunteer's grade was not published: %+v", d)
	}
	if b := w.account("bob"); b.Agreed != 0 || b.Strikes != 0 {
		t.Fatalf("bob: %+v", b)
	}
	if code, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); code != 200 || !strings.Contains(page, "no established grade") || strings.Contains(page, "set aside: no established grade") {
		t.Fatalf("the round's page: HTTP %d", code)
	}
	w.check()
}

func TestAScriptNobodyCanGradeIsDroppedAndTheOtherRunStands(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob")
	idx := w.draw()
	one, two := "{\"doc\": 1}\n", "{\"doc\": 2}\n"
	ca, cb := w.claim(anchor, true), w.claim(bob, true)
	w.edit(anchor, ca, one)
	w.edit(bob, cb, two)
	if g, status := w.grade(bob, true, 5.5); status != "pending" || g.Job.Script.SHA256 != shaOf(one) {
		t.Fatalf("bob's grade of the other run's script: %s %+v", status, g.Job)
	}
	// The only machine that may grade bob's script cannot.
	for i := 0; i < maxGradeAttempts; i++ {
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "grade" || c.Job.Script.SHA256 != shaOf(two) || c.Job.Run != 2 {
			t.Fatalf("attempt %d: %+v", i, c.Job)
		}
		w.release(anchor, c)
	}
	w.idle(anchor, true, "a sixth attempt")
	w.app.tick(context.Background())
	e := w.entry(idx, "tidy")
	if e.Status != "settled" || !e.Valid || e.Void || e.NRuns != 1 || e.Confirmed || e.RunsDiffer || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("the entry: %+v", e)
	}
	if len(e.Runs) != 1 || e.Runs[0].By != "anchor-one" || len(e.Dropped) != 1 || e.Dropped[0].By != "bob" || e.Dropped[0].N != 2 || !strings.Contains(e.Dropped[0].Reason, "could not grade") {
		t.Fatalf("its runs: %+v, dropped %+v", e.Runs, e.Dropped)
	}
	// A run is credited when it counts. One that was set aside is not held against its
	// runner, and is not to their credit either: nobody could check it.
	if b := w.account("bob"); b.Runs != 0 || b.Strikes != 0 {
		t.Fatalf("bob: %+v", b)
	}
	if a := w.account("anchor-one"); a.Runs != 1 {
		t.Fatalf("the run that stood was not credited: %+v", a)
	}
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); !strings.Contains(page, "set aside: the verifiers could not grade") {
		t.Fatal("the round's page does not show the run that was set aside")
	}
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
	before, _ := fitRatings(rows, noiseFloor, 0)
	// One more editor times out in round 2, and counts as the round's lowest gain. What it
	// counts as is its own business.
	low := gains["a"] + hard[2]
	after, diff := fitRatings(append(rows, entryRow{RoundIdx: 2, EditorID: "z", Gain: low, Valid: false}), noiseFloor, 0)
	for id := range gains {
		if math.Abs(before[id].Adjusted-after[id].Adjusted) > 1e-9 {
			t.Fatalf("%s moved from %.5f to %.5f because someone else's entry was invalid", id, before[id].Adjusted, after[id].Adjusted)
		}
	}
	// Its own rating starts from that gain, less the round's difficulty like any other.
	if z := after["z"]; math.Abs(z.Adjusted-(low-diff[2])) > 1e-9 || z.Invalid != 1 {
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
		// Nothing in the round was graded, so it has no baseline and counts in nobody's
		// rating: the failure counts as nothing there, and is not said to.
		if es[0].CountsAs != nil {
			t.Fatalf("a failed entry in a round that counts for nobody is said to count as %v", *es[0].CountsAs)
		}
		if c := w.claim(anchor, true); c.Job != nil {
			t.Fatalf("there is nothing to grade, but the anchor got %+v", c.Job)
		}
	}
	w.app.st.view(func(s *State) {
		if e := s.Editors[id]; e.Status != "retired" {
			t.Fatalf("a curator that failed %d times is still %s", retireAfterInvalid, e.Status)
		}
		if r := s.roundWithStatus("open"); r != nil {
			t.Fatal("a round whose only entry failed is still open")
		}
	})
}

// /files/ serves what is put in the files directory (the CLI wheel) and nothing else: nothing
// from the bucket, nothing outside the directory, and nothing anybody can upload, organisers
// included.
func TestFilesAreServedFromTheFilesDirectoryAlone(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	wheel := []byte("a wheel, put there by whoever deployed the server")
	if err := os.WriteFile(filepath.Join(w.cfg.FilesDir, "stet-9.9.9-py3-none-any.whl"), wheel, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := w.blobs.Put(context.Background(), "files/kept.zst", []byte("in the bucket"), "application/octet-stream"); err != nil {
		t.Fatal(err)
	}
	for path, want := range map[string]int{"/files/stet-9.9.9-py3-none-any.whl": 200, "/files/kept.zst": 404, "/files/..%2Fmeta": 404, "/files/nothing.zst": 404} {
		resp, err := http.Get(w.srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != want || (want == 200 && !bytes.Equal(got, wheel)) {
			t.Fatalf("%s: HTTP %d, %d bytes", path, resp.StatusCode, len(got))
		}
	}
	var out struct {
		Error string `json:"error"`
	}
	if code := anchor.do("PUT", "/api/v1/admin/files/kept.zst", []byte("x"), &out); code != 404 {
		t.Fatalf("an organiser uploading a file: HTTP %d %s", code, out.Error)
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
		t.Fatal("the curator is not on the board")
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
	w.standing("bob") // the round waits only for a volunteer who has finished a job before
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
			t.Fatalf("a grade with a volunteer still at work: HTTP %d %s", code, status)
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
		t.Fatal("an open round publishes its baseline, which is half of every grade")
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
	w.standing("mallory")
	// Mallory takes the job and trains nothing.
	cm := w.claim(mallory, false)
	if cm.Job == nil {
		t.Fatal("no job")
	}
	if _, status := w.report(anchor, w.claim(anchor, true), 5.5); status != "pending" {
		t.Fatalf("the grade should wait for the volunteer: %s", status)
	}
	// The wait runs out. The round settles without her.
	w.pass(w.cfg.ReplicaWait + time.Minute)
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

func TestOnlyEstablishedGradesSetAScore(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, m1, m2 := w.join("anchor-one", priv), w.join("mallory", nil), w.join("mallet", nil)
	// Mallory has had grades agree before, one short of enough to be established.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		acct := tx.s.byHandle["mallory"]
		acct.Agreed = w.cfg.EstablishAfter - 1
		tx.saveAccount(acct)
		return nil
	})
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.draw()
	w.editAll(anchor)
	// Two accounts report the same invented number, 0.3 bits per byte better than the truth.
	for _, m := range []*actor{m1, m2} {
		if _, status := w.grade(m, true, 5.2); status != "pending" {
			t.Fatalf("%s: %s", m.handle, status)
		}
	}
	if w.settled(idx) != 0 {
		t.Fatal("two accounts that are not established settled an entry between them")
	}
	if _, status := w.grade(anchor, true, 5.5); status != "settled" {
		t.Fatal(status)
	}
	es := w.entries(m1, idx)
	if *es[0].Loss != 5.5 {
		t.Fatalf("they moved the score: %v", *es[0].Loss)
	}
	// Both are on record as differing, with their numbers, and both are struck: that two
	// keys say the same thing is no evidence, since keys are free.
	if d := es[0].Detail; d.Disputed != 2 || d.DisputedTogether || len(d.Verifiers) != 1 || len(es[0].Runs[0].Grades) != 3 {
		t.Fatalf("the entry's detail: %+v", d)
	}
	for _, h := range []string{"mallory", "mallet"} {
		if m := w.account(h); m.Strikes != 1 || m.Streak != 0 {
			t.Fatalf("%s after disagreeing with the established grade: %+v", h, m)
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

func TestAnEditorTheVerifiersCannotRunDoesNotHoldTheRoundOpen(t *testing.T) {
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
			t.Fatal("the curator was retired for something that may be the verifiers' fault")
		}
		// The round does not count for the editor: not as one of its qualifying rounds,
		// and not as a gain of nothing in its rating.
		if e := s.ofEditor[id][0]; !e.Void || roundsRun(s, id) != 0 {
			t.Fatalf("a round the anchor could not run counts for the curator: %+v", e)
		}
	})
	for _, r := range append(w.app.board().Rated, w.app.board().Provisional...) {
		if r.ID == id {
			t.Fatalf("the curator has a rating from a round that was never run: %+v", r)
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
			t.Fatalf("round %d: the curator was tried %d times", n, tries)
		}
		w.app.tick(context.Background())
	}
	w.app.st.view(func(s *State) {
		if !failing(s, id) {
			t.Fatal("a curator the verifiers could not run in three rounds, while running others, is still being scheduled")
		}
		// Whether the verifiers ran the others is known only when the round has settled, and
		// that is when the editor stops: it is not drawn a fourth time.
		if e := s.Editors[id]; e.Status != "retired" {
			t.Fatalf("the curator is still %s", e.Status)
		}
	})
	next := w.draw()
	w.app.st.view(func(s *State) {
		for _, e := range s.inRound[next] {
			if e.EditorID == id {
				t.Fatal("it was drawn a fourth time")
			}
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

// play plays whole rounds with one machine, the organisers', doing all the work. loss gives each editor's
// loss in a round (against an unedited 6.0: its gain is 6.0 less that), by editor name.
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
	// How often a person submits in a day is not counted: what they have in play at once is.
	// After a dozen submissions, each under a name of its own, three are waiting.
	for i := 0; i < 12; i++ {
		if code := mallory.do("POST", "/api/v1/editors", bundle(t, fmt.Sprintf("again%d", i), fmt.Sprintf("print(%d)", i+2)), &out); code != 201 {
			t.Fatalf("submission %d of a day was refused: HTTP %d %s", i+1, code, out.Error)
		}
	}
	w.app.st.view(func(s *State) {
		n := 0
		for _, e := range s.Editors {
			if e.GithubID == 4242 && (e.Status == "queued" || e.Status == "running") {
				n++
			}
		}
		if n != w.app.cfg.EntriesInPlay {
			t.Fatalf("after a dozen submissions one person has %d entries in play, want %d", n, w.app.cfg.EntriesInPlay)
		}
	})
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
	code, body, _ := get("api.test", "/api/v1/season")
	if code != 200 {
		t.Fatalf("the API host: HTTP %d", code)
	}
	// The season says what an editor is promised: the rules point here for the numbers.
	var promised struct {
		Machine     Machine     `json:"machine"`
		Runner      RunnerFloor `json:"runner"`
		Assets      AssetLimits `json:"assets"`
		Environment struct {
			Python   string            `json:"python"`
			Packages map[string]string `json:"packages"`
		} `json:"environment"`
	}
	if err := json.Unmarshal([]byte(body), &promised); err != nil {
		t.Fatal(err)
	}
	if promised.Machine.Cores != 6 || promised.Runner.GPUTokensPerSecond <= 0 || promised.Assets.MaxFiles != 32 || promised.Environment.Python != "3.13" ||
		promised.Environment.Packages["torch"] == "" || promised.Environment.Packages["nodejs-wheel"] == "" {
		t.Fatalf("the season does not say what a curator is promised: %+v", promised)
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
		t.Fatalf("the second server holds %d curators before it has been told of the third", n)
	}
	other.app.st.sync(context.Background())
	if n := editors(); n != 3 {
		t.Fatalf("the second server holds %d curators after asking the database", n)
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
		t.Fatalf("a curator that has not run is visible by its id: HTTP %d", code)
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
	if _, _, err := parseBundle(pack([]*tar.Header{{Name: "stet.toml"}}, nil), testAssetLimits); err != nil {
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
		// What the site shows and what a verifier runs must be one file.
		"the same file twice":         pack([]*tar.Header{{Name: "stet.toml"}, {Name: "editor.py"}, {Name: "editor.py"}}, nil),
		"a file by a roundabout path": pack([]*tar.Header{{Name: "stet.toml"}, {Name: "x/../editor.py"}}, nil),
		"a file as ./file":            pack([]*tar.Header{{Name: "stet.toml"}, {Name: "./editor.py"}}, nil),
		"a book of file names":        pack(manyNames(400, 200), nil),
		"a manifest of a megabyte":    bigManifest(),
		// The decoder would take either for "name", and which one would be chance.
		"a field name in capitals":      manifest("name = \"tidy\"\nName = \"other\"\nentry = \"python editor.py\"\n"),
		"a field name in other letters": manifest("name = \"tidy\"\nentry = \"python editor.py\"\n\"licen\u017fe\" = \"x\"\n"),
	} {
		if _, _, err := parseBundle(blob, testAssetLimits); err == nil {
			t.Fatalf("a bundle with %s was accepted", what)
		}
	}
	if _, _, err := parseBundle(manifest("name = \"tidy\"\nentry = \"python editor.py\"\nmine = 3\n"), testAssetLimits); err != nil {
		t.Fatalf("a key of the entrant's own: %v", err)
	}
}

func manifest(body string) []byte {
	var tb bytes.Buffer
	tw := tar.NewWriter(&tb)
	tw.WriteHeader(&tar.Header{Name: "stet.toml", Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg})
	tw.Write([]byte(body))
	tw.Close()
	enc, _ := zstd.NewWriter(nil)
	return enc.EncodeAll(tb.Bytes(), nil)
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
var testAssetLimits = AssetLimits{MaxFiles: 32, MaxBytes: 4 << 30}

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
	for path, want := range map[string]int{"/api/v1/admin/rounds/" + roundID(w.app.season.ID, 1) + "/exam": 403, "/api/v1/jobs/1/script": 403} {
		method := "POST"
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

func TestAResultCarriesOneScoreForEachItemAndNothingElse(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.submit(anchor, "ref-a", "print(1)")
	w.draw()
	w.editAll(anchor)
	c := w.claim(bob, false)
	var out struct {
		Error string `json:"error"`
	}
	// Whatever is in a result is stored: one score more than the exam has items is refused,
	// and so is one fewer, a mean that is not theirs, and a score that is not a number.
	for what, change := range map[string]func(map[string]any){
		"an item too many": func(e map[string]any) { e["item_scores"] = append(e["item_scores"].([]float64), -5.5) },
		"an item short":    func(e map[string]any) { e["item_scores"] = e["item_scores"].([]float64)[1:] },
		"another mean":     func(e map[string]any) { e["mean"] = -4.0 },
		"nonsense": func(e map[string]any) {
			e["item_scores"].([]float64)[0], e["mean"] = -1e6, -1e6/float64(len(testTable()))
		},
	} {
		res := w.result(c, 5.5, 6.0)
		change(res["exam"].(map[string]any))
		if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), res, &out); code != 400 {
			t.Fatalf("a result with %s: HTTP %d %s", what, code, out.Error)
		}
	}
	if code := bob.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), w.result(c, 5.5, 6.0), &out); code != 200 {
		t.Fatalf("the result as it should be: HTTP %d %s", code, out.Error)
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
	// An organiser asks for the draw sooner. Verifiers have already been told the old slot was this round's.
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
	// Meanwhile its pages say what it waits for, not a countdown to a time it cannot keep.
	waiting := scheduled()
	page := "/rounds/" + roundID(w.app.season.ID, waiting.Idx)
	countdown := fmt.Sprintf(`data-until="%d"`, waiting.DrawAt)
	for _, path := range []string{page, "/rounds"} {
		if _, body := w.page(path); !strings.Contains(body, fmt.Sprintf("once round %d is graded", idx)) && !strings.Contains(body, fmt.Sprintf(`round %d</a> is graded`, idx)) || strings.Contains(body, countdown) {
			t.Errorf("%s, while round %d is graded: no word of the wait, or a countdown", path, idx)
		}
	}
	w.editAll(anchor)
	if _, status := w.report(anchor, w.claim(anchor, true), 5.5); status != "settled" {
		t.Fatal(status)
	}
	next := scheduled()
	// The first mark on the grid at least the notice away, passing over any that another
	// round was ever announced for (which ones those are depends on the time of day).
	want := w.app.season.nextGrid(now() + drawNotice)
	w.app.st.view(func(s *State) {
		for taken := announcedSlots(s, s.Rounds[next.Idx]); taken[w.app.season.Beacon.drandRoundAt(want)]; {
			want = w.app.season.nextGrid(want)
		}
	})
	if next.Postponed || next.DrawAt > want || next.DrawAt < now()+drawNotice {
		t.Fatalf("a put-off draw waits %d s after the round settled, not %d s", next.DrawAt-now(), want-now())
	}
	if _, body := w.page(page); !strings.Contains(body, fmt.Sprintf(`data-until="%d"`, next.DrawAt)) {
		t.Error("once the round before settled, the next one's page does not count down to its draw")
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
			t.Fatal("this test is about a curator that does not hold the record")
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
		t.Fatal("a rated curator with nobody above it is not on the ladder")
	}
	// On the ladder it runs every round. Now it times out twice.
	for n := 0; n < retireAfterInvalid; n++ {
		w.draw()
		c := w.claim(anchor, true)
		if c.Job == nil || c.Job.Kind != "edit" || c.Job.Editor.ID != id {
			t.Fatalf("the ladder curator was not drawn: %+v", c.Job)
		}
		if code, status := w.edit(anchor, c, ""); code != 200 {
			t.Fatal(status)
		}
	}
	if onLadder() {
		t.Fatal("a curator that failed twice running is still on the ladder, costing the verifiers an edit every round")
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
	manifest := "name = \"tidy\"\nentry = \"python editor.py\"\ndescription = \"nice \\u001b]0;owned\\u0007 curator\"\n"
	tw.WriteHeader(&tar.Header{Name: "stet.toml", Mode: 0o644, Size: int64(len(manifest)), Typeflag: tar.TypeReg})
	tw.Write([]byte(manifest))
	tw.Close()
	enc, _ := zstd.NewWriter(nil)
	if _, _, err := parseBundle(enc.EncodeAll(tb.Bytes(), nil), testAssetLimits); err == nil || !strings.Contains(err.Error(), "ordinary text") {
		t.Fatalf("a description with terminal escapes in it: %v", err)
	}
	w.check()
}

func TestOneEditorsTimeoutMovesNobodyElsesRating(t *testing.T) {
	// The references run every round. carol qualifies in rounds 1-3 and dave in 4-6, both
	// gaining 0.925 every time. zed shares carol's rounds, and times out in one of them.
	var es []entryRow
	for r := 1; r <= 6; r++ {
		for id, g := range map[string]float64{"identity": 0, "english": 0.912, "tidy": 0.933} {
			es = append(es, entryRow{RoundIdx: r, EditorID: id, Gain: g, Valid: true})
		}
		who := "carol"
		if r > 3 {
			who = "dave"
		}
		es = append(es, entryRow{RoundIdx: r, EditorID: who, Gain: 0.925, Valid: true})
	}
	es = append(es, entryRow{RoundIdx: 1, EditorID: "zed", Gain: 0.9, Valid: true}, entryRow{RoundIdx: 2, EditorID: "zed"}, entryRow{RoundIdx: 3, EditorID: "zed", Gain: 0.9, Valid: true})
	rt, diff := fitRatings(es, noiseFloor, 0)
	if d := rt["carol"].Adjusted - rt["dave"].Adjusted; math.Abs(d) > 1e-6 {
		t.Fatalf("carol %v, dave %v: someone else's timeout moved a rating by %v", rt["carol"].Adjusted, rt["dave"].Adjusted, d)
	}
	for r, d := range diff {
		if math.Abs(d) > 1e-6 {
			t.Fatalf("round %d is taken to be %v easier than the others; they were all alike", r, d)
		}
	}
	// zed's own rating still pays for the timeout.
	if z := rt["zed"]; math.Abs(z.Adjusted-0.6) > 1e-6 || z.Invalid != 1 {
		t.Fatalf("zed: %+v", z)
	}
}

func TestSmallRulesOfJudging(t *testing.T) {
	// A difference of exactly the tolerance agrees, whatever floating point makes of it.
	x, y := 4.48954, 4.40954 // variables: Go would subtract the constants exactly
	if d := x - y; d <= 0.08 || !within(d, 0.08) || within(0.0801, 0.08) {
		t.Fatalf("within: %v", d)
	}
	// A person is not handed their own entry under a second key, even when the first key
	// no longer has the GitHub identity the editor was entered under.
	j := &judge{s: &State{Accounts: map[int64]*Account{1: {ID: 1}}}}
	st := &entryState{OwnerID: 1, OwnerGH: 4242}
	if !j.owns(st, &Account{ID: 2, GithubID: 4242, Agreed: 1}) || !j.owns(st, &Account{ID: 1}) {
		t.Fatal("their own entry is not seen to be theirs")
	}
	if j.owns(st, &Account{ID: 3, GithubID: 77, Agreed: 1}) || j.owns(st, &Account{ID: 4}) {
		t.Fatal("somebody else's entry is taken for theirs")
	}
	// Two keys are one person when they share a GitHub identity, and never for want of one.
	if !samePerson(&Account{ID: 2, GithubID: 7}, &Account{ID: 3, GithubID: 7}) || samePerson(&Account{ID: 2}, &Account{ID: 3}) {
		t.Fatal("samePerson")
	}
	// The organisers' accounts start out established; counts from before one was theirs are
	// not agreements.
	w := newWorld(t)
	w.app.st.view(func(s *State) {
		v := w.app.verifierStats(s, &Account{ID: 9, Anchor: true, Agreed: 3, Streak: 3})
		if v["validated"] != 0 || v["streak"] != 0 || v["established"] != true || v["needs_github"] != false {
			t.Fatalf("an organisers' account's stats: %+v", v)
		}
		v = w.app.verifierStats(s, &Account{ID: 10, Agreed: 3, Runs: 2})
		if v["validated"] != 3 || v["established"] != false || v["establish_after"] != 5 || v["runs"] != 2 || v["needs_github"] != false {
			t.Fatalf("a volunteer's stats: %+v", v)
		}
	})
	// What a machine measures can be recorded, however bad.
	score := func(v float64) *ExamScore {
		items := make([]float64, 4)
		for i := range items {
			items[i] = v
		}
		return &ExamScore{Mean: v, ItemScores: items}
	}
	if err := checkItems(score(-45), 4, "exam"); err != nil {
		t.Fatalf("an item score of -45: %v", err)
	}
	if err := checkItems(score(-1e6), 4, "exam"); err == nil {
		t.Fatal("an item score of minus a million was taken")
	}
}

func TestAgreeingThatAScriptIsInvalidEarnsNoStanding(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	w.submit(alice, "tidy", "print('v1')")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	// Finding a script invalid takes no training: it is decided before any.
	invalid := func(a *actor) (int, string) {
		c := w.claim(a, a == anchor)
		p := w.result(c, 5.5, 6.0)
		p["valid"], p["invalid_reason"] = false, "line 3: this line holds text; a script is copies of the crawl and holds no text of its own"
		delete(p, "exam")
		var ack struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		code := a.do("POST", fmt.Sprintf("/api/v1/jobs/%d/result", c.Job.ID), p, &ack)
		return code, ack.Status + ack.Error
	}
	if code, status := invalid(bob); code != 200 {
		t.Fatalf("HTTP %d %s", code, status)
	}
	if code, status := invalid(anchor); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	es := w.entries(alice, idx)
	// A run whose script is invalid is a failed run, and says why.
	if es[0].Valid || !strings.Contains(es[0].Reason, "no text of its own") || len(es[0].Runs) != 1 || es[0].Runs[0].Status != "failed" || es[0].Runs[0].Gain != 0 {
		t.Fatalf("the entry: %+v %+v", es[0], es[0].Runs)
	}
	if g := es[0].Runs[0].Grades; len(g) != 2 || g[1].Handle != "bob" || g[1].Verdict != "agreed" || g[1].Valid || g[1].Loss != nil {
		t.Fatalf("its grades: %+v", g)
	}
	if b := w.account("bob"); b.Agreed != 0 || b.Streak != 0 || b.Strikes != 0 {
		t.Fatalf("bob after agreeing that a script is invalid: %+v", b)
	}
}

func TestARoundThatEndsByGivingAnEntryUpStillWaitsForItsVolunteers(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	good := w.submit(anchor, "ref-a", "print(1)")
	w.submit(bob, "trouble", "print(2)") // his own, so it is never his to grade
	idx := w.draw()
	for i := 0; i < 2; i++ {
		if code, status := w.edit(anchor, w.claim(anchor, true), fmt.Sprintf("{\"doc\": %d}\n", i)); code != 200 {
			t.Fatal(status)
		}
	}
	w.established("bob")
	ctx := context.Background()
	// The anchor grades one entry, and cannot grade the other however often it tries.
	tries := 0
	for {
		c := w.claim(anchor, true)
		if c.Job == nil {
			break
		}
		if c.Job.Editor.ID == good {
			if code, status := w.report(anchor, c, 5.5); code != 200 || status != "pending" {
				t.Fatalf("HTTP %d %s", code, status)
			}
			continue
		}
		tries++
		anchor.do("POST", fmt.Sprintf("/api/v1/jobs/%d/release", c.Job.ID), map[string]string{"reason": "the harness fell over"}, nil)
	}
	if tries != maxGradeAttempts {
		t.Fatalf("the anchor tried %d times", tries)
	}
	// All that took a long time: the one grade is far in the past. Just before the
	// other entry is given up, a volunteer is handed the graded one, because a round with
	// an ungraded entry is not yet closing.
	w.app.st.update(ctx, func(tx *Tx) error {
		for _, r := range tx.s.results(idx, good) {
			r.ReceivedAt = now() - int64(w.app.cfg.ReplicaWait.Seconds()) - 600
			tx.saveResult(r)
		}
		return nil
	})
	held := w.claim(bob, false)
	if held.Job == nil || held.Job.Editor.ID != good {
		t.Fatalf("the volunteer was given %+v", held.Job)
	}
	w.app.tick(ctx)
	// The round became ready a moment ago, when its last entry was given up. The volunteer
	// has the wait that every volunteer is promised, counted from then.
	w.app.st.view(func(s *State) {
		if c := s.Claims[held.Job.ID]; c.Status != "claimed" {
			t.Fatalf("a job a minute old, in a round that became ready this minute: %+v", c)
		}
		if s.Rounds[idx].Status != "open" {
			t.Fatal("the round settled under a volunteer who had just started")
		}
	})
	if code, status := w.report(bob, held, 5.52); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	var out claimed
	if code := bob.do("POST", "/api/v1/jobs/claim", w.hello(false), &out); code != 200 {
		t.Fatalf("the volunteer is refused work afterwards: HTTP %d", code)
	}
	w.check()
}

func TestTheSubmitReplySaysWhichRoundOnlyWhenItKnows(t *testing.T) {
	w := newWorld(t)
	w.app.cfg.RequireGithub = false
	w.app.tick(context.Background()) // schedules the first round
	reply := func(handle string) map[string]any {
		a := w.join(handle, nil)
		var out map[string]any
		if code := a.do("POST", "/api/v1/editors", bundle(t, "ed-"+handle, "print('"+handle+"')"), &out); code != 201 {
			t.Fatalf("submit: HTTP %d %v", code, out)
		}
		return out
	}
	// A round takes everyone waiting at its draw, however many they are.
	for i := 0; i < 12; i++ {
		if out := reply(fmt.Sprintf("early%d", i)); out["enters"] == nil || out["earliest"] != nil {
			t.Fatalf("entry %d, in time for the draw: %v", i, out)
		}
	}
	var in int
	w.app.st.view(func(s *State) {
		rd := s.roundWithStatus("scheduled")
		in = len(w.app.selectEntrants(s, rd.DrawAt, rd.Idx))
	})
	if in != 12 {
		t.Fatalf("the draw would enter %d of the 12 curators waiting", in)
	}
	// One submitted in the last seconds before the draw, after its entries have closed, is
	// told the earliest round it could be in and not that it is in this one.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		rd := tx.s.roundWithStatus("scheduled")
		rd.DrawAt = now() + commitMargin/2
		tx.saveRound(rd)
		return nil
	})
	if out := reply("late"); out["enters"] != nil || out["earliest"] == nil {
		t.Fatalf("an entry after the round's entries closed: %v", out)
	}
}

// Before its draw a round has no entries, and its page and the API say who the draw would
// enter if it were made now, and who waits for a later round: marked as not entered yet,
// with no link to code that is not public.
func TestARoundNotYetDrawnShowsWhoItWouldEnter(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.cfg.QualifyRounds = 3
	anchor := w.join("anchor-one", priv)
	a, b, c := w.submit(anchor, "ref-a", "print(1)"), w.submit(anchor, "ref-b", "print(2)"), w.submit(anchor, "ref-c", "print(3)")
	type row struct {
		EditorID string `json:"editor_id"`
		Why      string `json:"why"`
		Played   int    `json:"played"`
		Public   bool   `json:"public"`
	}
	look := func() (string, string, []row, []row) {
		t.Helper()
		w.app.tick(context.Background())
		rid := ""
		w.app.st.view(func(s *State) {
			if r := s.roundWithStatus("scheduled"); r != nil {
				rid = roundID(w.app.season.ID, r.Idx)
			}
		})
		if rid == "" {
			t.Fatal("no round scheduled")
		}
		code, page := w.page("/rounds/" + rid)
		if code != 200 {
			t.Fatalf("the round's page: HTTP %d", code)
		}
		var api struct {
			Entries  []map[string]any `json:"entries"`
			Expected []row            `json:"expected"`
			Later    []row            `json:"later"`
		}
		if _, body := w.page("/api/v1/rounds/" + rid); json.Unmarshal([]byte(body), &api) != nil || len(api.Entries) != 0 {
			t.Fatalf("the round's API: %s", body)
		}
		return rid, page, api.Expected, api.Later
	}

	// Nothing drawn yet. Every curator waiting enters the draw, however many: all three here.
	_, page, in, later := look()
	if len(in) != 3 || in[0].EditorID != a || in[1].EditorID != b || in[2].EditorID != c || in[0].Why != "new" || in[0].Public || len(later) != 0 {
		t.Fatalf("expected %+v, later %+v", in, later)
	}
	if strings.Contains(page, "Queued for the next round") {
		t.Error("the page says somebody waits for the next round, and nobody does")
	}
	// Two of them submitted in the last seconds before the draw, after its entries closed,
	// wait for the next one. Nobody's code is public.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		drawAt := tx.s.roundWithStatus("scheduled").DrawAt
		for i, id := range []string{b, c} {
			tx.s.Editors[id].CreatedAt = drawAt - 2 + int64(i)
			tx.saveEditor(tx.s.Editors[id])
		}
		return nil
	})
	_, page, in, later = look()
	if len(in) != 1 || in[0].EditorID != a || in[0].Why != "new" || in[0].Public || len(later) != 2 || later[0].EditorID != b || later[1].EditorID != c {
		t.Fatalf("expected %+v, later %+v", in, later)
	}
	for _, want := range []string{"Expected to play", "Not entered yet", "new · queued", "Queued for the next round", ">ref-b<", ">ref-c<"} {
		if !strings.Contains(page, want) {
			t.Errorf("the page does not say %q", want)
		}
	}
	for _, id := range []string{a, b, c} {
		if strings.Contains(page, "/editors/"+id[:12]) {
			t.Errorf("the page links to %s, whose code is not public", id[:12])
		}
	}

	// After a draw (the test's draw takes every curator waiting as submitted in time), each
	// curator it entered is qualifying in the next, and public.
	first := w.drawWithoutExam()
	_, page, in, later = look()
	if len(in) != 3 || in[0].EditorID != a || in[0].Why != "qualifying" || in[0].Played != 1 || !in[0].Public ||
		in[1].EditorID != b || in[1].Why != "qualifying" || in[2].EditorID != c || len(later) != 0 {
		t.Fatalf("expected %+v, later %+v", in, later)
	}
	if !strings.Contains(page, `href="/editors/`+a[:12]) || !strings.Contains(page, `href="/editors/`+b[:12]) || !strings.Contains(page, "qualifying · round 2 of 3") {
		t.Error("the page does not link the curators that are public now, or say where they are in qualifying")
	}
	// A round that has been drawn has entries, and nothing expected.
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, first)); strings.Contains(page, "Expected to play") {
		t.Error("a drawn round still shows who it would enter")
	}
	if _, body := w.page("/api/v1/rounds/" + roundID(w.app.season.ID, first)); strings.Contains(body, `"expected"`) {
		t.Error("a drawn round's API still says who it would enter")
	}
	w.check()
}

func TestAScriptThatDoesNotCompressIsServedWhole(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice := w.join("anchor-one", priv), w.join("alice", nil)
	w.submit(alice, "noisy", "print(1)")
	w.draw()
	// A script of exactly the season's limit, of bytes that will not compress: stored, it
	// is a little larger than the limit.
	w.app.season.Script.MaxBytes = 4096
	raw := make([]byte, 4096)
	rand.Read(raw)
	script := string(raw)
	if code, status := w.edit(anchor, w.claim(anchor, true), script); code != 200 {
		t.Fatalf("HTTP %d %s", code, status)
	}
	sum := sha256.Sum256(raw)
	// Sealed while the round is open: it goes to the machine that holds it as a grade job.
	if c := w.claim(anchor, true); c.Job == nil || c.Job.Kind != "grade" {
		t.Fatalf("the organisers' machine should be grading its script: %+v", c.Job)
	}
	code, blob := anchor.fetch(hex.EncodeToString(sum[:]))
	if code != 200 {
		t.Fatalf("the download: HTTP %d after %d bytes", code, len(blob))
	}
	if int64(len(blob)) <= w.app.season.Script.MaxBytes {
		t.Fatalf("the test is not testing anything: %d bytes stored", len(blob))
	}
	if got, err := unzstd(blob, w.app.season.Script.MaxBytes); err != nil || !bytes.Equal(got, raw) {
		t.Fatalf("what was served is not the script: %v", err)
	}
}

// --- runs by volunteers ---------------------------------------------------------------------

func TestTwoVolunteersConfirmAnEntryWithNoOrganiser(t *testing.T) {
	w := newWorld(t) // nobody's machine here is the organisers'
	alice, bob, carol := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol")
	idx := w.draw()
	rid := roundID(w.app.season.ID, idx)
	// Both are handed the entry to run, at once: neither waits for the other.
	cb, cc := w.claim(bob, true), w.claim(carol, true)
	if cb.Job == nil || cb.Job.Kind != "edit" || cc.Job == nil || cc.Job.Kind != "edit" || cb.Job.ID == cc.Job.ID {
		t.Fatalf("two runners of one entry: %+v %+v", cb.Job, cc.Job)
	}
	one, two := "{\"doc\": 1}\n", "{\"doc\": 2}\n"
	if code, status := w.edit(bob, cb, one); code != 200 || status != "edited" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	if code, status := w.edit(carol, cc, two); code != 200 || status != "edited" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	// Sealed: nothing public says who ran it, or names a script. A second run that could
	// read the first one's script could hand it in as its own.
	sealed := func(when string) {
		t.Helper()
		var raw bytes.Buffer
		for _, path := range []string{"/api/v1/rounds/" + rid, "/api/v1/editors/" + id, "/api/v1/rounds", "/api/v1/leaderboard", "/api/v1/verifiers"} {
			resp, err := http.Get(w.srv.URL + path)
			if err != nil || resp.StatusCode != 200 {
				t.Fatalf("%s: %v", path, err)
			}
			raw.ReadFrom(resp.Body)
			resp.Body.Close()
		}
		for _, path := range []string{"/rounds/" + rid, "/editors/" + id[:12], "/", "/verifiers", "/u/bob"} {
			_, page := w.page(path)
			raw.WriteString(page)
		}
		for _, secret := range []string{shaOf(one), shaOf(two), shaOf(one)[:8], shaOf(two)[:8], `"runs":[{`, `"edit":{`, `id="runs"`, `id="grades"`} {
			if strings.Contains(raw.String(), secret) {
				t.Fatalf("%s, something public shows %q", when, secret)
			}
		}
	}
	sealed("after the runs")
	// Neither is handed its own script. Each is handed the other's.
	gb := w.claim(bob, true)
	if gb.Job == nil || gb.Job.Kind != "grade" || gb.Job.Script.SHA256 != shaOf(two) || gb.Job.Run != 2 {
		t.Fatalf("bob should be grading carol's script: %+v", gb.Job)
	}
	gc := w.claim(carol, true)
	if gc.Job == nil || gc.Job.Kind != "grade" || gc.Job.Script.SHA256 != shaOf(one) || gc.Job.Run != 1 {
		t.Fatalf("carol should be grading bob's script: %+v", gc.Job)
	}
	if code, status := w.report(bob, gb, 5.50); code != 200 || status != "pending" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	sealed("after a grade")
	if code, status := w.report(carol, gc, 5.46); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	// Two runs that agree: the score is their mean, and it is confirmed.
	e := w.entry(idx, "tidy")
	if !e.Valid || !e.Confirmed || e.RunsDiffer || e.NRuns != 2 || math.Abs(*e.Gain-0.52) > 1e-9 || math.Abs(*e.Loss-5.48) > 1e-9 || len(e.Runs) != 2 {
		t.Fatalf("the entry: %+v", e)
	}
	if r := e.Runs[0]; r.By != "bob" || r.Status != "ok" || r.Script != shaOf(one) || math.Abs(r.Gain-0.54) > 1e-9 || r.Self || len(r.Grades) != 1 || r.Grades[0].Handle != "carol" || !r.Grades[0].Counted || r.Grades[0].Self {
		t.Fatalf("the first run: %+v", r)
	}
	if r := e.Runs[1]; r.By != "carol" || r.Status != "ok" || r.Script != shaOf(two) || math.Abs(r.Gain-0.5) > 1e-9 || r.Self || len(r.Grades) != 1 || r.Grades[0].Handle != "bob" {
		t.Fatalf("the second run: %+v", r)
	}
	if v := e.Detail.Verifiers; len(v) != 2 || v[0] != "carol" || v[1] != "bob" {
		t.Fatalf("who graded it: %v", v)
	}
	for _, h := range []string{"bob", "carol"} {
		if a := w.account(h); a.Runs != 1 || a.Strikes != 0 || a.GPUSeconds < 3 {
			t.Fatalf("%s: %+v", h, a)
		}
	}
	w.app.st.view(func(s *State) {
		if r := s.Rounds[idx]; r.Status != "settled" || r.BaselineLoss == nil || *r.BaselineLoss != 6.0 || r.BaselineN != 2 {
			t.Fatalf("the round: %+v", r)
		}
	})
	// And now it is all on show.
	_, page := w.page("/rounds/" + rid)
	for _, want := range []string{`id="runs"`, shaOf(one)[:8], shaOf(two)[:8], ">confirmed<", `<a href="/u/carol">carol</a>`} {
		if !strings.Contains(page, want) {
			t.Fatalf("the settled round's page does not show %q", want)
		}
	}
	if strings.Contains(page, ">graded by its own runner<") || strings.Contains(page, ">run once<") || strings.Contains(page, ">runs differed<") {
		t.Fatal("a confirmed entry is shown as something less")
	}
	var vs struct {
		Verifiers []VerifierRow `json:"verifiers"`
	}
	bob.do("GET", "/api/v1/verifiers", nil, &vs)
	if len(vs.Verifiers) != 2 || vs.Verifiers[0].Runs != 1 || !vs.Verifiers[0].Established || vs.Verifiers[0].Trusted {
		t.Fatalf("the verifier board: %+v", vs.Verifiers)
	}
	if _, page := w.page("/verifiers"); !strings.Contains(page, ">established<") || !strings.Contains(page, ">Runs<") {
		t.Fatal("the verifiers page does not show who is established, or how many curators each has run")
	}
	var me struct {
		Account  map[string]any `json:"account"`
		Verifier map[string]any `json:"verifier"`
	}
	bob.do("GET", "/api/v1/me", nil, &me)
	if me.Account["established"] != true || me.Verifier["established"] != true || me.Verifier["runs"] != 1.0 || me.Verifier["establish_after"] != 5.0 || me.Verifier["needs_github"] != false || me.Verifier["trusted"] != false {
		t.Fatalf("/api/v1/me: %+v", me)
	}
}

func TestAScriptTwoPeoplesRunsWroteIsNobodysOwn(t *testing.T) {
	w := newWorld(t)
	alice, bob, carol := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	id := w.submit(alice, "steady", "print('the same every time')")
	w.established("bob", "carol")
	idx := w.draw()
	cb, cc := w.claim(bob, true), w.claim(carol, true)
	same := "{\"doc\": 7}\n"
	w.edit(bob, cb, same)
	// While his is the only run that wrote it, the script is his own, and not his to grade.
	w.idle(bob, true, "a script only he wrote")
	w.edit(carol, cc, same)
	// Now two people's runs wrote the same bytes, so neither chose them. Either may grade.
	g := w.claim(bob, true)
	if g.Job == nil || g.Job.Kind != "grade" || g.Job.Script.SHA256 != shaOf(same) || g.Job.Run != 1 {
		t.Fatalf("bob should be grading the script both runs wrote: %+v", g.Job)
	}
	// It is the entry's only script, so one grade of it would be one machine's word for
	// the whole score. Carol is there, and grades it too.
	if code, status := w.report(bob, g, 5.5); code != 200 || status != "pending" {
		t.Fatalf("one grade of the entry's only script, with a second grader around: HTTP %d %s", code, status)
	}
	if _, status := w.grade(carol, true, 5.5); status != "settled" {
		t.Fatalf("the second grade: %s", status)
	}
	w.check()
	e := w.entry(idx, "steady")
	if !e.Valid || !e.Confirmed || e.NRuns != 2 || math.Abs(*e.Gain-0.5) > 1e-9 || e.Runs[0].Script != e.Runs[1].Script {
		t.Fatalf("the entry: %+v", e)
	}
	// One script, one set of grades, shown with both runs.
	for _, r := range e.Runs {
		if r.Self || len(r.Grades) != 2 || r.Grades[0].Handle != "bob" || r.Grades[1].Handle != "carol" || r.Grades[0].Self || !r.Grades[0].Counted || !r.Grades[1].Counted {
			t.Fatalf("run %d: %+v", r.N, r)
		}
	}
	w.app.st.view(func(s *State) {
		if n := len(s.results(idx, id)); n != 2 {
			t.Fatalf("%d grades of one script", n)
		}
	})
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); !strings.Contains(page, ">1 and 2<") {
		t.Fatal("the round's page does not say that both runs wrote the script")
	}
}

func TestTheOrganisersGradeStopsBeingItsOwnWhenAnotherRunWritesTheSameScript(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, zed, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("zed", nil), w.join("bob", nil)
	w.submit(alice, "one", "print(1)")
	w.submit(zed, "two", "print(2)")
	w.established("bob")
	idx := w.draw()
	script := func(name string) string { return fmt.Sprintf("{\"doc\": 1, \"by\": %q}\n", name) }
	// Alone, the organisers' machine runs both and grades one of its own scripts.
	for i := 0; i < 2; i++ {
		c := w.claim(anchor, true)
		w.edit(anchor, c, script(c.Job.Editor.Name))
	}
	if _, status := w.grade(anchor, true, 5.5); status != "pending" {
		t.Fatal(status)
	}
	// A volunteer arrives, and its runs of the same editors write the same bytes.
	for busy := true; busy; {
		busy = false
		for _, a := range []*actor{bob, anchor} {
			c := w.claim(a, true)
			if c.Job == nil {
				continue
			}
			busy = true
			if c.Job.Kind == "edit" {
				if code, status := w.edit(a, c, script(c.Job.Editor.Name)); code != 200 {
					t.Fatal(status)
				}
			} else if code, status := w.report(a, c, 5.5); code != 200 {
				t.Fatal(status)
			}
		}
	}
	w.check()
	if w.settled(idx) != 2 {
		t.Fatal("the round did not settle")
	}
	for _, name := range []string{"one", "two"} {
		e := w.entry(idx, name)
		if !e.Valid || !e.Confirmed || e.NRuns != 2 || e.Runs[0].Script != e.Runs[1].Script || math.Abs(*e.Gain-0.5) > 1e-9 {
			t.Fatalf("%s: %+v", name, e)
		}
		for _, r := range e.Runs {
			for _, g := range r.Grades {
				if r.Self || g.Self || !g.Counted {
					t.Fatalf("%s, run %d: a grade of a script two runs wrote is called its runner's own: %+v", name, r.N, g)
				}
			}
		}
	}
}

func TestRunsThatDifferGetAThirdAndTheMedianIsTheScore(t *testing.T) {
	good, other := "{\"doc\": 1}\n", "{\"doc\": 2}\n"
	start := func(third bool) (*world, int, *actor, *actor, *actor) {
		w := newWorld(t)
		alice, bob, carol, dave := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
		w.submit(alice, "flaky", "print('sometimes')")
		w.established("bob", "carol", "dave")
		idx := w.draw()
		cb, cc := w.claim(bob, true), w.claim(carol, true)
		if third {
			// Two runs are under way, which is all an entry wants unless they differ.
			w.idle(dave, true, "a third run before the first two are known to differ")
		}
		if code, status := w.edit(bob, cb, ""); code != 200 || status != "edited" { // the editor fails on bob's machine
			t.Fatalf("a failed run: HTTP %d %s", code, status)
		}
		w.edit(carol, cc, good)
		return w, idx, bob, carol, dave
	}

	// With somebody there to run it a third time.
	w, idx, bob, carol, dave := start(true)
	if g, status := w.grade(bob, true, 5.5); status != "pending" || g.Job.Script.SHA256 != shaOf(good) {
		t.Fatalf("one run failed and one did not, and somebody could run it again: %s", status)
	}
	cd := w.claim(dave, true)
	if cd.Job == nil || cd.Job.Kind != "edit" {
		t.Fatalf("dave should be running it a third time: %+v", cd.Job)
	}
	w.edit(dave, cd, other)
	if g, status := w.grade(carol, true, 5.4); status != "settled" || g.Job.Script.SHA256 != shaOf(other) || g.Job.Run != 3 {
		t.Fatalf("the last grade: %s %+v", status, g.Job)
	}
	w.check()
	e := w.entry(idx, "flaky")
	if !e.Valid || e.NRuns != 3 || !e.Confirmed || e.OneGrader || e.RunsDiffer || math.Abs(*e.Gain-0.5) > 1e-9 || math.Abs(*e.Loss-5.5) > 1e-9 || e.Script["sha256"] != shaOf(good) {
		t.Fatalf("the median of nothing, 0.5 and 0.6: %+v", e)
	}
	if r := e.Runs[0]; r.Status != "failed" || r.Gain != 0 || r.Script != "" || !strings.Contains(r.Reason, "out of time") || r.By != "bob" {
		t.Fatalf("the failed run: %+v", r)
	}

	// With nobody there: the score is the mean of the two, and says that they differed.
	w, idx, bob, _, _ = start(false)
	if _, status := w.grade(bob, true, 5.5); status != "settled" {
		t.Fatalf("nobody could run it a third time, and the round did not settle: %s", status)
	}
	w.check()
	e = w.entry(idx, "flaky")
	if !e.Valid || e.NRuns != 2 || e.Confirmed || !e.RunsDiffer || math.Abs(*e.Gain-0.25) > 1e-9 || math.Abs(*e.Loss-5.75) > 1e-9 {
		t.Fatalf("the mean of nothing and 0.5: %+v", e)
	}
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); !strings.Contains(page, ">runs differed<") {
		t.Fatal("the round's page does not say that the runs differed")
	}
}

func TestOneMachineAloneSettlesItsRoundsAsItFinishesThem(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, zed := w.join("anchor-one", priv), w.join("alice", nil), w.join("zed", nil)
	w.submit(alice, "one", "print(1)")
	w.submit(zed, "two", "print(2)")
	idx := w.draw()
	last, jobs := "", 0
	for {
		c := w.claim(anchor, true)
		if c.Job == nil {
			break
		}
		jobs++
		if c.Job.Kind == "edit" {
			w.edit(anchor, c, fmt.Sprintf("{\"doc\": 1, \"by\": %q}\n", c.Job.Editor.Name))
			continue
		}
		if _, last = w.report(anchor, c, 5.5); last == "settled" && w.settled(idx) != 2 {
			t.Fatal("an entry settled before its round")
		}
	}
	// No waiting for a second machine that is not there: the last grade settles the round.
	if jobs != 4 || last != "settled" || w.settled(idx) != 2 {
		t.Fatalf("%d jobs, the last %s, %d entries settled", jobs, last, w.settled(idx))
	}
	for _, name := range []string{"one", "two"} {
		e := w.entry(idx, name)
		if !e.Valid || e.NRuns != 1 || e.Confirmed || e.RunsDiffer || !e.Runs[0].Self || e.Runs[0].By != "anchor-one" || math.Abs(*e.Gain-0.5) > 1e-9 {
			t.Fatalf("%s: %+v", name, e)
		}
	}
	w.check()
}

func TestNobodyRunsTheirOwnEntryOrOneEntryTwice(t *testing.T) {
	w := newWorld(t)
	bob, bob2, carol, carol2 := w.join("bob", nil), w.join("bob-again", nil), w.join("carol", nil), w.join("carol-again", nil)
	github := func(handle string, id int64) {
		w.app.st.update(context.Background(), func(tx *Tx) error {
			c := tx.s.byHandle[handle]
			c.GithubID = id
			tx.saveAccount(c)
			return nil
		})
	}
	github("bob", 4242)
	id := w.submit(bob, "mine", "print('bob')")
	// Bob's GitHub identity moves to a second key of his. The entry is still his.
	github("bob", 0)
	github("bob-again", 4242)
	github("carol", 77)
	w.established("bob", "bob-again", "carol", "carol-again")
	idx := w.draw()
	w.idle(bob, true, "its owner")
	w.idle(bob2, true, "its owner, under another key")
	c := w.claim(carol, true)
	if c.Job == nil || c.Job.Kind != "edit" {
		t.Fatalf("somebody else should be running it: %+v", c.Job)
	}
	script := "{\"doc\": 1}\n"
	w.edit(carol, c, script)
	w.idle(carol, true, "a second run of an entry she has run, or her own script")
	// Carol's identity moves to a second key of hers. It is still one person: no second run
	// of the entry, and no grade of the script her first key wrote.
	github("carol", 0)
	github("carol-again", 77)
	w.idle(carol2, true, "the person who ran it, under another key")
	w.app.st.view(func(s *State) {
		edits := 0
		for _, c := range s.Claims {
			if c.Kind == "edit" {
				edits++
			}
		}
		if e := s.entry(idx, id); len(e.Runs) != 1 || e.Runs[0].GithubID != 77 || edits != 1 || len(s.results(idx, id)) != 0 {
			t.Fatalf("the entry's runs: %+v, %d edit jobs", e.Runs, edits)
		}
	})
	w.check()
}

func TestOnlyAnEstablishedAccountOnAMachineUpToTheFloorIsHandedEditors(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, eve := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("eve", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob")
	w.standing("eve")
	w.draw()
	w.app.cfg.Dev = false // as in production: the sandbox and the floor are required
	f := w.app.season.Runner
	if f.GPUTokensPerSecond <= 0 || f.Cores <= 0 {
		t.Fatalf("the season has no floor to test against: %+v", f)
	}
	bench := func() map[string]any {
		return map[string]any{"gpu_tokens_per_second": f.GPUTokensPerSecond, "cpu_mb_per_second": f.CPUMBPerSecond, "cores": f.Cores, "memory_gb": f.MemoryGB, "gpu_memory_gb": f.GPUMemoryGB}
	}
	hello := func(b map[string]any, sandbox string) map[string]any {
		h := w.hello(true)
		hw := map[string]any{"gpu": "Test GPU", "device": "cuda", "backend": "cuda"}
		if b != nil {
			hw["bench"] = b
		}
		h["hardware"], h["sandbox"] = hw, sandbox
		return h
	}
	refused := func(who *actor, h map[string]any, why string) {
		t.Helper()
		c := w.claimWith(who, h)
		reason, _ := c.Runner["reason"].(string)
		if c.Job != nil || c.Runner["eligible"] != false || !strings.Contains(reason, why) {
			t.Fatalf("%s was not refused curators for %q: job %+v, %+v", who.handle, why, c.Job, c.Runner)
		}
	}
	slow, small := bench(), bench()
	slow["gpu_tokens_per_second"] = f.GPUTokensPerSecond - 1
	small["cores"] = f.Cores - 1
	refused(eve, hello(bench(), "bwrap"), "established") // a fast machine, and an account without the record
	refused(bob, hello(nil, "bwrap"), "floor")           // no measurement of itself
	refused(bob, hello(slow, "bwrap"), "floor")          // too slow
	refused(bob, hello(small, "bwrap"), "floor")         // too small
	refused(bob, hello(bench(), "none"), "sandbox")      // nothing to run entrants' code in
	refused(anchor, hello(nil, "bwrap"), "floor")        // the organisers' machines are held to it too
	if a := w.account("bob"); a.RunnerSeen != 0 {
		t.Fatalf("a machine that may not run curators is counted as one that could: %+v", a)
	}
	c := w.claimWith(bob, hello(bench(), "bwrap"))
	if c.Job == nil || c.Job.Kind != "edit" || c.Runner["eligible"] != true {
		t.Fatalf("an established account on a machine up to the floor: %+v %+v", c.Job, c.Runner)
	}
	if a := w.account("bob"); a.RunnerSeen == 0 || a.Backend != "cuda" {
		t.Fatalf("the runner's record: %+v", a)
	}
	// An account that loses its standing while it holds the job cannot hand in the run.
	w.app.st.update(context.Background(), func(tx *Tx) error {
		b := tx.s.byHandle["bob"]
		b.Agreed = 0
		tx.saveAccount(b)
		return nil
	})
	if code, msg := w.edit(bob, c, "{\"doc\": 1}\n"); code != 403 {
		t.Fatalf("a run by an account that is no longer established: HTTP %d %s", code, msg)
	}
	w.check()
}

// After its exam a round hands out only the grades its scores wait for. Any other would
// hold the round back for as long as it trained, and while it did, the next machine to
// come free would be handed another to wait for. A script that two runs wrote waits for
// two grades, and both go out at once.
func TestAfterItsExamARoundHandsOutOnlyTheGradesItsScoresWaitFor(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, carol, dave := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol", "dave")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	// Bob's grade is the one the score waits for. Carol, free while he works, is handed nothing.
	cb := w.claim(bob, false)
	if cb.Job == nil || cb.Job.Kind != "grade" {
		t.Fatalf("an established grader: %+v", cb.Job)
	}
	w.idle(carol, false, "the score's grade is under way")
	if code, status := w.report(bob, cb, 5.5); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.idle(dave, false, "the round has settled")
	w.check()
	if e := w.entry(idx, "tidy"); len(e.Runs[0].Grades) != 1 {
		t.Fatalf("the grades behind the score: %+v", e.Runs[0].Grades)
	}
}

func TestBothGradesAScriptTwoRunsWroteWaitsForGoOutAtOnce(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, carol, dave, erin, frank := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil), w.join("erin", nil), w.join("frank", nil)
	w.submit(alice, "steady", "print('the same every time')")
	w.established("bob", "carol", "dave", "erin", "frank")
	idx := w.draw()
	cb, cc := w.claim(bob, true), w.claim(carol, true)
	same := "{\"doc\": 7}\n"
	w.edit(bob, cb, same)
	w.edit(carol, cc, same)
	wanted := func() int {
		var out Work
		if code := anchor.do("GET", "/api/v1/admin/work", nil, &out); code != 200 || len(out.Rounds) != 1 {
			t.Fatalf("the work: HTTP %d %+v", code, out)
		}
		return out.Rounds[0].GradesWanted
	}
	cd := w.claim(dave, false)
	if cd.Job == nil || cd.Job.Kind != "grade" {
		t.Fatalf("the first grade: %+v", cd.Job)
	}
	if n := wanted(); n != 1 {
		t.Fatalf("with one of its two grades under way, the script is still wanted: %d", n)
	}
	ce := w.claim(erin, false)
	if ce.Job == nil || ce.Job.Kind != "grade" || ce.Job.Script.SHA256 != cd.Job.Script.SHA256 {
		t.Fatalf("the second grade: %+v", ce.Job)
	}
	if n := wanted(); n != 0 {
		t.Fatalf("both grades are under way: %d", n)
	}
	w.idle(frank, false, "both grades the score waits for are under way")
	if code, status := w.report(dave, cd, 5.5); code != 200 || status != "pending" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	if code, status := w.report(erin, ce, 5.5); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	if e := w.entry(idx, "steady"); !e.Confirmed || len(e.Runs[0].Grades) != 2 {
		t.Fatalf("the entry: %+v", e)
	}
}

// A verifier that trains before the exam is handed a grade as soon as there is a script to
// grade: it trains while the exam is made, and sits it once it is posted. Its job does not
// run out while it waits, and once every grade a score waits for is in the hands of such
// machines, nothing more is handed out: the round settles minutes after its exam.
func TestAGraderThatTrainsBeforeTheExamIsHandedItsGradeEarly(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, carol, dave := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol", "dave")
	idx := w.drawWithoutExam()
	w.run(anchor, "{\"doc\": 1}\n")
	early := w.claimEarly
	work := func() WorkRound {
		var out Work
		if code := anchor.do("GET", "/api/v1/admin/work", nil, &out); code != 200 || len(out.Rounds) != 1 {
			t.Fatalf("the work: HTTP %d %+v", code, out)
		}
		return out.Rounds[0]
	}
	if wr := work(); wr.GradesEarly != 1 || wr.GradesWanted != 0 || wr.ExamAt != 0 || wr.DrawnAt == 0 {
		t.Fatalf("before the exam, one grade for a machine that trains early: %+v", wr)
	}
	w.idle(carol, false, "a verifier that does not train before the exam waits for it")
	cb := early(bob)
	if cb.Job == nil || cb.Job.Kind != "grade" || cb.Job.Round["exam_digest"] != nil {
		t.Fatalf("a grade before the exam: %+v", cb.Job)
	}
	if wr := work(); wr.GradesEarly != 0 || wr.GradesUnderWay != 1 {
		t.Fatalf("the early grade is under way: %+v", wr)
	}
	if code := bob.do("GET", fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)), nil, nil); code != 404 {
		t.Fatalf("an exam not yet posted: HTTP %d", code)
	}
	// However long the exam takes, the job waits for it.
	w.pass(w.cfg.Lease + time.Hour)
	if c := early(bob); c.Job == nil || c.Job.ID != cb.Job.ID {
		t.Fatalf("the job ran out while it waited for the exam: %+v", c.Job)
	}
	w.exam(idx)
	w.pass(w.cfg.Lease - time.Minute)
	if code := bob.do("GET", fmt.Sprintf("/api/v1/rounds/%s/exam", roundID(w.app.season.ID, idx)), nil, nil); code != 200 {
		t.Fatalf("the grader could not fetch the exam once it was posted: HTTP %d", code)
	}
	// All that is left is sitting it: a machine that comes free now is handed no grade it
	// would have to train for, however it trains.
	w.idle(dave, false, "the grade the score waits for was trained before the exam")
	if c := early(carol); c.Job != nil {
		t.Fatalf("carol was handed a grade after the exam: %+v", c.Job)
	}
	if code, status := w.report(bob, cb, 5.5); code != 200 || status != "settled" {
		t.Fatalf("the grade trained before the exam: HTTP %d %s", code, status)
	}
	w.check()
}

func TestTwoEstablishedGradesThatDisagreeGetAThird(t *testing.T) {
	start := func(third bool) (*world, int, *actor, *actor, *actor, *claimed, *claimed) {
		pub, priv := newKey()
		w := newWorld(t, pub)
		anchor, alice, bob, carol, dave := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
		w.submit(alice, "tidy", "print('v1')")
		w.established("bob", "carol", "dave")
		if third {
			w.idle(dave, false, "no round is open") // a third grader's machine is up
		}
		idx := w.drawWithoutExam()
		w.run(anchor, "{\"doc\": 1}\n")
		// Both take the script while the exam is made: the second as a replica, which only a
		// grade taken before the exam may be.
		cb, cc := w.claimEarly(bob), w.claimEarly(carol)
		if cb.Job == nil || cc.Job == nil || cb.Job.Kind != "grade" || cc.Job.Kind != "grade" {
			t.Fatalf("two established graders: %+v %+v", cb.Job, cc.Job)
		}
		w.exam(idx)
		// With them there, the machine that wrote the script is not asked to grade it.
		w.idle(anchor, true, "others are there to grade it")
		return w, idx, bob, carol, dave, cb, cc
	}

	// Nobody is there to say which of two is right: the gain is midway, and says so.
	w, idx, bob, carol, _, cb, cc := start(false)
	if code, status := w.report(bob, cb, 5.5); code != 200 || status != "pending" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	if code, status := w.report(carol, cc, 5.2); code != 200 || status != "settled" {
		t.Fatalf("HTTP %d %s", code, status)
	}
	w.check()
	e := w.entry(idx, "tidy")
	if !e.Valid || math.Abs(*e.Gain-0.65) > 1e-9 || !e.Detail.DisputedTogether || e.Detail.Disputed != 0 || len(e.Runs[0].Grades) != 2 {
		t.Fatalf("two grades that disagree and no third: %+v %+v", e, e.Detail)
	}
	for _, g := range e.Runs[0].Grades {
		if g.Verdict != "disputed" || !g.Counted {
			t.Fatalf("a disputed grade: %+v", g)
		}
	}
	for _, h := range []string{"bob", "carol"} {
		if a := w.account(h); a.Strikes != 0 || a.Agreed != 2*w.cfg.EstablishAfter {
			t.Fatalf("%s was struck, or earned standing, in a dispute nobody settled: %+v", h, a)
		}
	}
	if _, page := w.page("/rounds/" + roundID(w.app.season.ID, idx)); !strings.Contains(page, ">disputed<") {
		t.Fatal("the round's page does not show the dispute")
	}

	// A third established verifier is around: the round waits for it, and the median decides.
	w, idx, bob, carol, dave, cb, cc := start(true)
	w.report(bob, cb, 5.5)
	if code, status := w.report(carol, cc, 5.2); code != 200 || status != "pending" {
		t.Fatalf("two grades disagree and a third grader is around: HTTP %d %s", code, status)
	}
	if g, status := w.grade(dave, false, 5.45); status != "settled" || g.Job.Run != 1 {
		t.Fatalf("the tie-break: %s", status)
	}
	w.check()
	e = w.entry(idx, "tidy")
	if !e.Valid || math.Abs(*e.Gain-0.55) > 1e-9 || e.Detail.DisputedTogether || e.Detail.Disputed != 1 || len(e.Runs[0].Grades) != 3 {
		t.Fatalf("the median of three: %+v %+v", e, e.Detail)
	}
	if a := w.account("carol"); a.Strikes != 1 || a.Streak != 0 {
		t.Fatalf("the one the median went against: %+v", a)
	}
	for _, h := range []string{"bob", "dave"} {
		if a := w.account(h); a.Strikes != 0 || a.Agreed != 2*w.cfg.EstablishAfter+1 || a.Streak != 1 {
			t.Fatalf("%s, whose grade agreed with the median: %+v", h, a)
		}
	}
}

func TestARoundWaitsForAVerifierWhoIsAroundAndNotForOneWhoHasGone(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob, carol := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	w.submit(alice, "tidy", "print('v1')")
	w.established("bob", "carol")
	// Bob's machine is up, and may run editors. No round is open yet.
	w.idle(bob, true, "no round is open")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	// Somebody else is around who could grade the script, so its own runner is not asked to.
	w.idle(anchor, true, "others are around to grade it")
	if _, status := w.grade(carol, false, 5.5); status != "pending" {
		t.Fatalf("the entry could be scored on one run, but bob is around to give it a second: %s", status)
	}
	w.pass(w.cfg.ActiveWindow - time.Minute)
	if w.settled(idx) != 0 {
		t.Fatal("the round settled while a runner was around who had not run the entry")
	}
	// Bob never comes back. A little later nobody is waiting for him.
	w.pass(2 * time.Minute)
	if w.settled(idx) != 1 {
		t.Fatal("the round waits for ever for a verifier who has stopped asking for work")
	}
	e := w.entry(idx, "tidy")
	if !e.Valid || e.NRuns != 1 || e.Confirmed || e.Runs[0].Self || len(e.Runs[0].Grades) != 1 || e.Runs[0].Grades[0].Handle != "carol" || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("run once, and graded by somebody else: %+v %+v", e, e.Runs)
	}
	w.check()
}

func TestAVerifierAtWorkIsNotTakenForOneWhoHasGone(t *testing.T) {
	for _, how := range []string{"handed back", "ran out"} {
		pub, priv := newKey()
		w := newWorld(t, pub)
		anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
		w.submit(alice, "tidy", "print('v1')")
		w.established("bob")
		idx := w.draw()
		w.run(anchor, "{\"doc\": 1}\n")
		g := w.claim(anchor, true) // nobody else is around: it is handed its own script
		if g.Job == nil || g.Job.Kind != "grade" {
			t.Fatalf("%+v", g.Job)
		}
		// A volunteer's machine arrives and takes the entry's second run.
		cb := w.claim(bob, true)
		if cb.Job == nil || cb.Job.Kind != "edit" {
			t.Fatalf("bob should be running the entry: %+v", cb.Job)
		}
		if code, status := w.report(anchor, g, 5.5); code != 200 || status != "pending" {
			t.Fatalf("the entry has a score, and a second run under way: HTTP %d %s", code, status)
		}
		// It works for longer than a verifier that had gone would be waited for, and says
		// nothing meanwhile. It holds a job: it is at work, not gone.
		w.pass(w.cfg.ActiveWindow + w.cfg.ReplicaWait + time.Minute)
		w.app.st.view(func(s *State) {
			if s.Claims[cb.Job.ID].Status != "claimed" || s.Rounds[idx].Status != "open" {
				t.Fatalf("%s: the round settled on one run under a verifier at work on the second: %+v", how, s.Claims[cb.Job.ID])
			}
		})
		if how == "handed back" {
			w.release(bob, cb)
		} else {
			w.pass(w.cfg.Lease)
		}
		if w.settled(idx) != 1 {
			t.Fatalf("%s: the round did not settle once the job was over and its holder long silent", how)
		}
		if e := w.entry(idx, "tidy"); !e.Valid || e.NRuns != 1 || e.Confirmed || !e.Runs[0].Self {
			t.Fatalf("%s: %+v", how, e)
		}
		// Letting a job run out costs a day. Handing it back costs nothing.
		var out claimed
		if code := bob.do("POST", "/api/v1/jobs/claim", w.hello(true), &out); (code == 403) != (how == "ran out") {
			t.Fatalf("%s: bob asks for work afterwards: HTTP %d %s", how, code, out.Error)
		}
		w.check()
	}
}

func TestWhoeverHasBeenHandedAScriptIsNotHandedItsEntryToRun(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor, alice, bob := w.join("anchor-one", priv), w.join("alice", nil), w.join("bob", nil)
	id := w.submit(alice, "tidy", "print('v1')")
	w.established("bob")
	idx := w.draw()
	w.run(anchor, "{\"doc\": 1}\n")
	g := w.claim(bob, false)
	if g.Job == nil || g.Job.Kind != "grade" {
		t.Fatalf("%+v", g.Job)
	}
	// He has the first run's script now. He hands the job back and asks for the entry to
	// run: he could hand that script in as his own, and it would confirm itself.
	w.release(bob, g)
	w.idle(bob, true, "a run of an entry whose script he has been handed")
	// Nor is the job he handed back offered again, so nobody else can grade the script, and
	// its runner is asked to.
	if _, status := w.grade(anchor, true, 5.5); status != "settled" {
		t.Fatal(status)
	}
	w.app.st.view(func(s *State) {
		if e := s.entry(idx, id); len(e.Runs) != 1 || e.Confirmed {
			t.Fatalf("the entry: %+v", e)
		}
	})
	w.check()
}

func TestAVolunteersRunIsNotTakenOnItsWordAlone(t *testing.T) {
	ctx := context.Background()
	for _, script := range []string{"{\"doc\": 1}\n", ""} { // a run that wrote a script, and one that says the editor failed
		w := newWorld(t)
		alice, bob := w.join("alice", nil), w.join("bob", nil)
		id := w.submit(alice, "tidy", "print('v1')")
		w.established("bob")
		idx := w.draw()
		w.run(bob, script)
		// A volunteer is never handed a script that only it wrote, however long it asks.
		w.idle(bob, true, "its own script")
		w.app.tick(ctx)
		if w.settled(idx) != 0 {
			t.Fatal("the entry was given up before anybody else had a chance to come")
		}
		w.age(w.cfg.ReplicaWait + time.Minute)
		w.idle(bob, true, "its own script, later")
		w.app.tick(ctx)
		e := w.entry(idx, "tidy")
		if e.Status != "settled" || e.Valid || !e.Void || !strings.Contains(e.Reason, "no second verifier was there") || len(e.Runs) != 0 || len(e.Dropped) != 1 || e.Dropped[0].By != "bob" {
			t.Fatalf("script %q: %+v", script, e)
		}
		if why := e.Dropped[0].Reason; !strings.Contains(why, "nobody else was there") {
			t.Fatalf("why the run was set aside: %q", why)
		}
		w.app.st.view(func(s *State) {
			if r := s.Rounds[idx]; r.Status != "settled" {
				t.Fatal("the round is held open by a run nobody else can check")
			}
			// Counted neither for the entry nor against it.
			if en := s.entry(idx, id); !en.Unattended || failing(s, id) || roundsRun(s, id) != 0 || s.Editors[id].Status == "retired" {
				t.Fatalf("the entry afterwards: %+v, curator %s", en, s.Editors[id].Status)
			}
		})
		if b := w.account("bob"); b.Runs != 0 || b.Strikes != 0 { // (a run nobody could check is not credited)
			t.Fatalf("bob: %+v", b)
		}
		// Nothing is held against the volunteer either.
		w.idle(bob, true, "no round is open")
		w.check()
	}
	// A round nobody was there for never counts towards the three in a row that say an
	// entry cannot be run; three in which the verifiers ran others and failed on it do.
	rounds := func(unattended bool) *State {
		s := &State{ofEditor: map[string][]*Entry{}, inRound: map[int][]*Entry{}}
		for r := 1; r <= maxEditAttempts; r++ {
			e := &Entry{RoundIdx: r, EditorID: "x", Status: "settled", Void: true, Unattended: unattended}
			o := &Entry{RoundIdx: r, EditorID: "y", Status: "settled", Valid: true}
			s.ofEditor["x"], s.inRound[r] = append(s.ofEditor["x"], e), []*Entry{e, o}
		}
		return s
	}
	if failing(rounds(true), "x") || !failing(rounds(false), "x") {
		t.Fatal("which void rounds count against an entry")
	}
}

func TestARoundIsNotHeldUpByAnEntryNobodyAroundMayRun(t *testing.T) {
	w := newWorld(t)
	alice, bob, carol := w.join("alice", nil), w.join("bob", nil), w.join("carol", nil)
	w.submit(alice, "hers", "print('alice')")
	mine := w.submit(bob, "his", "print('bob')")
	w.established("bob", "carol")
	idx := w.draw()
	// Carol grades and does not run editors, so bob is the only runner: of alice's entry,
	// which carol then grades, and never of his own.
	w.run(bob, "{\"doc\": 1}\n")
	w.idle(bob, true, "his own entry")
	if _, status := w.grade(carol, false, 5.5); status != "pending" {
		t.Fatal(status)
	}
	// While nothing has waited long, the round waits for a runner to come.
	w.pass(w.cfg.ReplicaWait - time.Minute)
	if w.settled(idx) != 0 {
		t.Fatal("an entry was given up at once")
	}
	w.pass(2 * time.Minute)
	if w.settled(idx) != 2 {
		t.Fatal("one entry nobody around may run holds the other's score back for ever")
	}
	if e := w.entry(idx, "hers"); !e.Valid || e.NRuns != 1 || math.Abs(*e.Gain-0.5) > 1e-9 {
		t.Fatalf("the entry that was run: %+v", e)
	}
	e := w.entry(idx, "his")
	if !e.Void || e.Valid || !strings.Contains(e.Reason, "no verifier was there to run") {
		t.Fatalf("the entry nobody could run: %+v", e)
	}
	w.app.st.view(func(s *State) {
		if failing(s, mine) || roundsRun(s, mine) != 0 {
			t.Fatal("it was held against the entry")
		}
	})
	w.check()

	// A round in which nothing has been run does not give its entries up: it waits.
	w = newWorld(t)
	alice = w.join("alice", nil)
	w.submit(alice, "hers", "print('alice')")
	idx = w.draw()
	w.pass(10 * w.cfg.ReplicaWait)
	if w.settled(idx) != 0 {
		t.Fatal("a round nobody has come to yet threw its entries away")
	}
}

func TestASecondRunGoesToTheOtherMakeOfGPUWhenThereIsOne(t *testing.T) {
	w := newWorld(t)
	alice, zed, bob, carol, dave := w.join("alice", nil), w.join("zed", nil), w.join("bob", nil), w.join("carol", nil), w.join("dave", nil)
	w.submit(alice, "one", "print(1)")
	w.submit(zed, "two", "print(2)")
	w.established("bob", "carol", "dave")
	idx := w.draw()
	on := func(backend string) map[string]any {
		h := w.hello(true)
		h["hardware"] = map[string]any{"gpu": "Test GPU", "device": "cuda", "backend": backend}
		return h
	}
	// First runs come before second runs: the two entries go to the first two runners.
	cb := w.claimWith(bob, on("cuda"))
	w.edit(bob, cb, "{\"doc\": 1}\n")
	cc := w.claimWith(carol, on("rocm"))
	if cc.Job == nil || cc.Job.Kind != "edit" || cc.Job.Editor.ID == cb.Job.Editor.ID {
		t.Fatalf("an entry was given a second run while another had none: %+v", cc.Job)
	}
	w.edit(carol, cc, "{\"doc\": 2}\n")
	// Both want a second run. A third machine has an NVIDIA GPU, like the one that ran the
	// first entry: it is handed the entry that has only been run on AMD.
	cd := w.claimWith(dave, on("cuda"))
	if cd.Job == nil || cd.Job.Kind != "edit" || cd.Job.Editor.ID != cc.Job.Editor.ID {
		t.Fatalf("the second run went to the same make of GPU as the first: %+v", cd.Job)
	}
	w.app.st.view(func(s *State) {
		if e := s.entry(idx, cc.Job.Editor.ID); len(e.Runs) != 1 || e.Runs[0].Backend != "rocm" || e.Runs[0].GPU != "Test GPU" {
			t.Fatalf("what a run was made on: %+v", e.Runs)
		}
	})
	w.check()
}

func TestBeingEstablishedNeedsTheGithubIdentityThatEnteringDoes(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	w.app.cfg.RequireGithub = true
	anchor, bob := w.join("anchor-one", priv), w.join("bob", nil)
	w.submit(anchor, "ref-a", "print(1)")
	w.established("bob") // the grades, and no GitHub account
	w.draw()
	standing := func() map[string]any {
		var me struct {
			Verifier map[string]any `json:"verifier"`
		}
		bob.do("GET", "/api/v1/me", nil, &me)
		return me.Verifier
	}
	if v := standing(); v["established"] != false || v["needs_github"] != true || v["validated"] != 10.0 || v["establish_after"] != 5.0 {
		t.Fatalf("enough grades and no GitHub identity: %+v", v)
	}
	c := w.claim(bob, true)
	if why, _ := c.Runner["reason"].(string); c.Job != nil || c.Runner["eligible"] != false || !strings.Contains(why, "GitHub") {
		t.Fatalf("handed a curator without one: %+v %+v", c.Job, c.Runner)
	}
	w.app.st.update(context.Background(), func(tx *Tx) error {
		b := tx.s.byHandle["bob"]
		b.Github, b.GithubID, b.GithubCreatedAt = "bob", 5, now()-400*86400
		tx.saveAccount(b)
		return nil
	})
	if v := standing(); v["established"] != true || v["needs_github"] != false {
		t.Fatalf("with one: %+v", v)
	}
	if c := w.claim(bob, true); c.Job == nil || c.Job.Kind != "edit" || c.Runner["eligible"] != true {
		t.Fatalf("an established account with a GitHub identity: %+v %+v", c.Job, c.Runner)
	}
	// The organisers' accounts need none.
	var me struct {
		Verifier map[string]any `json:"verifier"`
	}
	anchor.do("GET", "/api/v1/me", nil, &me)
	if me.Verifier["established"] != true || me.Verifier["needs_github"] != false {
		t.Fatalf("the organisers' account: %+v", me.Verifier)
	}
}

// A new season starts on an empty database, so a verifier still running the last season's
// harness has no account in it. It upgrades itself when it is told to (426) and stops when it
// is told to log in (401): it is told to upgrade. A current verifier with no account is told
// to log in, as before, and nothing is said to a request that is not signed by its key.
func TestAVerifierFromTheLastSeasonWithNoAccountIsToldToUpgrade(t *testing.T) {
	pub, _ := newKey()
	w := newWorld(t, pub)
	_, priv := newKey()
	old := &actor{t: t, srv: w.srv, priv: priv, pub: hex.EncodeToString(priv.Public().(ed25519.PublicKey)), handle: "old"}
	var out struct {
		Error string `json:"error"`
	}
	last := w.hello(false)
	last["season"], last["season_digest"], last["harness"] = "s0", strings.Repeat("0", 64), "0.0.1"
	if code := old.do("POST", "/api/v1/jobs/claim", last, &out); code != 426 || !strings.Contains(out.Error, "Upgrade") {
		t.Fatalf("last season's verifier, with no account: HTTP %d %s", code, out.Error)
	}
	if code := old.do("POST", "/api/v1/jobs/claim", w.hello(false), &out); code != 401 || !strings.Contains(out.Error, "stet login") {
		t.Fatalf("this season's verifier, with no account: HTTP %d %s", code, out.Error)
	}
	if code := old.do("POST", "/api/v1/jobs/claim", []byte("{"), &out); code != 401 {
		t.Fatalf("a malformed request from a key with no account: HTTP %d %s", code, out.Error)
	}
	// Signed by another key than the one it names: nothing about the season is said.
	_, other := newKey()
	forger := &actor{t: t, srv: w.srv, priv: other, pub: old.pub, handle: "forger"}
	if code := forger.do("POST", "/api/v1/jobs/claim", last, &out); code != 401 || strings.Contains(out.Error, "Upgrade") {
		t.Fatalf("an old hello under a signature that is not its key's: HTTP %d %s", code, out.Error)
	}
	w.app.st.view(func(s *State) {
		if len(s.Accounts) != 0 || len(s.Claims) != 0 {
			t.Fatalf("a key with no account was given something: %d accounts, %d jobs", len(s.Accounts), len(s.Claims))
		}
	})
}
