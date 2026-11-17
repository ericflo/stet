package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// --- plumbing -------------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, err error) {
	var ae *apiError
	if errors.As(err, &ae) {
		if ae.Status == 503 {
			w.Header().Set("Retry-After", "5")
		}
		writeJSON(w, ae.Status, map[string]string{"error": ae.Message})
		return
	}
	if errors.Is(err, errNoBlob) {
		writeJSON(w, 404, map[string]string{"error": "not found"})
		return
	}
	slog.Error("internal", "err", err)
	writeJSON(w, 500, map[string]string{"error": "something went wrong on the server"})
}

func readBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if err != nil {
		return nil, fail(413, "the request is too large")
	}
	return body, nil
}

// signed authenticates a request and returns a copy of the account behind it. With
// mustExist, the key must belong to an account.
func (a *App) signed(w http.ResponseWriter, r *http.Request, limit int64, mustExist bool) (*Account, string, []byte, error) {
	body, err := readBody(w, r, limit)
	if err != nil {
		return nil, "", nil, err
	}
	key, err := verifySignature(r, body)
	if err != nil {
		return nil, "", nil, fail(401, "%s", err.Error())
	}
	acct := a.accountByKey(key)
	if acct == nil && mustExist {
		return nil, "", nil, fail(401, "no account for this key; run `stet login`")
	}
	return acct, key, body, nil
}

func (a *App) accountByKey(key string) *Account {
	var out *Account
	a.st.view(func(s *State) {
		if acct := s.byPubkey[key]; acct != nil {
			c := *acct
			out = &c
		}
	})
	return out
}

// limiter is a small in-memory guard against floods from one address.
type limiter struct {
	mu   sync.Mutex
	hits map[string][]int64
}

func newLimiter() *limiter { return &limiter{hits: map[string][]int64{}} }

func (l *limiter) allow(key string, max int, window int64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	t := now()
	kept := l.hits[key][:0]
	for _, h := range l.hits[key] {
		if t-h < window {
			kept = append(kept, h)
		}
	}
	if len(kept) >= max {
		l.hits[key] = kept
		return false
	}
	l.hits[key] = append(kept, t)
	if len(l.hits) > 50000 {
		l.hits = map[string][]int64{}
	}
	return true
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		first, _, _ := strings.Cut(xff, ",")
		return strings.TrimSpace(first)
	}
	host, _, _ := net.SplitHostPort(r.RemoteAddr)
	return host
}

// --- routes ---------------------------------------------------------------------------------

func (a *App) mountAPI(mux *http.ServeMux) {
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if !a.st.ready() {
			http.Error(w, "database unavailable", 503)
			return
		}
		w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("GET /api/v1/season", a.apiSeason)
	mux.HandleFunc("POST /api/v1/accounts", a.apiCreateAccount)
	mux.HandleFunc("POST /api/v1/accounts/github", a.apiLinkGithub)
	mux.HandleFunc("GET /api/v1/me", a.apiMe)
	mux.HandleFunc("POST /api/v1/editors", a.apiSubmit)
	mux.HandleFunc("GET /api/v1/editors/{ref}", a.apiEditor)
	mux.HandleFunc("GET /api/v1/editors/{ref}/bundle", a.apiBundle)
	mux.HandleFunc("GET /api/v1/rounds", a.apiRounds)
	mux.HandleFunc("GET /api/v1/rounds/{id}", a.apiRound)
	mux.HandleFunc("GET /api/v1/leaderboard", a.apiLeaderboard)
	mux.HandleFunc("GET /api/v1/verifiers", a.apiVerifiers)
	mux.HandleFunc("GET /api/v1/records", a.apiRecords)
	mux.HandleFunc("GET /api/v1/scripts/{sha}", a.apiGetScript)
	mux.HandleFunc("POST /api/v1/jobs/claim", a.apiClaim)
	mux.HandleFunc("POST /api/v1/jobs/{id}/script", a.apiScript)
	mux.HandleFunc("POST /api/v1/jobs/{id}/result", a.apiResult)
	mux.HandleFunc("POST /api/v1/jobs/{id}/release", a.apiRelease)
	mux.HandleFunc("POST /api/v1/admin/draw", a.apiAdminDraw)
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 404, map[string]string{"error": "no such endpoint"})
	})
}

func hostOf(rawURL string) string {
	rest := rawURL
	if _, after, ok := strings.Cut(rawURL, "://"); ok {
		rest = after
	}
	host, _, _ := strings.Cut(rest, "/")
	return strings.ToLower(host)
}

// routes serves three kinds of host from one process: the website, the API (which also
// carries the downloads the CLI needs), and names that only redirect to the website.
func (a *App) routes() http.Handler {
	site := http.NewServeMux()
	a.mountAPI(site)
	a.site.mountDownloads(site)
	a.site.mount(site)

	api := http.NewServeMux()
	a.mountAPI(api)
	a.site.mountDownloads(api)
	api.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			writeJSON(w, 404, map[string]string{"error": "no such endpoint", "site": a.cfg.BaseURL})
			return
		}
		writeJSON(w, 200, map[string]any{"service": "Stet API", "season": a.season.ID, "site": a.cfg.BaseURL, "docs": a.cfg.BaseURL + "/docs/api", "v1": a.cfg.APIURL + "/api/v1/season"})
	})

	apiHost, siteHost := hostOf(a.cfg.APIURL), hostOf(a.cfg.BaseURL)
	return logRequests(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := strings.ToLower(r.Host)
		switch {
		case a.cfg.RedirectHosts[host]:
			http.Redirect(w, r, a.cfg.BaseURL+r.URL.RequestURI(), http.StatusPermanentRedirect)
		case host == apiHost && apiHost != siteHost:
			api.ServeHTTP(w, r)
		default:
			site.ServeHTTP(w, r)
		}
	}))
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sw := &statusWriter{ResponseWriter: w, status: 200}
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		t0 := time.Now()
		next.ServeHTTP(sw, r)
		if r.URL.Path != "/healthz" && r.URL.Path != "/readyz" && !strings.HasPrefix(r.URL.Path, "/static/") {
			slog.Info("http", "method", r.Method, "host", r.Host, "path", r.URL.Path, "status", sw.status, "ms", time.Since(t0).Milliseconds())
		}
	})
}

// --- season ---------------------------------------------------------------------------------

// announcement is what the server commits to about the next draw before its beacon value
// exists: when (and so from which beacon round), and from which crawl.
func (a *App) announcement(s *State) map[string]any {
	r := s.roundWithStatus("scheduled")
	if r == nil {
		return nil
	}
	return map[string]any{"id": roundID(a.season.ID, r.Idx), "draw_at": r.DrawAt, "drand_round": r.DrandRound, "crawl": r.Crawl}
}

func (a *App) apiSeason(w http.ResponseWriter, r *http.Request) {
	out := map[string]any{
		"season": a.season.ID, "title": a.season.Title, "digest": a.season.Digest, "harness": a.season.Harness,
		"rules_version": a.season.RulesVersion, "scale": a.cfg.Scale, "student": a.season.Scales[a.cfg.Scale],
		"ink_frac": a.season.InkFrac, "ink_tokens": a.season.InkTokens[a.cfg.Scale],
		"round_interval_seconds": int(a.cfg.RoundInterval.Seconds()), "draw_grid_seconds": a.season.DrawGrid,
		"qualify_rounds": a.cfg.QualifyRounds, "rated_rounds": a.cfg.RatedRounds, "ladder_size": a.cfg.LadderSize,
		"beacon": a.season.Beacon, "pools": a.season.Pools, "tolerance": a.tolerance(), "record": a.season.Record,
		"grade_devices": a.season.GradeDevices, "site": a.cfg.BaseURL, "api": a.cfg.APIURL,
		"requires_github": a.cfg.RequireGithub, "server": revision,
	}
	if name, version := a.site.wheel(); name != "" {
		out["cli"] = map[string]any{"version": version, "wheel": a.cfg.APIURL + "/files/" + name}
	}
	a.st.view(func(s *State) {
		out["next_round"] = a.announcement(s)
		if open := s.roundWithStatus("open"); open != nil {
			out["open_round"] = open.description(a.season)
		}
		if c := a.savedCrawl(s); c != nil {
			out["crawl"] = c
		}
	})
	writeJSON(w, 200, out)
}

// --- accounts -------------------------------------------------------------------------------

func accountJSON(c *Account) map[string]any {
	return map[string]any{"handle": c.Handle, "github": c.Github, "created_at": c.CreatedAt, "anchor": c.Anchor, "trusted": c.reliable()}
}

func (a *App) apiCreateAccount(w http.ResponseWriter, r *http.Request) {
	existing, key, body, err := a.signed(w, r, 1<<16, false)
	if err != nil {
		writeErr(w, err)
		return
	}
	if existing != nil {
		writeJSON(w, 200, map[string]any{"account": accountJSON(existing)})
		return
	}
	var req struct {
		Handle string `json:"handle"`
	}
	json.Unmarshal(body, &req)
	handle := strings.ToLower(strings.TrimSpace(req.Handle))
	if !handleRe.MatchString(handle) || strings.Contains(handle, "--") {
		writeErr(w, fail(400, "a handle is 3-24 lowercase letters, digits and single hyphens, starting and ending with a letter or digit"))
		return
	}
	if reservedHandles[handle] && !a.cfg.AnchorKeys[key] { // the organisers' own accounts hold these
		writeErr(w, fail(409, "that handle is reserved"))
		return
	}
	if !a.cfg.Dev && !a.limits.allow("acct:"+clientIP(r), 5, 86400) {
		writeErr(w, fail(429, "too many new accounts from this address today"))
		return
	}
	var out map[string]any
	status := 201
	err = a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		if acct := s.byPubkey[key]; acct != nil { // two requests raced; the first one won
			out, status = accountJSON(acct), 200
			return nil
		}
		if s.byHandle[handle] != nil {
			return fail(409, "the handle %q is taken", handle)
		}
		anchor := a.cfg.AnchorKeys[key]
		acct := &Account{ID: s.Meta.NextAccount, Handle: handle, Pubkey: key, CreatedAt: now(), Anchor: anchor}
		s.Meta.NextAccount++
		tx.put(metaKey, &s.Meta)
		tx.saveAccount(acct)
		out = accountJSON(acct)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, status, map[string]any{"account": out})
}

var gistRe = regexp.MustCompile(`([0-9a-f]{20,40})/?$`)

// githubLookup is how the server asks GitHub about a gist and its owner. Tests replace it.
var githubLookup = func(ctx context.Context, gistID string) (login string, id int64, created int64, content string, err error) {
	raw, err := httpGet(ctx, "https://api.github.com/gists/"+gistID, 4<<20)
	if err != nil {
		return "", 0, 0, "", err
	}
	var gist struct {
		Owner struct {
			Login string `json:"login"`
			ID    int64  `json:"id"`
		} `json:"owner"`
		Files map[string]struct {
			Content string `json:"content"`
		} `json:"files"`
	}
	if json.Unmarshal(raw, &gist) != nil || gist.Owner.Login == "" {
		return "", 0, 0, "", errors.New("the gist has no owner")
	}
	var all strings.Builder
	for _, f := range gist.Files {
		all.WriteString(f.Content)
		all.WriteString("\n")
	}
	raw, err = httpGet(ctx, "https://api.github.com/users/"+gist.Owner.Login, 1<<20)
	if err != nil {
		return "", 0, 0, "", err
	}
	var user struct {
		CreatedAt time.Time `json:"created_at"`
	}
	if json.Unmarshal(raw, &user) != nil || user.CreatedAt.IsZero() {
		return "", 0, 0, "", errors.New("could not read the GitHub account's age")
	}
	return gist.Owner.Login, gist.Owner.ID, user.CreatedAt.Unix(), all.String(), nil
}

// apiLinkGithub proves a GitHub identity without OAuth: the user publishes their Stet key
// in a gist, and the gist's owner is the proof.
func (a *App) apiLinkGithub(w http.ResponseWriter, r *http.Request) {
	acct, key, body, err := a.signed(w, r, 1<<16, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	var req struct {
		Gist string `json:"gist"`
	}
	json.Unmarshal(body, &req)
	m := gistRe.FindStringSubmatch(strings.TrimSpace(req.Gist))
	if m == nil {
		writeErr(w, fail(400, "that does not look like a gist URL"))
		return
	}
	if !a.limits.allow("gist:"+acct.Handle, 10, 3600) {
		writeErr(w, fail(429, "too many attempts; try again in an hour"))
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	login, ghID, created, content, err := githubLookup(ctx, m[1])
	if err != nil {
		writeErr(w, fail(502, "could not read that gist from GitHub; is it public?"))
		return
	}
	if !strings.Contains(content, "stet-key:"+key) {
		writeErr(w, fail(400, "the gist must contain the line stet-key:%s", key))
		return
	}
	var out map[string]any
	err = a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		me := s.Accounts[acct.ID]
		// One GitHub identity, one account: linking moves it to whoever proved it last.
		for _, other := range s.Accounts {
			if other.ID != me.ID && (strings.EqualFold(other.Github, login) || (ghID != 0 && other.GithubID == ghID)) {
				other.Github, other.GithubID, other.GithubCreatedAt = "", 0, 0
				tx.saveAccount(other)
			}
		}
		me.Github, me.GithubID, me.GithubCreatedAt = login, ghID, created
		tx.saveAccount(me)
		out = accountJSON(me)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"account": out})
}

func verifierStats(s *State, c *Account) map[string]any {
	recent := 0
	for _, r := range s.Results {
		if r.AccountID == c.ID && r.Verdict == "agreed" && r.ReceivedAt > now()-14*86400 {
			recent++
		}
	}
	return map[string]any{"validated": c.Agreed, "recent": recent, "streak": c.Streak, "trusted": c.reliable(), "strikes": c.Strikes, "gpu_hours": c.GPUSeconds / 3600, "priority": recent}
}

func (a *App) apiMe(w http.ResponseWriter, r *http.Request) {
	acct, _, _, err := a.signed(w, r, 1<<10, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	board := a.board()
	var out map[string]any
	a.st.view(func(s *State) {
		me := s.Accounts[acct.ID]
		editors := []map[string]any{}
		for _, e := range editorsOf(s, me.ID, 50) {
			j := a.editorJSON(s, e, board)
			es := s.ofEditor[e.ID]
			for i := len(es) - 1; i >= 0; i-- {
				if es[i].Status == "settled" {
					j["last"] = map[string]any{"round": roundID(a.season.ID, es[i].RoundIdx), "gain": es[i].Gain, "valid": es[i].Valid}
					break
				}
			}
			editors = append(editors, j)
		}
		out = map[string]any{"account": accountJSON(me), "verifier": verifierStats(s, me), "editors": editors, "next_round": a.announcement(s)}
	})
	writeJSON(w, 200, out)
}

// --- editors --------------------------------------------------------------------------------

func (a *App) editorJSON(s *State, e *Editor, b *Board) map[string]any {
	owner := s.owner(e)
	j := map[string]any{
		"id": e.ID, "name": e.Name, "version": e.Version, "owner": owner.Handle, "github": owner.Github, "description": e.Description,
		"technique": e.Technique, "parent": e.Parent, "status": e.Status, "created_at": e.CreatedAt, "size": e.Size, "public": e.Public,
		"rounds": 0, "url": a.cfg.BaseURL + "/editors/" + e.ID[:12],
	}
	if row := b.ByID[e.ID]; row != nil {
		j["rounds"], j["rating"], j["mean_gain"], j["se"] = row.Rounds, row.Rating, row.MeanGain, row.SE
		j["multiplier"], j["capped"], j["provisional"], j["record"], j["ladder"] = row.Multiplier, row.Capped, row.Provisional, row.Record, row.Ladder
		if row.Rank > 0 {
			j["rank"] = row.Rank
		}
	}
	return j
}

func (a *App) apiSubmit(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signed(w, r, maxBundleBytes+1024, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	sub, err := a.submit(r.Context(), acct.ID, body)
	if err != nil {
		writeErr(w, err)
		return
	}
	board := a.board()
	var out map[string]any
	a.st.view(func(s *State) {
		ed := s.Editors[sub.Editor.ID]
		out = map[string]any{"editor": a.editorJSON(s, ed, board), "url": a.cfg.BaseURL + "/editors/" + ed.ID[:12], "message": sub.Message}
		if sub.Replaced != "" {
			out["replaced"] = sub.Replaced
		}
		if next := s.roundWithStatus("scheduled"); next != nil {
			out["enters"] = map[string]any{"round": roundID(a.season.ID, next.Idx), "draw_at": next.DrawAt}
		}
	})
	writeJSON(w, 201, out)
}

// viewer returns the account behind an optionally signed GET, or nil.
func (a *App) viewer(r *http.Request) *Account {
	if r.Header.Get("X-Stet-Key") == "" {
		return nil
	}
	key, err := verifySignature(r, nil)
	if err != nil {
		return nil
	}
	return a.accountByKey(key)
}

// entryJSON is the public face of an entry. Sealed until settled: an open entry says
// nothing about any grade it may have.
func (a *App) entryJSON(s *State, e *Entry) map[string]any {
	ed, rd := s.Editors[e.EditorID], s.Rounds[e.RoundIdx]
	j := map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID, "name": ed.Name, "version": ed.Version,
		"owner": s.owner(ed).Handle, "status": e.Status, "stage": "editing"}
	if e.EditedAt > 0 {
		j["stage"] = "grading"
		j["edit"] = e.Edit
		if e.Script != "" {
			j["script"] = map[string]any{"sha256": e.Script, "size": e.ScriptSize, "url": a.cfg.APIURL + "/api/v1/scripts/" + e.Script}
		}
	}
	if e.Status != "settled" {
		return j
	}
	j["stage"] = "settled"
	j["valid"], j["results"], j["settled_at"] = e.Valid, e.NResults, e.SettledAt
	if rd.BaselineLoss != nil {
		j["baseline"] = *rd.BaselineLoss
	}
	if e.Valid && e.Loss != nil {
		j["loss"], j["gain"], j["by_source"] = *e.Loss, e.Gain, e.BySource
	} else {
		j["gain"], j["reason"] = 0.0, e.Reason
	}
	if e.Detail != nil {
		j["detail"] = e.Detail
	}
	return j
}

// entriesOfRound lists a round's entries, best first.
func entriesOfRound(s *State, idx int) []*Entry {
	es := append([]*Entry(nil), s.inRound[idx]...)
	sort.SliceStable(es, func(i, j int) bool {
		if (es[i].Status == "settled") != (es[j].Status == "settled") {
			return es[i].Status == "settled"
		}
		if es[i].Gain != es[j].Gain {
			return es[i].Gain > es[j].Gain
		}
		return es[i].EditorID < es[j].EditorID
	})
	return es
}

func (a *App) apiEditor(w http.ResponseWriter, r *http.Request) {
	viewer := a.viewer(r)
	board := a.board()
	var out map[string]any
	a.st.view(func(s *State) {
		e := findEditor(s, r.PathValue("ref"))
		if e == nil || (!e.Public && (viewer == nil || viewer.ID != e.AccountID)) {
			return
		}
		j := a.editorJSON(s, e, board)
		j["files"], j["entry"] = e.Files, e.Entry
		results := []map[string]any{}
		es := s.ofEditor[e.ID]
		for i := len(es) - 1; i >= 0; i-- {
			results = append(results, a.entryJSON(s, es[i]))
		}
		out = map[string]any{"editor": j, "results": results}
	})
	if out == nil {
		writeErr(w, fail(404, "no such editor (entries become public when they are first drawn into a round)"))
		return
	}
	writeJSON(w, 200, out)
}

func (a *App) apiBundle(w http.ResponseWriter, r *http.Request) {
	viewer := a.viewer(r)
	var id, name string
	a.st.view(func(s *State) {
		e := findEditor(s, r.PathValue("ref"))
		if e == nil || (!e.Public && (viewer == nil || viewer.ID != e.AccountID)) {
			return
		}
		id, name = e.ID, fmt.Sprintf("%s-v%d.tar.zst", e.Name, e.Version)
	})
	if id == "" {
		writeErr(w, fail(404, "no such editor"))
		return
	}
	blob, err := a.readBundle(r.Context(), id)
	if err != nil {
		writeErr(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
	w.Write(blob)
}

// apiGetScript serves an edit script: the plain-data record of what an editor did in a
// round, which is all anyone needs to grade it again.
func (a *App) apiGetScript(w http.ResponseWriter, r *http.Request) {
	sha := strings.ToLower(r.PathValue("sha"))
	blob, err := a.readScript(r.Context(), sha)
	if err != nil {
		writeErr(w, fail(404, "no such script"))
		return
	}
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	w.Header().Set("Content-Disposition", `attachment; filename="`+sha[:12]+`.script.jsonl.zst"`)
	w.Write(blob)
}

// --- rounds ---------------------------------------------------------------------------------

func (a *App) roundJSON(s *State, r *Round) map[string]any {
	j := r.description(a.season)
	j["entries"] = len(s.inRound[r.Idx])
	if r.Crawl != nil {
		j["crawl_id"] = r.Crawl.Crawl
	}
	if r.BaselineLoss != nil {
		j["baseline_loss"], j["baseline"], j["baseline_verifiers"] = *r.BaselineLoss, r.Baseline, r.BaselineN
	}
	if r.CorpusSHA != "" {
		j["corpus_sha256"], j["exam_sha256"] = r.CorpusSHA, r.ExamSHA
	}
	if r.SettledAt > 0 {
		j["settled_at"] = r.SettledAt
	}
	return j
}

func (a *App) apiRounds(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 || limit > 200 {
		limit = 30
	}
	var out map[string]any
	a.st.view(func(s *State) {
		rounds := []map[string]any{}
		for _, rd := range s.roundsDesc(limit) {
			rounds = append(rounds, a.roundJSON(s, rd))
		}
		out = map[string]any{"rounds": rounds, "scale": a.cfg.Scale, "next": a.announcement(s)}
	})
	writeJSON(w, 200, out)
}

var roundIDRe = regexp.MustCompile(`^([a-z0-9]+)-o(\d{1,9})$`)

func (a *App) parseRoundID(id string) (int, bool) {
	m := roundIDRe.FindStringSubmatch(id)
	if m == nil || m[1] != a.season.ID {
		return 0, false
	}
	idx, _ := strconv.Atoi(m[2])
	return idx, true
}

func (a *App) apiRound(w http.ResponseWriter, r *http.Request) {
	idx, ok := a.parseRoundID(r.PathValue("id"))
	var out map[string]any
	a.st.view(func(s *State) {
		rd := s.Rounds[idx]
		if !ok || rd == nil {
			return
		}
		entries := []map[string]any{}
		for _, e := range entriesOfRound(s, idx) {
			entries = append(entries, a.entryJSON(s, e))
		}
		out = map[string]any{"round": a.roundJSON(s, rd), "entries": entries}
	})
	if out == nil {
		writeErr(w, fail(404, "no such round"))
		return
	}
	writeJSON(w, 200, out)
}

// --- standings ------------------------------------------------------------------------------

func (a *App) apiLeaderboard(w http.ResponseWriter, r *http.Request) {
	b := a.board()
	rated, prov := b.Rated, b.Provisional
	if rated == nil {
		rated = []*BoardRow{}
	}
	if prov == nil {
		prov = []*BoardRow{}
	}
	writeJSON(w, 200, map[string]any{"scale": a.cfg.Scale, "editors": rated, "provisional": prov, "rated_rounds": a.cfg.RatedRounds})
}

type VerifierRow struct {
	Handle    string  `json:"handle"`
	Github    string  `json:"github,omitempty"`
	Validated int     `json:"validated"`
	Recent    int     `json:"recent"`
	Streak    int     `json:"streak"`
	Trusted   bool    `json:"trusted"`
	Anchor    bool    `json:"anchor,omitempty"`
	GPUHours  float64 `json:"gpu_hours"`
	GPU       string  `json:"gpu,omitempty"`
	LastSeen  int64   `json:"last_seen"`
}

// verifiers lists everyone who grades, most active in the last two weeks first.
func verifiers(s *State) []VerifierRow {
	since := now() - 14*86400
	recent := map[int64]int{}
	for _, r := range s.Results {
		if r.Verdict == "agreed" && r.ReceivedAt > since {
			recent[r.AccountID]++
		}
	}
	out := []VerifierRow{}
	for _, c := range s.Accounts {
		if c.Banned || (c.Agreed == 0 && c.LastSeen <= since) {
			continue
		}
		var h struct {
			GPU    string `json:"gpu"`
			Device string `json:"device"`
		}
		json.Unmarshal(c.Hardware, &h)
		if h.GPU == "" {
			h.GPU = h.Device
		}
		out = append(out, VerifierRow{Handle: c.Handle, Github: c.Github, Validated: c.Agreed, Recent: recent[c.ID], Streak: c.Streak,
			Trusted: c.reliable(), Anchor: c.Anchor, GPUHours: c.GPUSeconds / 3600, GPU: h.GPU, LastSeen: c.LastSeen})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Recent != out[j].Recent {
			return out[i].Recent > out[j].Recent
		}
		if out[i].Validated != out[j].Validated {
			return out[i].Validated > out[j].Validated
		}
		if out[i].LastSeen != out[j].LastSeen {
			return out[i].LastSeen > out[j].LastSeen
		}
		return out[i].Handle < out[j].Handle
	})
	if len(out) > 200 {
		out = out[:200]
	}
	return out
}

func (a *App) apiVerifiers(w http.ResponseWriter, r *http.Request) {
	var out []VerifierRow
	a.st.view(func(s *State) { out = verifiers(s) })
	writeJSON(w, 200, map[string]any{"verifiers": out})
}

type RecordRow struct {
	Record
	Name      string `json:"name"`
	Version   int    `json:"version"`
	Owner     string `json:"owner"`
	Technique string `json:"technique"`
	RoundID   string `json:"round_id"`
}

func (a *App) records(s *State) []RecordRow {
	out := []RecordRow{}
	for _, rec := range s.Records {
		ed := s.Editors[rec.EditorID]
		if ed == nil {
			continue
		}
		out = append(out, RecordRow{Record: *rec, Name: ed.Name, Version: ed.Version, Owner: s.owner(ed).Handle, Technique: ed.Technique, RoundID: roundID(a.season.ID, rec.RoundIdx)})
	}
	return out
}

func (a *App) apiRecords(w http.ResponseWriter, r *http.Request) {
	var out []RecordRow
	a.st.view(func(s *State) { out = a.records(s) })
	writeJSON(w, 200, map[string]any{"records": out})
}

// --- verifying ------------------------------------------------------------------------------

func (a *App) apiClaim(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signed(w, r, 1<<16, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	var h Hello
	if json.Unmarshal(body, &h) != nil {
		writeErr(w, fail(400, "malformed request"))
		return
	}
	job, err := a.claim(r.Context(), acct.ID, &h)
	if err != nil {
		writeErr(w, err)
		return
	}
	out := map[string]any{"job": job}
	a.st.view(func(s *State) { out["announced"] = a.announcement(s) })
	if job == nil {
		out["retry_after"] = 60
		out["message"] = "No work right now."
		if !acct.reliable() {
			out["message"] = "No work right now. A new verifier grades entries that a trusted verifier also grades; more will come with the next round."
		}
	}
	writeJSON(w, 200, out)
}

func (a *App) apiScript(w http.ResponseWriter, r *http.Request) {
	// base64 of a compressed script, plus the report around it
	acct, _, body, err := a.signed(w, r, a.season.Script.MaxBytes*4/3+(1<<20), true)
	if err != nil {
		writeErr(w, err)
		return
	}
	id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	var p ScriptPayload
	if json.Unmarshal(body, &p) != nil {
		writeErr(w, fail(400, "malformed edit report"))
		return
	}
	ack, err := a.submitScript(r.Context(), acct.ID, id, &p)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, ack)
}

func (a *App) apiResult(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signed(w, r, 1<<20, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	var p ResultPayload
	if json.Unmarshal(body, &p) != nil {
		writeErr(w, fail(400, "malformed result"))
		return
	}
	ack, err := a.submitResult(r.Context(), acct.ID, id, &p)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, ack)
}

func (a *App) apiRelease(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signed(w, r, 1<<16, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	id, _ := strconv.ParseInt(r.PathValue("id"), 10, 64)
	var req struct {
		Reason string `json:"reason"`
	}
	json.Unmarshal(body, &req)
	slog.Info("job released", "job", id, "verifier", acct.Handle, "reason", clip(req.Reason, 300))
	if err := a.release(r.Context(), acct.ID, id); err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"released": true})
}

// apiAdminDraw lets an anchor bring the next draw forward to the next moment on the grid
// instead of waiting for the schedule's.
func (a *App) apiAdminDraw(w http.ResponseWriter, r *http.Request) {
	acct, _, _, err := a.signed(w, r, 1<<10, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	if !acct.Anchor {
		writeErr(w, fail(403, "anchors only"))
		return
	}
	rd, err := a.bringForward(r.Context())
	if err != nil {
		writeErr(w, err)
		return
	}
	if a.cfg.Dev {
		ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		defer cancel()
		if err := a.maybeDraw(ctx, true); err != nil {
			writeErr(w, fail(409, "%s", err.Error()))
			return
		}
		a.st.view(func(s *State) { *rd = *s.Rounds[rd.Idx] })
	}
	writeJSON(w, 200, map[string]any{"round": rd.description(a.season)})
}
