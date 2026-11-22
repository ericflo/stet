package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/netip"
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
	return a.signedBy(w, r, limit, mustExist, anyone)
}

// Who a request must come from.
const (
	anyone     = iota
	runners    // established accounts: the ones that are handed editors to run
	organisers // the organisers' accounts
)

// A body larger than this is an upload: only so many are held in memory at once.
const uploadBytes = 2 << 20

// signedBy is signed, with the option of admitting only runners or only organisers.
//
// An upload is proved before any of it is read. The client sends the hash of the body it
// signed, so the signature can be checked from the headers alone; only then is the upload
// counted against the account and given one of the few places for large bodies. A key is
// public, and naming someone else's must cost them nothing. The place is kept until the
// handler has finished with the body, and runners have places of their own, so that no
// amount of entrants' traffic keeps a run's script from arriving.
func (a *App) signedBy(w http.ResponseWriter, r *http.Request, limit int64, mustExist bool, who int) (*Account, string, []byte, error) {
	key, err := claimedKey(r)
	if err != nil {
		return nil, "", nil, fail(401, "%s", err.Error())
	}
	acct := a.accountByKey(key)
	if acct == nil && (mustExist || who != anyone) {
		return nil, "", nil, fail(401, "no account for this key; run `stet login`")
	}
	switch {
	case who == organisers && !acct.administers():
		return nil, "", nil, fail(403, "organisers only")
	case who == runners && !a.established(acct):
		return nil, "", nil, fail(403, "curators are run by established verifiers only")
	}
	sum, proved := strings.ToLower(r.Header.Get("X-Stet-Body")), false
	if limit > uploadBytes {
		if !sha256Re.MatchString(sum) {
			return nil, "", nil, fail(426, "this copy of stet is out of date: it cannot send an upload the way the server now needs. Upgrade: curl -fsSL %s/install.sh | sh", a.cfg.BaseURL)
		}
		if _, err := verifySigned(r, sum); err != nil {
			return nil, "", nil, fail(401, "%s", err.Error())
		}
		proved = true
		// The only upload that is not a runner's or an organiser's is an entry, so whoever
		// may not enter has no call on a place either.
		if who == anyone && acct != nil {
			if err := a.mayEnter(acct); err != nil {
				return nil, "", nil, err
			}
		}
		if !a.limits.allow("upload:"+key, 40, 3600) {
			return nil, "", nil, fail(429, "that is a great many uploads in an hour; try again later")
		}
		pool := a.uploads
		if who != anyone {
			pool = a.runnerUploads
		}
		select {
		case pool <- struct{}{}:
			context.AfterFunc(r.Context(), func() { <-pool }) // when the handler returns
		default:
			return nil, "", nil, fail(503, "the server is busy with other uploads; try again in a moment")
		}
	}
	body, err := readBody(w, r, limit)
	if err != nil {
		return nil, "", nil, err
	}
	got := sha256Hex(body)
	if sum != "" && sum != got {
		return nil, "", nil, fail(400, "the body that arrived is not the one that was signed")
	}
	if !proved {
		if _, err := verifySigned(r, got); err != nil {
			return nil, "", nil, fail(401, "%s", err.Error())
		}
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
	// Bounded, without forgetting who is over a limit: drop the keys with nothing recent
	// first, and only then, if a flood of new keys has filled it, the rest.
	if len(l.hits) > 50000 {
		for k, hs := range l.hits {
			if len(hs) == 0 || t-hs[len(hs)-1] > 3600 {
				delete(l.hits, k)
			}
		}
		for k := range l.hits {
			if len(l.hits) <= 40000 {
				break
			}
			if k != key {
				delete(l.hits, k)
			}
		}
	}
	return true
}

// clientIP is the address the proxy in front of the server saw the request come from: the
// last entry of X-Forwarded-For, which the proxy appends. Everything to its left is
// whatever the client chose to send. An IPv6 address counts as its /64, which is what one
// subscriber is given.
func clientIP(r *http.Request) string {
	ip, _, _ := net.SplitHostPort(r.RemoteAddr)
	// A proxy may append to the client's header line or add a line of its own; either
	// way its entry is the last one of all.
	if xff := strings.Join(r.Header.Values("X-Forwarded-For"), ","); xff != "" {
		parts := strings.Split(xff, ",")
		ip = strings.TrimSpace(parts[len(parts)-1])
	}
	if addr, err := netip.ParseAddr(ip); err == nil && addr.Is6() && !addr.Is4In6() {
		if p, err := addr.Prefix(64); err == nil {
			return p.String()
		}
	}
	return ip
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
	mux.HandleFunc("GET /api/v1/rounds/{id}/exam", a.apiExamGet)
	mux.HandleFunc("GET /api/v1/rounds/{id}/scores", a.apiScoresGet)
	mux.HandleFunc("POST /api/v1/admin/rounds/{id}/exam", a.apiExamPost)
	mux.HandleFunc("GET /api/v1/leaderboard", a.apiLeaderboard)
	mux.HandleFunc("GET /api/v1/verifiers", a.apiVerifiers)
	mux.HandleFunc("GET /api/v1/records", a.apiRecords)
	mux.HandleFunc("GET /api/v1/scripts/{sha}", a.apiGetScript)
	mux.HandleFunc("POST /api/v1/jobs/claim", a.apiClaim)
	mux.HandleFunc("POST /api/v1/jobs/{id}/script", a.apiScript)
	mux.HandleFunc("POST /api/v1/jobs/{id}/result", a.apiResult)
	mux.HandleFunc("POST /api/v1/jobs/{id}/release", a.apiRelease)
	mux.HandleFunc("POST /api/v1/admin/draw", a.apiAdminDraw)
	mux.HandleFunc("POST /api/v1/admin/pause", a.apiAdminPause)
	mux.HandleFunc("POST /api/v1/admin/farm/batches", a.apiFarmBatch)
	mux.HandleFunc("GET /api/v1/admin/farm/batches/{id}", a.apiFarmBatchGet)
	mux.HandleFunc("POST /api/v1/farm/claim", a.apiFarmClaim)
	mux.HandleFunc("GET /api/v1/farm/batches/{id}/items", a.apiFarmItems)
	mux.HandleFunc("POST /api/v1/farm/batches/{id}/tasks/{i}", a.apiFarmResult)
	mux.HandleFunc("POST /api/v1/farm/batches/{id}/tasks/{i}/release", a.apiFarmRelease)
	mux.HandleFunc("PUT /api/v1/admin/farm/scripts/{sha}", a.apiFarmScriptPut)
	mux.HandleFunc("GET /api/v1/farm/scripts/{sha}", a.apiFarmScript)
	mux.HandleFunc("GET /api/v1/admin/accounts/{handle}", a.apiAdminAccount)
	mux.HandleFunc("GET /api/v1/admin/work", a.apiAdminWork)
	mux.HandleFunc("GET /api/v1/admin/workers", a.apiWorkers)
	mux.HandleFunc("POST /api/v1/admin/harvests", a.apiHarvestPost)
	mux.HandleFunc("GET /api/v1/harvests", a.apiHarvests)
	mux.HandleFunc("POST /api/v1/admin/workers", a.apiWorkerGrant)
	mux.HandleFunc("DELETE /api/v1/admin/workers/{key}", a.apiWorkerRevoke)
	mux.HandleFunc("POST /api/v1/admin/workers/{key}/release", a.apiWorkerRelease)
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
		case r.URL.Path == "/healthz" || r.URL.Path == "/readyz":
			site.ServeHTTP(w, r)
		case a.cfg.RedirectHosts[host]:
			http.Redirect(w, r, a.cfg.BaseURL+r.URL.RequestURI(), http.StatusPermanentRedirect)
		case !strings.HasPrefix(r.URL.Path, "/static/") && !a.st.ready():
			// The last change could not be stored and has not been read back, so what is in
			// memory may never have happened. Say so rather than show it.
			w.Header().Set("Retry-After", "20")
			if host == apiHost && apiHost != siteHost || strings.HasPrefix(r.URL.Path, "/api/") {
				writeJSON(w, 503, map[string]string{"error": errUnavailable.Message})
			} else {
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				w.WriteHeader(503)
				io.WriteString(w, unavailablePage)
			}
		case host == apiHost && apiHost != siteHost:
			api.ServeHTTP(w, r)
		default:
			site.ServeHTTP(w, r)
		}
	}))
}

// unavailablePage is what the site says while the server cannot vouch for what it remembers.
const unavailablePage = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Stet · back in a moment</title>
<body style="margin:0;background:#f2eee2;color:#1d1b16;font:19px/1.6 Georgia,serif">
<div style="max-width:560px;margin:18vh auto;padding:0 24px"><h1 style="font-size:40px;margin:0 0 12px">Back in a moment.</h1>
<p>Stet cannot reach its records right now, and will not show you numbers it cannot vouch for. Nothing is lost. This page tries again every twenty seconds.</p></div>
`

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
	if r == nil || s.Meta.Paused != 0 {
		return nil // while the rounds are paused, none is to be drawn
	}
	return map[string]any{"id": roundID(a.season.ID, r.Idx), "draw_at": r.DrawAt, "drand_round": r.DrandRound, "crawl": r.Crawl}
}

func (a *App) apiSeason(w http.ResponseWriter, r *http.Request) {
	out := map[string]any{
		"season": a.season.ID, "title": a.season.Title, "digest": a.season.Digest, "harness": a.season.Harness,
		"rules_version": a.season.RulesVersion, "scale": a.cfg.Scale, "student": a.season.Scales[a.cfg.Scale],
		"ink_frac": a.season.InkFrac, "ink_tokens": a.season.InkTokens[a.cfg.Scale],
		"round_interval_seconds": int(a.cfg.RoundInterval.Seconds()), "draw_grid_seconds": a.season.DrawGrid,
		"round_min_gap_seconds": int(a.cfg.RoundMinGap.Seconds()), "entries_in_play": a.cfg.EntriesInPlay, "submissions_per_day": a.cfg.SubmitsPerDay,
		"qualify_rounds": a.cfg.QualifyRounds, "rated_rounds": a.cfg.RatedRounds, "ladder_size": a.cfg.LadderSize,
		"beacon": a.season.Beacon, "pools": a.season.Pools, "tolerance": a.tolerance(), "record": a.season.Record,
		"grade_devices": a.season.GradeDevices, "site": a.cfg.BaseURL, "api": a.cfg.APIURL,
		"requires_github": a.cfg.RequireGithub, "server": revision,
		"establish_after": a.cfg.EstablishAfter, "runs_per_entry": a.cfg.RunsPerEntry,
		// What a score is: the unit gains and ratings are in, the exam's pillars (each counting
		// the same: an entry's by_source is its gain on each), and whether a rating has a
		// multiplier at all. Each round's exam is made after its draw: see the round.
		"unit": unit, "pillars": a.season.ExamSources, "has_multiplier": a.hasMultiplier(),
		// What an editor is promised, whoever runs it: its machine, the slowest machine that
		// may run it, how much it may name by reference, and what it may import.
		"machine": a.season.Machine, "runner": a.season.Runner, "assets": a.season.Assets,
		"environment": map[string]any{"python": a.season.Environment.Python, "packages": a.season.Environment.Packages},
	}
	if name, version := a.site.wheel(); name != "" {
		out["cli"] = map[string]any{"version": version, "wheel": a.cfg.APIURL + "/files/" + name}
	}
	a.st.view(func(s *State) {
		out["next_round"] = a.announcement(s)
		if s.Meta.Paused != 0 {
			out["paused"] = s.Meta.Paused
		}
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

// accountJSON is an account as the API shows it. "anchor" and "trusted" both say that it is
// one of the organisers' accounts; "established" says that its grades count towards scores
// and that it may run editors, which the organisers' accounts do from the start and anyone
// else earns.
func (a *App) accountJSON(c *Account) map[string]any {
	out := map[string]any{"handle": c.Handle, "github": c.Github, "created_at": c.CreatedAt, "anchor": c.Anchor, "trusted": c.Anchor,
		"established": a.established(c)}
	if c.Worker {
		out["worker"] = true // a machine the organisers rent: it administers nothing
	}
	return out
}

func (a *App) apiCreateAccount(w http.ResponseWriter, r *http.Request) {
	existing, key, body, err := a.signed(w, r, 1<<16, false)
	if err != nil {
		writeErr(w, err)
		return
	}
	if existing != nil {
		writeJSON(w, 200, map[string]any{"account": a.accountJSON(existing)})
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
	// The organisers' keys, and those they vouch for, cost nothing to let through: a pool of
	// rented machines' accounts is made from one address in a day.
	var anchor, worker bool
	a.st.view(func(s *State) { anchor, worker = s.organiser(a.cfg.AnchorKeys, key) })
	if !a.cfg.Dev && !anchor && !worker && !a.limits.allow("acct:"+clientIP(r), 5, 86400) {
		writeErr(w, fail(429, "too many new accounts from this address today"))
		return
	}
	var out map[string]any
	status := 201
	err = a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		if acct := s.byPubkey[key]; acct != nil { // two requests raced; the first one won
			out, status = a.accountJSON(acct), 200
			return nil
		}
		if s.byHandle[handle] != nil {
			return fail(409, "the handle %q is taken", handle)
		}
		anchor, worker := s.organiser(a.cfg.AnchorKeys, key)
		acct := &Account{ID: s.Meta.NextAccount, Handle: handle, Pubkey: key, CreatedAt: now(), Anchor: anchor, Worker: worker}
		s.Meta.NextAccount++
		tx.put(metaKey, &s.Meta)
		tx.saveAccount(acct)
		out = a.accountJSON(acct)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, status, map[string]any{"account": out})
}

var gistRe = regexp.MustCompile(`([0-9a-f]{20,40})/?$`)

// githubToken, when set, goes with the server's questions to GitHub. Without one GitHub
// answers 60 an hour for the server's whole address, which is 30 links.
var githubToken string

// githubLookup is how the server asks GitHub about a gist and its owner. Tests replace it.
var githubLookup = func(ctx context.Context, gistID string) (login string, id int64, created int64, content string, err error) {
	raw, err := httpGetAs(ctx, "https://api.github.com/gists/"+gistID, githubToken, 4<<20)
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
	raw, err = httpGetAs(ctx, "https://api.github.com/users/"+gist.Owner.Login, githubToken, 1<<20)
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
	// Per account and per address: accounts are free, and GitHub answers this server only
	// so many times an hour for everybody put together.
	if !a.limits.allow("gist:"+acct.Handle, 10, 3600) || !a.limits.allow("gistip:"+clientIP(r), 10, 3600) {
		writeErr(w, fail(429, "too many attempts; try again in an hour"))
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	login, ghID, created, content, err := githubLookup(ctx, m[1])
	if errors.Is(err, errBusy) {
		writeErr(w, fail(503, "GitHub is limiting how often this server may ask it things. Nothing is wrong with your gist; try again in an hour"))
		return
	}
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
		// One GitHub identity, one account: linking moves it to whoever proved it last, and
		// the entries it was vouching for leave play with it. Otherwise one person could
		// keep an entry in play under every key they cared to make.
		for _, other := range s.Accounts {
			if other.ID != me.ID && (strings.EqualFold(other.Github, login) || (ghID != 0 && other.GithubID == ghID)) {
				other.Github, other.GithubID, other.GithubCreatedAt = "", 0, 0
				tx.saveAccount(other)
				for _, e := range s.ownedBy[other.ID] {
					if e.Status == "queued" || e.Status == "running" {
						e.Status = "withdrawn"
						tx.saveEditor(e)
					}
				}
			}
		}
		me.Github, me.GithubID, me.GithubCreatedAt = login, ghID, created
		tx.saveAccount(me)
		out = a.accountJSON(me)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"account": out})
}

// verifierStats is an account's standing as a verifier. An account is established when
// "validated" has reached "establish_after", it is not muted, and (where entering needs a
// GitHub identity) it has one: "needs_github" says that is what is missing.
func (a *App) verifierStats(s *State, c *Account) map[string]any {
	recent, agreed, streak := recentAgreed(s)[c.ID], c.Agreed, c.Streak
	if c.Anchor {
		// The organisers' accounts start out established, and their grades earn them
		// nothing. (An account that became one may carry counts from before.)
		recent, agreed, streak = 0, 0, 0
	}
	return map[string]any{"validated": agreed, "recent": recent, "streak": streak, "trusted": c.Anchor, "established": a.established(c),
		"establish_after": a.cfg.EstablishAfter, "needs_github": !c.Anchor && a.mayEnter(c) != nil, "runs": c.Runs, "strikes": c.Strikes,
		"muted": muted(c, now()), "gpu_hours": c.GPUSeconds / 3600, "priority": recentWork(s)[c.ID]}
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
				if !sealed(s, es[i]) {
					last := map[string]any{"round": roundID(a.season.ID, es[i].RoundIdx), "gain": places5(es[i].Gain), "valid": es[i].Valid}
					if as, ok := countsAs(s, es[i]); ok {
						last["counts_as"] = places5(as)
					}
					j["last"] = last
					break
				}
			}
			editors = append(editors, j)
		}
		out = map[string]any{"account": a.accountJSON(me), "verifier": a.verifierStats(s, me), "editors": editors, "next_round": a.announcement(s)}
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
		j["rounds"], j["rating"], j["mean_gain"], j["se"] = row.Rounds, places5p(row.Rating), places5p(row.MeanGain), places5p(row.SE)
		j["adjusted"], j["lower"] = places5p(row.Adjusted), places5p(row.Lower)
		j["multiplier"], j["capped"], j["provisional"], j["record"], j["ladder"] = row.Multiplier, row.Capped, row.Provisional, row.Record, row.Ladder
		if row.Tied {
			j["tied"] = true
		}
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
		if sub.InPlay > 0 {
			out["in_play"], out["in_play_limit"] = sub.InPlay, a.cfg.EntriesInPlay
		}
		if next := s.roundWithStatus("scheduled"); next != nil {
			// "It enters round X" only when it will: there is room for it, and it is in
			// time. Otherwise X is the earliest it could be.
			ahead := 0
			for _, e := range s.Editors {
				if e.Status == "queued" && e.ID != ed.ID && e.CreatedAt <= ed.CreatedAt {
					ahead++
				}
			}
			key := "earliest"
			if ed.Status == "queued" && ahead < a.cfg.MaxNewPerRound && ed.CreatedAt <= next.DrawAt-a.freeze() {
				key = "enters"
			}
			out[key] = map[string]any{"round": roundID(a.season.ID, next.Idx), "draw_at": next.DrawAt}
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

// places5 is a number as the API prints it: scores are measured to five places, and
// arithmetic on them should not print as 0.9124699999999999.
func places5(v float64) float64 { return math.Round(v*1e5) / 1e5 }

func places5p(v *float64) *float64 {
	if v == nil {
		return nil
	}
	r := places5(*v)
	return &r
}

// RunView is one run of an entry as the API and the site show it, once it is public.
type RunView struct {
	N           int      `json:"n"`
	By          string   `json:"by"`
	GPU         string   `json:"gpu,omitempty"`
	Backend     string   `json:"backend,omitempty"`
	EditSeconds float64  `json:"edit_seconds"`
	Sandbox     string   `json:"sandbox,omitempty"` // bwrap, or container on the organisers' rented machines
	Status      string   `json:"status"`            // ok | failed; or dropped: nobody could check it, and it counts for nothing
	Reason      string   `json:"reason,omitempty"`
	Script      string   `json:"script,omitempty"`
	ScriptSize  int64    `json:"script_size,omitempty"`
	Gain        float64  `json:"gain"`
	Loss        *float64 `json:"loss,omitempty"` // the unedited student's bits per byte less the gain
	Self        bool     `json:"self,omitempty"` // its only established grade is its own runner's
	Grades      []Grade  `json:"grades"`
}

// runViews lists a settled entry's runs, with every grade of each. An open entry has none
// to show: runs are sealed with the grades.
func (a *App) runViews(s *State, e *Entry) []RunView {
	out := []RunView{}
	rd := s.Rounds[e.RoundIdx]
	if e.Status != "settled" || rd == nil || rd.Status != "settled" { // an entry given up early does not unseal anything
		return out
	}
	handle := func(id int64) string {
		if acct := s.Accounts[id]; acct != nil {
			return acct.Handle
		}
		return "unknown"
	}
	lossOf := func(gain float64) *float64 {
		if rd == nil || rd.Status != "settled" || rd.BaselineLoss == nil {
			return nil
		}
		loss := places5(*rd.BaselineLoss - gain)
		return &loss
	}
	for _, run := range e.Runs {
		v := RunView{N: run.N, By: handle(run.By), GPU: run.GPU, Backend: run.Backend, Status: "failed", Reason: run.Reason,
			Script: run.Script, ScriptSize: run.ScriptSize, Gain: places5(run.Gain), Self: run.Self, Grades: run.Grades}
		if run.Edit != nil {
			v.EditSeconds, v.Sandbox = run.Edit.Seconds, run.Edit.Sandbox
		}
		switch {
		case run.Dropped:
			v.Status = "dropped"
		case run.Valid:
			v.Status, v.Loss = "ok", lossOf(run.Gain)
		}
		if v.Grades == nil {
			v.Grades = []Grade{}
		}
		out = append(out, v)
	}
	return out
}

// sealed reports whether an entry is still under seal: it is open, or its round is. A
// round settles as a whole. An entry that was given up early (nobody could run it, or
// check its run) is closed in the records before its round is, and says nothing until then.
func sealed(s *State, e *Entry) bool {
	rd := s.Rounds[e.RoundIdx]
	return e.Status != "settled" || rd == nil || rd.Status != "settled"
}

// entryJSON is the public face of an entry. Sealed until its round settles: before that it
// says nothing about its runs, the scripts they wrote, any grade of them, or whether it
// has been given up.
func (a *App) entryJSON(s *State, e *Entry) map[string]any {
	ed, rd := s.Editors[e.EditorID], s.Rounds[e.RoundIdx]
	j := map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID, "name": ed.Name, "version": ed.Version,
		"owner": s.owner(ed).Handle, "status": e.Status, "stage": "editing"}
	if len(e.Runs) > 0 || e.EditedAt > 0 {
		j["stage"] = "grading"
	}
	if sealed(s, e) {
		j["status"] = "open"
		return j
	}
	j["stage"] = "settled"
	// The run the score is nearest to stands for the entry where one script is wanted.
	if e.EditedAt > 0 {
		j["edit"] = e.Edit
		if e.Script != "" {
			j["script"] = map[string]any{"sha256": e.Script, "size": e.ScriptSize, "url": a.cfg.APIURL + "/api/v1/scripts/" + e.Script}
		}
	}
	// The runs the score is made of, and apart from them any that nobody could check.
	runs, dropped := []RunView{}, []RunView{}
	for _, run := range a.runViews(s, e) {
		if run.Status == "dropped" {
			dropped = append(dropped, run)
		} else {
			runs = append(runs, run)
		}
	}
	j["runs"], j["n_runs"], j["confirmed"], j["runs_differ"] = runs, len(runs), e.Confirmed, e.RunsDiffer
	if e.OneGrader {
		j["one_grader"] = true
	}
	if e.OneRunner {
		j["one_runner"] = true
	}
	if len(dropped) > 0 {
		j["dropped_runs"] = dropped
	}
	j["valid"], j["results"], j["settled_at"] = e.Valid, e.NResults, e.SettledAt
	if rd.BaselineLoss != nil && rd.Status == "settled" { // an entry settled invalid does not unseal the round
		j["baseline"] = *rd.BaselineLoss
	}
	if e.Valid && e.Loss != nil {
		j["loss"], j["gain"], j["by_source"] = *e.Loss, places5(e.Gain), e.BySource
	} else {
		j["gain"], j["reason"] = 0.0, e.Reason
		if e.Void {
			j["void"] = true
		}
		// It scored nothing, and what the round counts as for it is not nothing: the lowest
		// gain of the round's valid entries, or zero if that is lower.
		if as, ok := countsAs(s, e); ok {
			j["counts_as"] = places5(as)
		}
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
		// (An entry given up before its round settles is in no different a place from the
		// rest: where it stood would say that it had been.)
		if x, y := !sealed(s, es[i]), !sealed(s, es[j]); x != y {
			return x
		}
		if sealed(s, es[i]) {
			return es[i].EditorID < es[j].EditorID
		}
		// An entry that failed is below every entry that did not, whatever those scored.
		if es[i].Valid != es[j].Valid {
			return es[i].Valid
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
		e := findEditor(s, r.PathValue("ref"), viewerID(viewer))
		if e == nil {
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
		writeErr(w, fail(404, "no such curator (entries become public when they are first drawn into a round)"))
		return
	}
	writeJSON(w, 200, out)
}

func (a *App) apiBundle(w http.ResponseWriter, r *http.Request) {
	viewer := a.viewer(r)
	var id, name string
	a.st.view(func(s *State) {
		e := findEditor(s, r.PathValue("ref"), viewerID(viewer))
		if e == nil {
			return
		}
		id, name = e.ID, fmt.Sprintf("%s-v%d.tar.zst", e.Name, e.Version)
	})
	if id == "" {
		writeErr(w, fail(404, "no such curator"))
		return
	}
	// Streamed from storage: a slow reader holds a connection, not the bundle in memory.
	f, size, err := a.st.blobs.Open(r.Context(), bundleKey(id))
	if err != nil {
		writeErr(w, err)
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
	io.Copy(w, io.LimitReader(f, maxBundleBytes))
}

func viewerID(viewer *Account) int64 {
	if viewer == nil {
		return 0
	}
	return viewer.ID
}

// scriptAccess says who may read a script. Once a round that holds it has settled it is
// public: the plain record of what an editor did, which is all anyone needs to grade it
// again. Until then it is sealed, and goes only to an account that holds it as a grade job
// (one under way, or one it has reported). A script no entry's run wrote is nobody's to read.
func scriptAccess(s *State, sha string, viewer *Account) (known, public, holder bool) {
	for _, e := range s.Entries {
		has := e.Script == sha
		for _, run := range e.Runs {
			has = has || run.Script == sha
		}
		if !has {
			continue
		}
		known = true
		if rd := s.Rounds[e.RoundIdx]; rd != nil && rd.Status == "settled" {
			public = true
		}
	}
	if viewer != nil {
		for _, c := range s.Claims {
			if c.AccountID == viewer.ID && c.Kind == "grade" && c.Script == sha && (c.live(now()) || c.Status == "done") {
				holder = true
			}
		}
	}
	return known, public, holder
}

// apiGetScript serves an edit script: to anyone once its round has settled, and before
// that only to a signed request from the account that holds it as a grade job.
func (a *App) apiGetScript(w http.ResponseWriter, r *http.Request) {
	sha := strings.ToLower(r.PathValue("sha"))
	if !sha256Re.MatchString(sha) {
		writeErr(w, fail(404, "no such script"))
		return
	}
	viewer := a.viewer(r)
	var known, public, holder bool
	a.st.view(func(s *State) { known, public, holder = scriptAccess(s, sha, viewer) })
	switch {
	case !known:
		writeErr(w, fail(404, "no such script"))
		return
	case !public && !holder:
		w.Header().Set("Cache-Control", "no-store")
		writeErr(w, fail(403, "this script's round has not settled. Until it does the script is sealed: it is served only to the verifier that holds it as a grade job, in a signed request. (A verifier that is told this about its own job is out of date. Upgrade: curl -fsSL %s/install.sh | sh)", a.cfg.BaseURL))
		return
	}
	// Streamed from storage, like a bundle: a script may be tens of megabytes, and every
	// volunteer starting on a round asks for the same ones at once.
	f, size, err := a.st.blobs.Open(r.Context(), scriptKey(sha))
	if errors.Is(err, errNoBlob) {
		writeErr(w, fail(404, "no such script"))
		return
	}
	if err != nil {
		slog.Error("reading a script", "sha", sha, "err", err)
		writeErr(w, fail(503, "the file store is not answering; try again in a moment"))
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", "application/zstd")
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	if public {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		w.Header().Set("Cache-Control", "private, no-store") // sealed: for this account, for now
	}
	w.Header().Set("Content-Disposition", `attachment; filename="`+sha[:12]+`.script.jsonl.zst"`)
	// All of it. What is stored is the compressed script, and a script that does not
	// compress is a little larger stored than raw: cut at the raw limit, it could never be
	// downloaded whole, and so never graded. Its size was bounded when it was uploaded.
	io.Copy(w, f)
}

// --- rounds ---------------------------------------------------------------------------------

func (a *App) roundJSON(s *State, r *Round) map[string]any {
	j := r.description(a.season)
	j["entries"] = len(s.inRound[r.Idx])
	if r.Crawl != nil {
		j["crawl_id"] = r.Crawl.Crawl
	}
	// Sealed with the grades: a published baseline would tell a grader what to report.
	if r.BaselineLoss != nil && r.Status == "settled" {
		j["baseline_loss"], j["baseline_verifiers"] = *r.BaselineLoss, r.BaselineN
	}
	if len(r.Differs) > 0 && r.Status == "settled" {
		j["differs"] = r.Differs // verifiers turned away: another corpus or exam, or another unedited baseline
	}
	if r.CorpusSHA != "" {
		j["corpus_sha256"], j["exam_sha256"] = r.CorpusSHA, r.ExamSHA
	}
	if r.SettledAt > 0 {
		j["settled_at"] = r.SettledAt
	}
	// Every item score behind the round's scores, once it has been put where it is served from.
	if scoresArchived(r) {
		j["scores"] = a.cfg.APIURL + "/api/v1/rounds/" + roundID(a.season.ID, r.Idx) + "/scores"
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
		// Before the draw a round has no entries. Who its draw would enter now is not one.
		if rd.Status == "scheduled" {
			in, later := a.expected(s, rd)
			out["expected"], out["later"] = append([]expectedRow{}, in...), append([]expectedRow{}, later...)
		}
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
	// The same numbers as every other route prints, to the same five places.
	printed := func(rows []*BoardRow) []*BoardRow {
		out := make([]*BoardRow, len(rows))
		for i, r := range rows {
			c := *r
			c.Rating, c.MeanGain, c.SE, c.Multiplier = places5p(r.Rating), places5p(r.MeanGain), places5p(r.SE), places5p(r.Multiplier)
			c.Adjusted, c.Lower = places5p(r.Adjusted), places5p(r.Lower)
			out[i] = &c
		}
		return out
	}
	writeJSON(w, 200, map[string]any{"scale": a.cfg.Scale, "editors": printed(b.Rated), "provisional": printed(b.Provisional), "rated_rounds": a.cfg.RatedRounds})
}

// gpuNameRe is what a GPU's name looks like. The name is whatever the client sent, and it
// is shown on the site.
var gpuNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 ()\[\]._+/@:-]{0,59}$`)

type VerifierRow struct {
	Handle      string  `json:"handle"`
	Github      string  `json:"github,omitempty"`
	Validated   int     `json:"validated"`
	Recent      int     `json:"recent"`
	Streak      int     `json:"streak"`
	Runs        int     `json:"runs"`             // editors it has run
	Established bool    `json:"established"`      // its grades count towards scores, and it may run editors
	Trusted     bool    `json:"trusted"`          // one of the organisers' accounts
	Anchor      bool    `json:"anchor,omitempty"` // the same, under its older name
	Worker      bool    `json:"worker,omitempty"` // one of the machines the organisers rent
	GPUHours    float64 `json:"gpu_hours"`
	GPU         string  `json:"gpu,omitempty"`
	LastSeen    int64   `json:"last_seen"`
}

// verifiers lists the organisers' accounts and every volunteer with a grade that has agreed
// or an editor run, most active in the last two weeks first. Asking for work puts nobody
// on it.
func (a *App) verifiers(s *State) []VerifierRow {
	recent := recentAgreed(s)
	out := []VerifierRow{}
	for _, c := range s.Accounts {
		// A rented machine is listed once it has been at work: the organisers may keep many
		// keys for them, and most are not in use at any one time.
		if muted(c, now()) || (c.Agreed == 0 && c.Runs == 0 && !c.Anchor) || (c.Worker && c.LastSeen == 0) {
			continue
		}
		row := VerifierRow{Handle: c.Handle, Github: c.Github, Validated: c.Agreed, Recent: recent[c.ID], Streak: c.Streak, Runs: c.Runs,
			Established: a.established(c), Trusted: c.Anchor, Anchor: c.Anchor, Worker: c.Worker, GPUHours: c.GPUSeconds / 3600,
			GPU: readHardware(c.Hardware).gpuName(), LastSeen: c.LastSeen}
		if c.Anchor { // established from the start: its grades earn it nothing
			row.Validated, row.Recent, row.Streak = 0, 0, 0
		}
		out = append(out, row)
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
	a.st.view(func(s *State) { out = a.verifiers(s) })
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
	handed, err := a.claim(r.Context(), acct.ID, &h)
	if err != nil {
		writeErr(w, err)
		return
	}
	out := map[string]any{"job": handed.Job}
	if handed.Runner != nil {
		out["runner"] = handed.Runner // whether this machine is handed editors to run, and if not, why
	}
	open := false
	a.st.view(func(s *State) { out["announced"], open = a.announcement(s), len(openRounds(s)) > 0 })
	if handed.Job == nil {
		out["retry_after"] = 60
		if handed.Retry > 0 {
			out["retry_after"] = min(60, handed.Retry)
		}
		out["message"] = "No work right now. Curators to run and scripts to grade appear when a round is drawn."
		if open {
			// A verifier is offered each job once, never its own entry, and never a script
			// that it alone wrote: with a round open, that is why there is nothing for it.
			out["message"] = "Nothing in the open round is this machine's to do right now: it is waiting on other verifiers."
		}
		if handed.Note != "" {
			out["message"] = handed.Note
		}
	}
	writeJSON(w, 200, out)
}

func (a *App) apiScript(w http.ResponseWriter, r *http.Request) {
	// base64 of a compressed script, plus the report around it
	acct, _, body, err := a.signedBy(w, r, a.season.maxScriptBlob(), true, runners)
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
	if err := a.release(r.Context(), acct.ID, id, req.Reason); err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"released": true})
}

// apiAdminAccount tells an organiser an account's public key. That is what the list of
// organisers' keys is made of, so adding a machine to it needs nothing carried from that
// machine by hand.
func (a *App) apiAdminAccount(w http.ResponseWriter, r *http.Request) {
	if _, _, _, err := a.signedBy(w, r, 1<<10, true, organisers); err != nil {
		writeErr(w, err)
		return
	}
	handle := strings.ToLower(r.PathValue("handle"))
	var out map[string]any
	a.st.view(func(s *State) {
		if acct := s.byHandle[handle]; acct != nil {
			out = map[string]any{"handle": acct.Handle, "key": acct.Pubkey, "anchor": acct.Anchor, "worker": acct.Worker, "github": acct.Github, "created_at": acct.CreatedAt}
		}
	})
	if out == nil {
		writeErr(w, fail(404, "no account has that handle"))
		return
	}
	writeJSON(w, 200, out)
}

// apiAdminDraw lets an organiser bring the next draw forward to the next moment on the grid
// instead of waiting for the schedule's.
// apiAdminPause pauses the rounds, or resumes them: while they are paused no round is drawn,
// and the machines have nothing official to do (the round under way finishes). For the
// organisers, between seasons or while they work on the next one.
func (a *App) apiAdminPause(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signed(w, r, 1<<10, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	if !acct.administers() {
		writeErr(w, fail(403, "organisers only"))
		return
	}
	var req struct {
		Paused *bool `json:"paused"`
	}
	if json.Unmarshal(body, &req) != nil || req.Paused == nil {
		writeErr(w, fail(400, `send {"paused": true} or {"paused": false}`))
		return
	}
	var out Meta
	err = a.st.update(r.Context(), func(tx *Tx) error {
		m := &tx.s.Meta
		switch {
		case *req.Paused && m.Paused == 0:
			m.Paused, m.PausedBy = now(), acct.Handle
			tx.event("pause", map[string]any{"by": acct.Handle})
		case !*req.Paused && m.Paused != 0:
			m.Paused, m.PausedBy = 0, ""
			tx.event("resume", map[string]any{"by": acct.Handle})
		}
		tx.put(metaKey, m)
		out = *m
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	slog.Info("rounds paused or resumed", "paused", out.Paused != 0, "by", acct.Handle)
	writeJSON(w, 200, map[string]any{"paused": out.Paused, "paused_by": out.PausedBy})
}

func (a *App) apiAdminDraw(w http.ResponseWriter, r *http.Request) {
	acct, _, _, err := a.signed(w, r, 1<<10, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	if !acct.administers() {
		writeErr(w, fail(403, "organisers only"))
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
