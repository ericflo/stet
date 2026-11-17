package main

import (
	"bytes"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/parser"
	"github.com/yuin/goldmark/renderer/html"
)

//go:embed web/templates/*.html web/static
var webFS embed.FS

// Site serves the website, the downloads and the install scripts. Every page is rendered
// from the server's records on request; there is no client-side application.
type Site struct {
	app     *App
	pages   map[string]*template.Template
	md      goldmark.Markdown
	version string // cache-buster for static files

	docMu sync.Mutex
	docs  map[string]*docPage
}

type docSpec struct{ Slug, File, Title string }

// The handbook, in reading order. Files missing from the docs directory are left out.
var docSpecs = []docSpec{
	{"rules", "RULES.md", "Rules"},
	{"agents", "AGENTS.md", "For agents"},
	{"verifying", "VERIFYING.md", "Verifying"},
	{"design", "DESIGN.md", "Design"},
	{"experiments", "EXPERIMENTS.md", "Experiments"},
	{"api", "API.md", "API"},
	{"faq", "FAQ.md", "FAQ"},
}

type docPage struct {
	Spec    docSpec
	HTML    template.HTML
	ModTime time.Time
}

func newSite(a *App) (*Site, error) {
	s := &Site{app: a, pages: map[string]*template.Template{}, docs: map[string]*docPage{}}
	s.md = goldmark.New(
		goldmark.WithExtensions(extension.GFM),
		goldmark.WithParserOptions(parser.WithAutoHeadingID()),
		goldmark.WithRendererOptions(html.WithUnsafe()), // our own documents, not user input
	)
	css, _ := webFS.ReadFile("web/static/site.css")
	js, _ := webFS.ReadFile("web/static/site.js")
	s.version = fmt.Sprintf("%x", len(css)*31+len(js)*17+int(sumBytes(css)+sumBytes(js))%100000)
	names, err := fs.Glob(webFS, "web/templates/*.html")
	if err != nil {
		return nil, err
	}
	for _, n := range names {
		base := filepath.Base(n)
		if base == "layout.html" {
			continue
		}
		t, err := template.New("layout.html").Funcs(s.funcs()).ParseFS(webFS, "web/templates/layout.html", n)
		if err != nil {
			return nil, fmt.Errorf("template %s: %w", base, err)
		}
		s.pages[strings.TrimSuffix(base, ".html")] = t
	}
	return s, nil
}

func sumBytes(b []byte) (n int64) {
	for _, c := range b {
		n += int64(c)
	}
	return
}

// mountDownloads serves what the command line needs: on the API's host and the website's.
func (s *Site) mountDownloads(mux *http.ServeMux) {
	mux.HandleFunc("GET /files/{name}", s.file)
	mux.HandleFunc("GET /install.sh", s.installSh)
	mux.HandleFunc("GET /install.ps1", s.installPs1)
	mux.HandleFunc("GET /AGENTS.md", s.agentsMd)
}

func (s *Site) mount(mux *http.ServeMux) {
	static, _ := fs.Sub(webFS, "web/static")
	fileServer := http.FileServer(http.FS(static))
	mux.HandleFunc("GET /static/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		http.StripPrefix("/static/", fileServer).ServeHTTP(w, r)
	})
	mux.HandleFunc("GET /robots.txt", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "User-agent: *\nAllow: /\nDisallow: /api/\n")
	})
	mux.HandleFunc("GET /favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/static/favicon.svg", http.StatusMovedPermanently)
	})
	mux.HandleFunc("GET /{$}", s.home)
	mux.HandleFunc("GET /leaderboard", s.leaderboard)
	mux.HandleFunc("GET /editors/{ref}", s.editor)
	mux.HandleFunc("GET /rounds", s.rounds)
	mux.HandleFunc("GET /rounds/{id}", s.round)
	mux.HandleFunc("GET /verifiers", s.verifiersPage)
	mux.HandleFunc("GET /u/{handle}", s.account)
	mux.HandleFunc("GET /docs", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/docs/rules", http.StatusFound)
	})
	mux.HandleFunc("GET /docs/{slug}", s.doc)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { s.notFound(w, r, "") })
}

// --- rendering ------------------------------------------------------------------------------

type Page struct {
	Title    string
	Desc     string
	Nav      string
	Path     string
	Season   *Season
	Scale    Scale
	ScaleID  string
	BaseURL  string
	APIURL   string
	Version  string
	Dev      bool
	Now      int64
	Wheel    string
	CLI      string
	Docs     []docSpec
	Interval string
	Github   bool // entering needs a linked GitHub account
	D        any
}

func (s *Site) page(r *http.Request, title, nav string, d any) *Page {
	a := s.app
	wheel, version := s.wheel()
	return &Page{
		Title: title, Nav: nav, Path: r.URL.Path, Season: a.season, Scale: a.season.Scales[a.cfg.Scale], ScaleID: a.cfg.Scale,
		BaseURL: a.cfg.BaseURL, APIURL: a.cfg.APIURL, Version: s.version, Dev: a.cfg.Dev, Now: now(), Wheel: wheel, CLI: version,
		Docs: s.docList(), Interval: plainDuration(a.cfg.RoundInterval), Github: a.cfg.RequireGithub, D: d,
		Desc: "An open competition for agents that edit pretraining data. The model is fixed. The trainer is fixed. The only thing you can touch is the data.",
	}
}

func (s *Site) render(w http.ResponseWriter, status int, name string, p *Page) {
	t := s.pages[name]
	if t == nil {
		http.Error(w, "missing template "+name, 500)
		return
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, p); err != nil {
		slog.Error("render", "page", name, "err", err)
		http.Error(w, "something went wrong rendering this page", 500)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(status)
	w.Write(buf.Bytes())
}

func (s *Site) notFound(w http.ResponseWriter, r *http.Request, what string) {
	if what == "" {
		what = "There is nothing at this address."
	}
	s.render(w, 404, "error", s.page(r, "Not found", "", map[string]any{"Code": 404, "Message": what}))
}

func plainDuration(d time.Duration) string {
	switch {
	case d%(24*time.Hour) == 0 && d >= 24*time.Hour:
		n := int(d / (24 * time.Hour))
		if n == 1 {
			return "day"
		}
		return fmt.Sprintf("%d days", n)
	case d%time.Hour == 0 && d >= time.Hour:
		n := int(d / time.Hour)
		if n == 1 {
			return "hour"
		}
		return fmt.Sprintf("%d hours", n)
	default:
		return fmt.Sprintf("%d minutes", int(d/time.Minute))
	}
}

func ago(ts int64) string {
	if ts <= 0 {
		return "never"
	}
	d := now() - ts
	switch {
	case d < 0:
		return "in " + span(-d)
	case d < 45:
		return "just now"
	default:
		return span(d) + " ago"
	}
}

func span(d int64) string {
	switch {
	case d < 90:
		return fmt.Sprintf("%ds", d)
	case d < 90*60:
		return fmt.Sprintf("%dm", (d+30)/60)
	case d < 36*3600:
		return fmt.Sprintf("%dh", (d+1800)/3600)
	default:
		return fmt.Sprintf("%dd", (d+43200)/86400)
	}
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f kB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func comma(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	if neg {
		return "-" + s
	}
	return s
}

func compact(n int64) string {
	f := float64(n)
	switch {
	case n >= 1_000_000_000:
		return trimZero(fmt.Sprintf("%.1f", f/1e9)) + "B"
	case n >= 1_000_000:
		return trimZero(fmt.Sprintf("%.1f", f/1e6)) + "M"
	case n >= 10_000:
		return trimZero(fmt.Sprintf("%.0f", f/1e3)) + "k"
	}
	return comma(n)
}

func trimZero(s string) string { return strings.TrimSuffix(s, ".0") }

func signed(v float64, prec int) string {
	s := strconv.FormatFloat(v, 'f', prec, 64)
	if v > 0 {
		return "+" + s
	}
	if v < 0 {
		return "−" + s[1:]
	}
	return s
}

func deref(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

// multText writes a data multiplier: "1.42×", or ">8×" when the gain runs off the end of
// the measured curve.
func multText(p *float64, capped bool) string {
	if p == nil {
		return "—"
	}
	s := fmt.Sprintf("%.2f×", *p)
	if *p >= 10 || capped {
		s = trimZero(fmt.Sprintf("%.1f", *p)) + "×"
	}
	if capped {
		if *p <= 1 {
			return "<" + s
		}
		return ">" + s
	}
	return s
}

func (s *Site) funcs() template.FuncMap {
	return template.FuncMap{
		"ago":      ago,
		"span":     span,
		"bytes":    humanBytes,
		"comma":    func(n any) string { return comma(toInt(n)) },
		"compact":  func(n any) string { return compact(toInt(n)) },
		"short":    func(id string) string { return clipID(id, 12) },
		"tiny":     func(id string) string { return clipID(id, 8) },
		"date":     func(ts int64) string { return time.Unix(ts, 0).UTC().Format("2 Jan 2006") },
		"datetime": func(ts int64) string { return time.Unix(ts, 0).UTC().Format("2 Jan 2006, 15:04 UTC") },
		"iso":      func(ts int64) string { return time.Unix(ts, 0).UTC().Format(time.RFC3339) },
		"f1":       func(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) },
		"f2":       func(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) },
		"f3":       func(v float64) string { return strconv.FormatFloat(v, 'f', 3, 64) },
		"gain":     func(v float64) string { return signed(v, 3) },
		"ptr":      deref,
		"has":      func(p *float64) bool { return p != nil },
		"mult":     multText,
		"pct":      func(v float64) string { return fmt.Sprintf("%.0f%%", 100*v) },
		"pct1":     func(v float64) string { return trimZero(fmt.Sprintf("%.1f", 100*v)) + "%" },
		"spark":    sparkSVG,
		"add":      func(a, b int) int { return a + b },
		"lower":    strings.ToLower,
		"join":     strings.Join,
		"roundid":  func(idx int) string { return roundID(s.app.season.ID, idx) },
		"plural": func(n int, one, many string) string {
			if n == 1 {
				return one
			}
			return many
		},
		"gh":      func(login string) template.URL { return template.URL("https://github.com/" + login) },
		"minutes": func(seconds any) string { return trimZero(fmt.Sprintf("%.1f", float64(toInt(seconds))/60)) },
		"howlong": func(seconds any) string {
			n := toInt(seconds)
			switch {
			case n == 60:
				return "one minute"
			case n%60 == 0:
				return fmt.Sprintf("%d minutes", n/60)
			}
			return fmt.Sprintf("%d seconds", n)
		},
		"title": func(s string) string {
			if s == "" {
				return s
			}
			return strings.ToUpper(s[:1]) + s[1:]
		},
		"frac": func(done, total int) float64 {
			if total <= 0 {
				return 0
			}
			return 100 * float64(done) / float64(total)
		},
	}
}

func toInt(n any) int64 {
	switch v := n.(type) {
	case int:
		return int64(v)
	case int64:
		return v
	case float64:
		return int64(v)
	}
	return 0
}

func clipID(id string, n int) string {
	if len(id) > n {
		return id[:n]
	}
	return id
}

// --- views of the state ---------------------------------------------------------------------

// ScriptStats is what a grader reports about the edit script it trained on.
type ScriptStats struct {
	KeptDocFrac  float64 `json:"kept_doc_frac"`
	KeptByteFrac float64 `json:"kept_byte_frac"`
	InkFrac      float64 `json:"ink_frac"`
	InkTokens    int64   `json:"ink_tokens"`
	OutDocs      int64   `json:"out_docs"`
	SrcDocs      int64   `json:"src_docs"`
	Passes       float64 `json:"passes"`
	Truncated    bool    `json:"truncated"`
}

type EntryView struct {
	Round      string
	RoundIdx   int
	EditorID   string
	Name       string
	Version    int
	Owner      string
	Stage      string // editing | grading | settled
	Settled    bool
	Valid      bool
	Loss       float64
	Gain       float64
	Baseline   float64
	BySource   map[string]float64
	Reason     string
	Results    int
	SettledAt  int64
	Audited    bool
	Verifiers  []string
	Edit       *EditReport
	ScriptSHA  string
	ScriptSize int64
	HasStats   bool
	Script     ScriptStats
	Train      *TrainInfo
}

// entryView is the public face of an entry. Sealed until settled: an open entry says
// nothing about any grade it may have.
func (a *App) entryView(s *State, e *Entry) EntryView {
	ed := s.Editors[e.EditorID]
	v := EntryView{Round: roundID(a.season.ID, e.RoundIdx), RoundIdx: e.RoundIdx, EditorID: e.EditorID, Name: ed.Name, Version: ed.Version,
		Owner: s.owner(ed).Handle, Stage: "editing"}
	if e.EditedAt > 0 {
		v.Stage, v.Edit, v.ScriptSHA, v.ScriptSize = "grading", e.Edit, e.Script, e.ScriptSize
	}
	if e.Status != "settled" {
		return v
	}
	v.Stage, v.Settled, v.Valid, v.Gain, v.BySource = "settled", true, e.Valid, e.Gain, e.BySource
	v.Reason, v.Results, v.SettledAt = e.Reason, e.NResults, e.SettledAt
	v.Loss = deref(e.Loss)
	v.Baseline = deref(s.Rounds[e.RoundIdx].BaselineLoss)
	if d := e.Detail; d != nil {
		v.Audited, v.Verifiers, v.Train = d.Audited, d.Verifiers, d.Train
		v.HasStats = len(d.Script) > 0 && json.Unmarshal(d.Script, &v.Script) == nil
	}
	return v
}

type FeedItem struct {
	At   int64
	HTML template.HTML
}

func feed(s *State, limit int) []FeedItem {
	var out []FeedItem
	for i := len(s.Events) - 1; i >= 0 && len(out) < limit; i-- {
		ev := s.Events[i]
		d := ev.Data
		str := func(k string) string { v, _ := d[k].(string); return v }
		num := func(k string) float64 {
			switch v := d[k].(type) {
			case float64:
				return v
			case int:
				return float64(v)
			case int64:
				return float64(v)
			}
			return 0
		}
		var h string
		switch ev.Kind {
		case "submit":
			h = fmt.Sprintf(`<a href="/u/%s">%s</a> entered <b>%s</b> v%d`, esc(str("owner")), esc(str("owner")), esc(str("name")), int(num("version")))
		case "draw":
			h = fmt.Sprintf(`Round <a href="/rounds/%s">%s</a> drawn: %d %s on %s`, esc(str("round")), esc(str("round")), int(num("entries")),
				map[bool]string{true: "editor", false: "editors"}[int(num("entries")) == 1], esc(str("crawl")))
		case "settle":
			h = fmt.Sprintf(`Round <a href="/rounds/%s">%s</a> settled; best gain <span class="num">%s</span>`, esc(str("round")), esc(str("round")), signed(num("best_gain"), 3))
		case "record":
			h = fmt.Sprintf(`<span class="tag red">record</span> <a href="/editors/%s"><b>%s</b></a> by %s took the record at <span class="num">%s</span> nats`,
				esc(clipID(str("editor"), 12)), esc(str("name")), esc(str("owner")), signed(num("rating"), 3))
		case "crawl":
			h = fmt.Sprintf(`New crawl in play: <b>%s</b> (%s files)`, esc(str("crawl")), comma(int64(num("files"))))
		default:
			continue
		}
		out = append(out, FeedItem{At: ev.At, HTML: template.HTML(h)})
	}
	return out
}

// wheel is the newest CLI wheel in the files directory, and its version.
func (s *Site) wheel() (name, version string) {
	names, _ := filepath.Glob(filepath.Join(s.app.cfg.FilesDir, "stet_cli-*.whl"))
	for _, n := range names {
		base := filepath.Base(n)
		parts := strings.Split(base, "-")
		if len(parts) < 2 {
			continue
		}
		if name == "" || versionLess(version, parts[1]) {
			name, version = base, parts[1]
		}
	}
	return
}

type roundSummary struct {
	Round
	ID       string
	Entries  int
	Edited   int
	Settled  int
	BestGain *float64
	BestName string
	BestID   string
}

func (a *App) roundSummary(s *State, r *Round) roundSummary {
	rs := roundSummary{Round: *r, ID: roundID(a.season.ID, r.Idx)}
	for _, e := range s.inRound[r.Idx] {
		rs.Entries++
		if e.EditedAt > 0 || e.Status == "settled" {
			rs.Edited++
		}
		if e.Status != "settled" {
			continue
		}
		rs.Settled++
		if r.Status == "settled" && e.Valid && (rs.BestGain == nil || e.Gain > *rs.BestGain) {
			g := e.Gain
			rs.BestGain, rs.BestName, rs.BestID = &g, s.Editors[e.EditorID].Name, e.EditorID
		}
	}
	return rs
}

func (a *App) roundSummaries(s *State, limit int) []roundSummary {
	var out []roundSummary
	for _, r := range s.roundsDesc(limit) {
		out = append(out, a.roundSummary(s, r))
	}
	return out
}

// --- pages ----------------------------------------------------------------------------------

func (s *Site) home(w http.ResponseWriter, r *http.Request) {
	a := s.app
	b := a.board()
	top, provisional := b.Rated, false
	if len(top) == 0 && len(b.Provisional) > 0 {
		top, provisional = b.Provisional, true
	}
	if len(top) > 8 {
		top = top[:8]
	}
	d := map[string]any{"Top": top, "Provisional": provisional, "TrustAfter": trustAfterAgreements}
	a.st.view(func(st *State) {
		recs := a.records(st)
		if len(recs) > 0 {
			d["Record"] = &recs[len(recs)-1]
		}
		d["Records"], d["RecordChart"] = recs, recordChart(recs)
		vs := verifiers(st)
		active := 0
		var gpuHours float64
		for _, v := range vs {
			if v.LastSeen > now()-14*86400 {
				active++
			}
			gpuHours += v.GPUHours
		}
		if len(vs) > 5 {
			vs = vs[:5]
		}
		d["Verifiers"], d["Active"], d["GPUHours"] = vs, active, gpuHours
		settled := 0
		for _, rs := range a.roundSummaries(st, 0) {
			if rs.Status != "settled" {
				continue
			}
			settled++
			if d["Latest"] == nil && rs.BestGain != nil && rs.BaselineLoss != nil {
				rs := rs
				d["Latest"] = &rs
			}
		}
		d["Rounds"] = settled
		if next := st.roundWithStatus("scheduled"); next != nil {
			c := *next
			d["Next"] = &c
		}
		if open := st.roundWithStatus("open"); open != nil {
			rs := a.roundSummary(st, open)
			d["Open"] = &rs
		}
		public, queued := 0, 0
		for _, e := range st.Editors {
			switch {
			case e.Public:
				public++
			case e.Status == "queued":
				queued++
			}
		}
		d["Editors"], d["Queued"], d["Feed"] = public, queued, feed(st, 9)
		if c := a.savedCrawl(st); c != nil {
			d["Crawl"] = *c
		}
	})
	s.render(w, 200, "home", s.page(r, "", "home", d))
}

func (s *Site) leaderboard(w http.ResponseWriter, r *http.Request) {
	a := s.app
	b := a.board()
	var recs []RecordRow
	a.st.view(func(st *State) { recs = a.records(st) })
	chart := recordChart(recs)
	sort.Slice(recs, func(i, j int) bool { return recs[i].N > recs[j].N })
	_, hasCurve := a.season.Curves[a.curveKey()]
	s.render(w, 200, "leaderboard", s.page(r, "Leaderboard", "leaderboard", map[string]any{
		"Rated": b.Rated, "Provisional": b.Provisional, "Records": recs, "RecordChart": chart, "RatedRounds": a.cfg.RatedRounds,
		"QualifyRounds": a.cfg.QualifyRounds, "LadderSize": a.cfg.LadderSize, "HasCurve": hasCurve, "Rule": a.season.Record,
		"Margin": a.season.Record.Margin[a.cfg.Scale],
	}))
}

var fileRefRe = regexp.MustCompile(`^[A-Za-z0-9._/@+ -]{1,300}$`)

type editorLink struct {
	ID      string
	Name    string
	Version int
	Owner   string
	Status  string
	At      int64
}

func linkTo(s *State, e *Editor) editorLink {
	return editorLink{ID: e.ID, Name: e.Name, Version: e.Version, Owner: s.owner(e).Handle, Status: e.Status, At: e.CreatedAt}
}

func (s *Site) editor(w http.ResponseWriter, r *http.Request) {
	a := s.app
	b := a.board()
	var e *Editor
	d := map[string]any{}
	owner, github := "", ""
	a.st.view(func(st *State) {
		found := findEditor(st, r.PathValue("ref"))
		if found == nil || !found.Public {
			return
		}
		c := *found
		e = &c
		owner, github = st.owner(found).Handle, st.owner(found).Github
		var results []EntryView
		es := st.ofEditor[e.ID]
		for i := len(es) - 1; i >= 0; i-- {
			results = append(results, a.entryView(st, es[i]))
		}
		d["Results"], d["GainChart"] = results, gainChart(results)
		for i := range results {
			if results[i].Settled && results[i].Valid {
				latest := results[i]
				d["Latest"] = &latest
				d["SourceChart"] = sourceChart(a.season.ExamSources, st.Rounds[latest.RoundIdx].Baseline, latest.BySource)
				if latest.Train != nil {
					d["CurveChart"] = curveChart(latest.Train.Curve)
				}
				break
			}
		}
		if p := st.Editors[e.Parent]; p != nil && p.Public {
			l := linkTo(st, p)
			d["Parent"] = &l
		}
		var forks, versions []editorLink
		for _, o := range st.Editors {
			if o.Parent == e.ID && o.Public {
				forks = append(forks, linkTo(st, o))
			}
		}
		sort.Slice(forks, func(i, j int) bool { return forks[i].At > forks[j].At })
		if len(forks) > 20 {
			forks = forks[:20]
		}
		for _, o := range st.ownedBy[e.AccountID] {
			if o.Name == e.Name && o.Public {
				versions = append(versions, linkTo(st, o))
			}
		}
		sort.Slice(versions, func(i, j int) bool { return versions[i].Version > versions[j].Version })
		d["Forks"], d["Versions"] = forks, versions
	})
	if e == nil {
		s.notFound(w, r, "No such editor. Entries become public when they are first drawn into a round.")
		return
	}
	d["E"], d["Owner"], d["Github"], d["Row"] = e, owner, github, b.ByID[e.ID]

	// Source viewer: one file at a time, read out of the stored bundle.
	want := r.URL.Query().Get("file")
	if want == "" {
		for _, pref := range []string{"editor.py", "stet.toml"} {
			for _, f := range e.Files {
				if f.Path == pref && want == "" {
					want = pref
				}
			}
		}
		if want == "" && len(e.Files) > 0 {
			want = e.Files[0].Path
		}
	}
	if want != "" && fileRefRe.MatchString(want) {
		if blob, err := a.readBundle(r.Context(), e.ID); err == nil {
			if text, binary, truncated, err := bundleFile(blob, want); err == nil {
				d["File"], d["FileText"], d["FileBinary"], d["FileTruncated"] = want, text, binary, truncated
			}
		}
	}
	p := s.page(r, fmt.Sprintf("%s v%d", e.Name, e.Version), "leaderboard", d)
	if e.Technique != "" {
		p.Desc = fmt.Sprintf("%s v%d by %s: %s", e.Name, e.Version, owner, e.Technique)
	}
	s.render(w, 200, "editor", p)
}

func (s *Site) rounds(w http.ResponseWriter, r *http.Request) {
	a := s.app
	var rounds []roundSummary
	a.st.view(func(st *State) { rounds = a.roundSummaries(st, 100) })
	s.render(w, 200, "rounds", s.page(r, "Rounds", "rounds", map[string]any{"Rounds": rounds, "MaxNew": a.cfg.MaxNewPerRound}))
}

func (s *Site) round(w http.ResponseWriter, r *http.Request) {
	a := s.app
	idx, ok := a.parseRoundID(r.PathValue("id"))
	var rs *roundSummary
	var entries []EntryView
	a.st.view(func(st *State) {
		rd := st.Rounds[idx]
		if !ok || rd == nil {
			return
		}
		sum := a.roundSummary(st, rd)
		rs = &sum
		for _, e := range entriesOfRound(st, idx) {
			entries = append(entries, a.entryView(st, e))
		}
	})
	if rs == nil {
		s.notFound(w, r, "No such round.")
		return
	}
	sc := a.season.Scales[rs.Scale]
	d := map[string]any{"R": rs, "ID": rs.ID, "Entries": entries, "Scale": sc, "Tolerance": a.tolerance()}
	if rs.Status != "scheduled" && rs.Crawl != nil {
		if seed, err := hex.DecodeString(rs.Seed); err == nil && rs.Crawl.NShards > 0 {
			want := sc.CorpusUnits * a.season.Pools["official"].ShardsPerUnit
			if want > rs.Crawl.NShards {
				want = rs.Crawl.NShards
			}
			d["Shards"] = sampleWithoutReplacement(seed, "shard", rs.Crawl.NShards, want)
		}
		relay := "https://api.drand.sh"
		if len(a.season.Beacon.Relays) > 0 {
			relay = a.season.Beacon.Relays[0]
		}
		d["BeaconURL"] = template.URL(fmt.Sprintf("%s/%s/public/%d", relay, a.season.Beacon.ChainHash, rs.DrandRound))
		d["SeedInput"] = fmt.Sprintf("stet|%s|official|%d|%d|%s|%s|%s", a.season.ID, rs.Idx, rs.DrandRound, rs.Randomness, rs.Crawl.Crawl, rs.Crawl.PathsSHA256)
	}
	if slot, onGrid := a.season.drawSlot(rs.DrandRound); onGrid {
		d["Slot"] = slot
	}
	if rs.Baseline != nil {
		d["BaselineChart"] = baselineChart(a.season.ExamSources, rs.Baseline)
	}
	s.render(w, 200, "round", s.page(r, "Round "+rs.ID, "rounds", d))
}

func (s *Site) verifiersPage(w http.ResponseWriter, r *http.Request) {
	a := s.app
	var vs []VerifierRow
	a.st.view(func(st *State) { vs = verifiers(st) })
	var hours float64
	var validated int
	for _, v := range vs {
		hours += v.GPUHours
		validated += v.Validated
	}
	s.render(w, 200, "verifiers", s.page(r, "Verifiers", "verifiers", map[string]any{
		"Verifiers": vs, "Hours": hours, "Validated": validated, "TrustAfter": trustAfterAgreements, "BanAfter": banAfterStrikes,
		"Tolerance": a.tolerance(), "Devices": a.season.GradeDevices,
	}))
}

func (s *Site) account(w http.ResponseWriter, r *http.Request) {
	a := s.app
	b := a.board()
	type row struct {
		E   editorLink
		Row *BoardRow
	}
	var acct *Account
	var eds []row
	var stats map[string]any
	a.st.view(func(st *State) {
		found := st.byHandle[strings.ToLower(r.PathValue("handle"))]
		if found == nil {
			return
		}
		c := *found
		acct, stats = &c, verifierStats(st, found)
		for _, e := range editorsOf(st, found.ID, 100) {
			if e.Public {
				eds = append(eds, row{linkTo(st, e), b.ByID[e.ID]})
			}
		}
	})
	if acct == nil {
		s.notFound(w, r, "No such account.")
		return
	}
	s.render(w, 200, "account", s.page(r, acct.Handle, "", map[string]any{"A": acct, "Editors": eds, "V": stats}))
}

// --- documents ------------------------------------------------------------------------------

// docPath is where a handbook page's Markdown lives.
func (s *Site) docPath(file string) string {
	if file == "AGENTS.md" && s.app.cfg.AgentsFile != "" {
		return s.app.cfg.AgentsFile
	}
	return filepath.Join(s.app.cfg.DocsDir, file)
}

func (s *Site) docList() []docSpec {
	var out []docSpec
	for _, d := range docSpecs {
		if _, err := os.Stat(s.docPath(d.File)); err == nil {
			out = append(out, d)
		}
	}
	return out
}

func (s *Site) loadDoc(spec docSpec) (*docPage, error) {
	p := s.docPath(spec.File)
	st, err := os.Stat(p)
	if err != nil {
		return nil, err
	}
	s.docMu.Lock()
	defer s.docMu.Unlock()
	if d := s.docs[spec.Slug]; d != nil && d.ModTime.Equal(st.ModTime()) {
		return d, nil
	}
	src, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := s.md.Convert(src, &buf); err != nil {
		return nil, err
	}
	out := strings.ReplaceAll(buf.String(), "<table>", `<div class="table-wrap"><table>`)
	out = strings.ReplaceAll(out, "</table>", "</table></div>")
	d := &docPage{Spec: spec, HTML: template.HTML(out), ModTime: st.ModTime()}
	s.docs[spec.Slug] = d
	return d, nil
}

func (s *Site) doc(w http.ResponseWriter, r *http.Request) {
	slug := r.PathValue("slug")
	for _, spec := range docSpecs {
		if spec.Slug != slug {
			continue
		}
		d, err := s.loadDoc(spec)
		if err != nil {
			break
		}
		s.render(w, 200, "doc", s.page(r, spec.Title, "docs", map[string]any{"Doc": d, "Slug": slug}))
		return
	}
	s.notFound(w, r, "No such page in the handbook.")
}

func (s *Site) agentsMd(w http.ResponseWriter, r *http.Request) {
	b, err := os.ReadFile(s.docPath("AGENTS.md"))
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=300")
	w.Write(b)
}

// --- downloads ------------------------------------------------------------------------------

var fileNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]{0,200}$`)

// file serves the CLI wheel from the files directory, and anything else (the exam pack)
// from the bucket, under files/.
func (s *Site) file(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !fileNameRe.MatchString(name) {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Cache-Control", "public, max-age=3600")
	p := filepath.Join(s.app.cfg.FilesDir, name)
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		http.ServeFile(w, r, p)
		return
	}
	body, size, err := s.app.st.blobs.Open(r.Context(), "files/"+name)
	if err != nil {
		if err != errNoBlob {
			slog.Error("file", "name", name, "err", err)
		}
		http.NotFound(w, r)
		return
	}
	defer body.Close()
	w.Header().Set("Content-Type", "application/octet-stream")
	if size > 0 {
		w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	}
	if r.Method != http.MethodHead {
		io.Copy(w, body)
	}
}

const installShTmpl = `#!/bin/sh
# Installs the Stet command line. Usage:  curl -fsSL {{.Base}}/install.sh | sh
set -eu
WHEEL="{{.Files}}/files/{{.Wheel}}"
SPEC="stet-cli @ $WHEEL"

say() { printf '%s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  if command -v pipx >/dev/null 2>&1; then
    say "Installing stet with pipx..."
    pipx install --force "$WHEEL"
    pipx inject stet-cli torch >/dev/null 2>&1 || say "Could not add PyTorch; see https://pytorch.org/get-started/locally/"
    say ""
    say "Done. Next:  stet doctor"
    exit 0
  fi
  say "Stet installs with uv (https://docs.astral.sh/uv/). Install uv first:"
  say ""
  say "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  say ""
  say "then run this installer again."
  exit 1
fi

say "Installing stet with uv..."
# PyTorch does the training. On Linux the default build targets NVIDIA; on a Mac it uses
# Metal. For AMD (ROCm) or anything unusual, set STET_TORCH_INDEX to the right wheel index
# from https://pytorch.org/get-started/locally/ before running this.
if [ -n "${STET_TORCH_INDEX:-}" ]; then
  uv tool install --force --python ">=3.11" --with torch --index "$STET_TORCH_INDEX" --index-strategy unsafe-best-match "$SPEC"
elif [ "${STET_NO_TORCH:-}" = "1" ]; then
  uv tool install --force --python ">=3.11" "$SPEC"
else
  uv tool install --force --python ">=3.11" --with torch "$SPEC"
fi

say ""
say "Done. Next:"
say "  stet doctor              check this machine"
say "  stet init my-editor      start an editor"
say "  stet verify              lend your GPU to the leaderboard"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) say ""; say "If 'stet' is not found, run:  uv tool update-shell" ;;
esac
`

const installPs1Tmpl = `# Installs the Stet command line. Usage:  irm {{.Base}}/install.ps1 | iex
$ErrorActionPreference = "Stop"
$wheel = "{{.Files}}/files/{{.Wheel}}"
$spec = "stet-cli @ $wheel"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Stet installs with uv (https://docs.astral.sh/uv/). Install uv first:"
  Write-Host ""
  Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
  Write-Host ""
  Write-Host "then run this installer again."
  return
}

Write-Host "Installing stet with uv..."
# On Windows the default PyTorch build is CPU-only; with an NVIDIA card, use the CUDA build.
$index = $env:STET_TORCH_INDEX
if (-not $index -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  $index = "https://download.pytorch.org/whl/cu124"
}
if ($index) {
  uv tool install --force --python ">=3.11" --with torch --index $index --index-strategy unsafe-best-match $spec
} else {
  uv tool install --force --python ">=3.11" --with torch $spec
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  stet doctor              check this machine"
Write-Host "  stet init my-editor      start an editor"
Write-Host "  stet verify              lend your GPU to the leaderboard"
`

func (s *Site) installScript(w http.ResponseWriter, r *http.Request, tmpl string) {
	wheel, _ := s.wheel()
	if wheel == "" {
		http.Error(w, "echo 'The Stet CLI is not published on this server yet.'; exit 1\n", 503)
		return
	}
	out := strings.NewReplacer("{{.Base}}", s.app.cfg.BaseURL, "{{.Files}}", s.app.cfg.APIURL, "{{.Wheel}}", wheel).Replace(tmpl)
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=300")
	w.Write([]byte(out))
}

func (s *Site) installSh(w http.ResponseWriter, r *http.Request) {
	s.installScript(w, r, installShTmpl)
}

func (s *Site) installPs1(w http.ResponseWriter, r *http.Request) {
	s.installScript(w, r, installPs1Tmpl)
}
