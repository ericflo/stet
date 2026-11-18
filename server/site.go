package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log/slog"
	"math"
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

	srcMu   sync.Mutex
	src     map[string]sourceView
	srcGate chan struct{} // bundles being unpacked for the source viewer at one time
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
	s := &Site{app: a, pages: map[string]*template.Template{}, docs: map[string]*docPage{}, src: map[string]sourceView{}, srcGate: make(chan struct{}, 2)}
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
	mux.HandleFunc("GET /constraints.txt", s.constraints)
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
	mux.HandleFunc("GET /editors/{handle}/{name}", s.editorByName)
	mux.HandleFunc("GET /rounds", s.rounds)
	mux.HandleFunc("GET /rounds/{id}", s.round)
	mux.HandleFunc("GET /verifiers", s.verifiersPage)
	mux.HandleFunc("GET /enter", s.guide("enter", "Enter the competition", "You compete by writing a curator: a program that edits pretraining data. Test it on your own machine as often as you like, and enter it to be scored on volunteers' GPUs and ranked on the leaderboard."))
	mux.HandleFunc("GET /verify", s.guide("verify", "Verify scores", "Lend a GPU to the leaderboard: one command retrains a fixed model on entrants' edit scripts. No one else's code runs on your machine, and it keeps itself up to date."))
	mux.HandleFunc("GET /run", s.guide("run", "Run entries", "Run entrants' curators in a sandbox on a Linux machine with a GPU. What it needs, what the sandbox does, and how to leave it running."))
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
		Desc: "Stet is an open competition to make the best training data for a language model. You enter a curator: a program that cuts, keeps and reorders raw web crawl. The model, the trainer and the budget are fixed. The only thing you can touch is the data.",
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
// gpuTime writes an amount of GPU time in the unit that suits its size.
func gpuTime(hours float64) [2]string {
	switch minutes := math.Round(hours * 60); {
	case minutes < 1:
		return [2]string{"0", "minutes"}
	case minutes == 1:
		return [2]string{"1", "minute"}
	case minutes < 90:
		return [2]string{strconv.Itoa(int(minutes)), "minutes"}
	case hours < 100:
		return [2]string{strconv.FormatFloat(hours, 'f', 1, 64), "hours"}
	}
	return [2]string{comma(int64(math.Round(hours))), "hours"}
}

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
		"f0":       func(v float64) string { return trimZero(strconv.FormatFloat(v, 'f', 1, 64)) },
		"f1":       func(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) },
		"gpunum":   func(hours float64) string { return gpuTime(hours)[0] },
		"gpuunit":  func(hours float64) string { return gpuTime(hours)[1] },
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
	Void       bool // nobody could run it, or check its run; the round does not count for it
	Results    int
	SettledAt  int64
	Replicas   int         // graders, beyond the first, whose grades agreed
	Disputed   int         // and those whose grade did not
	Together   bool        // two established graders disagreed, with nobody to say which was right
	Verifiers  []string    // those whose grades are the score first
	Runs       []RunView   // every run of it, with every grade of each
	NRuns      int         // those that count
	Confirmed  bool        // two people's runs agree, and none rests on its own runner's grade
	RunsDiffer bool        // no two runs agree
	OneGrader  bool        // the runs agree, and one person graded them all
	OneRunner  bool        // the runs agree, and one person's machines made them all
	Own        bool        // a run's gain rests on a grade by its own runner
	Edit       *EditReport // the run the score is nearest to: its edit, its script
	ScriptSHA  string
	ScriptSize int64
	HasStats   bool
	Script     ScriptStats
	Train      *TrainInfo
}

// entryView is the public face of an entry. Sealed until settled: an open entry says
// nothing about its runs, the scripts they wrote, or any grade of them.
func (a *App) entryView(s *State, e *Entry) EntryView {
	ed := s.Editors[e.EditorID]
	v := EntryView{Round: roundID(a.season.ID, e.RoundIdx), RoundIdx: e.RoundIdx, EditorID: e.EditorID, Name: ed.Name, Version: ed.Version,
		Owner: s.owner(ed).Handle, Stage: "editing"}
	if len(e.Runs) > 0 || e.EditedAt > 0 {
		v.Stage = "grading"
	}
	if sealed(s, e) { // until its round settles: not even whether it has been given up
		return v
	}
	v.Stage, v.Settled, v.Valid, v.Gain, v.BySource = "settled", true, e.Valid, e.Gain, e.BySource
	v.Reason, v.Results, v.SettledAt, v.Void = e.Reason, e.NResults, e.SettledAt, e.Void
	v.Loss = deref(e.Loss)
	if e.EditedAt > 0 {
		v.Edit, v.ScriptSHA, v.ScriptSize = e.Edit, e.Script, e.ScriptSize
	}
	v.Runs, v.Confirmed, v.RunsDiffer, v.OneGrader, v.OneRunner = a.runViews(s, e), e.Confirmed, e.RunsDiffer, e.OneGrader, e.OneRunner
	for _, run := range v.Runs {
		if run.Status != "dropped" {
			v.NRuns++
			v.Own = v.Own || run.Self
		}
	}
	if rd := s.Rounds[e.RoundIdx]; rd.Status == "settled" { // an entry settled invalid does not unseal the round
		v.Baseline = deref(rd.BaselineLoss)
	}
	if d := e.Detail; d != nil {
		v.Verifiers, v.Train, v.Disputed, v.Replicas = d.Verifiers, d.Train, d.Disputed, max(0, len(d.Verifiers)-1)
		v.Together = d.DisputedTogether
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
				map[bool]string{true: "entry", false: "entries"}[int(num("entries")) == 1], esc(str("crawl")))
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
	if r.Status != "settled" {
		// Sealed with the grades: a volunteer has to measure the unedited loss, not read it.
		rs.BaselineLoss, rs.Baseline, rs.BaselineN, rs.Differs = nil, nil, 0, nil
	}
	for _, e := range s.inRound[r.Idx] {
		rs.Entries++
		if len(e.Runs) > 0 || e.EditedAt > 0 || e.Status == "settled" {
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
	d := map[string]any{"Top": top, "Provisional": provisional}
	if len(b.Rated) > 0 {
		d["Leader"] = &b.Rated[0] // the ranked entry ahead, while nobody holds the record
	}
	a.st.view(func(st *State) {
		recs := a.records(st)
		if len(recs) > 0 {
			d["Record"] = &recs[len(recs)-1]
		}
		d["Records"], d["RecordChart"] = recs, recordChart(recs)
		vs := a.verifiers(st)
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
		a.nowFacts(st, d)
		if c := a.savedCrawl(st); c != nil {
			d["Crawl"] = *c
		}
	})
	s.render(w, 200, "home", s.page(r, "", "home", d))
}

// guide is one of the three pages written for somebody who came to do one thing: enter
// an editor, verify scores, or run editors. Each says what to type, what happens next and
// what it costs, with the numbers that are true now.
func (s *Site) guide(name, title, desc string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		a := s.app
		d := map[string]any{
			"InPlay": a.cfg.EntriesInPlay, "PerDay": a.cfg.SubmitsPerDay, "Qualify": a.cfg.QualifyRounds, "Ladder": a.cfg.LadderSize,
			"Establish": a.cfg.EstablishAfter, "GithubDays": a.cfg.GithubMinAgeDays, "CopyFrac": 1 - a.season.InkFrac,
		}
		a.st.view(func(st *State) { a.nowFacts(st, d) })
		p := s.page(r, title, name, d)
		p.Desc = desc
		s.render(w, 200, name, p)
	}
}

// nowFacts is what the pages say about this moment: who is waiting, who is working, what
// is open and what comes next.
func (a *App) nowFacts(st *State, d map[string]any) {
	recs := a.records(st)
	if len(recs) > 0 {
		d["Record"] = &recs[len(recs)-1]
	}
	active, runners, queued, volunteers := 0, 0, 0, 0
	var gpuHours float64
	for _, v := range a.verifiers(st) {
		if v.LastSeen > now()-14*86400 {
			active++
		}
		gpuHours += v.GPUHours
	}
	for _, acct := range st.Accounts {
		if acct.RunnerSeen != 0 && acct.LastSeen > now()-3600 && a.established(acct) {
			runners++
		}
		// Established people who are not the organisers, and have been here lately: while
		// there are none, every score is one machine's, and the pages say so.
		if !acct.Anchor && a.established(acct) && acct.LastSeen > now()-14*86400 {
			volunteers++
		}
	}
	d["Volunteers"] = volunteers
	for _, e := range st.Editors {
		if e.Status == "queued" {
			queued++
		}
	}
	d["Active"], d["GPUHours"], d["Runners"], d["Queued"] = active, gpuHours, runners, queued
	if next := st.roundWithStatus("scheduled"); next != nil {
		c := *next
		d["Next"] = &c
	}
	if open := st.roundWithStatus("open"); open != nil {
		rs := a.roundSummary(st, open)
		d["Open"] = &rs
	}
	for _, rd := range st.roundsDesc(0) {
		if rd.Status == "settled" && rd.BaselineLoss != nil {
			d["LastSettled"] = roundID(a.season.ID, rd.Idx)
			break
		}
	}
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

// editorByName sends /editors/<handle>/<name> to that editor's newest version. An editor's
// own address is its content hash, which never changes meaning.
func (s *Site) editorByName(w http.ResponseWriter, r *http.Request) {
	id := ""
	s.app.st.view(func(st *State) {
		if e := findEditor(st, r.PathValue("handle")+"/"+r.PathValue("name"), 0); e != nil {
			id = e.ID
		}
	})
	if id == "" {
		s.notFound(w, r, "No such curator. It may not have been drawn yet: a curator is published when it is first drawn into a round.")
		return
	}
	http.Redirect(w, r, "/editors/"+id[:12], http.StatusFound)
}

func (s *Site) editor(w http.ResponseWriter, r *http.Request) {
	a := s.app
	b := a.board()
	var e *Editor
	d := map[string]any{}
	owner, github := "", ""
	a.st.view(func(st *State) {
		found := findEditor(st, r.PathValue("ref"), 0)
		if found == nil {
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
		s.notFound(w, r, "No such curator. Entries become public when they are first drawn into a round.")
		return
	}
	d["E"], d["Owner"], d["Github"], d["Row"] = e, owner, github, b.ByID[e.ID]

	// Source viewer: one file at a time, read out of the stored bundle.
	want := r.URL.Query().Get("file")
	if want == "" {
		for _, pref := range []string{"curator.py", "editor.py", "stet.toml"} {
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
	// Only a file the editor is known to have is worth opening the bundle for, and what is
	// read out is remembered: the page is public and the bundle can be 50 MB.
	listed := false
	for _, f := range e.Files {
		listed = listed || f.Path == want
	}
	if listed && fileRefRe.MatchString(want) {
		if v, err := s.sourceFile(r.Context(), e.ID, want); err == nil {
			d["File"], d["FileText"], d["FileBinary"], d["FileTruncated"] = want, v.text, v.binary, v.truncated
		}
	}
	p := s.page(r, fmt.Sprintf("%s v%d", e.Name, e.Version), "leaderboard", d)
	if e.Technique != "" {
		p.Desc = fmt.Sprintf("%s v%d by %s: %s", e.Name, e.Version, owner, e.Technique)
	}
	s.render(w, 200, "editor", p)
}

type sourceView struct {
	text              string
	binary, truncated bool
}

// sourceFile reads one file out of an editor's bundle for the source viewer. Bundles never
// change, so what was read is kept (a few hundred files at most, each at most maxViewBytes),
// and only one bundle is unpacked at a time.
func (s *Site) sourceFile(ctx context.Context, id, path string) (sourceView, error) {
	key := id + "/" + path
	cached := func() (sourceView, bool) {
		s.srcMu.Lock()
		defer s.srcMu.Unlock()
		v, ok := s.src[key]
		return v, ok
	}
	if v, ok := cached(); ok {
		return v, nil
	}
	// The lock is never held while a bundle is fetched and unpacked: one large bundle and
	// one patient crawler would otherwise stall every editor page there is.
	select {
	case s.srcGate <- struct{}{}:
		defer func() { <-s.srcGate }()
	case <-ctx.Done():
		return sourceView{}, ctx.Err()
	}
	if v, ok := cached(); ok {
		return v, nil
	}
	blob, err := s.app.readBundle(ctx, id)
	if err != nil {
		return sourceView{}, err
	}
	var v sourceView
	if v.text, v.binary, v.truncated, err = bundleFile(blob, path); err != nil {
		return sourceView{}, err
	}
	s.srcMu.Lock()
	defer s.srcMu.Unlock()
	for k := range s.src { // make room by dropping a few, not all
		if len(s.src) < 300 {
			break
		}
		delete(s.src, k)
	}
	s.src[key] = v
	return v, nil
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
	d := map[string]any{"R": rs, "ID": rs.ID, "Entries": entries, "Scale": sc, "Tolerance": a.tolerance(), "RunsPerEntry": a.cfg.RunsPerEntry}
	// Every run of the round's entries, and every grade of every script they wrote, whatever
	// it said. A script that two runs wrote has one set of grades.
	type runRow struct {
		RunView
		Name, Owner, EditorID string
	}
	type gradeRow struct {
		Grade
		Name, Owner, EditorID string
		Runs                  string // the runs that wrote the script
		ScriptGain            float64
		ScriptValid, Dropped  bool
	}
	var runs []runRow
	var grades []gradeRow
	for _, e := range entries {
		seen := map[string]bool{}
		for _, run := range e.Runs {
			runs = append(runs, runRow{run, e.Name, e.Owner, e.EditorID})
			if run.Script == "" || seen[run.Script] {
				continue
			}
			seen[run.Script] = true
			var wrote []string
			for _, o := range e.Runs {
				if o.Script == run.Script {
					wrote = append(wrote, strconv.Itoa(o.N))
				}
			}
			for _, g := range run.Grades {
				grades = append(grades, gradeRow{g, e.Name, e.Owner, e.EditorID, strings.Join(wrote, " and "), run.Gain, run.Status == "ok", run.Status == "dropped"})
			}
		}
	}
	d["Runs"], d["Grades"] = runs, grades
	// The command that re-grades an entry names it. Two versions of one editor can share a
	// round (a rated editor and its successor), and then the name needs its version.
	if len(entries) > 0 {
		first, n := entries[0], 0
		for _, e := range entries {
			if e.Owner == first.Owner && e.Name == first.Name {
				n++
			}
		}
		d["Example"] = first.Owner + "/" + first.Name
		if n > 1 {
			d["Example"] = fmt.Sprintf("%s/%s@v%d", first.Owner, first.Name, first.Version)
		}
	}
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
	a.st.view(func(st *State) { vs = a.verifiers(st) })
	var hours float64
	var validated, runs, established int
	for _, v := range vs {
		hours += v.GPUHours
		validated += v.Validated
		runs += v.Runs
		if v.Established {
			established++
		}
	}
	s.render(w, 200, "verifiers", s.page(r, "Verifiers", "verifiers", map[string]any{
		"Verifiers": vs, "Hours": hours, "Validated": validated, "Runs": runs, "Established": established, "MuteAfter": muteAfterStrikes,
		"MuteDays": muteFor / 86400, "ReplicaWait": a.cfg.ReplicaWait.Seconds(), "MaxReplicas": maxReplicas,
		"Tolerance": a.tolerance(), "Devices": a.season.GradeDevices, "EstablishAfter": a.cfg.EstablishAfter, "RunsPerEntry": a.cfg.RunsPerEntry,
		"Floor": a.season.Runner,
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
		acct, stats = &c, a.verifierStats(st, found)
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

var docLinkRe = regexp.MustCompile(`href="(?:[./A-Za-z]*/)?([A-Z]+\.md)(#[^"]*)?"`)

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
	// The handbook's pages link to each other by file name, which is what works on GitHub.
	out = docLinkRe.ReplaceAllStringFunc(out, func(m string) string {
		parts := docLinkRe.FindStringSubmatch(m)
		for _, d := range docSpecs {
			if d.File == parts[1] {
				return `href="/docs/` + d.Slug + parts[2] + `"`
			}
		}
		return m
	})
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
	p := filepath.Join(s.app.cfg.FilesDir, name)
	if st, err := os.Stat(p); err == nil && !st.IsDir() {
		w.Header().Set("Cache-Control", "public, max-age=3600")
		http.ServeFile(w, r, p)
		return
	}
	body, size, err := s.app.st.blobs.Open(r.Context(), "files/"+name)
	if errors.Is(err, errNoBlob) {
		http.NotFound(w, r)
		return
	}
	if err != nil { // the store's trouble, not the file's absence: nothing a cache should keep
		slog.Error("file", "name", name, "err", err)
		http.Error(w, "the file store is not answering; try again in a moment", http.StatusBadGateway)
		return
	}
	defer body.Close()
	w.Header().Set("Cache-Control", "public, max-age=3600")
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
#
# It installs its own Python {{.Python}} and, with it, the libraries an editor may use, at the
# season's exact versions: the same on every machine, so that an editor behaves the same
# wherever it is run. PyTorch is among them, in the build for this machine's GPU.
#
#   STET_LIGHT=1         only what "stet submit", "stet status" and friends need: no
#                        PyTorch and no libraries. Seconds to install, and megabytes.
#   STET_TORCH_INDEX=... take PyTorch from this wheel index, not the one chosen below.
#   STET_UPGRADE=1       replacing an install: fetch everything before touching it.
#                        "stet upgrade", and a running "stet verify", set this.
set -eu
WHEEL="{{.Files}}/files/{{.Wheel}}"
PY="{{.Python}}"

say() { printf '%s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  say "Stet installs with uv (https://docs.astral.sh/uv/). Install uv first:"
  say ""
  say "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  say ""
  say "then run this installer again."
  exit 1
fi

if [ "${STET_LIGHT:-}" = "1" ] || [ "${STET_NO_TORCH:-}" = "1" ]; then
  say "Installing stet (light: no PyTorch, no libraries)..."
  uv tool install --force --python "$PY" "stet-cli @ $WHEEL"
else
  # Which build of PyTorch. With an NVIDIA GPU, the default one. With an AMD GPU and no
  # NVIDIA one, the ROCm build. On a Mac the default build uses Metal.
  INDEX="${STET_TORCH_INDEX:-}"
  if [ -z "$INDEX" ] && [ "$(uname -s)" = "Linux" ]; then
    VENDORS="$(cat /sys/class/drm/card*/device/vendor 2>/dev/null || true)"
    case "$VENDORS" in
      *0x10de*) ;;
      *0x1002*) INDEX="{{.AMDIndex}}"; say "AMD GPU found: taking the ROCm build of PyTorch." ;;
    esac
  fi
  TMP="$(mktemp -d)"
  PRE=""
  trap 'rm -rf "$TMP" ${PRE:+"$PRE"}' EXIT
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "{{.Base}}/constraints.txt" -o "$TMP/constraints.txt"
  else
    wget -qO "$TMP/constraints.txt" "{{.Base}}/constraints.txt"
  fi
  if [ "${STET_UPGRADE:-}" = "1" ]; then
    # An upgrade replaces an install that works, and a verifier may be doing it with
    # nobody watching. So fetch and build everything first, in a scratch environment:
    # if a download fails, the install that is there is left alone.
    PRE="${STET_CACHE:-$HOME/.cache/stet}/upgrade-env"
    rm -rf "$PRE"
    say "Fetching the new version before replacing the one in use..."
    uv venv --quiet --python "$PY" "$PRE"
    set -- uv pip install --quiet --python "$PRE/bin/python" --constraints "$TMP/constraints.txt"
    if [ -n "$INDEX" ]; then
      set -- "$@" --index "$INDEX" --index-strategy unsafe-best-match
    fi
    if ! "$@" "stet-cli[run] @ $WHEEL"; then
      rm -rf "$PRE"
      say "The new version could not be fetched whole. Nothing was changed."
      exit 1
    fi
    rm -rf "$PRE"
  fi
  say "Installing stet with Python $PY, PyTorch and the curator libraries (a few GB the first time)..."
  set -- uv tool install --force --python "$PY" --constraints "$TMP/constraints.txt"
  if [ -n "$INDEX" ]; then
    set -- "$@" --index "$INDEX" --index-strategy unsafe-best-match
  fi
  if ! "$@" "stet-cli[run] @ $WHEEL"; then
    say ""
    say "The full set of curator libraries could not be installed on this machine."
    say "Installing stet with PyTorch alone: enough to grade, and to run curators that"
    say "need nothing else. 'stet doctor' will say what is missing."
    set -- uv tool install --force --python "$PY" --with "torch=={{.Torch}}"
    if [ -n "$INDEX" ]; then
      set -- "$@" --index "$INDEX" --index-strategy unsafe-best-match
    fi
    "$@" "stet-cli @ $WHEEL"
  fi
fi

say ""
say "Done. Next:"
say "  stet doctor              check this machine"
say "  stet init my-curator      start a curator"
say "  stet verify              lend your GPU to the leaderboard"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) say ""; say "If 'stet' is not found, run:  uv tool update-shell" ;;
esac
`

const installPs1Tmpl = `# Installs the Stet command line. Usage:  irm {{.Base}}/install.ps1 | iex
#
# It installs its own Python {{.Python}} and the libraries an editor may use, at the season's
# exact versions. Set $env:STET_LIGHT = "1" for the command line alone.
$ErrorActionPreference = "Stop"
$wheel = "{{.Files}}/files/{{.Wheel}}"
$py = "{{.Python}}"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Stet installs with uv (https://docs.astral.sh/uv/). Install uv first:"
  Write-Host ""
  Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
  Write-Host ""
  Write-Host "then run this installer again."
  return
}

if ($env:STET_LIGHT -eq "1" -or $env:STET_NO_TORCH -eq "1") {
  Write-Host "Installing stet (light: no PyTorch, no libraries)..."
  uv tool install --force --python $py "stet-cli @ $wheel"
} else {
  # On Windows the default PyTorch build is CPU-only; with an NVIDIA card, use the CUDA build.
  $index = $env:STET_TORCH_INDEX
  if (-not $index -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $index = "{{.NvidiaWindowsIndex}}"
  }
  $constraints = Join-Path ([System.IO.Path]::GetTempPath()) "stet-constraints.txt"
  Invoke-WebRequest -UseBasicParsing "{{.Base}}/constraints.txt" -OutFile $constraints
  Write-Host "Installing stet with Python $py, PyTorch and the curator libraries (a few GB the first time)..."
  $more = @()
  if ($index) { $more = @("--index", $index, "--index-strategy", "unsafe-best-match") }
  uv tool install --force --python $py --constraints $constraints @more "stet-cli[run] @ $wheel"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "The full set of curator libraries could not be installed here. Installing stet with PyTorch alone."
    uv tool install --force --python $py --with "torch=={{.Torch}}" @more "stet-cli @ $wheel"
  }
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  stet doctor              check this machine"
Write-Host "  stet init my-curator      start a curator"
Write-Host "  stet verify              lend your GPU to the leaderboard"
`

// constraints is every library of the season's environment at its exact version: what the
// installer holds the install to, so that what an editor imports is the same everywhere.
func (s *Site) constraints(w http.ResponseWriter, r *http.Request) {
	env := s.app.season.Environment
	only := map[string]bool{}
	for _, name := range env.Linux {
		only[name] = true
	}
	var lines []string
	for _, set := range []map[string]string{env.Packages, env.Also} {
		for name, version := range set {
			line := name + "==" + version
			if only[name] {
				line += " ; sys_platform == 'linux' and platform_machine == 'x86_64'"
			}
			lines = append(lines, line)
		}
	}
	sort.Strings(lines)
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=300")
	w.Write([]byte(strings.Join(lines, "\n") + "\n"))
}

func (s *Site) installScript(w http.ResponseWriter, r *http.Request, tmpl string) {
	wheel, _ := s.wheel()
	if wheel == "" {
		http.Error(w, "echo 'The Stet CLI is not published on this server yet.'; exit 1\n", 503)
		return
	}
	env := s.app.season.Environment
	out := strings.NewReplacer("{{.Base}}", s.app.cfg.BaseURL, "{{.Files}}", s.app.cfg.APIURL, "{{.Wheel}}", wheel,
		"{{.Python}}", env.Python, "{{.Torch}}", env.Packages["torch"],
		"{{.AMDIndex}}", env.TorchIndex["amd"], "{{.NvidiaWindowsIndex}}", env.TorchIndex["nvidia_windows"]).Replace(tmpl)
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
