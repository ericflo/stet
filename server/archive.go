package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io/fs"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// --- finished seasons -------------------------------------------------------------------------
//
// The server runs one season. A finished one keeps its pages as they stood when it closed:
// its last server build was run on a copy of its records, and tools/archive_season.py kept
// what each page said (title, description, content), with links rewritten to stay under
// /seasons/<id>/. They are shown inside this site's own masthead and footer, under a bar
// that names the season and leads around it. Each season is one file in archive/, zstd-
// compressed JSON Lines, read once. Nothing here reads the records: an archived season
// cannot change.

//go:embed archive
var archiveFS embed.FS

// archiveMeta is what the archiver says about a season, for the list of seasons.
type archiveMeta struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	About    string `json:"about"`
	Rounds   int    `json:"rounds"`
	Curators int    `json:"curators"`
	First    int64  `json:"first"` // its first draw
	Last     int64  `json:"last"`  // its last settlement
	Scale    string `json:"scale"`
	Pages    int    `json:"pages"`
}

// archivedPage is one page of a finished season: what lay between its masthead and footer.
type archivedPage struct {
	Title string `json:"title"`
	Desc  string `json:"desc"`
	HTML  string `json:"html"`
}

type archivedSeason struct {
	Meta  archiveMeta
	pages map[string]*archivedPage // "leaderboard", "rounds/s0-o00001", "editors/1d28422e1acd?file=editor.py"
}

var archives struct {
	once sync.Once
	all  []*archivedSeason // newest first
	err  error
}

// loadArchives reads every season in archive/, once: they are part of the build.
func loadArchives() ([]*archivedSeason, error) {
	archives.once.Do(func() { archives.all, archives.err = readArchives(archiveFS) })
	return archives.all, archives.err
}

func readArchives(fsys fs.FS) ([]*archivedSeason, error) {
	names, err := fs.Glob(fsys, "archive/*.jsonl.zst")
	if err != nil {
		return nil, err
	}
	dec, err := zstd.NewReader(nil)
	if err != nil {
		return nil, err
	}
	defer dec.Close()
	var out []*archivedSeason
	for _, name := range names {
		packed, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, err
		}
		raw, err := dec.DecodeAll(packed, nil)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		a := &archivedSeason{pages: map[string]*archivedPage{}}
		for i, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
			var row struct {
				Season *archiveMeta `json:"season"`
				Page   string       `json:"page"`
				archivedPage
			}
			if err := json.Unmarshal(line, &row); err != nil {
				return nil, fmt.Errorf("%s line %d: %w", name, i+1, err)
			}
			switch {
			case i == 0 && row.Season != nil:
				a.Meta = *row.Season
			case i > 0 && row.Page != "":
				p := row.archivedPage
				a.pages[row.Page] = &p
			default:
				return nil, fmt.Errorf("%s line %d: neither the season (first) nor a page", name, i+1)
			}
		}
		if want := strings.TrimSuffix(path.Base(name), ".jsonl.zst"); a.Meta.ID != want {
			return nil, fmt.Errorf("%s holds season %q", name, a.Meta.ID)
		}
		if a.pages["leaderboard"] == nil {
			return nil, fmt.Errorf("%s has no leaderboard, its front page", name)
		}
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Meta.Last > out[j].Meta.Last })
	return out, nil
}

func (s *Site) archived(id string) *archivedSeason {
	for _, a := range s.archives {
		if a.Meta.ID == id {
			return a
		}
	}
	return nil
}

// seasons lists the seasons: the one running, then the finished ones.
func (s *Site) seasons(w http.ResponseWriter, r *http.Request) {
	s.render(w, 200, "seasons", s.page(r, "Seasons", "", nil))
}

// archiveTab is which of a finished season's own pages the bar marks as the one being read.
func archiveTab(key string) string {
	first, _, _ := strings.Cut(key, "/")
	first, _, _ = strings.Cut(first, "?")
	switch first {
	case "leaderboard", "editors":
		return "leaderboard"
	case "u":
		return "verifiers"
	case "docs":
		return "rules"
	}
	return first // rounds, record, verifiers, glossary
}

// archivePage shows one page of a finished season, as it was saved, inside today's site.
func (s *Site) archivePage(w http.ResponseWriter, r *http.Request) {
	a := s.archived(r.PathValue("season"))
	if a == nil {
		s.notFound(w, r, "There is no such season.")
		return
	}
	key := r.PathValue("rest")
	if key == "" {
		key = "leaderboard" // its front page
	}
	// The one query a saved page is chosen by: which of a curator's files it shows. It is
	// saved under the query as the archiver wrote it, which is how Go encodes it too.
	if f := r.URL.Query().Get("file"); f != "" {
		key += "?" + url.Values{"file": {f}}.Encode()
	}
	p := a.pages[key]
	if p == nil {
		s.notFound(w, r, a.Meta.Title+" has no page at this address.")
		return
	}
	title := a.Meta.Title
	if p.Title != "" {
		title = p.Title + " · " + a.Meta.Title
	}
	pg := s.page(r, title, "", map[string]any{
		"Season": a.Meta,
		"Tab":    archiveTab(key),
		"HTML":   template.HTML(p.HTML), // as the season's own server rendered and escaped it
	})
	if p.Desc != "" {
		pg.Desc = p.Desc
	}
	s.render(w, 200, "archived", pg)
}
