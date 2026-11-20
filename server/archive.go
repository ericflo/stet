package main

import (
	"archive/tar"
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"mime"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// --- finished seasons -------------------------------------------------------------------------
//
// The server runs one season. A finished one keeps its site as it stood when it closed: its
// last server build was run on a copy of its records, and tools/archive_season.py saved every
// page as static HTML, with links rewritten to stay under /seasons/<id>/. Each season is one
// zstd tarball in archive/, unpacked into memory when the server starts. Nothing here reads
// the records: an archived season cannot change.

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

type archivedSeason struct {
	Meta  archiveMeta
	files map[string][]byte // "rounds/s0-o00001", "editors/1d28422e1acd?file=curator.py", "static/site.css", "index"
}

// loadArchives unpacks every season in archive/, newest first.
func loadArchives() ([]*archivedSeason, error) {
	names, err := fs.Glob(archiveFS, "archive/*.tar.zst")
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
		packed, err := archiveFS.ReadFile(name)
		if err != nil {
			return nil, err
		}
		raw, err := dec.DecodeAll(packed, nil)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		a := &archivedSeason{files: map[string][]byte{}}
		tr := tar.NewReader(bytes.NewReader(raw))
		for {
			h, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			b, err := io.ReadAll(tr)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			a.files[h.Name] = b
		}
		if err := json.Unmarshal(a.files["meta.json"], &a.Meta); err != nil || a.Meta.ID == "" {
			return nil, fmt.Errorf("%s: no meta.json naming the season", name)
		}
		if want := strings.TrimSuffix(path.Base(name), ".tar.zst"); a.Meta.ID != want {
			return nil, fmt.Errorf("%s holds season %q", name, a.Meta.ID)
		}
		delete(a.files, "meta.json")
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

// archivePage serves one page of a finished season, as it was saved.
func (s *Site) archivePage(w http.ResponseWriter, r *http.Request) {
	a := s.archived(r.PathValue("season"))
	if a == nil {
		s.notFound(w, r, "There is no such season.")
		return
	}
	key := r.PathValue("rest")
	if key == "" {
		key = "index"
	}
	// The one query a saved page is chosen by: which of a curator's files it shows. It is
	// saved under the query as the archiver wrote it, which is how Go encodes it too.
	if f := r.URL.Query().Get("file"); f != "" {
		key += "?" + url.Values{"file": {f}}.Encode()
	}
	b, ok := a.files[key]
	if !ok {
		s.notFound(w, r, a.Meta.Title+" has no page at this address.")
		return
	}
	ctype := "text/html; charset=utf-8"
	if strings.HasPrefix(key, "static/") {
		switch ext := path.Ext(key); ext {
		case ".woff2":
			ctype = "font/woff2"
		default:
			if t := mime.TypeByExtension(ext); t != "" {
				ctype = t
			}
		}
		w.Header().Set("Cache-Control", "public, max-age=86400")
	} else {
		w.Header().Set("Cache-Control", "public, max-age=3600")
	}
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Write(b)
}
