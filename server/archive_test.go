package main

import (
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"testing"
)

// A finished season is kept as it stood: its pages are served under /seasons/<id>/, every
// link in them stays inside the archive or goes to how to take part now, and the live site
// points to it only quietly.
func TestAFinishedSeasonKeepsItsPages(t *testing.T) {
	pub, _ := newKey()
	w := newWorld(t, pub)
	if len(w.app.site.archives) == 0 {
		t.Skip("no season is archived")
	}
	get := func(path string) (int, string, string) {
		t.Helper()
		req, _ := http.NewRequest("GET", w.srv.URL+path, nil)
		req.Host = "site.test"
		resp, err := http.DefaultTransport.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header.Get("Content-Type"), string(body)
	}
	for _, a := range w.app.site.archives {
		m := a.Meta
		prefix := "/seasons/" + m.ID
		if m.Rounds == 0 || m.Pages == 0 || m.First == 0 || m.Last < m.First {
			t.Errorf("%s: its description is incomplete: %+v", m.ID, m)
		}
		// Every saved page answers, says the season is over, and links only to itself, to
		// fragments, to the live site's root and seasons, to the guides, or off the site.
		allowed := regexp.MustCompile(`^(#|https?://|mailto:|` + regexp.QuoteMeta(prefix) + `/|/$|/seasons$|/enter$|/verify$|/run$)`)
		pages := 0
		for key := range a.files {
			if strings.HasPrefix(key, "static/") {
				continue
			}
			pages++
			path := prefix + "/" + key
			if key == "index" {
				path = prefix + "/"
			}
			code, ctype, body := get(path)
			if code != 200 || !strings.HasPrefix(ctype, "text/html") {
				t.Fatalf("%s: HTTP %d %s", path, code, ctype)
			}
			if !strings.Contains(body, `class="archived"`) || !strings.Contains(body, m.Title+" is over") {
				t.Errorf("%s does not say the season is over", path)
			}
			for _, l := range regexp.MustCompile(`(?:href|src)="([^"]*)"`).FindAllStringSubmatch(body, -1) {
				if !allowed.MatchString(l[1]) {
					t.Errorf("%s links to %s", path, l[1])
				}
				// An archived link names a saved page, and the archive holds it.
				if rest, ok := strings.CutPrefix(l[1], prefix+"/"); ok {
					rest, _, _ = strings.Cut(rest, "#")
					rest = strings.ReplaceAll(rest, "&amp;", "&")
					if rest == "" {
						rest = "index"
					}
					if _, held := a.files[rest]; !held {
						t.Errorf("%s links to %s, which the archive does not hold", path, l[1])
					}
				}
			}
		}
		if pages != m.Pages {
			t.Errorf("%s holds %d pages, its description says %d", m.ID, pages, m.Pages)
		}
		if code, ctype, _ := get(prefix + "/static/site.css"); code != 200 || !strings.HasPrefix(ctype, "text/css") {
			t.Errorf("%s's stylesheet: HTTP %d %s", m.ID, code, ctype)
		}
		if code, _, _ := get(prefix + "/rounds/nothing-here"); code != 404 {
			t.Errorf("a page the archive does not hold: HTTP %d", code)
		}
		if code, _, _ := get(prefix); code != http.StatusMovedPermanently {
			t.Errorf("%s without its slash: HTTP %d", prefix, code)
		}
	}
	if code, _, _ := get("/seasons/s999/"); code != 404 {
		t.Errorf("a season that never was: HTTP %d", code)
	}
	// The live site: a list of seasons, a footer link to it, and a line under the rounds.
	code, _, body := get("/seasons")
	if code != 200 || !strings.Contains(body, "running now") || !strings.Contains(body, `href="/seasons/`+w.app.site.archives[0].Meta.ID+`/"`) {
		t.Fatalf("the seasons page: HTTP %d", code)
	}
	if _, _, home := get("/"); !strings.Contains(home, `<a href="/seasons">Past seasons</a>`) {
		t.Error("the footer does not link to the past seasons")
	}
	if _, _, rounds := get("/rounds"); !strings.Contains(rounds, "Rounds of earlier seasons") {
		t.Error("the rounds page does not mention the earlier seasons' rounds")
	}
}

// The copy a finished season is archived from is read into memory, and never with a real
// database or bucket at hand.
func TestAnArchiveCopyKeepsNothing(t *testing.T) {
	dir := t.TempDir()
	good := dir + "/good.jsonl"
	os.WriteFile(good, []byte(`{"key":"meta/counters","value":{"paused":1}}`+"\n"+`{"key":"round/00000001","value":{"idx":1}}`+"\n"), 0o600)
	m, err := seededMemKV(good)
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := m.Load(t.Context(), []KVScan{{Prefix: ""}})
	if err != nil || len(got[0]) != 2 || got[0][1].Key != "round/00000001" {
		t.Fatalf("the copy read back as %+v (%v)", got, err)
	}
	bad := dir + "/bad.jsonl"
	os.WriteFile(bad, []byte("not json\n"), 0o600)
	if _, err := seededMemKV(bad); err == nil {
		t.Error("a line that is not a key and value was taken")
	}
	season := loadSeason()
	cfg := Config{Scale: season.OfficialScale, ArchiveSeed: good, DBURL: "http://db.test"}
	if err := cfg.check(season); err == nil {
		t.Error("an archive copy was allowed a real database")
	}
	cfg.DBURL = ""
	if err := cfg.check(season); err != nil {
		t.Errorf("an archive copy with nothing real at hand was refused: %v", err)
	}
}
