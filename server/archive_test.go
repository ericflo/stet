package main

import (
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"testing"
)

// A finished season is kept as it stood, inside today's site: each of its pages is shown
// under the live masthead and footer, with a bar naming the season; every link in what was
// saved stays inside the archive, goes to how to take part now, or leaves the site; and the
// live site points to it only quietly.
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
	link := regexp.MustCompile(`(?:href|src)="([^"]*)"`)
	for _, a := range w.app.site.archives {
		m := a.Meta
		prefix := "/seasons/" + m.ID
		if m.Rounds == 0 || m.First == 0 || m.Last < m.First || m.Pages != len(a.pages) {
			t.Errorf("%s: its description does not match it: %+v, %d pages", m.ID, m, len(a.pages))
		}
		allowed := regexp.MustCompile(`^(#|https?://|mailto:|` + regexp.QuoteMeta(prefix) + `/|/enter$|/verify$|/run$|/static/)`)
		for key, p := range a.pages {
			if strings.Contains(p.HTML, "<header") || strings.Contains(p.HTML, "<footer") || strings.Contains(p.HTML, "<body") {
				t.Errorf("%s: what was saved is more than the page's content", key)
			}
			for _, l := range link.FindAllStringSubmatch(p.HTML, -1) {
				if !allowed.MatchString(l[1]) {
					t.Errorf("%s links to %s", key, l[1])
				}
				// An archived link names a saved page, and the archive holds it.
				if rest, ok := strings.CutPrefix(l[1], prefix+"/"); ok {
					rest, _, _ = strings.Cut(rest, "#")
					rest = strings.ReplaceAll(rest, "&amp;", "&")
					if rest == "" {
						rest = "leaderboard"
					}
					if a.pages[rest] == nil {
						t.Errorf("%s links to %s, which the archive does not hold", key, l[1])
					}
				}
			}
			path := prefix + "/" + key
			code, ctype, body := get(path)
			if code != 200 || !strings.HasPrefix(ctype, "text/html") {
				t.Fatalf("%s: HTTP %d %s", path, code, ctype)
			}
			// Today's masthead and footer, the season's bar, then what the page said.
			for _, want := range []string{`<a class="logo" href="/"`, `<a href="/seasons">Past seasons</a>`, `class="season-bar"`, `<b>` + m.Title + `</b>`, strings.TrimSpace(p.HTML)} {
				if !strings.Contains(body, want) {
					t.Errorf("%s does not show %.60q", path, want)
				}
			}
		}
		// The season's front page is its leaderboard, and its bar marks where one is.
		_, _, front := get(prefix + "/")
		if !strings.Contains(front, strings.TrimSpace(a.pages["leaderboard"].HTML)) || !strings.Contains(front, `href="`+prefix+`/" aria-current="page">Leaderboard`) {
			t.Errorf("%s/ is not the season's leaderboard", prefix)
		}
		for path, tab := range map[string]string{"/rounds": "Rounds", "/record": "The record", "/verifiers": "Verifiers", "/docs/rules": "Rules"} {
			if _, _, body := get(prefix + path); a.pages[strings.TrimPrefix(path, "/")] != nil && !strings.Contains(body, `aria-current="page">`+tab+`</a>`) {
				t.Errorf("%s%s: the bar does not mark %s", prefix, path, tab)
			}
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
