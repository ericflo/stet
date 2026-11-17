package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

const testCommit = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

func assetTable(path, file string, sha string, size int64) string {
	return fmt.Sprintf("[[assets]]\npath = %q\nurl = \"https://huggingface.co/owner/model/resolve/%s/%s\"\nsha256 = %q\nsize = %d\n", path, testCommit, file, sha, size)
}

// hub stands in for the host of the assets: what it says about each file.
type hub struct {
	files map[string]hubFile // by the file's path in the repository
	asked []string
	down  bool
}

type hubFile struct {
	status int
	body   string // a small file, sent on GET
	sha    string // a large file's sha256, said without sending it
	size   int64
}

func (h *hub) RoundTrip(r *http.Request) (*http.Response, error) {
	h.asked = append(h.asked, r.Method+" "+r.URL.Path)
	if h.down {
		return nil, errors.New("no route to host")
	}
	answer := func(status int, header http.Header, body string) (*http.Response, error) {
		return &http.Response{StatusCode: status, Header: header, Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
	}
	name := r.URL.Path[strings.Index(r.URL.Path, testCommit)+len(testCommit)+1:]
	f, ok := h.files[name]
	if !ok {
		return answer(404, http.Header{}, "")
	}
	if f.status != 0 {
		return answer(f.status, http.Header{}, "")
	}
	head := http.Header{}
	if f.sha != "" {
		head.Set("X-Linked-Size", fmt.Sprint(f.size))
		head.Set("X-Linked-Etag", `"`+f.sha+`"`)
		return answer(302, head, "")
	}
	head.Set("X-Linked-Size", fmt.Sprint(len(f.body)))
	head.Set("X-Linked-Etag", `"ccf3dac2f33905d152adbdf3886b5e38a4eb7579"`) // git's name for it, not a sha256
	if r.Method == http.MethodHead {
		return answer(307, head, "")
	}
	return answer(200, head, f.body)
}

func withHub(t *testing.T, h *hub) {
	old := hubTransport
	hubTransport = h
	t.Cleanup(func() { hubTransport = old })
}

func shaOf(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func TestAnAssetsTableIsHeldToTheRules(t *testing.T) {
	sha := strings.Repeat("a", 64)
	ok := "name = \"an-editor\"\nentry = \"python editor.py\"\n"
	m, _, err := parseBundle(manifest(ok+assetTable("smol/model.safetensors", "model.safetensors", sha, 269060552)+assetTable("smol/config.json", "config.json", sha, 704)), testAssetLimits)
	if err != nil || len(m.Files) != 2 || m.Files[0].Path != "smol/model.safetensors" || m.Files[0].Size != 269060552 || m.Files[1].SHA256 != sha {
		t.Fatalf("a well-formed table: %v %+v", err, m)
	}
	url := "https://huggingface.co/owner/model/resolve/" + testCommit + "/model.safetensors"
	table := func(path, u, s, size string) string {
		return fmt.Sprintf("[[assets]]\npath = %s\nurl = %s\nsha256 = %s\nsize = %s\n", path, u, s, size)
	}
	q := func(s string) string { return fmt.Sprintf("%q", s) }
	for why, body := range map[string]string{
		"pinned to a commit: a branch":           table(q("m"), q(strings.Replace(url, testCommit, "main", 1)), q(sha), "5"),
		"pinned to a commit: another host":       table(q("m"), q(strings.Replace(url, "huggingface.co", "example.test", 1)), q(sha), "5"),
		"pinned to a commit: plain http":         table(q("m"), q(strings.Replace(url, "https://", "http://", 1)), q(sha), "5"),
		"pinned to a commit: a query":            table(q("m"), q(url+"?download=true"), q(sha), "5"),
		"pinned to a commit: dots":               table(q("m"), q(strings.Replace(url, "/model.safetensors", "/../x", 1)), q(sha), "5"),
		"64 lowercase hex: capitals":             table(q("m"), q(url), q(strings.ToUpper(sha)), "5"),
		"64 lowercase hex: short":                table(q("m"), q(url), q(sha[:40]), "5"),
		"length in bytes: zero":                  table(q("m"), q(url), q(sha), "0"),
		"length in bytes: a fraction":            table(q("m"), q(url), q(sha), "1.5"),
		"length in bytes: text":                  table(q("m"), q(url), q(sha), q("5")),
		"relative path of plain names: absolute": table(q("/etc/passwd"), q(url), q(sha), "5"),
		"relative path of plain names: up":       table(q("../up"), q(url), q(sha), "5"),
		"relative path of plain names: hidden":   table(q("smol/.hidden"), q(url), q(sha), "5"),
		"are required, and are strings":          table("7", q(url), q(sha), "5"),
		"are required, and are strings: missing": "[[assets]]\npath = \"m\"\nsize = 5\n",
		"unknown key":                            table(q("m"), q(url), q(sha), "5") + "mirror = \"https://example.test\"\n",
		"collides":                               table(q("smol"), q(url), q(sha), "5") + table(q("Smol/model"), q(url), q(sha), "5"),
		"more than the limit of 4 GiB":           table(q("a"), q(url), q(sha), fmt.Sprint(int64(4)<<30)) + table(q("b"), q(url), q(sha), "1"),
		"[[assets]] tables":                      "assets = \"none\"\n",
	} {
		_, _, err := parseBundle(manifest(ok+body), testAssetLimits)
		if want := strings.SplitN(why, ":", 2)[0]; err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: got %v", why, err)
		}
	}
	many := ok
	for i := 0; i < 33; i++ {
		many += assetTable(fmt.Sprintf("m/%d", i), "f", sha, 1)
	}
	if _, _, err := parseBundle(manifest(many), testAssetLimits); err == nil || !strings.Contains(err.Error(), "33 assets; the limit is 32") {
		t.Errorf("too many: %v", err)
	}
	if _, _, err := parseBundle(manifest(ok+assetTable("m", "f", sha, 1)), AssetLimits{}); err == nil || !strings.Contains(err.Error(), "does not allow") {
		t.Errorf("a season without assets: %v", err)
	}
}

func TestTheHostIsAskedAboutEveryAssetBeforeAnEntryIsTaken(t *testing.T) {
	big := strings.Repeat("b", 64)
	config := `{"hidden_size": 576}`
	h := &hub{files: map[string]hubFile{
		"model.safetensors": {sha: big, size: 269060552},
		"config.json":       {body: config},
		"gated.safetensors": {status: 401},
	}}
	withHub(t, h)
	as := func(file, sha string, size int64) Asset {
		return Asset{Path: "smol/" + file, URL: "https://huggingface.co/owner/model/resolve/" + testCommit + "/" + file, SHA256: sha, Size: size}
	}
	good := []Asset{as("model.safetensors", big, 269060552), as("config.json", shaOf(config), int64(len(config)))}
	if err := checkAssets(context.Background(), good); err != nil {
		t.Fatalf("what the host confirms is accepted: %v", err)
	}
	// The large file was never downloaded: the host says what it is. The small one was.
	gets := 0
	for _, a := range h.asked {
		if strings.HasPrefix(a, "GET") {
			gets++
			if !strings.HasSuffix(a, "config.json") {
				t.Errorf("fetched %s", a)
			}
		}
	}
	if gets != 1 {
		t.Errorf("%d files fetched, want the one small file", gets)
	}
	for why, a := range map[string]Asset{
		"the file's sha256 is":                  as("model.safetensors", strings.Repeat("c", 64), 269060552),
		"bytes; the entry says":                 as("model.safetensors", big, 5),
		"no such file":                          as("missing.bin", big, 5),
		"needs an account":                      as("gated.safetensors", big, 5),
		"the file's sha256 is " + shaOf(config): as("config.json", strings.Repeat("d", 64), int64(len(config))),
	} {
		if err := checkAssets(context.Background(), []Asset{good[0], a}); err == nil || !strings.Contains(err.Error(), why) || errors.Is(err, errHubAway) {
			t.Errorf("%s: got %v", why, err)
		}
	}
	// A host that cannot be asked is not the entry's fault, and is not a yes either.
	h.down = true
	if err := checkAssets(context.Background(), good); !errors.Is(err, errHubAway) {
		t.Errorf("the host is away: got %v", err)
	}
}

func TestAnEntryWithAssetsIsCheckedAndShowsThem(t *testing.T) {
	w := newWorld(t)
	al := w.join("alice", nil)
	weights := strings.Repeat("e", 64)
	h := &hub{files: map[string]hubFile{"model.safetensors": {sha: weights, size: 1000}}}
	withHub(t, h)
	withAssets := func(sha string) []byte {
		return bundleWith(t, "reader", "print(1)", assetTable("smol/model.safetensors", "model.safetensors", sha, 1000))
	}
	var out struct {
		Editor *Editor `json:"editor"`
		Error  string  `json:"error"`
	}
	if code := al.do("POST", "/api/v1/editors", withAssets(strings.Repeat("f", 64)), &out); code != 400 || !strings.Contains(out.Error, "the file's sha256 is") {
		t.Fatalf("a wrong hash: HTTP %d %s", code, out.Error)
	}
	h.down = true
	if code := al.do("POST", "/api/v1/editors", withAssets(weights), &out); code != 503 || !strings.Contains(out.Error, "Nothing was submitted") {
		t.Fatalf("the host away: HTTP %d %s", code, out.Error)
	}
	h.down = false
	if code := al.do("POST", "/api/v1/editors", withAssets(weights), &out); code != 201 {
		t.Fatalf("a good entry: HTTP %d %s", code, out.Error)
	}
	var ed *Editor
	w.app.st.view(func(s *State) { ed = s.Editors[out.Editor.ID] })
	if ed == nil || len(ed.Assets) != 1 || ed.Assets[0].SHA256 != weights || ed.AssetBytes != 1000 {
		t.Fatalf("the editor's record: %+v", ed)
	}
	w.check()
}
