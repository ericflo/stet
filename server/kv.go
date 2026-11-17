package main

// Where the server's records live.
//
// Stet's state is small (accounts, editors, rounds, entries, results) and has one writer,
// so the server keeps all of it in memory and treats storage as an ordered key-value map
// with atomic commits. In production that map is a durable-db database
// (github.com/ericflo/durable-db), whose only state is an object-storage bucket: the
// server itself keeps nothing on disk and can be thrown away and restarted at will.
//
// Every commit is conditional on the database still being at the revision the server last
// saw. If anything else has written in between, the commit is refused and the server reads
// everything again before retrying, so two servers can never interleave their decisions.

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
)

type KVItem struct {
	Key   string          `json:"key"`
	Value json.RawMessage `json:"value"`
}

type KVWrite struct {
	Key    string          `json:"key"`
	Value  json.RawMessage `json:"value,omitempty"`
	Delete bool            `json:"delete,omitempty"`
}

// KVScan selects the keys with a prefix, in order. Limit 0 means all of them.
type KVScan struct {
	Prefix  string
	Reverse bool
	Limit   int
}

var errConflict = errors.New("the database changed underneath this write")

type KV interface {
	// Load answers several scans as of one revision.
	Load(ctx context.Context, scans []KVScan) ([][]KVItem, int64, error)
	// Commit applies the writes atomically if the database is still at ifRevision, and
	// returns the new revision. errConflict means it was not, and nothing was written.
	Commit(ctx context.Context, ifRevision int64, writes []KVWrite) (int64, error)
	// Revision is the database's current revision: a cheap way to ask whether anyone
	// else has written.
	Revision(ctx context.Context) (int64, error)
	Describe() string
}

// --- in memory: tests, and development without a durable-db at hand ---------------------------

type memKV struct {
	mu   sync.Mutex
	rev  int64
	data map[string]json.RawMessage
	// failNext makes the next commit fail after (if applied) or before applying it: tests
	// use it to check that the server recovers from an outcome it cannot know.
	failNext string
}

func newMemKV() *memKV { return &memKV{data: map[string]json.RawMessage{}} }

func (m *memKV) Describe() string { return "memory (nothing is kept across restarts)" }

func (m *memKV) Load(ctx context.Context, scans []KVScan) ([][]KVItem, int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	keys := make([]string, 0, len(m.data))
	for k := range m.data {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([][]KVItem, len(scans))
	for i, sc := range scans {
		for _, k := range keys {
			if strings.HasPrefix(k, sc.Prefix) {
				out[i] = append(out[i], KVItem{Key: k, Value: m.data[k]})
			}
		}
		if sc.Reverse {
			for a, b := 0, len(out[i])-1; a < b; a, b = a+1, b-1 {
				out[i][a], out[i][b] = out[i][b], out[i][a]
			}
		}
		if sc.Limit > 0 && len(out[i]) > sc.Limit {
			out[i] = out[i][:sc.Limit]
		}
	}
	return out, m.rev, nil
}

func (m *memKV) Revision(ctx context.Context) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.rev, nil
}

func (m *memKV) Commit(ctx context.Context, ifRevision int64, writes []KVWrite) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	fail := m.failNext
	m.failNext = ""
	if fail == "before" {
		return 0, errors.New("storage unavailable (injected)")
	}
	if ifRevision != m.rev {
		return 0, errConflict
	}
	for _, w := range writes {
		if w.Delete {
			delete(m.data, w.Key)
		} else {
			m.data[w.Key] = append(json.RawMessage(nil), w.Value...)
		}
	}
	m.rev++
	if fail == "after" {
		return 0, errors.New("the answer was lost (injected)")
	}
	return m.rev, nil
}

// --- durable-db -------------------------------------------------------------------------------

type ddbKV struct {
	root string // http://host:port
	name string
	key  []byte // the deployment's signing key: this server mints its own tokens
	http *http.Client
}

// parseDBKey reads a signing key as durable-db's key file holds it: hex, the first line
// signing.
func parseDBKey(text string) ([]byte, error) {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, err := hex.DecodeString(line)
		if err != nil || len(key) < 32 {
			return nil, errors.New("the durable-db signing key must be at least 32 bytes of hex, as in its admin.key")
		}
		return key, nil
	}
	return nil, errors.New("the durable-db signing key is empty")
}

func newDurableDB(baseURL, database string, key []byte) *ddbKV {
	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConnsPerHost:   16,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		DisableCompression:    true,
	}
	return &ddbKV{root: strings.TrimRight(baseURL, "/"), name: database, key: key, http: &http.Client{Transport: transport, Timeout: 2 * time.Minute}}
}

// token mints a short-lived bearer token, exactly as `durable-db token` does. It only has
// to outlive one request.
func (d *ddbKV) token(db, access string) string {
	grant, _ := json.Marshal(map[string]any{"db": db, "access": access, "exp": time.Now().Add(10 * time.Minute).Unix(), "generation": 0})
	payload := base64.RawURLEncoding.EncodeToString(grant)
	mac := hmac.New(sha256.New, d.key)
	mac.Write([]byte(payload))
	return "ddb_" + payload + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func (d *ddbKV) base() string { return d.root + "/v1/databases/" + url.PathEscape(d.name) }

// ensure creates the database if this is the first time the server has run against this
// deployment. Repeating it is harmless.
func (d *ddbKV) ensure(ctx context.Context) error {
	var last error
	for attempt := 0; attempt < 30; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPut, d.base(), strings.NewReader("{}"))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+d.token("*", "admin"))
		req.Header.Set("Content-Type", "application/json")
		resp, err := d.http.Do(req)
		if err == nil {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
			resp.Body.Close()
			if resp.StatusCode < 300 || (resp.StatusCode == 409 && strings.Contains(string(body), `"exists"`)) {
				return nil
			}
			last = fmt.Errorf("durable-db %d: %s", resp.StatusCode, clip(string(body), 200))
			if resp.StatusCode != 429 && resp.StatusCode < 500 {
				return last
			}
		} else {
			last = err
		}
		select {
		case <-ctx.Done():
			return last
		case <-time.After(2 * time.Second):
		}
	}
	return last
}

func (d *ddbKV) Describe() string { return "durable-db at " + d.base() }

type ddbError struct {
	Status   int
	Code     string
	Message  string
	Revision int64
}

func (e *ddbError) Error() string {
	return fmt.Sprintf("durable-db %d %s: %s", e.Status, e.Code, e.Message)
}

// post sends one JSON request. The identical bytes are sent again after a network error,
// a 429 or a 5xx, which durable-db makes safe: reads change nothing and every commit
// carries an idempotency key.
func (d *ddbKV) post(ctx context.Context, path string, body, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	var last error
	for attempt := 0; attempt < 8; attempt++ {
		if attempt > 0 {
			wait := time.Duration(100<<min(attempt, 5)) * time.Millisecond
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(wait):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.base()+path, bytes.NewReader(raw))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+d.token(d.name, "write"))
		req.Header.Set("Content-Type", "application/json")
		resp, err := d.http.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			last = err
			continue
		}
		answer, err := io.ReadAll(io.LimitReader(resp.Body, 256<<20))
		resp.Body.Close()
		if err != nil {
			last = err
			continue
		}
		if resp.StatusCode < 300 {
			return json.Unmarshal(answer, out)
		}
		var e struct {
			Error struct {
				Code     string `json:"code"`
				Message  string `json:"message"`
				Revision int64  `json:"revision"`
			} `json:"error"`
		}
		json.Unmarshal(answer, &e)
		if e.Error.Code == "" {
			e.Error.Code, e.Error.Message = fmt.Sprintf("http_%d", resp.StatusCode), clip(string(answer), 200)
		}
		last = &ddbError{Status: resp.StatusCode, Code: e.Error.Code, Message: e.Error.Message, Revision: e.Error.Revision}
		if resp.StatusCode != 429 && resp.StatusCode < 500 {
			return last
		}
	}
	return last
}

func (d *ddbKV) Load(ctx context.Context, scans []KVScan) ([][]KVItem, int64, error) {
	type scanReq struct {
		Prefix  string `json:"prefix,omitempty"`
		Reverse bool   `json:"reverse,omitempty"`
		Limit   int    `json:"limit"`
		After   string `json:"after,omitempty"`
	}
	type page struct {
		Revision int64 `json:"revision"`
		Scans    []struct {
			Items []KVItem `json:"items"`
			Next  *string  `json:"next"`
		} `json:"scans"`
	}
attempt:
	for try := 0; try < 6; try++ {
		out := make([][]KVItem, len(scans))
		revision := int64(-1)
		for i, sc := range scans {
			after := ""
			for {
				limit := 10000
				if sc.Limit > 0 && sc.Limit-len(out[i]) < limit {
					limit = sc.Limit - len(out[i])
				}
				var p page
				body := map[string]any{"consistency": "strong", "scans": []scanReq{{Prefix: sc.Prefix, Reverse: sc.Reverse, Limit: limit, After: after}}}
				if err := d.post(ctx, "/read", body, &p); err != nil {
					return nil, 0, err
				}
				if len(p.Scans) != 1 {
					return nil, 0, errors.New("durable-db answered a read with the wrong number of scans")
				}
				if revision >= 0 && p.Revision != revision {
					continue attempt // something wrote while we were reading: start over
				}
				revision = p.Revision
				out[i] = append(out[i], p.Scans[0].Items...)
				if p.Scans[0].Next == nil || (sc.Limit > 0 && len(out[i]) >= sc.Limit) {
					break
				}
				after = *p.Scans[0].Next
			}
		}
		if revision < 0 {
			revision = 0
		}
		return out, revision, nil
	}
	return nil, 0, errors.New("the database kept changing while it was being read")
}

func (d *ddbKV) Revision(ctx context.Context) (int64, error) {
	var p struct {
		Revision int64 `json:"revision"`
	}
	// A strong read of nothing much: durable-db answers it with one HEAD of its bucket.
	body := map[string]any{"consistency": "strong", "keys": []string{"meta"}}
	if err := d.post(ctx, "/read", body, &p); err != nil {
		return 0, err
	}
	return p.Revision, nil
}

func (d *ddbKV) Commit(ctx context.Context, ifRevision int64, writes []KVWrite) (int64, error) {
	if len(writes) == 0 {
		return ifRevision, nil
	}
	if len(writes) > 9000 {
		return 0, fmt.Errorf("one commit of %d writes is more than durable-db takes", len(writes))
	}
	idem := make([]byte, 16)
	rand.Read(idem)
	body := map[string]any{"if_revision": ifRevision, "writes": writes, "idempotency_key": "stet-" + hex.EncodeToString(idem)}
	var out struct {
		Revision int64 `json:"revision"`
	}
	err := d.post(ctx, "/commit", body, &out)
	var de *ddbError
	if errors.As(err, &de) && de.Code == "conflict" {
		return 0, errConflict
	}
	if err != nil {
		return 0, err
	}
	return out.Revision, nil
}
