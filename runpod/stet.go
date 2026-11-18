package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// --- keys --------------------------------------------------------------------------------------
//
// A key is stet's own key.json: an Ed25519 private key (its 32-byte seed), the public key,
// and the account's handle, all as `stet login` writes them. A rented machine is given its
// key file whole and runs the stet command line with it, as any verifier does.

type Key struct {
	Private string `json:"private"`
	Public  string `json:"public"`
	Handle  string `json:"handle"`
	priv    ed25519.PrivateKey
}

func newKey(handle string) (*Key, error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	return &Key{Private: hex.EncodeToString(priv.Seed()), Public: hex.EncodeToString(pub), Handle: handle, priv: priv}, nil
}

func loadKey(path string) (*Key, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var k Key
	if err := json.Unmarshal(raw, &k); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	seed, err := hex.DecodeString(k.Private)
	if err != nil || len(seed) != ed25519.SeedSize {
		return nil, fmt.Errorf("%s: not a stet key", path)
	}
	k.priv = ed25519.NewKeyFromSeed(seed)
	if got := hex.EncodeToString(k.priv.Public().(ed25519.PublicKey)); k.Public != "" && got != k.Public {
		return nil, fmt.Errorf("%s: its public key does not go with its private key", path)
	}
	k.Public = hex.EncodeToString(k.priv.Public().(ed25519.PublicKey))
	return &k, nil
}

// save writes the key where only its owner can read it, whole or not at all.
func (k *Key) save(path string) error {
	raw, err := json.MarshalIndent(k, "", " ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// file is the key as stet keeps it, for a rented machine to write to its own disk.
func (k *Key) file() []byte {
	raw, _ := json.Marshal(k)
	return raw
}

// sign returns the headers that put this key's name to a request (docs/API.md, Signing).
func (k *Key) sign(method, pathAndQuery string, body []byte, now time.Time) http.Header {
	sum := sha256.Sum256(body)
	ts := strconv.FormatInt(now.Unix(), 10)
	msg := strings.Join([]string{"stet-v1", strings.ToUpper(method), pathAndQuery, ts, hex.EncodeToString(sum[:])}, "\n")
	h := http.Header{}
	h.Set("X-Stet-Key", k.Public)
	h.Set("X-Stet-Time", ts)
	h.Set("X-Stet-Sig", hex.EncodeToString(ed25519.Sign(k.priv, []byte(msg))))
	h.Set("X-Stet-Body", hex.EncodeToString(sum[:]))
	return h
}

// --- the Stet API ------------------------------------------------------------------------------

type Stet struct {
	api  string
	http *http.Client
	now  func() time.Time
}

func newStet(api string) *Stet {
	return &Stet{api: strings.TrimRight(api, "/"), http: &http.Client{Timeout: 60 * time.Second}, now: time.Now}
}

// StetError is the server saying no. Its message is written to be shown to a person.
type StetError struct {
	Status  int
	Message string
}

func (e *StetError) Error() string {
	return fmt.Sprintf("the Stet server said %d: %s", e.Status, e.Message)
}

func statusOf(err error) int {
	var se *StetError
	if errors.As(err, &se) {
		return se.Status
	}
	return 0
}

func (s *Stet) do(ctx context.Context, method, path string, body any, key *Key, into any) error {
	var raw []byte
	if body != nil {
		var err error
		if raw, err = json.Marshal(body); err != nil {
			return err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, s.api+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "stet-runpod")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key != nil {
		for k, v := range key.sign(method, path, raw, s.now()) {
			req.Header[k] = v
		}
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return fmt.Errorf("the Stet server could not be reached: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		var e struct {
			Error string `json:"error"`
		}
		json.Unmarshal(data, &e)
		if e.Error == "" {
			e.Error = strings.TrimSpace(string(data))
		}
		return &StetError{Status: resp.StatusCode, Message: e.Error}
	}
	if into != nil {
		return json.Unmarshal(data, into)
	}
	return nil
}

// Work is what GET /api/v1/admin/work says (server/work.go).
type Work struct {
	Rounds []struct {
		ID             string `json:"id"`
		Entries        int    `json:"entries"`
		RunsWaiting    int    `json:"runs_waiting"`
		EditsUnderWay  int    `json:"edits_under_way"`
		GradesWanted   int    `json:"grades_wanted"`
		GradesUnderWay int    `json:"grades_under_way"`
	} `json:"rounds"`
	Runners    int       `json:"runners"`
	NextDrawAt int64     `json:"next_draw_at"`
	Machines   []Machine `json:"machines"`
}

// Machine is one of the organisers' accounts as the server sees it.
type Machine struct {
	Handle   string `json:"handle"`
	Worker   bool   `json:"worker"`
	LastSeen int64  `json:"last_seen"`
	Busy     string `json:"busy"`
}

func (w *Work) machine(handle string) *Machine {
	for i := range w.Machines {
		if w.Machines[i].Handle == handle {
			return &w.Machines[i]
		}
	}
	return nil
}

// work asks what is waiting. Only an organisers' own key may.
func (s *Stet) work(ctx context.Context, op *Key) (*Work, error) {
	var w Work
	if err := s.do(ctx, "GET", "/api/v1/admin/work", nil, op, &w); err != nil {
		if statusOf(err) == 404 {
			return nil, fmt.Errorf("this Stet server does not say what is waiting (GET /api/v1/admin/work): it is older than this tool")
		}
		if statusOf(err) == 403 {
			return nil, fmt.Errorf("%s is not one of the organisers' own keys, and only they may ask what is waiting (--key)", op.Handle)
		}
		return nil, err
	}
	return &w, nil
}

// account makes sure a rented machine's key has its account, and says whether the server
// takes it as one of the organisers' rented machines (STET_WORKER_KEYS). Asking twice is
// harmless: an account that is there is returned as it is.
func (s *Stet) account(ctx context.Context, k *Key) (worker bool, err error) {
	var out struct {
		Account struct {
			Handle string `json:"handle"`
			Anchor bool   `json:"anchor"`
			Worker bool   `json:"worker"`
		} `json:"account"`
	}
	if err := s.do(ctx, "POST", "/api/v1/accounts", map[string]string{"handle": k.Handle}, k, &out); err != nil {
		return false, err
	}
	if out.Account.Handle != "" && out.Account.Handle != k.Handle {
		return false, fmt.Errorf("the key kept for %s belongs to the account %q", k.Handle, out.Account.Handle)
	}
	return out.Account.Anchor && out.Account.Worker, nil
}

// lookup says what the server knows of a handle: nothing (no account yet), an account, or
// an organisers' rented machine. Asked with the organisers' own key.
func (s *Stet) lookup(ctx context.Context, op *Key, handle string) (exists, worker bool, err error) {
	var out struct {
		Key    string `json:"key"`
		Worker bool   `json:"worker"`
	}
	if err := s.do(ctx, "GET", "/api/v1/admin/accounts/"+handle, nil, op, &out); err != nil {
		if statusOf(err) == 404 {
			return false, false, nil
		}
		return false, false, err
	}
	return true, out.Worker, nil
}

// site is the website whose installer a rented machine runs.
func (s *Stet) site(ctx context.Context) (string, error) {
	var out struct {
		Site string `json:"site"`
	}
	if err := s.do(ctx, "GET", "/api/v1/season", nil, nil, &out); err != nil {
		return "", err
	}
	if out.Site == "" {
		return "", errors.New("the Stet server did not say where its website is")
	}
	return strings.TrimRight(out.Site, "/"), nil
}
