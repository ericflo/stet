package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// --- the markets machines are rented from ------------------------------------------------------
//
// Every market is reached through its own API, and only that: no aggregator or price feed of
// anybody else's is asked anything. Each is one Market here, and the fleet treats every (market,
// card) as a kind of its own, chosen by the same rules whichever market it is on.

// Market is one place machines are rented from.
type Market interface {
	Name() string
	// Offers are what could be rented now, that the market says meet the need: its price all
	// in, an hour.
	Offers(ctx context.Context, n Need) ([]Offer, error)
	// Rent rents one machine. errNoRoom: that offer is gone, and the next may do.
	Rent(ctx context.Context, o Offer, l Launch) (*Rental, error)
	// Rentals are the machines on the account, live or stopped. Ended ones may not be listed.
	Rentals(ctx context.Context) ([]Rental, error)
	// End ends a machine for good; one already gone is not an error.
	End(ctx context.Context, id string) error
	// Check asks, reading only, whether the key works, and what is left to spend if it says.
	Check(ctx context.Context) (Account, error)
	Caps() Caps
}

// marketOrder is the order the markets are listed in, and asked in.
var marketOrder = []string{"runpod", "vast", "lium", "salad", "clore"}

// marketTitle is a market's name as people write it.
func marketTitle(name string) string {
	switch name {
	case "runpod":
		return "RunPod"
	case "vast":
		return "Vast.ai"
	case "lium":
		return "Lium"
	case "salad":
		return "Salad"
	case "clore":
		return "Clore.ai"
	}
	return name
}

// Need is what a machine must have.
type Need struct {
	MinGPUMemGB, MinCPUs, MinRAMGB, DiskGB int
	MinCUDA                                string // "13.0": a host driver of 580 or newer
}

// Offer is one kind of machine a market has now: on RunPod a kind of GPU in one of its clouds,
// elsewhere often one host's machine.
type Offer struct {
	Market   string
	Card     string // the GPU as stet's grades name it, without the maker's name: "RTX 4090"
	Model    string // the market's own name for it, as its API gives it
	Name     string // the market's short name for it, where it has another
	ID       string // the market's own: RunPod's GPU ID, a Vast ask, a Lium executor, a Salad GPU class, a Clore server
	Maker    string // NVIDIA, AMD
	MemoryGB int
	Price    float64 // dollars an hour, all in: the GPU, the disk, the market's fees
	Ask      float64 // the market's own price, as it lists it (Clore: a day, before its fees)
	Stock    int     // machines like it free now; -1 when the market says only Level
	Level    string  // the market's own word for how many are free: low, medium, high
	CUDA     string  // the newest CUDA the host's driver takes, where the market says
	Trust    string  // the market's word for its hosts: secure, verified, datacenter, community
	Region   string
	// Rate is how often a machine of this kind has been seen to start, of those rented (0: not
	// known, taken as all): what it costs is ranked as its price over this.
	Rate float64
}

// rankPrice is what an offer is ranked by: its price, over how often such a machine starts.
// A $0.18 card that starts one time in three is ranked as $0.54. The hourly limits go by
// the price itself.
func (o Offer) rankPrice() float64 {
	if o.Rate <= 0 {
		return o.Price
	}
	return o.Price / max(o.Rate, 0.2)
}

// Launch is what a machine is rented with.
type Launch struct {
	Name   string            // the worker key's handle
	Image  string            // the container image
	Env    map[string]string // the worker's settings, its key among them
	Script string            // worker.sh
	Need   Need
	SSHKey string
}

// Rental is one machine rented on a market, as the market says.
type Rental struct {
	Market  string
	ID      string
	Name    string // the worker key's handle, where the market keeps a name; "" where it does not
	Card    string
	Price   float64 // dollars an hour
	State   string  // starting, running, stopped; anything else is the market's own word
	Status  string  // the market's own word
	Created time.Time
	// What the market says of the machine itself, for seeing why one does not start: whether
	// its container is up ("yes", "no", or "" where the market does not say), its phase in
	// the market's own words (one that changes only as the machine moves on: "pulling image",
	// "no container yet"), and more that changes as it goes ("container up 4m").
	Up     string
	Phase  string
	Detail string
	// How long its container has been up, where the market says: one shorter than at the
	// last look was started again.
	UpFor time.Duration
}

// logger is a market that can say what a machine's container and the market itself have said
// of it: the last n lines.
type logger interface {
	Logs(ctx context.Context, id string, n int) ([]string, error)
}

// logLines are the lines a market's log answer holds, whatever its shape: strings, or objects
// with the line in one of a few fields.
func logLines(data []byte) []string {
	var v any
	if json.Unmarshal(data, &v) != nil {
		return strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	}
	var out []string
	var walk func(v any)
	walk = func(v any) {
		switch x := v.(type) {
		case []any:
			for _, e := range x {
				walk(e)
			}
		case map[string]any:
			for _, k := range []string{"items", "logs", "entries", "data"} {
				if inner, ok := x[k]; ok {
					walk(inner)
					return
				}
			}
			line := ""
			for _, k := range []string{"line", "log_text", "text_log", "message", "msg", "text", "event_name", "log"} {
				if s, ok := x[k].(string); ok && s != "" {
					line = s
					break
				}
			}
			if line == "" {
				if j, ok := x["json_log"].(map[string]any); ok {
					raw, _ := json.Marshal(j)
					line = string(raw)
				}
			}
			if line == "" {
				return
			}
			for _, k := range []string{"ts", "time", "event_time", "received_at", "receive_time", "created_at", "timestamp"} {
				if s, ok := x[k].(string); ok && len(s) >= 19 {
					line = s[11:19] + "  " + line
					break
				}
			}
			out = append(out, line)
		case string:
			out = append(out, strings.Split(x, "\n")...)
		}
	}
	walk(v)
	return out
}

func lastLines(lines []string, n int) []string {
	return lines[max(0, len(lines)-n):]
}

// live is a machine that is, or is about to be, billed for.
func (r *Rental) live() bool { return r.State == "starting" || r.State == "running" }

// Account is what a market says of the account.
type Account struct {
	Balance float64 // dollars left to spend, when Known
	Known   bool
	Note    string
}

// Caps is what a market allows.
type Caps struct {
	Rents    bool // renting here is built: a market can be armed only then
	SelfEnd  bool // a machine can end itself with a key that reaches only it
	Curators bool // stet's container sandbox is known to hold there
	Logs     bool
	SSH      bool
}

// errNotBuilt is a market asked to do what is not built for it yet.
var errNotBuilt = errors.New("not built for this market yet")

// ledgerKey is how a machine is known in the record of what is rented: RunPod's by its pod ID
// alone, as stet-runpod kept them, and the others' with the market's name before it.
func ledgerKey(market, id string) string {
	if market == "" || market == "runpod" {
		return id
	}
	return market + ":" + id
}

// --- credentials ----------------------------------------------------------------------------------
//
// The markets' keys are kept in one file, credentials.env, as KEY=value lines, and read again at
// every look: a key added later needs no restart. One set in the environment is taken over the
// file's. A market whose key is missing is off.

func defaultCredentials() string {
	if p := os.Getenv("STET_CLOUD_CREDENTIALS"); p != "" {
		return p
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "stet-cloud", "credentials.env")
}

// credentialNames are the settings credentials.env holds.
var credentialNames = []string{"RUNPOD_API_KEY", "VAST_API_KEY", "LIUM_API_KEY", "SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT", "CLORE_API_KEY"}

// readCredentials reads the file, then the environment over it. A file that is not there is
// no keys; one that others may read is refused, since it holds keys that spend money.
func readCredentials(path string) (map[string]string, error) {
	creds := map[string]string{}
	fh, err := os.Open(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return nil, err
	default:
		defer fh.Close()
		if st, err := fh.Stat(); err == nil && st.Mode().Perm()&0o077 != 0 {
			return nil, fmt.Errorf("%s may be read by others (mode %o): chmod 600 it", path, st.Mode().Perm())
		}
		sc := bufio.NewScanner(fh)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			line = strings.TrimPrefix(line, "export ")
			k, v, ok := strings.Cut(line, "=")
			if !ok {
				continue
			}
			k, v = strings.TrimSpace(k), strings.TrimSpace(v)
			if len(v) >= 2 && (v[0] == '"' && v[len(v)-1] == '"' || v[0] == '\'' && v[len(v)-1] == '\'') {
				v = v[1 : len(v)-1]
			}
			creds[k] = v
		}
		if err := sc.Err(); err != nil {
			return nil, err
		}
	}
	for _, k := range credentialNames {
		if v := os.Getenv(k); v != "" {
			creds[k] = v
		}
	}
	return creds, nil
}

// --- arming ---------------------------------------------------------------------------------------
//
// A market with a key is only looked at until it is armed: nothing is rented on it before
// then. Which are armed is kept in markets.json, with when and why. With no such file, RunPod
// alone is armed, as stet-runpod rented there.

type Arming struct {
	Armed map[string]Armed `json:"armed"`
	// Curators are the markets, besides RunPod, where stet's container sandbox was shown to
	// hold (stet-cloud probe), and machines may run curators. Each machine checks its own
	// sandbox before it takes one, and only grades where it cannot be made.
	Curators map[string]Armed `json:"curators,omitempty"`
}

type Armed struct {
	At   int64  `json:"at"`
	Note string `json:"note,omitempty"`
}

func armingPath(home string) string { return filepath.Join(home, "markets.json") }

func readArming(home string) (*Arming, error) {
	a := &Arming{}
	raw, err := os.ReadFile(armingPath(home))
	if errors.Is(err, os.ErrNotExist) {
		a.Armed = map[string]Armed{"runpod": {Note: "stet-runpod rented here"}}
		return a, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, a); err != nil {
		return nil, fmt.Errorf("%s: %w", armingPath(home), err)
	}
	if a.Armed == nil {
		a.Armed = map[string]Armed{}
	}
	return a, nil
}

func (a *Arming) save(home string) error {
	raw, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(home, 0o700); err != nil {
		return err
	}
	tmp := armingPath(home) + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, armingPath(home))
}

// --- the markets with keys ------------------------------------------------------------------------

// openMarkets makes a client for each market a key is found for, in marketOrder; off says why
// each of the others is off.
func openMarkets(o *Options, creds map[string]string) (on []Market, off map[string]string) {
	off = map[string]string{}
	for _, name := range marketOrder {
		m, why := openMarket(o, name, creds)
		if m == nil {
			off[name] = why
			continue
		}
		on = append(on, m)
	}
	return on, off
}

func openMarket(o *Options, name string, creds map[string]string) (Market, string) {
	missing := func(k string) string { return k + " is not in " + o.Credentials + " or the environment" }
	switch name {
	case "runpod":
		if creds["RUNPOD_API_KEY"] == "" {
			return nil, missing("RUNPOD_API_KEY")
		}
		return newRunPodMarket(newRunPod(o.RunPodAPI, creds["RUNPOD_API_KEY"]), &o.Cloud), ""
	case "vast":
		if creds["VAST_API_KEY"] == "" {
			return nil, missing("VAST_API_KEY")
		}
		return newVast(o.VastAPI, creds["VAST_API_KEY"]), ""
	case "lium":
		if creds["LIUM_API_KEY"] == "" {
			return nil, missing("LIUM_API_KEY")
		}
		return newLium(o.LiumAPI, creds["LIUM_API_KEY"]), ""
	case "salad":
		for _, k := range []string{"SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT"} {
			if creds[k] == "" {
				return nil, missing(k)
			}
		}
		return newSalad(o.SaladAPI, creds["SALAD_API_KEY"], creds["SALAD_ORG"], creds["SALAD_PROJECT"]), ""
	case "clore":
		if creds["CLORE_API_KEY"] == "" {
			return nil, missing("CLORE_API_KEY")
		}
		return newClore(o.CloreAPI, creds["CLORE_API_KEY"]), ""
	}
	return nil, "no such market"
}

// --- one JSON API, for the markets but RunPod -----------------------------------------------------

// api is one market's JSON API: its base, and the header its key goes in.
type api struct {
	market string
	base   string
	header string // Authorization, X-API-Key, Salad-Api-Key, auth
	value  string // what the header says: "Bearer " + the key, or the key
	http   *http.Client
	sleep  func(context.Context, time.Duration) error
}

func newAPI(market, base, header, value string) *api {
	return &api{market: market, base: strings.TrimRight(base, "/"), header: header, value: value,
		http: &http.Client{Timeout: 60 * time.Second}, sleep: sleepCtx}
}

// APIError is a market saying no.
type APIError struct {
	Market string
	Status int
	Detail string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s said %d: %s", marketTitle(e.Market), e.Status, e.Detail)
}

// do makes one call. A 429 is waited out; a read that fails on the way, or that the market
// fails at, is asked again. Nothing else is: a rental asked twice could be two.
func (a *api) do(ctx context.Context, method, path string, body any, into any) (int, error) {
	return a.doWith(ctx, method, path, nil, body, into)
}

// doWith is do, with more headers.
func (a *api) doWith(ctx context.Context, method, path string, headers map[string]string, body any, into any) (int, error) {
	var raw []byte
	if body != nil {
		var err error
		if raw, err = json.Marshal(body); err != nil {
			return 0, err
		}
	}
	reads := method == "GET" || (method == "POST" && into != nil && a.readOnlyPost(path))
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, a.base+path, bytes.NewReader(raw))
		if err != nil {
			return 0, err
		}
		if a.header != "" && a.value != "" {
			req.Header.Set(a.header, a.value)
		}
		req.Header.Set("Accept", "application/json")
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := a.http.Do(req)
		if err != nil {
			if reads && attempt < 3 && ctx.Err() == nil {
				if err := a.sleep(ctx, time.Duration(attempt+1)*2*time.Second); err != nil {
					return 0, err
				}
				continue
			}
			return 0, fmt.Errorf("%s could not be reached: %w", marketTitle(a.market), err)
		}
		data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
		resp.Body.Close()
		if err != nil {
			return resp.StatusCode, err
		}
		switch {
		case resp.StatusCode == 429 && attempt < 6:
			wait, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
			if err := a.sleep(ctx, time.Duration(max(wait, 2))*time.Second); err != nil {
				return 0, err
			}
			continue
		case resp.StatusCode >= 500 && reads && attempt < 3:
			if err := a.sleep(ctx, time.Duration(attempt+1)*3*time.Second); err != nil {
				return 0, err
			}
			continue
		case resp.StatusCode >= 300:
			return resp.StatusCode, &APIError{Market: a.market, Status: resp.StatusCode, Detail: errorDetail(data)}
		}
		if into != nil && len(data) > 0 {
			if err := json.Unmarshal(data, into); err != nil {
				return resp.StatusCode, fmt.Errorf("%s's answer to %s %s: %w", marketTitle(a.market), method, path, err)
			}
		}
		return resp.StatusCode, nil
	}
}

// readOnlyPost is a POST that only asks: Vast's offer search, Salad's availability.
func (a *api) readOnlyPost(path string) bool {
	return strings.HasPrefix(path, "/api/v0/bundles") || strings.Contains(path, "/availability/") || strings.HasSuffix(path, "/log-entries")
}

// errorDetail is what an error answer says, from whichever field the market puts it in.
func errorDetail(data []byte) string {
	var e map[string]any
	if json.Unmarshal(data, &e) == nil {
		for _, k := range []string{"detail", "msg", "message", "error", "title"} {
			if s, ok := e[k].(string); ok && s != "" {
				if v, ok := e["validation_errors"]; ok { // Lium says which fields
					raw, _ := json.Marshal(v)
					s += ": " + string(raw)
				}
				return s
			}
		}
	}
	s := strings.TrimSpace(string(data))
	if len(s) > 300 {
		s = s[:300] + "…"
	}
	if s == "" {
		s = "(nothing said)"
	}
	return s
}

// --- shared helpers --------------------------------------------------------------------------------

// workerCommand starts worker.sh from the environment, where it is kept base64'd in
// STET_WORKER_SCRIPT, on a market whose start command is one line: as the container's only
// process, which never exits.
const workerCommand = `bash -c "` + workerInner + `"`

// workerInner is what bash -c is given to start it.
const workerInner = `echo $STET_WORKER_SCRIPT | base64 -d > /root/stet-worker.sh && exec bash /root/stet-worker.sh`

// sshKey is the public key a market that will have one is given: --ssh's, or this machine's
// own, so that a machine there can be looked into.
func sshKey(l Launch) string {
	if l.SSHKey != "" {
		return l.SSHKey
	}
	home, _ := os.UserHomeDir()
	for _, name := range []string{"id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"} {
		if raw, err := os.ReadFile(filepath.Join(home, ".ssh", name)); err == nil {
			return strings.TrimSpace(string(raw))
		}
	}
	return ""
}

// newID is a random ID, for idempotency keys and names that must be unique.
func newID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// expectedLife is how long a rented machine is taken to live, to spread a fee paid once a
// machine over its hours when offers are compared.
const expectedLife = time.Hour

// cudaAtLeast says whether a CUDA version ("13.2") is at least another ("13.0"); an unknown
// one is not.
func cudaAtLeast(have, want string) bool {
	if want == "" {
		return true
	}
	if _, err := strconv.ParseFloat(strings.TrimSpace(have), 64); err != nil {
		return false
	}
	// 13.10 is newer than 13.2: major, then minor, each as a number.
	hm, hn := splitVersion(have)
	wm, wn := splitVersion(want)
	if hm != wm {
		return hm > wm
	}
	return hn >= wn
}

func splitVersion(v string) (int, int) {
	major, minor, _ := strings.Cut(strings.TrimSpace(v), ".")
	m, _ := strconv.Atoi(major)
	n, _ := strconv.Atoi(minor)
	return m, n
}

// driverCUDA is the newest CUDA an NVIDIA driver takes, by its major version: 580 is CUDA
// 13.0, 590 13.1, 595 13.2; older ones are under 13 and say "12".
func driverCUDA(driver string) string {
	major, _, _ := strings.Cut(strings.TrimSpace(driver), ".")
	n, err := strconv.Atoi(major)
	switch {
	case err != nil:
		return ""
	case n >= 595:
		return "13.2"
	case n >= 590:
		return "13.1"
	case n >= 580:
		return "13.0"
	}
	return "12"
}

// sortOffers puts the cheapest first, the more trusted first where two cost the same.
func sortOffers(os []Offer) {
	rank := func(t string) int {
		switch t {
		case "secure", "datacenter":
			return 0
		case "verified":
			return 1
		}
		return 2
	}
	sort.SliceStable(os, func(i, j int) bool {
		if os[i].Price != os[j].Price {
			return os[i].Price < os[j].Price
		}
		return rank(os[i].Trust) < rank(os[j].Trust)
	})
}
