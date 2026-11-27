package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// --- Vast.ai (https://docs.vast.ai/api-reference) ----------------------------------------------
//
// Vast lists each host's machine as an offer of its own, with what its host charges for the
// GPU, for disk by the month, and for each GB downloaded. Its API takes a key limited to
// searching offers and to its own instances (misc, instance_read, instance_write).

const vastAPI = "https://console.vast.ai"

type vastMarket struct{ a *api }

func newVast(base, key string) *vastMarket {
	return &vastMarket{a: newAPI("vast", base, "Authorization", "Bearer "+key)}
}

func (m *vastMarket) Name() string { return "vast" }

// A machine on Vast can end itself with CONTAINER_API_KEY, a key that reaches only it.
func (m *vastMarket) Caps() Caps { return Caps{Rents: true, SelfEnd: true} }

// vastOffer is one offer, of the fields read here.
type vastOffer struct {
	ID           int64   `json:"id"`
	GPUName      string  `json:"gpu_name"`
	GPURAM       float64 `json:"gpu_ram"` // MB
	NumGPUs      int     `json:"num_gpus"`
	DPHBase      float64 `json:"dph_base"`
	DPHTotal     float64 `json:"dph_total"`
	StorageCost  float64 `json:"storage_cost"`   // dollars a GB a month
	InetDownCost float64 `json:"inet_down_cost"` // dollars a GB
	InetDown     float64 `json:"inet_down"`      // Mbps
	CUDAMaxGood  float64 `json:"cuda_max_good"`
	Verification string  `json:"verification"`
	HostingType  int     `json:"hosting_type"` // 1: a data centre
	Geolocation  string  `json:"geolocation"`
	GPUArch      string  `json:"gpu_arch"`
	MachineID    int64   `json:"machine_id"`
	CPUName      string  `json:"cpu_name"`
}

// vastNames are Vast's names for cards whose names elsewhere differ from them.
var vastNames = map[string]string{
	"RTX 4080S": "RTX 4080 SUPER", "RTX 4070S": "RTX 4070 SUPER", "RTX 4070S Ti": "RTX 4070 Ti SUPER",
	"RTX 2000Ada": "RTX 2000 Ada Generation", "RTX 4000Ada": "RTX 4000 Ada Generation",
	"RTX 5000Ada": "RTX 5000 Ada Generation", "RTX 6000Ada": "RTX 6000 Ada Generation",
	"RTX PRO 6000 WS": "RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000 S": "RTX PRO 6000 Blackwell Server Edition",
	"RTX PRO 4000": "RTX PRO 4000 Blackwell", "RTX PRO 4500": "RTX PRO 4500 Blackwell", "RTX PRO 5000": "RTX PRO 5000 Blackwell",
}

// vastHost is the machine an offer is on, as Vast numbers it ("": not said).
func vastHost(id int64) string {
	if id <= 0 {
		return ""
	}
	return strconv.FormatInt(id, 10)
}

func vastCard(name string) string {
	if c, ok := vastNames[name]; ok {
		return c
	}
	return shortGPU(name)
}

// vastQuery is the search for offers that meet the need: verified hosts, one NVIDIA GPU, a
// driver new enough, the CPUs, memory and disk asked for, a host that keeps its machines up and
// downloads quickly, and is there for three hours more at least.
func vastQuery(n Need) map[string]any {
	q := map[string]any{
		"type": "ondemand", "limit": 2000, "order": [][]string{{"dph_total", "asc"}},
		"verified": map[string]any{"eq": true}, "rentable": map[string]any{"eq": true}, "rented": map[string]any{"eq": false},
		"external": map[string]any{"eq": false}, "num_gpus": map[string]any{"eq": 1}, "gpu_arch": map[string]any{"eq": "nvidia"},
		"reliability": map[string]any{"gte": 0.97}, "inet_down": map[string]any{"gte": 100}, "duration": map[string]any{"gte": 3 * 3600},
	}
	if n.MinGPUMemGB > 0 {
		q["gpu_ram"] = map[string]any{"gte": n.MinGPUMemGB * 1000}
	}
	if n.MinCUDA != "" {
		if v, err := strconv.ParseFloat(n.MinCUDA, 64); err == nil {
			q["cuda_max_good"] = map[string]any{"gte": v}
		}
	}
	if n.MinCPUs > 0 {
		q["cpu_cores_effective"] = map[string]any{"gte": n.MinCPUs}
	}
	if n.MinRAMGB > 0 {
		q["cpu_ram"] = map[string]any{"gte": n.MinRAMGB * 1000}
	}
	if n.DiskGB > 0 {
		q["disk_space"] = map[string]any{"gte": n.DiskGB}
	}
	return q
}

// vastPrice is an offer's price all in, an hour: the GPU, the disk for the hour, and the
// image and the season's libraries downloaded once (about 3 GB), over the machine's life.
func vastPrice(o vastOffer, n Need) float64 {
	return o.DPHBase + float64(n.DiskGB)*o.StorageCost/720 + 3*o.InetDownCost/expectedLife.Hours()
}

func (m *vastMarket) Offers(ctx context.Context, n Need) ([]Offer, error) {
	var out struct {
		Offers []vastOffer `json:"offers"`
	}
	if _, err := m.a.do(ctx, "POST", "/api/v0/bundles/", vastQuery(n), &out); err != nil {
		return nil, err
	}
	var offers []Offer
	for _, o := range out.Offers {
		if o.NumGPUs != 1 {
			continue
		}
		trust := "verified"
		if o.HostingType == 1 {
			trust = "datacenter"
		}
		offers = append(offers, Offer{
			Market: "vast", Card: vastCard(o.GPUName), Model: o.GPUName, Name: o.GPUName, ID: strconv.FormatInt(o.ID, 10),
			Maker: "NVIDIA", MemoryGB: int(math.Round(o.GPURAM / 1024)), Price: vastPrice(o, n), Stock: 1,
			CUDA: strconv.FormatFloat(o.CUDAMaxGood, 'f', -1, 64), Trust: trust, Region: o.Geolocation,
			Host: vastHost(o.MachineID), CPU: strings.TrimSpace(o.CPUName),
		})
	}
	return offers, nil
}

// Rent rents one offer: the image, with the worker as its only process (bash -c, as on
// RunPod), its settings in its environment, and the disk asked for, labelled with the worker
// key's handle. Vast refuses an offer that has gone (404, 410): the next may do. An answer lost
// on the way is looked for by its label before anything is said, so that none is rented twice.
func (m *vastMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	env := map[string]string{"STET_MARKET": "vast"}
	for k, v := range l.Env {
		env[k] = v
	}
	body := map[string]any{
		"client_id": "me", "image": l.Image, "label": l.Name, "disk": l.Need.DiskGB, "env": env,
		"runtype": "args", "args": []string{"bash", "-c", l.Script}, "cancel_unavail": true, "target_state": "running",
	}
	var out struct {
		Success     bool   `json:"success"`
		NewContract int64  `json:"new_contract"`
		Msg         string `json:"msg"`
	}
	status, err := m.a.do(ctx, "PUT", "/api/v0/asks/"+url.PathEscape(o.ID)+"/", body, &out)
	switch {
	case err == nil && out.Success && out.NewContract > 0:
		return &Rental{Market: "vast", ID: strconv.FormatInt(out.NewContract, 10), Name: l.Name, Card: o.Card,
			Price: o.Price, State: "starting", Status: "created", Created: time.Now()}, nil
	case err == nil:
		return nil, fmt.Errorf("Vast.ai answered %d to a new instance, without one: %s", status, out.Msg)
	case status == 404 || status == 410 || (status == 400 && strings.Contains(err.Error(), "no_such_ask")):
		// Gone between the search and the rent: Vast says so with a 400 too.
		return nil, fmt.Errorf("%w (%v)", errNoRoom, err)
	case status == 0 || status >= 500:
		if rs, lerr := m.Rentals(ctx); lerr == nil {
			for i := range rs {
				if rs[i].Name == l.Name && rs[i].live() {
					return &rs[i], nil
				}
			}
		}
	}
	return nil, err
}

// vastPort is one host port mapping reported on a Vast.ai instance.
type vastPort struct {
	HostPort string `json:"HostPort"`
}

// vastInstance is one of the account's instances, of the fields read here.
type vastInstance struct {
	ID             int64                 `json:"id"`
	Label          *string               `json:"label"`
	ActualStatus   *string               `json:"actual_status"`
	IntendedStatus string                `json:"intended_status"`
	CurState       string                `json:"cur_state"`
	StatusMsg      *string               `json:"status_msg"`
	GPUName        string                `json:"gpu_name"`
	DPHTotal       float64               `json:"dph_total"`
	StartDate      float64               `json:"start_date"`
	PublicIPAddr   string                `json:"public_ipaddr"`
	Ports          map[string][]vastPort `json:"ports"`
}

// vastState is what an instance's state comes to: starting until its container runs, and
// stopped once it has exited, gone offline, or been stopped (which still bills its disk).
func vastState(i vastInstance) (state, status string) {
	s := ""
	if i.ActualStatus != nil {
		s = *i.ActualStatus
	}
	switch {
	case s == "running":
		return "running", s
	case s == "exited" || s == "offline" || s == "stopped" || i.IntendedStatus == "stopped" || i.CurState == "stopped":
		return "stopped", s
	case s == "" || s == "loading" || s == "created" || s == "scheduling":
		return "starting", s
	}
	return s, s
}

func (m *vastMarket) Rentals(ctx context.Context) ([]Rental, error) {
	var all []Rental
	after := ""
	for page := 0; page < 100; page++ {
		q := url.Values{"limit": {"25"}}
		if after != "" {
			q.Set("after_token", after)
		}
		var out struct {
			Instances []vastInstance `json:"instances"`
			NextToken *string        `json:"next_token"`
		}
		// The path ends in a slash: without it Vast answers 301.
		if _, err := m.a.do(ctx, "GET", "/api/v1/instances/?"+q.Encode(), nil, &out); err != nil {
			return nil, err
		}
		for _, i := range out.Instances {
			state, status := vastState(i)
			name := ""
			if i.Label != nil {
				name = *i.Label
			}
			r := Rental{Market: "vast", ID: strconv.FormatInt(i.ID, 10), Name: name, Card: vastCard(i.GPUName),
				Price: i.DPHTotal, State: state, Status: status, Created: time.Unix(int64(i.StartDate), 0)}
			r.Phase = status
			if r.Phase == "" {
				r.Phase = "scheduling"
			}
			r.Up = map[bool]string{true: "yes", false: "no"}[status == "running"]
			if i.StatusMsg != nil {
				r.Detail = strings.TrimSpace(*i.StatusMsg)
			}
			if r.Detail == "" {
				r.Detail = r.Phase
			}
			all = append(all, r)
		}
		if out.NextToken == nil || *out.NextToken == "" {
			return all, nil
		}
		after = *out.NextToken
	}
	return all, fmt.Errorf("Vast listed more than 2,500 instances")
}

// Logs are what the instance's container has said: Vast puts the last n lines where they can be
// fetched, a moment after it is asked.
func (m *vastMarket) Logs(ctx context.Context, id string, n int) ([]string, error) {
	var out struct {
		ResultURL string `json:"result_url"`
	}
	if _, err := m.a.do(ctx, "PUT", "/api/v0/instances/request_logs/"+url.PathEscape(id)+"/", map[string]any{"tail": strconv.Itoa(n)}, &out); err != nil {
		return nil, err
	}
	if out.ResultURL == "" {
		return nil, errors.New("Vast gave no place to fetch the logs from")
	}
	for try := 0; try < 8; try++ {
		if err := m.a.sleep(ctx, 3*time.Second); err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, "GET", out.ResultURL, nil)
		if err != nil {
			return nil, err
		}
		resp, err := m.a.http.Do(req)
		if err != nil {
			continue
		}
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if resp.StatusCode == 200 && !bytes.Contains(data, []byte("<Error>")) {
			return lastLines(strings.Split(strings.TrimRight(string(data), "\n"), "\n"), n), nil
		}
	}
	return nil, errors.New("Vast's logs were not there to fetch")
}

// End destroys an instance: stopped, its disk would still be billed. One already gone is not
// an error.
func (m *vastMarket) End(ctx context.Context, id string) error {
	status, err := m.a.do(ctx, "DELETE", "/api/v0/instances/"+url.PathEscape(id)+"/", nil, nil)
	if status == 404 {
		return nil
	}
	return err
}

// Check asks who the key belongs to, and the credit left.
func (m *vastMarket) Check(ctx context.Context) (Account, error) {
	var u struct {
		Credit float64 `json:"credit"` // prepaid, and not yet spent
		CanPay bool    `json:"can_pay"`
	}
	if _, err := m.a.do(ctx, "GET", "/api/v0/users/current/", nil, &u); err != nil {
		return Account{}, err
	}
	if _, err := m.a.do(ctx, "GET", "/api/v1/instances/?limit=1", nil, nil); err != nil {
		return Account{}, err
	}
	a := Account{Balance: u.Credit, Known: true}
	if !u.CanPay {
		a.Note = "Vast says the account cannot pay"
	}
	return a, nil
}

// ExaminerOffers searches Vast.ai for verified single RTX 5090 hosts suited to serving the
// Season 3 pinned Inkling-2-Mini vLLM container, preferring gigabit+ hosts (>=1500 Mbps) so the
// vLLM image + 18 GB model weights pull quickly, and falling back to >=500 / >=200 Mbps when
// supply is tight.
func (m *vastMarket) ExaminerOffers(ctx context.Context, maxPrice float64) ([]ExaminerOffer, error) {
	for _, minDown := range []int{1500, 500, 200} {
		q := map[string]any{
			"type": "ondemand", "limit": 50, "order": [][]string{{"dph_total", "asc"}},
			"verified": map[string]any{"eq": true}, "rentable": map[string]any{"eq": true}, "rented": map[string]any{"eq": false},
			"external": map[string]any{"eq": false}, "num_gpus": map[string]any{"eq": 1}, "gpu_name": map[string]any{"eq": "RTX 5090"},
			"cuda_max_good": map[string]any{"gte": 13.2}, "disk_space": map[string]any{"gte": examinerDiskGB},
			"reliability": map[string]any{"gte": 0.97}, "inet_down": map[string]any{"gte": minDown}, "duration": map[string]any{"gte": 3600},
		}
		if maxPrice > 0 {
			q["dph_total"] = map[string]any{"lte": maxPrice}
		}
		var out struct {
			Offers []vastOffer `json:"offers"`
		}
		if _, err := m.a.do(ctx, "POST", "/api/v0/bundles/", q, &out); err != nil {
			return nil, err
		}
		var offers []ExaminerOffer
		for _, o := range out.Offers {
			if o.NumGPUs != 1 {
				continue
			}
			price := o.DPHTotal
			if price <= 0 {
				price = o.DPHBase
			}
			if maxPrice > 0 && price > maxPrice+1e-9 {
				continue
			}
			offers = append(offers, ExaminerOffer{
				ID: strconv.FormatInt(o.ID, 10), Card: vastCard(o.GPUName), Price: price,
				CUDA: o.CUDAMaxGood, Region: o.Geolocation, InetDown: o.InetDown,
			})
		}
		if len(offers) > 0 {
			return offers, nil
		}
	}
	return nil, nil
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// examinerOnstartScript builds the Vast.ai startup script that:
//  1. Materializes the per-instance TLS certificate & private key (0600) in /tmp from hex-encoded
//     environment variables (with /etc/environment fallback).
//  2. Spawns a detached container-side Python 3 self-destruct timer calling
//     DELETE https://console.vast.ai/api/v0/instances/$CONTAINER_ID/ after STET_EXAMINER_TTL_SECONDS
//     so hard timeout enforcement survives stet-cloud exiting or losing connectivity.
//  3. Launches vLLM over HTTPS (--ssl-certfile / --ssl-keyfile) with the per-instance Bearer key,
//     streaming non-secret startup logs to container stdout (/proc/1/fd/1) so Vast logs remain
//     observable while onstart exits 0 promptly, and self-destructing if vLLM exits unexpectedly.
func examinerOnstartScript(l ExaminerLaunch, servedName string) string {
	args := []string{"vllm", "serve", shellQuote(l.Model)}
	if l.Revision != "" {
		args = append(args, "--revision", shellQuote(l.Revision))
	}
	args = append(args,
		"--served-model-name", shellQuote(servedName),
		"--reasoning-parser", "inkling",
		"--tensor-parallel-size", "1",
		"--max-model-len", "4096",
		"--max-num-seqs", "32",
		"--gpu-memory-utilization", "0.88",
		"--host", "0.0.0.0", "--port", "8000",
		"--ssl-certfile", "/tmp/stet-examiner-cert.pem",
		"--ssl-keyfile", "/tmp/stet-examiner-key.pem",
		"--api-key", `"$STET_EXAMINER_API_KEY"`,
	)
	return strings.Join([]string{
		`set -eu`,
		`export PATH="/usr/local/bin:/opt/venv/bin:/venv/bin:$PATH"`,
		`LOG=/tmp/stet-vllm.log`,
		`[ -w /proc/1/fd/1 ] && LOG=/proc/1/fd/1`,
		`echo "[stet-examiner] onstart begin" >>"$LOG" 2>&1 || true`,
		`python3 -c 'import os` + "\n" +
			`def _env(n):` + "\n" +
			`    v = os.environ.get(n, "").strip()` + "\n" +
			`    if v: return v` + "\n" +
			`    try:` + "\n" +
			`        for line in open("/etc/environment", encoding="utf-8", errors="ignore"):` + "\n" +
			`            line = line.strip()` + "\n" +
			`            if line.startswith("export "): line = line[7:].strip()` + "\n" +
			`            if line.startswith(n + "="): return line.split("=", 1)[1].strip().strip("\"\x27")` + "\n" +
			`    except OSError: pass` + "\n" +
			`    return ""` + "\n" +
			`for p,k in (("/tmp/stet-examiner-cert.pem","STET_EXAMINER_CERT_HEX"),("/tmp/stet-examiner-key.pem","STET_EXAMINER_KEY_HEX")):` + "\n" +
			`    raw = bytes.fromhex(_env(k))` + "\n" +
			`    if not raw: raise SystemExit(f"missing {k}")` + "\n" +
			`    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)` + "\n" +
			`    with os.fdopen(fd, "wb") as fh:` + "\n" +
			`        fh.write(raw)' >>"$LOG" 2>&1`,
		`cat >/tmp/stet-selfdestruct.py <<'PY'`,
		`import os, sys, time, urllib.error, urllib.request`,
		`def _env(name):`,
		`    v = os.environ.get(name, "").strip()`,
		`    if v: return v`,
		`    try:`,
		`        for line in open("/etc/environment", encoding="utf-8", errors="ignore"):`,
		`            line = line.strip()`,
		`            if line.startswith("export "): line = line[7:].strip()`,
		`            if line.startswith(name + "="):`,
		`                return line.split("=", 1)[1].strip().strip("\"'")`,
		`    except OSError:`,
		`        pass`,
		`    return ""`,
		`delay = max(0, int(sys.argv[1])) if len(sys.argv) > 1 else 0`,
		`if delay > 0: time.sleep(delay)`,
		`cid, key = _env("CONTAINER_ID"), _env("CONTAINER_API_KEY")`,
		`if not cid or not key: sys.exit(0)`,
		`url = f"https://console.vast.ai/api/v0/instances/{cid}/"`,
		`for attempt in range(30):`,
		`    req = urllib.request.Request(url, method="DELETE", headers={"Authorization": f"Bearer {key}"})`,
		`    try:`,
		`        with urllib.request.urlopen(req, timeout=15) as r:`,
		`            if r.status in (200, 204, 404): break`,
		`    except urllib.error.HTTPError as e:`,
		`        if e.code == 404: break`,
		`    except Exception:`,
		`        pass`,
		`    time.sleep(30 if attempt < 10 else 120)`,
		`PY`,
		`cat >/tmp/stet-vllm.sh <<'SH'`,
		`#!/bin/sh`,
		`export PATH="/usr/local/bin:/opt/venv/bin:/venv/bin:$PATH"`,
		`if [ -z "${STET_EXAMINER_API_KEY:-}" ] && [ -r /etc/environment ]; then`,
		`  STET_EXAMINER_API_KEY="$(python3 -c 'import re` + "\n" +
			`for l in open("/etc/environment", encoding="utf-8", errors="ignore"):` + "\n" +
			`    m = re.match(r"^(?:export\s+)?STET_EXAMINER_API_KEY=(.*)$", l.strip())` + "\n" +
			`    if m: print(m.group(1).strip().strip("\"\x27")); break')"`,
		`  export STET_EXAMINER_API_KEY`,
		`fi`,
		`echo "[stet-examiner] launching vllm ($(command -v vllm || echo missing))"`,
		strings.Join(args, " "),
		`rc=$?`,
		`echo "[stet-examiner] vllm exited rc=$rc; self-destructing in 15s"`,
		`python3 /tmp/stet-selfdestruct.py 15`,
		`SH`,
		`chmod 700 /tmp/stet-vllm.sh`,
		`nohup python3 /tmp/stet-selfdestruct.py "${STET_EXAMINER_TTL_SECONDS:-2700}" >/tmp/stet-ttl.log 2>&1 &`,
		`nohup sh /tmp/stet-vllm.sh >>"$LOG" 2>&1 &`,
	}, "\n")
}

// RentExaminer rents one Vast.ai RTX 5090 running the pinned immutable vLLM OpenAI-compatible
// server container over peer-verified HTTPS (--ssl-certfile/--ssl-keyfile) with a per-instance
// Bearer API key, TCP port 8000 exposed, and a container-side self-destruct TTL watchdog.
func (m *vastMarket) RentExaminer(ctx context.Context, offerID string, l ExaminerLaunch) (string, error) {
	if len(l.CertPEM) == 0 || len(l.KeyPEM) == 0 {
		return "", errors.New("examiner launch requires per-instance TLS cert and key")
	}
	ttl := l.TTLSeconds
	if ttl <= 0 {
		ttl = 45 * 60
	}
	env := map[string]string{
		"-p 8000:8000":              "1",
		"STET_EXAMINER_CERT_HEX":    hex.EncodeToString(l.CertPEM),
		"STET_EXAMINER_KEY_HEX":     hex.EncodeToString(l.KeyPEM),
		"STET_EXAMINER_API_KEY":     l.APIKey,
		"STET_EXAMINER_TTL_SECONDS": strconv.Itoa(ttl),
	}
	if l.HFToken != "" {
		env["HF_TOKEN"] = l.HFToken
	}
	disk := l.DiskGB
	if disk <= 0 {
		disk = examinerDiskGB
	}
	servedName := l.Model
	if l.Revision != "" {
		servedName = l.Model + "@" + l.Revision
	}
	body := map[string]any{
		"client_id":      "me",
		"image":          l.Image,
		"label":          l.Label,
		"disk":           disk,
		"env":            env,
		"runtype":        "ssh",
		"onstart":        examinerOnstartScript(l, servedName),
		"cancel_unavail": true,
		"target_state":   "running",
	}
	var out struct {
		Success     bool   `json:"success"`
		NewContract int64  `json:"new_contract"`
		Msg         string `json:"msg"`
	}
	status, err := m.a.do(ctx, "PUT", "/api/v0/asks/"+url.PathEscape(offerID)+"/", body, &out)
	switch {
	case err == nil && out.Success && out.NewContract > 0:
		return strconv.FormatInt(out.NewContract, 10), nil
	case err == nil:
		return "", fmt.Errorf("Vast.ai answered %d to a new examiner instance, without one: %s", status, out.Msg)
	case status == 404 || status == 410 || (status == 400 && strings.Contains(err.Error(), "no_such_ask")):
		return "", fmt.Errorf("%w (%v)", errNoRoom, err)
	case status == 0 || status >= 500:
		if rs, lerr := m.ListExaminers(ctx); lerr == nil {
			for _, r := range rs {
				if r.Label == l.Label && r.State != "stopped" {
					return r.ID, nil
				}
			}
		}
	}
	return "", err
}

// ListExaminers lists every Vast.ai instance labelled as a stet-cloud ephemeral examiner,
// including its mapped public IP and host port for container port 8000/tcp.
func (m *vastMarket) ListExaminers(ctx context.Context) ([]ExaminerInstance, error) {
	var all []ExaminerInstance
	after := ""
	for page := 0; page < 100; page++ {
		q := url.Values{"limit": {"25"}}
		if after != "" {
			q.Set("after_token", after)
		}
		var out struct {
			Instances []vastInstance `json:"instances"`
			NextToken *string        `json:"next_token"`
		}
		if _, err := m.a.do(ctx, "GET", "/api/v1/instances/?"+q.Encode(), nil, &out); err != nil {
			return nil, err
		}
		for _, i := range out.Instances {
			label := ""
			if i.Label != nil {
				label = *i.Label
			}
			if !strings.HasPrefix(label, examinerLabelPrefix) {
				continue
			}
			state, status := vastState(i)
			hostPort := ""
			for _, p := range i.Ports["8000/tcp"] {
				if hp := strings.TrimSpace(p.HostPort); hp != "" {
					hostPort = hp
					break
				}
			}
			var created time.Time
			if i.StartDate > 0 {
				created = time.Unix(int64(i.StartDate), 0)
			}
			all = append(all, ExaminerInstance{
				ID: strconv.FormatInt(i.ID, 10), Label: label, State: state, Status: status,
				PublicIP: strings.TrimSpace(i.PublicIPAddr), HostPort: hostPort,
				Price: i.DPHTotal, Created: created,
			})
		}
		if out.NextToken == nil || *out.NextToken == "" {
			return all, nil
		}
		after = *out.NextToken
	}
	return all, fmt.Errorf("Vast listed more than 2,500 instances")
}

func (m *vastMarket) EndExaminer(ctx context.Context, id string) error {
	return m.End(ctx, id)
}
