package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// --- Lium (https://docs.lium.io) ---------------------------------------------------------------
//
// Lium lists each provider's node as an executor, with its GPUs, its driver and a price for each
// GPU. Its nodes are in two tiers: secure, and spot, which a provider may take back at any time;
// only secure ones are offered. Its key is sent as X-API-Key.

const liumAPI = "https://lium.io/api"

type liumMarket struct{ a *api }

func newLium(base, key string) *liumMarket {
	return &liumMarket{a: newAPI("lium", base, "X-API-Key", key)}
}

func (m *liumMarket) Name() string { return "lium" }

func (m *liumMarket) Caps() Caps { return Caps{Rents: true} }

// The image a machine on Lium runs: Lium's own Ubuntu 24.04 with CUDA 13.0, which its nodes
// keep pulled, and whose pull is not billed (a pod is billed from RUNNING).
const liumImage, liumImageTag = "daturaai/pytorch", "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind-lium1"

// liumCommand is the worker as Lium's start command: Lium refuses one with a shell's
// characters unless it is all one bash -c '…'.
const liumCommand = `bash -c '` + workerInner + `'`

// liumHours is how long Lium lets a pod live, whatever becomes of the fleet: its own backstop.
const liumHours = 3

// liumExecutor is one node, of the fields read here. Memory and disk are in kB.
type liumExecutor struct {
	ID                   string  `json:"id"`
	MachineName          string  `json:"machine_name"`
	PricePerGPU          float64 `json:"price_per_gpu"`
	GPUCount             int     `json:"gpu_count"`
	AvailableGPUCount    int     `json:"available_gpu_count"`
	MinGPUCountForRental *int    `json:"min_gpu_count_for_rental"`
	Tier                 string  `json:"tier"`
	FrequentRentFailures bool    `json:"frequent_rent_failures"`
	MaxCUDAVersion       float64 `json:"max_cuda_version"`
	Specs                struct {
		CPU struct {
			Count int `json:"count"`
		} `json:"cpu"`
		GPU struct {
			Driver  string `json:"driver"`
			Details []struct {
				Name     string  `json:"name"`
				Capacity float64 `json:"capacity"` // MiB
			} `json:"details"`
		} `json:"gpu"`
		RAM struct {
			Total float64 `json:"total"`
		} `json:"ram"`
		HardDisk struct {
			Free float64 `json:"free"`
		} `json:"hard_disk"`
	} `json:"specs"`
	Location struct {
		CountryCode string `json:"country_code"`
	} `json:"location"`
}

// liumOffer is a node as an offer of one of its GPUs, if it has one to give that meets the need:
// a secure node, free, that rents a GPU at a time, with a driver new enough, and its CPUs,
// memory and disk shared out among its GPUs enough for one.
func liumOffer(e liumExecutor, n Need) (Offer, bool) {
	gpus := max(e.GPUCount, 1)
	if e.Tier != "secure" || e.AvailableGPUCount < 1 || e.FrequentRentFailures || e.PricePerGPU <= 0 {
		return Offer{}, false
	}
	if e.MinGPUCountForRental != nil && *e.MinGPUCountForRental > 1 {
		return Offer{}, false
	}
	cuda := driverCUDA(e.Specs.GPU.Driver)
	if e.MaxCUDAVersion > 0 {
		cuda = strconv.FormatFloat(e.MaxCUDAVersion, 'f', -1, 64)
	}
	if !cudaAtLeast(cuda, n.MinCUDA) {
		return Offer{}, false
	}
	if e.Specs.CPU.Count/gpus < n.MinCPUs || e.Specs.RAM.Total/1e6/float64(gpus) < float64(n.MinRAMGB) || e.Specs.HardDisk.Free/1e6 < float64(n.DiskGB) {
		return Offer{}, false
	}
	mem := 0.0
	if len(e.Specs.GPU.Details) > 0 {
		mem = e.Specs.GPU.Details[0].Capacity
	}
	maker := "NVIDIA"
	if !strings.HasPrefix(e.MachineName, "NVIDIA") {
		maker = strings.SplitN(e.MachineName, " ", 2)[0]
	}
	return Offer{
		Market: "lium", Card: shortGPU(e.MachineName), Model: e.MachineName, Name: shortGPU(e.MachineName), ID: e.ID,
		Maker: maker, MemoryGB: int(math.Round(mem / 1024)), Price: e.PricePerGPU, Stock: e.AvailableGPUCount,
		CUDA: cuda, Trust: e.Tier, Region: e.Location.CountryCode,
	}, true
}

// Offers are the nodes Lium lists, of those that meet the need. The list is the one its site
// shows, which needs no key.
func (m *liumMarket) Offers(ctx context.Context, n Need) ([]Offer, error) {
	var nodes []liumExecutor
	if _, err := m.a.do(ctx, "GET", "/executors?view=summary&reliable_only=true", nil, &nodes); err != nil {
		return nil, err
	}
	var out []Offer
	for _, e := range nodes {
		if o, ok := liumOffer(e, n); ok {
			out = append(out, o)
		}
	}
	return out, nil
}

// Rent rents one GPU on a node. Lium takes a container's settings only from a template, so each
// machine has a template of its own, used once: its environment, the worker script among it,
// and the worker as the container's command. The rent pays no more than the price offered
// (409 when it rose: the next offer may do), ends the pod by itself after liumHours, and is
// asked again, with the same idempotency key, when its answer is lost.
func (m *liumMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	key := sshKey(l)
	if key == "" {
		return nil, errors.New("Lium rents only with an SSH public key: there is none in ~/.ssh")
	}
	// A Lium container's own disk is a few GB, most of it the image's: stet, its libraries
	// and what it fetches go on the pod's volume, /workspace.
	env := map[string]string{"STET_MARKET": "lium", "STET_WORKER_SCRIPT": gzipB64(l.Script),
		"STET_WORKER_HOME": "/workspace/root", "STET_WORKER_OPT": "/workspace/opt/stet"}
	for k, v := range l.Env {
		env[k] = v
	}
	var tpl struct {
		ID string `json:"id"`
	}
	if _, err := m.a.do(ctx, "POST", "/templates", map[string]any{
		"name": "stet-" + l.Name + "-" + newID()[:8], "docker_image": liumImage, "docker_image_tag": liumImageTag,
		"category": "PYTORCH", "volumes": []string{"/workspace"}, "environment": env, "startup_commands": liumCommand,
		"is_private": true, "one_time_template": true,
	}, &tpl); err != nil {
		return nil, fmt.Errorf("Lium's template for %s: %w", l.Name, err)
	}
	if tpl.ID == "" {
		return nil, errors.New("Lium made a template, and gave no ID for it")
	}
	body := map[string]any{
		"pod_name": l.Name, "user_public_key": key, "template_id": tpl.ID, "gpu_count": 1, "termination_hours": liumHours,
		"expected_price_per_gpu_hour": o.Price, "enable_backups": false, "enable_jupyter": false, "enable_volume_encryption": false,
	}
	idem := map[string]string{"Idempotency-Key": newID()}
	var out struct {
		Success bool   `json:"success"`
		PodID   string `json:"pod_id"`
	}
	var status int
	var err error
	for try := 0; try < 2; try++ {
		status, err = m.a.doWith(ctx, "POST", "/executors/"+url.PathEscape(o.ID)+"/rent", idem, body, &out)
		if !(status == 0 || status >= 500) || ctx.Err() != nil {
			break
		}
	}
	switch {
	case err == nil && out.PodID != "":
		return &Rental{Market: "lium", ID: out.PodID, Name: l.Name, Card: o.Card, Price: o.Price, State: "starting", Status: "PENDING", Created: time.Now()}, nil
	case err == nil:
		return nil, fmt.Errorf("Lium answered %d to a rent, without a pod", status)
	case status == 400 || status == 404 || status == 409 || status == 410:
		return nil, fmt.Errorf("%w (%v)", errNoRoom, err)
	}
	return nil, err
}

// liumPod is one of the account's pods, of the fields read here.
type liumPod struct {
	ID          string   `json:"id"`
	PodName     string   `json:"pod_name"`
	GPUName     string   `json:"gpu_name"`
	Status      string   `json:"status"`
	Price       *float64 `json:"price"`
	CreatedAt   string   `json:"created_at"` // without a zone: UTC
	Phase       *string  `json:"phase"`      // while it is created: "pulling image", "starting container", …
	SSH         string   `json:"ssh_connect_cmd"`
	NodeOffline bool     `json:"node_offline"` // its node fails its health checks
}

// liumTime reads Lium's times, which come with or without a zone (UTC when without).
func liumTime(s string) time.Time {
	for _, layout := range []string{time.RFC3339Nano, "2006-01-02T15:04:05.999999999"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t
		}
	}
	return time.Time{}
}

// liumState is what a pod's status comes to.
func liumState(status string) string {
	switch status {
	case "PENDING", "START_PENDING", "REBOOT_PENDING":
		return "starting"
	case "RUNNING":
		return "running"
	case "STOPPED", "STOP_PENDING", "FAILED", "CREATION_FAILED", "BROKEN", "REBOOT_FAILED":
		return "stopped"
	case "DELETING":
		return "gone"
	}
	return strings.ToLower(status)
}

// Rentals are the pods the key may see: with pod visibility "own", those it rented.
func (m *liumMarket) Rentals(ctx context.Context) ([]Rental, error) {
	var pods []liumPod
	if _, err := m.a.do(ctx, "GET", "/pods", nil, &pods); err != nil {
		return nil, err
	}
	out := make([]Rental, 0, len(pods))
	for _, p := range pods {
		price := 0.0
		if p.Price != nil {
			price = *p.Price
		}
		r := Rental{Market: "lium", ID: p.ID, Name: p.PodName, Card: shortGPU(p.GPUName), Price: price,
			State: liumState(p.Status), Status: p.Status, Created: liumTime(p.CreatedAt)}
		r.Phase = strings.ToLower(p.Status)
		if p.Phase != nil && *p.Phase != "" {
			r.Phase += ": " + *p.Phase
		}
		r.Up = map[bool]string{true: "yes", false: "no"}[p.Status == "RUNNING"]
		r.Detail = r.Phase
		if p.NodeOffline {
			r.Phase += " (node offline)"
			r.Detail = r.Phase
		}
		out = append(out, r)
	}
	return out, nil
}

// sshTo is where to log in to a pod: "root@host", and the port.
func (m *liumMarket) sshTo(ctx context.Context, id string) (string, int, error) {
	var pods []liumPod
	if _, err := m.a.do(ctx, "GET", "/pods", nil, &pods); err != nil {
		return "", 0, err
	}
	for _, p := range pods {
		if p.ID == id {
			var who string
			var port int
			if _, err := fmt.Sscanf(p.SSH, "ssh %s -p %d", &who, &port); err == nil {
				return who, port, nil
			}
			return "", 0, fmt.Errorf("Lium gives no way in to it yet (%q)", p.SSH)
		}
	}
	return "", 0, errors.New("Lium does not list it")
}

// Logs are what Lium has of a pod: its container's, and its own of making it.
func (m *liumMarket) Logs(ctx context.Context, id string, n int) ([]string, error) {
	var raw json.RawMessage
	if _, err := m.a.do(ctx, "GET", "/pods/"+url.PathEscape(id)+"/logs", nil, &raw); err != nil {
		var pod struct {
			PodStatus json.RawMessage `json:"pod_status"`
		}
		if _, perr := m.a.do(ctx, "GET", "/pods/"+url.PathEscape(id), nil, &pod); perr != nil || len(pod.PodStatus) == 0 {
			return nil, err
		}
		return lastLines(logLines(pod.PodStatus), n), nil
	}
	return lastLines(logLines(raw), n), nil
}

// End deletes a pod; one already gone is not an error.
func (m *liumMarket) End(ctx context.Context, id string) error {
	status, err := m.a.do(ctx, "DELETE", "/pods/"+url.PathEscape(id), nil, nil)
	if status == 404 {
		return nil
	}
	return err
}

// Check asks for the account: the key works, and the balance left.
func (m *liumMarket) Check(ctx context.Context) (Account, error) {
	var me struct {
		Balance float64 `json:"balance"`
	}
	if _, err := m.a.do(ctx, "GET", "/users/me", nil, &me); err != nil {
		return Account{}, err
	}
	if _, err := m.a.do(ctx, "GET", "/pods", nil, nil); err != nil {
		return Account{}, err
	}
	return Account{Balance: me.Balance, Known: true}, nil
}
