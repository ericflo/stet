package main

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// runpodMarket is RunPod behind the Market interface: the client stet-runpod had, unchanged.
type runpodMarket struct {
	rp    *RunPod
	cloud *string // SECURE, COMMUNITY or ANY (--cloud), as the settings say now
}

func newRunPodMarket(rp *RunPod, cloud *string) *runpodMarket {
	return &runpodMarket{rp: rp, cloud: cloud}
}

func (m *runpodMarket) Name() string { return "runpod" }

func (m *runpodMarket) Caps() Caps {
	return Caps{Rents: true, SelfEnd: true, Curators: true, Logs: true, SSH: true}
}

// clouds are the clouds --cloud lets machines be rented in.
func (m *runpodMarket) clouds() []string {
	if m.cloud == nil || *m.cloud == "ANY" || *m.cloud == "" {
		return []string{"COMMUNITY", "SECURE"}
	}
	return []string{*m.cloud}
}

// Offers are every kind of GPU RunPod has any of free now, in each cloud allowed, at its price
// there.
func (m *runpodMarket) Offers(ctx context.Context, n Need) ([]Offer, error) {
	var out []Offer
	for _, cloud := range m.clouds() {
		gpus, err := m.rp.gpus(ctx, cloud, n.MinCUDA)
		if err != nil {
			return nil, err
		}
		for _, g := range gpus {
			p := g.price(cloud)
			if !validRate(p) || g.Availability == "NONE" || g.Availability == "" {
				continue
			}
			out = append(out, Offer{
				Market: "runpod", Card: shortGPU(g.ID), Model: g.ID, Name: g.Name, ID: g.ID, Maker: g.Manufacturer,
				MemoryGB: g.Memory, Price: p, Stock: -1, Level: strings.ToLower(g.Availability), Trust: strings.ToLower(cloud),
			})
		}
	}
	return out, nil
}

// cloudOf is the RunPod cloud an offer is in.
func cloudOf(o Offer) string { return strings.ToUpper(o.Trust) }

func (m *runpodMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	env := l.Env
	var ports []string
	if l.SSHKey != "" {
		env = map[string]string{}
		for k, v := range l.Env {
			env[k] = v
		}
		env["PUBLIC_KEY"] = l.SSHKey
		ports = []string{"22/tcp"}
	}
	pod, err := m.rp.create(ctx, NewPod{
		Name: l.Name, Image: l.Image, Cloud: cloudOf(o), Disk: l.Need.DiskGB, Env: env, Ports: ports,
		GPU:        NewPodGPU{ID: o.ID, Count: 1, MinRAMPerGPU: l.Need.MinRAMGB, MinVCPUPerGPU: l.Need.MinCPUs, MinCUDAVersion: l.Need.MinCUDA},
		Entrypoint: []string{"/bin/bash", "-c"}, Cmd: []string{l.Script},
	}, l.KnownIDs)
	if err != nil {
		return nil, err
	}
	r := podRental(pod)
	if !validRate(r.Price) && validRate(o.Price) {
		r.Price = o.Price
	}
	return &r, nil
}

func (m *runpodMarket) Rentals(ctx context.Context) ([]Rental, error) {
	pods, err := m.rp.pods(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Rental, len(pods))
	for i := range pods {
		out[i] = podRental(&pods[i])
	}
	return out, nil
}

// podRental is a pod as the fleet sees any market's machine.
func podRental(p *Pod) Rental {
	state := strings.ToLower(p.Status)
	switch p.Status {
	case "PROVISIONING", "STARTING":
		state = "starting"
	case "RUNNING":
		state = "running"
	case "EXITED", "ERROR", "TERMINATED":
		state = "stopped"
	}
	r := Rental{Market: "runpod", ID: p.ID, Name: p.Name, Card: p.gpuName(), Price: p.Cost, State: state, Status: p.Status, Created: p.CreatedAt}
	// Only one pod's own record says whether its container is up (runtime); the list does not.
	switch {
	case p.Runtime != nil:
		r.UpFor = time.Duration(p.Runtime.Uptime) * time.Second
		r.Up, r.Phase, r.Detail = "yes", "container up", "container up "+r.UpFor.Round(time.Minute).String()
	default:
		r.Phase, r.Detail = strings.ToLower(p.Status), strings.ToLower(p.Status)
	}
	return r
}

// inspect is what RunPod says of one pod: its own record says whether its container is up.
func (m *runpodMarket) inspect(ctx context.Context, id string) (*Rental, error) {
	var p Pod
	if _, err := m.rp.do(ctx, "GET", "/pods/"+url.PathEscape(id), nil, &p); err != nil {
		return nil, err
	}
	r := podRental(&p)
	if p.Runtime == nil && r.live() {
		r.Up, r.Phase, r.Detail = "no", "no container yet", "no container yet"
	}
	return &r, nil
}

// Lookup verifies one pod directly by ID when a listing omits it.
func (m *runpodMarket) Lookup(ctx context.Context, id string) (*Rental, bool, error) {
	var p Pod
	status, err := m.rp.do(ctx, "GET", "/pods/"+url.PathEscape(id), nil, &p)
	if status == 404 {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	if p.ID == "" || p.ID != id {
		return nil, false, fmt.Errorf("RunPod lookup for %q returned unexpected pod id %q", id, p.ID)
	}
	r := podRental(&p)
	if p.Runtime == nil && r.live() {
		r.Up, r.Phase, r.Detail = "no", "no container yet", "no container yet"
	}
	return &r, true, nil
}

// Logs are what RunPod and the pod's container have said of it: the last n lines.
func (m *runpodMarket) Logs(ctx context.Context, id string, n int) ([]string, error) {
	var lines []string
	err := m.rp.logs(ctx, id, false, func(l LogLine) {
		ts := l.TS
		if len(ts) >= 19 {
			ts = ts[11:19]
		}
		lines = append(lines, ts+"  "+l.Source+": "+l.Line)
	})
	return lastLines(lines, n), err
}

func (m *runpodMarket) End(ctx context.Context, id string) error { return m.rp.terminate(ctx, id) }

// Check asks for one cloud's kinds of GPU: the key works, and may read the catalog. A key made
// for stet-cloud may not say what is left to spend (RunPod's balance is behind its GraphQL API).
func (m *runpodMarket) Check(ctx context.Context) (Account, error) {
	q := url.Values{"include": {"AVAILABILITY"}, "product": {"POD"}, "cloud": {"SECURE"}}
	var out struct {
		GPUs []GPUType `json:"gpus"`
	}
	if _, err := m.rp.do(ctx, "GET", "/catalog/gpus?"+q.Encode(), nil, &out); err != nil {
		return Account{}, err
	}
	if _, err := m.rp.do(ctx, "GET", "/pods?limit=1", nil, nil); err != nil {
		return Account{}, err
	}
	return Account{Note: "the catalog and pods can be read"}, nil
}
