package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// --- Salad (https://docs.salad.com/reference) --------------------------------------------------
//
// Salad rents GPUs in people's own PCs, by class of GPU ("RTX 4090 (24 GB)") rather than by
// machine, at a price an hour for each priority: at high priority a machine is not taken back
// for somebody else's work. The price takes in the CPUs and memory. A machine is a container
// group of one replica, in one project of one organization. Its key, Salad-Api-Key, is a user's,
// and reaches every organization the user is in: the fleet's is a second user's, in stet's
// organization alone.

const saladAPI = "https://api.salad.com/api/public"

// saladPriority is the priority machines are rented at, and priced by.
const saladPriority = "high"

type saladMarket struct {
	a            *api
	org, project string
	mu           sync.Mutex
	classes      []saladClass
	classesAt    time.Time
	stock        map[string]int // by class ID, as last asked
	stockAt      map[string]time.Time
	now          func() time.Time
}

func newSalad(base, key, org, project string) *saladMarket {
	return &saladMarket{a: newAPI("salad", base, "Salad-Api-Key", key), org: org, project: project,
		stock: map[string]int{}, stockAt: map[string]time.Time{}, now: time.Now}
}

func (m *saladMarket) Name() string { return "salad" }

func (m *saladMarket) Caps() Caps { return Caps{Rents: true} }

func (m *saladMarket) orgPath() string { return "/organizations/" + url.PathEscape(m.org) }

type saladClass struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Type   string `json:"gpu_class_type"`
	Prices []struct {
		Price    string `json:"price"`
		Priority string `json:"priority"`
	} `json:"prices"`
}

func (c saladClass) price(priority string) float64 {
	for _, p := range c.Prices {
		if p.Priority == priority {
			v, _ := strconv.ParseFloat(p.Price, 64)
			return v
		}
	}
	return 0
}

// saladName is a class's name in two: "RTX 4090 (24 GB)" is an RTX 4090 with 24 GB.
var saladName = regexp.MustCompile(`^(.*?)\s*\((\d+)\s*GB\)$`)

func saladCard(name string) (card string, memGB int) {
	if m := saladName.FindStringSubmatch(name); m != nil {
		memGB, _ = strconv.Atoi(m[2])
		return m[1], memGB
	}
	return name, 0
}

func saladMaker(name string) string {
	if strings.HasPrefix(name, "AMD") {
		return "AMD"
	}
	return "NVIDIA"
}

// gpuClasses are the classes Salad rents, asked again every ten minutes at most.
func (m *saladMarket) gpuClasses(ctx context.Context) ([]saladClass, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.classes != nil && m.now().Sub(m.classesAt) < 10*time.Minute {
		return m.classes, nil
	}
	var out struct {
		Items []saladClass `json:"items"`
	}
	if _, err := m.a.do(ctx, "GET", m.orgPath()+"/gpu-classes", nil, &out); err != nil {
		return nil, err
	}
	m.classes, m.classesAt = out.Items, m.now()
	return m.classes, nil
}

// available is how many machines of a class are free at the priority rented at, for a machine
// of the size needed. Salad asks that this be asked sparingly: each class's answer is kept for
// ten minutes.
func (m *saladMarket) available(ctx context.Context, class string, n Need) (int, error) {
	m.mu.Lock()
	if at, ok := m.stockAt[class]; ok && m.now().Sub(at) < 10*time.Minute {
		s := m.stock[class]
		m.mu.Unlock()
		return s, nil
	}
	m.mu.Unlock()
	body := map[string]any{"gpu_classes": []string{class}}
	if n.MinCPUs > 0 {
		body["cpu"] = n.MinCPUs
	}
	if n.MinRAMGB > 0 {
		body["memory"] = n.MinRAMGB * 1024
	}
	if n.DiskGB > 0 {
		body["storage_amount"] = int64(n.DiskGB) << 30
	}
	var out struct {
		High   int `json:"available_gpu_high"`
		Medium int `json:"available_gpu_medium"`
		Low    int `json:"available_gpu_low"`
		Batch  int `json:"available_gpu_batch"`
	}
	if _, err := m.a.do(ctx, "POST", m.orgPath()+"/availability/sce-gpu-availability", body, &out); err != nil {
		return 0, err
	}
	s := map[string]int{"high": out.High, "medium": out.Medium, "low": out.Low, "batch": out.Batch}[saladPriority]
	m.mu.Lock()
	m.stock[class], m.stockAt[class] = s, m.now()
	m.mu.Unlock()
	return s, nil
}

// Offers are the classes that could hold a grade, with how many are free now. Only those that
// fit are asked about. Salad does not say the hosts' drivers: the machine checks its own.
func (m *saladMarket) Offers(ctx context.Context, n Need) ([]Offer, error) {
	classes, err := m.gpuClasses(ctx)
	if err != nil {
		return nil, err
	}
	var out []Offer
	for _, c := range classes {
		card, mem := saladCard(c.Name)
		price := c.price(saladPriority)
		if saladMaker(c.Name) != "NVIDIA" || mem < n.MinGPUMemGB || unfitGPU.MatchString(c.Name) || price <= 0 {
			continue
		}
		stock, err := m.available(ctx, c.ID, n)
		if err != nil {
			return nil, err
		}
		if stock <= 0 {
			continue
		}
		out = append(out, Offer{Market: "salad", Card: card, Model: c.Name, Name: card, ID: c.ID, Maker: "NVIDIA",
			MemoryGB: mem, Price: price, Stock: stock, Trust: c.Type})
	}
	return out, nil
}

// Rent makes a container group of one replica on the class offered, at the priority rented
// at, with the worker as its command and the size needed. Its name has more after the handle,
// since Salad never takes a name twice; its display name is the handle. An answer lost on the
// way is looked for by its name.
func (m *saladMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	name := l.Name + "-" + newID()[:5]
	env := map[string]string{"STET_MARKET": "salad", "STET_WORKER_SCRIPT": base64.StdEncoding.EncodeToString([]byte(l.Script))}
	for k, v := range l.Env {
		env[k] = v
	}
	body := map[string]any{
		"name": name, "display_name": l.Name, "replicas": 1, "autostart_policy": true, "restart_policy": "never",
		"container": map[string]any{
			"image": l.Image, "command": []string{"bash", "-c", workerInner}, "environment_variables": env, "priority": saladPriority,
			"resources": map[string]any{"cpu": max(l.Need.MinCPUs, 1), "memory": max(l.Need.MinRAMGB, 1) * 1024,
				"gpu_classes": []string{o.ID}, "storage_amount": int64(max(l.Need.DiskGB, 1)) << 30},
		},
	}
	path := m.orgPath() + "/projects/" + url.PathEscape(m.project) + "/containers"
	var out saladGroup
	status, err := m.a.do(ctx, "POST", path, body, &out)
	if err != nil && (status == 0 || status >= 500) {
		if _, gerr := m.a.do(ctx, "GET", path+"/"+url.PathEscape(name), nil, &out); gerr == nil {
			err = nil
		}
	}
	if err != nil {
		return nil, err
	}
	return &Rental{Market: "salad", ID: name, Name: l.Name, Card: o.Card, Price: o.Price, State: "starting", Status: out.CurrentState.Status, Created: time.Now()}, nil
}

// saladGroup is one container group, of the fields read here.
type saladGroup struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	DisplayName  string    `json:"display_name"`
	Priority     string    `json:"priority"`
	CreateTime   time.Time `json:"create_time"`
	CurrentState struct {
		Status      string  `json:"status"`
		Description *string `json:"description"`
		Counts      struct {
			Allocating int `json:"allocating_count"`
			Creating   int `json:"creating_count"`
			Running    int `json:"running_count"`
			Stopping   int `json:"stopping_count"`
		} `json:"instance_status_counts"`
	} `json:"current_state"`
	Container struct {
		Resources struct {
			GPUClasses []string `json:"gpu_classes"`
		} `json:"resources"`
	} `json:"container"`
}

func saladState(status string) string {
	switch status {
	case "pending", "deploying":
		return "starting"
	case "running":
		return "running"
	case "stopped", "succeeded", "failed":
		return "stopped"
	}
	return status
}

// Rentals are the project's container groups, each one machine. A group's display name is
// the worker key's handle; its name has more after it, since Salad never takes a name twice.
func (m *saladMarket) Rentals(ctx context.Context) ([]Rental, error) {
	var out struct {
		Items []saladGroup `json:"items"`
	}
	if _, err := m.a.do(ctx, "GET", m.orgPath()+"/projects/"+url.PathEscape(m.project)+"/containers", nil, &out); err != nil {
		return nil, err
	}
	classes, _ := m.gpuClasses(ctx) // for the price; without them it is not known
	byID := map[string]saladClass{}
	for _, c := range classes {
		byID[c.ID] = c
	}
	var rentals []Rental
	for _, g := range out.Items {
		r := Rental{Market: "salad", ID: g.Name, Name: g.DisplayName, State: saladState(g.CurrentState.Status), Status: g.CurrentState.Status, Created: g.CreateTime}
		c := g.CurrentState.Counts
		switch {
		case c.Running > 0:
			r.Up, r.Phase = "yes", "running"
		case c.Creating > 0:
			r.Up, r.Phase = "no", "creating: pulling the image"
		case c.Allocating > 0:
			r.Up, r.Phase = "no", "allocating: looking for a node"
		default:
			r.Up, r.Phase = "no", g.CurrentState.Status
		}
		r.Detail = r.Phase
		if d := g.CurrentState.Description; d != nil && *d != "" {
			r.Detail += ": " + *d
		}
		if cs := g.Container.Resources.GPUClasses; len(cs) > 0 {
			if c, ok := byID[cs[0]]; ok {
				r.Card, _ = saladCard(c.Name)
				r.Price = c.price(g.Priority)
			}
		}
		rentals = append(rentals, r)
	}
	return rentals, nil
}

// Logs are what the container group's container has said, from Salad's log store (queries are
// limited a day, so this is asked only of a machine that failed, or a probe); failing that, what
// Salad's system has said of the group.
func (m *saladMarket) Logs(ctx context.Context, id string, n int) ([]string, error) {
	now := m.now().UTC()
	q := fmt.Sprintf(`resource.type = "container" AND resource.labels.project_name = "%s" AND resource.labels.container_group_name = "%s"`, m.project, id)
	var raw json.RawMessage
	_, err := m.a.do(ctx, "POST", m.orgPath()+"/log-entries", map[string]any{
		"query": q, "start_time": now.Add(-6 * time.Hour).Format(time.RFC3339), "end_time": now.Format(time.RFC3339),
		"sort_order": "asc", "page_size": max(n, 1),
	}, &raw)
	if err == nil {
		if lines := logLines(raw); len(lines) > 0 {
			return lastLines(lines, n), nil
		}
	}
	var sys json.RawMessage
	if _, serr := m.a.do(ctx, "GET", m.orgPath()+"/projects/"+url.PathEscape(m.project)+"/containers/"+url.PathEscape(id)+"/system-logs", nil, &sys); serr != nil {
		if err != nil {
			return nil, err
		}
		return nil, serr
	}
	return lastLines(logLines(sys), n), nil
}

// End deletes a container group; one already gone is not an error.
func (m *saladMarket) End(ctx context.Context, id string) error {
	status, err := m.a.do(ctx, "DELETE", m.orgPath()+"/projects/"+url.PathEscape(m.project)+"/containers/"+url.PathEscape(id), nil, nil)
	if status == 404 {
		return nil
	}
	return err
}

// Check asks for the organization's quotas and the project: the key works, the organization is
// the user's, and the project is there. Salad does not say the balance.
func (m *saladMarket) Check(ctx context.Context) (Account, error) {
	var q struct {
		Groups struct {
			Quota int `json:"container_replicas_quota"`
			Used  int `json:"container_replicas_used"`
		} `json:"container_groups_quotas"`
	}
	if _, err := m.a.do(ctx, "GET", m.orgPath()+"/quotas", nil, &q); err != nil {
		var e *APIError
		if errors.As(err, &e) && e.Status == 404 {
			return Account{}, fmt.Errorf("Salad has no organization %q that this key's user is in (SALAD_ORG)", m.org)
		}
		return Account{}, err
	}
	// Salad's API has no call that reads a project; listing its container groups is refused
	// with 404 (project_not_found_exception) when there is no such project.
	if _, err := m.a.do(ctx, "GET", m.orgPath()+"/projects/"+url.PathEscape(m.project)+"/containers", nil, nil); err != nil {
		var e *APIError
		if errors.As(err, &e) && e.Status == 404 {
			return Account{}, fmt.Errorf("the organization %q has no project %q: make it at portal.salad.com, or set SALAD_PROJECT to the one there", m.org, m.project)
		}
		return Account{}, err
	}
	return Account{Note: fmt.Sprintf("%d of %d replicas in use", q.Groups.Used, q.Groups.Quota)}, nil
}
