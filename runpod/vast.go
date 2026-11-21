package main

import (
	"context"
	"fmt"
	"math"
	"net/url"
	"strconv"
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
	StorageCost  float64 `json:"storage_cost"`   // dollars a GB a month
	InetDownCost float64 `json:"inet_down_cost"` // dollars a GB
	CUDAMaxGood  float64 `json:"cuda_max_good"`
	Verification string  `json:"verification"`
	HostingType  int     `json:"hosting_type"` // 1: a data centre
	Geolocation  string  `json:"geolocation"`
	GPUArch      string  `json:"gpu_arch"`
}

// vastNames are Vast's names for cards whose names elsewhere differ from them.
var vastNames = map[string]string{
	"RTX 4080S": "RTX 4080 SUPER", "RTX 4070S": "RTX 4070 SUPER", "RTX 4070S Ti": "RTX 4070 Ti SUPER",
	"RTX 2000Ada": "RTX 2000 Ada Generation", "RTX 4000Ada": "RTX 4000 Ada Generation",
	"RTX 5000Ada": "RTX 5000 Ada Generation", "RTX 6000Ada": "RTX 6000 Ada Generation",
	"RTX PRO 6000 WS": "RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000 S": "RTX PRO 6000 Blackwell Server Edition",
	"RTX PRO 4000": "RTX PRO 4000 Blackwell", "RTX PRO 4500": "RTX PRO 4500 Blackwell", "RTX PRO 5000": "RTX PRO 5000 Blackwell",
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
	case status == 404 || status == 410:
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

// vastInstance is one of the account's instances, of the fields read here.
type vastInstance struct {
	ID             int64   `json:"id"`
	Label          *string `json:"label"`
	ActualStatus   *string `json:"actual_status"`
	IntendedStatus string  `json:"intended_status"`
	CurState       string  `json:"cur_state"`
	GPUName        string  `json:"gpu_name"`
	DPHTotal       float64 `json:"dph_total"`
	StartDate      float64 `json:"start_date"`
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
			all = append(all, Rental{Market: "vast", ID: strconv.FormatInt(i.ID, 10), Name: name, Card: vastCard(i.GPUName),
				Price: i.DPHTotal, State: state, Status: status, Created: time.Unix(int64(i.StartDate), 0)})
		}
		if out.NextToken == nil || *out.NextToken == "" {
			return all, nil
		}
		after = *out.NextToken
	}
	return all, fmt.Errorf("Vast listed more than 2,500 instances")
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
