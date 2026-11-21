package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// --- Clore.ai (https://docs.clore.ai/for-hosts/api) --------------------------------------------
//
// Clore lists each host's server, with a price a day in each currency it takes. Orders are paid
// from a balance kept in USD-Blockchain, bought with USDT or USDC. Its key, sent as "auth", is
// the account's own, with no scopes. It answers 200 even when it says no, with a code: 0 is yes.

const cloreAPI = "https://api.clore.ai/v1"

// cloreCurrency is the currency orders are paid in.
const cloreCurrency = "USD-Blockchain"

// cloreFee is what Clore adds to an on-demand order, and once to each order that lives past ten
// minutes.
const cloreFee, cloreOrderFee = 0.05, 0.10

type cloreMarket struct {
	a         *api
	lastOrder time.Time // Clore takes one order every five seconds
	sleep     func(context.Context, time.Duration) error
}

func newClore(base, key string) *cloreMarket {
	return &cloreMarket{a: newAPI("clore", base, "auth", key), sleep: sleepCtx}
}

func (m *cloreMarket) Name() string { return "clore" }

func (m *cloreMarket) Caps() Caps { return Caps{Rents: true} }

// cloreHours is how long Clore lets an order live, whatever becomes of the fleet: its own
// backstop.
const cloreHours = 3

// cloreEnvValue is the longest value Clore takes for one variable.
const cloreEnvValue = 1536

// do is a call to Clore, with its code read: anything but 0 is an error.
func (m *cloreMarket) do(ctx context.Context, method, path string, body any, into func([]byte) error) error {
	var raw json.RawMessage
	if _, err := m.a.do(ctx, method, path, body, &raw); err != nil {
		return err
	}
	var c struct {
		Code  *int   `json:"code"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return err
	}
	if c.Code == nil || *c.Code != 0 {
		code := -1
		if c.Code != nil {
			code = *c.Code
		}
		return &APIError{Market: "clore", Status: 200, Detail: cloreCode(code, c.Error)}
	}
	if into != nil {
		return into(raw)
	}
	return nil
}

func cloreCode(code int, detail string) string {
	what := map[int]string{1: "the database failed", 2: "a request it does not take", 3: "the key is not valid", 4: "the endpoint is not there", 5: "too many requests", 6: "an error"}[code]
	if what == "" {
		what = "it said no"
	}
	if detail != "" {
		what += ": " + detail
	}
	return fmt.Sprintf("code %d, %s", code, what)
}

// cloreServer is one server on the marketplace, of the fields read here.
type cloreServer struct {
	ID          int64    `json:"id"`
	MRL         int      `json:"mrl"` // the longest it may be rented for, hours
	Rented      bool     `json:"rented"`
	AllowedCoin []string `json:"allowed_coins"`
	Price       struct {
		OnDemand map[string]float64 `json:"on_demand"` // a day
	} `json:"price"`
	Specs struct {
		CPUs   string  `json:"cpus"` // cores/threads
		RAM    float64 `json:"ram"`  // GB
		Disk   string  `json:"disk"` // "<name> 247.3GB"
		GPU    string  `json:"gpu"`  // "1x NVIDIA GeForce RTX 3060"
		GPURAM float64 `json:"gpuram"`
		Net    struct {
			CC   string  `json:"cc"`
			Down float64 `json:"down"` // Mbit/s
		} `json:"net"`
	} `json:"specs"`
	CUDA string `json:"cuda_version"`
}

var cloreDisk = regexp.MustCompile(`([\d.]+)\s*GB\s*$`)
var cloreGPU = regexp.MustCompile(`^(\d+)x\s+(.*)$`)

// clorePrice is an on-demand price a day as it comes to all in, an hour.
func clorePrice(perDay float64) float64 {
	return perDay/24*(1+cloreFee) + cloreOrderFee/expectedLife.Hours()
}

// cloreOffer is a server as an offer, if it meets the need: free, one GPU, paid in the
// currency used, with a driver new enough, CPU threads, memory and disk enough, a connection
// that downloads quickly, and rentable for two hours at least.
func cloreOffer(s cloreServer, n Need) (Offer, bool) {
	g := cloreGPU.FindStringSubmatch(s.Specs.GPU)
	price := s.Price.OnDemand[cloreCurrency]
	if s.Rented || g == nil || g[1] != "1" || price <= 0 || s.MRL < 2 || !cudaAtLeast(s.CUDA, n.MinCUDA) {
		return Offer{}, false
	}
	takes := false
	for _, c := range s.AllowedCoin {
		takes = takes || c == cloreCurrency
	}
	threads := 0
	if _, t, ok := strings.Cut(s.Specs.CPUs, "/"); ok {
		threads, _ = strconv.Atoi(t)
	} else {
		threads, _ = strconv.Atoi(s.Specs.CPUs)
	}
	disk := 0.0
	if d := cloreDisk.FindStringSubmatch(s.Specs.Disk); d != nil {
		disk, _ = strconv.ParseFloat(d[1], 64)
	}
	if !takes || threads < n.MinCPUs || s.Specs.RAM < float64(n.MinRAMGB)-0.5 || disk < float64(n.DiskGB) || s.Specs.Net.Down < 100 {
		return Offer{}, false
	}
	model := g[2]
	maker := "NVIDIA"
	if !strings.HasPrefix(model, "NVIDIA") {
		maker = strings.SplitN(model, " ", 2)[0]
	}
	return Offer{
		Market: "clore", Card: shortGPU(model), Model: model, Name: shortGPU(model), ID: strconv.FormatInt(s.ID, 10),
		Maker: maker, MemoryGB: int(s.Specs.GPURAM + 0.5), Price: clorePrice(price), Ask: price, Stock: 1, CUDA: s.CUDA,
		Trust: "community", Region: s.Specs.Net.CC,
	}, true
}

// Offers are the servers on the marketplace, of those that meet the need. The marketplace needs
// no key.
func (m *cloreMarket) Offers(ctx context.Context, n Need) ([]Offer, error) {
	var out struct {
		Servers []cloreServer `json:"servers"`
	}
	if err := m.do(ctx, "GET", "/marketplace", nil, func(raw []byte) error { return json.Unmarshal(raw, &out) }); err != nil {
		return nil, err
	}
	var offers []Offer
	for _, s := range out.Servers {
		if o, ok := cloreOffer(s, n); ok {
			offers = append(offers, o)
		}
	}
	return offers, nil
}

// cloreScript is worker.sh in the environment, base64'd and cut into values Clore takes, and
// the command that puts it back together and runs it.
func cloreScript(script string) (env map[string]string, command string) {
	b := base64.StdEncoding.EncodeToString([]byte(script))
	env = map[string]string{}
	var parts []string
	for i := 0; len(b) > 0; i++ {
		n := min(len(b), cloreEnvValue)
		k := fmt.Sprintf("STET_WORKER_SCRIPT_%d", i)
		env[k], b = b[:n], b[n:]
		parts = append(parts, "$"+k)
	}
	return env, `bash -c "echo ` + strings.Join(parts, "") + ` | base64 -d > /root/stet-worker.sh && exec bash /root/stet-worker.sh"`
}

// Rent orders a server on demand, paid in USD-Blockchain, at the price it was offered at,
// ended by Clore after cloreHours whatever becomes of the fleet. Clore does not say what order
// it made: it is found among the account's by its server, made since it was asked for.
func (m *cloreMarket) Rent(ctx context.Context, o Offer, l Launch) (*Rental, error) {
	server, err := strconv.ParseInt(o.ID, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("a Clore server's ID: %q", o.ID)
	}
	if wait := 5*time.Second - time.Since(m.lastOrder); wait > 0 {
		if err := m.sleep(ctx, wait); err != nil {
			return nil, err
		}
	}
	env, command := cloreScript(l.Script)
	env["STET_MARKET"] = "clore"
	for k, v := range l.Env {
		env[k] = v
	}
	body := map[string]any{
		"currency": cloreCurrency, "image": l.Image, "renting_server": server, "type": "on-demand",
		"env": env, "command": command, "rental_hours": cloreHours, "ports": map[string]string{},
	}
	if o.Ask > 0 {
		body["required_price"] = o.Ask
	}
	asked := time.Now().Add(-time.Minute)
	m.lastOrder = time.Now()
	if err := m.do(ctx, "POST", "/create_order", body, nil); err != nil {
		var e *APIError
		if errors.As(err, &e) && e.Status == 200 && strings.HasPrefix(e.Detail, "code 6") && !strings.Contains(e.Detail, "balance") {
			return nil, fmt.Errorf("%w (%v)", errNoRoom, err)
		}
		return nil, err
	}
	for try := 0; try < 6; try++ {
		rs, err := m.orders(ctx)
		if err == nil {
			for _, r := range rs {
				if r.server == server && !r.Created.Before(asked) && r.State != "gone" {
					r.Rental.Name, r.Rental.State = l.Name, "starting"
					return &r.Rental, nil
				}
			}
		}
		if err := m.sleep(ctx, 2*time.Second); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("Clore took an order for server %d, and does not list it", server)
}

// cloreOrder is one of the account's orders, of the fields read here. An order has no name:
// which worker key it was rented for is in the fleet's record.
type cloreOrder struct {
	ID       int64   `json:"id"`
	Price    float64 `json:"price"` // a day
	Currency string  `json:"currency"`
	CT       int64   `json:"ct"`
	SI       int64   `json:"si"`
	Expired  bool    `json:"expired"`
	Spot     bool    `json:"spot"`
	Specs    struct {
		GPU string `json:"gpu"`
	} `json:"specs"`
}

type cloreRental struct {
	Rental
	server int64
}

func (m *cloreMarket) Rentals(ctx context.Context) ([]Rental, error) {
	os, err := m.orders(ctx)
	if err != nil {
		return nil, err
	}
	rentals := make([]Rental, len(os))
	for i := range os {
		rentals[i] = os[i].Rental
	}
	return rentals, nil
}

// orders are the account's orders, with the server each is on.
func (m *cloreMarket) orders(ctx context.Context) ([]cloreRental, error) {
	var out struct {
		Orders []cloreOrder `json:"orders"`
	}
	if err := m.do(ctx, "GET", "/my_orders", nil, func(raw []byte) error { return json.Unmarshal(raw, &out) }); err != nil {
		return nil, err
	}
	var rentals []cloreRental
	for _, o := range out.Orders {
		card := o.Specs.GPU
		if g := cloreGPU.FindStringSubmatch(card); g != nil {
			card = g[2]
		}
		state := "running"
		if o.Expired {
			state = "gone"
		}
		status := map[bool]string{true: "expired", false: "active"}[o.Expired]
		rentals = append(rentals, cloreRental{Rental{Market: "clore", ID: strconv.FormatInt(o.ID, 10), Card: shortGPU(card),
			Price: o.Price / 24 * (1 + cloreFee), State: state, Status: status, Created: time.Unix(o.CT, 0),
			Phase: "order " + status, Detail: "order " + status + " (Clore says no more of it)"}, o.SI})
	}
	return rentals, nil
}

// End cancels an order. One Clore no longer lists as active is gone already, and not an error.
func (m *cloreMarket) End(ctx context.Context, id string) error {
	n, err := strconv.ParseInt(id, 10, 64)
	if err != nil {
		return fmt.Errorf("a Clore order's ID: %q", id)
	}
	err = m.do(ctx, "POST", "/cancel_order", map[string]any{"id": n}, nil)
	if err == nil {
		return nil
	}
	if os, lerr := m.orders(ctx); lerr == nil {
		for _, o := range os {
			if o.ID == id && o.State != "gone" {
				return err
			}
		}
		return nil
	}
	return err
}

// Check asks for the wallets: the key works, and the balance in the currency orders are paid in.
func (m *cloreMarket) Check(ctx context.Context) (Account, error) {
	var out struct {
		Wallets []struct {
			Name    string  `json:"name"`
			Balance float64 `json:"balance"`
		} `json:"wallets"`
	}
	if err := m.do(ctx, "GET", "/wallets", nil, func(raw []byte) error { return json.Unmarshal(raw, &out) }); err != nil {
		return Account{}, err
	}
	for _, w := range out.Wallets {
		if w.Name == cloreCurrency {
			return Account{Balance: w.Balance, Known: true}, nil
		}
	}
	return Account{Note: "no " + cloreCurrency + " wallet"}, nil
}
