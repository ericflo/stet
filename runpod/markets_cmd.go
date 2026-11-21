package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// --- offers, markets, migrate --------------------------------------------------------------------

func cmdOffers(ctx context.Context, f *Fleet, every bool) error {
	if err := f.listOffers(ctx, false, every); err != nil || !f.o.LongFarm {
		return err
	}
	fmt.Fprintln(f.out)
	return f.listOffers(ctx, true, every)
}

// listOffers says what would be rented, in the order it would be tried, on every market a key
// is found for: those not armed are marked, and nothing is rented on them. Where a market lists
// each host's machine on its own, a card's machines there are one line, at the cheapest's
// price, unless every machine is asked for.
func (f *Fleet) listOffers(ctx context.Context, long, every bool) error {
	f.calibrate(ctx)
	offers, err := f.choices(ctx, long, true)
	if err != nil {
		return err
	}
	what := "What would be rented, the fastest first (the cheapest where they are as fast)"
	if f.o.Cheapest {
		what = "What would be rented, by what a grade costs on each"
	}
	if long {
		what = "For the organisers' bigger students, by what a student costs on each, the quicker card first where that is about the same"
	}
	if len(offers) == 0 {
		fmt.Fprintf(f.out, "%s: no market has a GPU free now that would do (RunPod's %s cloud%s).\n", what, strings.ToLower(f.o.Cloud), f.ceiling(long))
	} else {
		fmt.Fprintf(f.out, "%s, in this order (RunPod's %s cloud%s, CUDA %s or newer, hosts with %d GB and %d vCPUs a GPU, %d GB of disk):\n\n",
			what, strings.ToLower(f.o.Cloud), f.ceiling(long), f.o.MinCUDA, f.o.MinRAM, f.o.MinCPU, f.o.Disk)
	}
	type line struct {
		o        Offer
		n, stock int
	}
	var lines []*line
	byKind := map[string]*line{}
	for _, o := range offers {
		kind := o.Market + "|" + o.Card + "|" + o.Trust
		if every {
			kind += "|" + o.ID
		}
		if l := byKind[kind]; l != nil {
			l.n++
			l.stock += max(o.Stock, 0)
			continue
		}
		l := &line{o: o, n: 1, stock: max(o.Stock, 0)}
		byKind[kind] = l
		lines = append(lines, l)
	}
	for _, l := range lines {
		c := l.o
		free := c.Level
		switch {
		case c.Stock < 0:
		case l.n > 1:
			free = fmt.Sprintf("%d hosts", l.n)
		default:
			free = fmt.Sprintf("%d free", l.stock)
		}
		mark := ""
		if c.Rate > 0 && c.Rate < 0.995 {
			mark = fmt.Sprintf("  (starts %.0f%%)", 100*c.Rate)
		}
		if !f.armed(c.Market) {
			mark += "  (not armed)"
		}
		fmt.Fprintf(f.out, "  $%.2f/h  %-7s %-10s %-30s %3d GB  %-9s", c.Price, c.Market, c.Trust, c.Card, c.MemoryGB, free)
		p, seen := f.pass(c.Card)
		how := "about"
		if seen {
			how = "seen"
		}
		fmt.Fprintf(f.out, "  a pass %s %.0f min (%.1fx a 3090), a round's first grade %.0f min: $%.2f%s\n",
			how, p.Minutes(), f.speedOf(c.Card), f.lead(c).Minutes(), c.Price*f.lead(c).Hours(), mark)
	}
	if len(offers) > 0 {
		started := "until three have been seen to"
		if len(f.calibration().Boot) >= 3 {
			started = "as seen"
		}
		fmt.Fprintf(f.out, "\nEach is tried in turn, on the markets armed, until one has a machine of that kind free with room for the rest. A machine is taken to start in %.0f minutes, %s.\n", f.boot().Minutes(), started)
	}
	f.sayMarketsLeftOut()
	return nil
}

// sayMarketsLeftOut says which markets were not asked, or did not answer, and why.
func (f *Fleet) sayMarketsLeftOut() {
	for _, name := range marketOrder {
		if why, off := f.off[name]; off {
			fmt.Fprintf(f.out, "%s is off: %s\n", marketTitle(name), why)
		} else if err := f.offerErrs[name]; err != nil {
			fmt.Fprintf(f.out, "%s's offers could not be had: %v\n", marketTitle(name), err)
		}
	}
}

// cmdMarkets says, for each market, whether a key is found, whether it works, what is left to
// spend, whether it is armed, and what it has that fits now; or arms or disarms one. Nothing is
// rented.
func cmdMarkets(ctx context.Context, f *Fleet, args []string, note string) error {
	if err := f.refresh(); err != nil {
		return err
	}
	verb, names := "", marketOrder
	if len(args) > 0 {
		verb = args[0]
		if len(args) != 2 || (verb != "check" && verb != "arm" && verb != "disarm") {
			return errors.New("markets takes nothing, or check, arm or disarm and one market's name")
		}
		if !slices.Contains(marketOrder, args[1]) {
			return fmt.Errorf("there is no market %q: it is one of %s", args[1], strings.Join(marketOrder, ", "))
		}
		names = args[1:]
	}
	switch verb {
	case "arm":
		return f.arm(ctx, names[0], note)
	case "disarm":
		if !f.armed(names[0]) {
			f.say("%s is not armed", marketTitle(names[0]))
			return nil
		}
		delete(f.arming.Armed, names[0])
		if err := f.arming.save(f.o.Home); err != nil {
			return err
		}
		f.say("disarmed %s: nothing more is rented there. The machines it has run until they end", marketTitle(names[0]))
		return nil
	}
	for _, name := range names {
		f.describeMarket(ctx, name)
	}
	if verb == "" {
		fmt.Fprintf(f.out, "\nKeys are read from %s, and from the environment over it, at every look. Which markets are armed is kept in %s.\n", f.o.Credentials, armingPath(f.o.Home))
	}
	return nil
}

// describeMarket says what one market is to the fleet now.
func (f *Fleet) describeMarket(ctx context.Context, name string) {
	m := f.market(name)
	armed := "not armed"
	if f.armed(name) {
		a := f.arming.Armed[name]
		armed = "armed"
		if a.At > 0 {
			armed += " since " + time.Unix(a.At, 0).UTC().Format("Jan 2 15:04") + " UTC"
		}
		if a.Note != "" {
			armed += " (" + a.Note + ")"
		}
	}
	if m == nil {
		fmt.Fprintf(f.out, "%-9s off: %s\n", marketTitle(name), f.off[name])
		return
	}
	acct, err := m.Check(ctx)
	if err != nil {
		fmt.Fprintf(f.out, "%-9s the key does not work: %v · %s\n", marketTitle(name), err, armed)
		return
	}
	said := "the key works"
	if acct.Known {
		said += fmt.Sprintf(" · $%.2f to spend", acct.Balance)
	}
	if acct.Note != "" {
		said += " · " + acct.Note
	}
	if !m.Caps().Rents {
		said += " · renting here is not built yet"
	}
	fmt.Fprintf(f.out, "%-9s %s · %s\n", marketTitle(name), said, armed)
	offers, err := m.Offers(ctx, f.needOn(m))
	if err != nil {
		fmt.Fprintf(f.out, "          its offers could not be had: %v\n", err)
		return
	}
	var fit []Offer
	ceiling := f.maxPrice(false)
	for _, o := range offers {
		if f.fits(o) && (ceiling <= 0 || o.Price <= ceiling+1e-9) {
			fit = append(fit, o)
		}
	}
	if len(fit) == 0 {
		fmt.Fprintf(f.out, "          nothing free now that fits%s\n", f.ceiling(false))
		return
	}
	byGrade(fit, f.speedOf)
	cards := map[string]bool{}
	for _, o := range fit {
		cards[o.Card] = true
	}
	best := fit[0]
	fmt.Fprintf(f.out, "          %d offers that fit%s, %d kinds of card; the cheapest a grade: %s at $%.2f an hour, about $%.2f a grade\n",
		len(fit), f.ceiling(false), len(cards), best.Card, best.Price, best.Price*f.lead(best).Hours())
}

// arm lets machines be rented on a market, once renting there is built and its key works.
func (f *Fleet) arm(ctx context.Context, name, note string) error {
	m := f.market(name)
	if m == nil {
		return fmt.Errorf("%s has no key: %s", marketTitle(name), f.off[name])
	}
	if !m.Caps().Rents {
		return fmt.Errorf("renting on %s is not built yet, so it cannot be armed", marketTitle(name))
	}
	if _, err := m.Check(ctx); err != nil {
		return fmt.Errorf("%s's key does not work: %w", marketTitle(name), err)
	}
	if a, ok := f.arming.Armed[name]; ok && a.At > 0 {
		f.say("%s is armed already", marketTitle(name))
		return nil
	}
	f.arming.Armed[name] = Armed{At: f.now().Unix(), Note: note}
	if err := f.arming.save(f.o.Home); err != nil {
		return err
	}
	f.say("armed %s: from the next look, machines are rented there too, under the same limits", marketTitle(name))
	return nil
}

// --- moving stet-runpod's pool over ----------------------------------------------------------------

// oldHome is where stet-runpod kept its pool.
func oldHome() string {
	if h := os.Getenv("STET_RUNPOD_HOME"); h != "" {
		return h
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "stet-runpod")
}

// migratedFiles are what is copied from stet-runpod's folder: the pool of keys, the record of
// what is rented and was spent, what was learned, and the service's settings.
var migratedFiles = []string{"fleet.json", "spent.jsonl", "calibration.json", "service.json"}

// cmdMigrate copies stet-runpod's pool and records into stet-cloud's folder, once. stet-runpod's
// folder is left as it was. Two fleets on one pool would rent two machines for a key: the
// stet-runpod service is stopped first, and stet-cloud's started in its place.
func cmdMigrate(o *Options, out io.Writer) error {
	from, to := oldHome(), o.Home
	if filepath.Clean(from) == filepath.Clean(to) {
		return fmt.Errorf("stet-cloud's folder is stet-runpod's (%s): there is nothing to copy", to)
	}
	if keys, _ := filepath.Glob(filepath.Join(to, "workers", "*.json")); len(keys) > 0 {
		return fmt.Errorf("%s has a pool of keys already: nothing is copied over it", to)
	}
	keys, err := filepath.Glob(filepath.Join(from, "workers", "*.json"))
	if err != nil {
		return err
	}
	if len(keys) == 0 {
		return fmt.Errorf("%s has no pool of keys to copy", from)
	}
	if err := os.MkdirAll(filepath.Join(to, "workers"), 0o700); err != nil {
		return err
	}
	copied := 0
	for _, k := range keys {
		if err := copyFile(k, filepath.Join(to, "workers", filepath.Base(k))); err != nil {
			return err
		}
		copied++
	}
	for _, name := range migratedFiles {
		src := filepath.Join(from, name)
		if _, err := os.Stat(src); errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err := copyFile(src, filepath.Join(to, name)); err != nil {
			return err
		}
		copied++
	}
	fmt.Fprintf(out, "Copied %d keys and stet-runpod's records from %s to %s; %s is left as it was.\n", len(keys), from, to, from)
	fmt.Fprintln(out, "Only one of stet-runpod and stet-cloud may rent with this pool: stop stet-runpod's service before stet-cloud's starts.")
	return nil
}

func copyFile(from, to string) error {
	raw, err := os.ReadFile(from)
	if err != nil {
		return err
	}
	tmp := to + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, to)
}
