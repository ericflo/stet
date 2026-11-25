// Command stet-cloud rents GPUs to grade for Stet while the organisers' own machines have more
// to grade than they can get through, and lets them go when there is nothing to grade. It rents
// from several markets, each through its own API (market.go): RunPod, Vast.ai, Lium, Salad and
// Clore.ai. It grew out of stet-runpod, and keeps its commands and settings.
//
// Each rented machine runs the stet command line, `stet verify`, with a key of its own
// from a pool kept here. The organisers' own key (--key) vouches for each, by a grant the
// server keeps: from then on the server hands it work as it would any of the organisers'
// machines, and gives it nothing else: it administers nothing. Nothing in the
// server's configuration changes. A machine stops by itself after
// a while with nothing to do and ends its own machine where the market lets it; stet-cloud
// rents more when work is waiting, ends what is idle when nothing is, and ends anything that
// has gone quiet.
//
// It is built on its own, apart from the stet command line and the server:
//
//	cd runpod && go build -o stet-cloud .
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"
)

const usage = `stet-cloud: rent GPUs to grade for Stet, as many as the work wants, from RunPod,
Vast.ai, Lium, Salad and Clore.ai.

  stet-cloud markets             each market: its key, what is left to spend, whether it is
                                 armed, and what it has that fits now (renting nothing)
  stet-cloud markets check M     the same, for one market
  stet-cloud markets arm M       let machines be rented on a market (--note why);
                                 markets disarm M: no more there
  stet-cloud offers [--each]     what would be rented, in order, on every market with a
                                 key, armed or not (gpus: the same)
  stet-cloud keys [--add N]      the pool of keys the rented machines use; --add makes
                                 more and vouches for them with the organisers' key
                                 (--vouch: any not vouched for; --revoke HANDLE...)
  stet-cloud status              what is waiting, what is rented, what it costs
  stet-cloud scale [--follow]    rent and end machines to match the work (once, or
                                 every minute until stopped)
  stet-cloud up [N]              rent N machines now, whatever is waiting (default 1)
  stet-cloud down [--force] [HANDLE...]
                                 end machines that hold no job; --force ends the rest
                                 too, and their jobs are handed back
  stet-cloud machine HANDLE [-n N]
                                 what was seen of the last N machines rented with a key:
                                 each phase, its work, how it ended and why, its last words
  stet-cloud logs HANDLE [-f]    what a machine on RunPod has said: its install, stet verify
  stet-cloud ssh HANDLE          log in to one on RunPod (rented with --ssh)
  stet-cloud pause | resume      stop drawing rounds, so that the machines turn to the
                                 organisers' students, or start again (the round under way finishes)
  stet-cloud migrate             copy stet-runpod's pool and records over, once
  stet-cloud probe --market M    rent one machine there that takes no work, and say what it
                                 is and whether stet's container sandbox holds on it
                                 (tools/cage_check.py); markets curators M --note: let
                                 machines there run curators

The markets' keys are read from ~/.config/stet-cloud/credentials.env (--credentials), and
the environment over it, at every look; a market without one is off. Nothing is rented on
a market until it is armed. The organisers' own stet key (--key) vouches for machines and
asks what is waiting.
Run any command with -h for its settings.
`

// Options are the settings every command shares.
type Options struct {
	Service             *Service // the settings the service runs with, when status judges by them
	Home                string
	Credentials         string // the markets' keys (market.go)
	API                 string
	RunPodAPI           string
	VastAPI             string
	LiumAPI             string
	SaladAPI            string
	CloreAPI            string
	OperatorKey         string
	Cloud               string
	Market              string // up and offers: this market alone; up rents there armed or not
	GPUs                []string
	Image               string
	Disk                int
	MinCUDA             string
	MinRAM              int
	MinCPU              int
	MinGPUMemory        int
	MinCuratorGPUMemory int
	MaxPrice            float64 // dollars an hour for one machine; 0: any
	LongMaxPrice        float64 // the same, for a machine rented for the organisers' bigger students; 0: MaxPrice
	LongMinutes         int     // the longest of the organisers' students a machine takes, minutes; 0: the short ones only
	Cheapest            bool    // rent the cheapest a grade first, not the fastest
	// For grades taken before an exam: how long after it the round may settle.
	Settle time.Duration
	Idle   int
	SSHKey string
	DryRun bool
	Every  time.Duration
	Policy
}

func defaultHome() string {
	if h := os.Getenv("STET_CLOUD_HOME"); h != "" {
		return h
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "stet-cloud")
}

func defaultOperatorKey() string {
	if c := os.Getenv("STET_CONFIG"); c != "" {
		return filepath.Join(c, "key.json")
	}
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "stet", "key.json")
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func flags(name string, o *Options) *flag.FlagSet {
	fs := flag.NewFlagSet("stet-cloud "+name, flag.ContinueOnError)
	fs.StringVar(&o.Home, "home", defaultHome(), "where the pool of keys and the record of spending are kept (STET_CLOUD_HOME)")
	fs.StringVar(&o.Credentials, "credentials", defaultCredentials(), "the markets' keys, KEY=value lines, read at every look (STET_CLOUD_CREDENTIALS)")
	fs.StringVar(&o.API, "api", envOr("STET_API", "https://api.stetrun.com"), "the Stet API (STET_API)")
	fs.StringVar(&o.RunPodAPI, "runpod-api", runpodAPI, "RunPod's API")
	fs.StringVar(&o.VastAPI, "vast-api", vastAPI, "Vast.ai's API")
	fs.StringVar(&o.LiumAPI, "lium-api", liumAPI, "Lium's API")
	fs.StringVar(&o.SaladAPI, "salad-api", saladAPI, "Salad's API")
	fs.StringVar(&o.CloreAPI, "clore-api", cloreAPI, "Clore.ai's API")
	fs.StringVar(&o.OperatorKey, "key", defaultOperatorKey(), "the organisers' own stet key, to ask what is waiting")
	fs.StringVar(&o.Cloud, "cloud", "any", "RunPod's clouds: any (the cheapest of both), community (strangers' machines: cheapest, but a host could read a machine's key), or secure (RunPod's data centres)")
	fs.StringVar(&o.Cloud, "runpod-cloud", "any", "the same as --cloud")
	fs.StringVar(&o.Market, "market", "", "with up or offers: this market alone (runpod, vast, lium, salad, clore); up rents there by hand, armed or not")
	gpus := fs.String("gpu", "", "GPU kinds to rent, in order of preference, comma-separated (default: whatever fits, the fastest first)")
	fs.BoolVar(&o.Cheapest, "cheapest", false, "rent the cheapest a grade first, not the fastest: a round settles only when its slowest grade is in, so this saves money and takes longer")
	fs.StringVar(&o.Image, "image", "ubuntu:24.04", "the container image; stet is installed on it when the pod starts")
	fs.IntVar(&o.Disk, "disk", 120, "container disk, GB: stet and the season's libraries, a round's corpus and assets, and the 80 GB a machine that runs curators keeps free")
	fs.StringVar(&o.MinCUDA, "min-cuda", "13.0", "the oldest CUDA the host's driver may have: the season's PyTorch is built for CUDA 13")
	fs.IntVar(&o.MinRAM, "min-ram", 24, "host memory a GPU, GB: the season's floor for a machine that runs curators (a curator is given 20)")
	fs.IntVar(&o.MinCPU, "min-cpus", 6, "host vCPUs a GPU: the season's floor for a machine that runs curators")
	fs.IntVar(&o.MinGPUMemory, "min-gpu-memory", 16, "GPU memory, GB: floor for grading the Season 3 0.6B student")
	fs.IntVar(&o.MinCuratorGPUMemory, "min-curator-gpu-memory", 24, "GPU memory, GB: floor for machines rented to run curators (a curator is given 22)")
	fs.IntVar(&o.Idle, "idle", 20, "minutes a machine waits with nothing to do before it ends itself")
	noEdits := fs.Bool("no-edits", false, "rent machines that only grade, and leave curators to the organisers' own machines")
	ssh := fs.Bool("ssh", false, "let ~/.ssh/id_ed25519.pub log in to the machines rented (for `stet-cloud ssh`)")
	fs.BoolVar(&o.DryRun, "dry-run", false, "say what would be rented and ended, and do none of it")
	fs.IntVar(&o.MaxWorkers, "max", 6, "never more machines than this")
	fs.IntVar(&o.LongMinutes, "long-farm", 0, "rent machines that also take the organisers' bigger students, given up to this many minutes each, and one for each waiting (0: none)")
	fs.Float64Var(&o.MaxPrice, "max-price", 0, "never rent a machine that costs more than this many dollars an hour, however much work is waiting (0: any)")
	fs.Float64Var(&o.LongMaxPrice, "long-max-price", 0, "the same, for a machine rented for the organisers' bigger students, which is chosen for what a student costs on it rather than its price (0: --max-price)")
	fs.Float64Var(&o.MaxHourly, "max-hourly", 4, "never rent past this many dollars an hour, all machines together")
	fs.Float64Var(&o.DailyBudget, "daily-budget", 0, "rent nothing more once about this many dollars have been spent today, UTC (0: no limit)")
	fs.Float64Var(&o.PerRunner, "per-runner", 2, "with --no-edits: machines that keep pace with one of the organisers' machines running curators")
	fs.IntVar(&o.PerWorker, "per-machine", 6, "scripts waiting to be graded that justify one more machine")
	fs.IntVar(&o.PerFarm, "per-farm", 4, "the organisers' students waiting to be trained that justify one more machine (0: rent none for them)")
	fs.DurationVar(&o.Every, "every", time.Minute, "with scale --follow: how often to look")
	fs.DurationVar(&o.Early, "early", 90*time.Minute, "rent graders for a round's exam from at most this long before it is due, each kind of GPU just in time, the cheapest a round first (0: only once it is posted)")
	fs.DurationVar(&o.Settle, "settle-within", 15*time.Minute, "with --early: how long after its exam a round may settle; a kind of GPU is rented in time to report by then, and a quicker, dearer one only once the cheaper ones can no longer")
	fs.DurationVar(&o.ExamAfter, "exam-after", 5*time.Minute, "how long after its draw a round's exam is posted, about, when the server does not say: in Season 2 the exam is made at the draw from hours of the day before, already harvested")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: stet-cloud %s [settings]\n\n", name)
		fs.PrintDefaults()
	}
	o.Grace, o.Silence = 25*time.Minute, 20*time.Minute
	finish := func() error {
		if o.Market != "" && !slices.Contains(marketOrder, o.Market) {
			return fmt.Errorf("--market is one of %s, not %q", strings.Join(marketOrder, ", "), o.Market)
		}
		if o.Market != "" && !marketOnlyFor[name] {
			return errors.New("--market is for up and offers alone: what is rented by itself goes by which markets are armed")
		}
		o.LongFarm = o.LongMinutes > 0
		o.Edits = !*noEdits
		o.Cloud = strings.ToUpper(o.Cloud)
		if o.Cloud != "SECURE" && o.Cloud != "COMMUNITY" && o.Cloud != "ANY" {
			return fmt.Errorf("--cloud is any, community or secure, not %q", strings.ToLower(o.Cloud))
		}
		o.GPUs = nil
		for _, g := range strings.Split(*gpus, ",") {
			if g = strings.TrimSpace(g); g != "" {
				o.GPUs = append(o.GPUs, g)
			}
		}
		if *ssh {
			home, _ := os.UserHomeDir()
			raw, err := os.ReadFile(filepath.Join(home, ".ssh", "id_ed25519.pub"))
			if err != nil {
				return fmt.Errorf("--ssh: %w", err)
			}
			o.SSHKey = strings.TrimSpace(string(raw))
		}
		return nil
	}
	finishers[fs] = finish
	return fs
}

var finishers = map[*flag.FlagSet]func() error{}

// marketOnlyFor are the commands --market is for: renting by hand, and looking.
var marketOnlyFor = map[string]bool{"up": true, "offers": true, "gpus": true, "probe": true}

// parse reads the settings wherever they are among the words, and returns the rest. Go's
// flag package stops at the first word that is not a setting, so that `up 2 --dry-run`
// would rent two machines and take --dry-run for a second number; here it does not.
func parse(fs *flag.FlagSet, args []string) ([]string, error) {
	var rest []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if fs.NArg() == 0 {
			break
		}
		rest = append(rest, fs.Arg(0))
		args = fs.Args()[1:]
	}
	if err := finishers[fs](); err != nil {
		return nil, err
	}
	return rest, nil
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(0)
		}
		fmt.Fprintf(os.Stderr, "stet-cloud: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 || args[0] == "help" || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprint(stdout, usage)
		return nil
	}
	cmd, args := args[0], args[1:]
	o := &Options{}
	fs := flags(cmd, o)

	fs.SetOutput(stderr)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	switch cmd {
	case "keys":
		add := fs.Int("add", 0, "make this many more keys, and vouch for them")
		prefix := fs.String("prefix", "stet-rp", "their handles start with this")
		vouch := fs.Bool("vouch", false, "vouch for every key in the pool the server does not take yet")
		revoke := fs.Bool("revoke", false, "take back the grant for the keys named, and forget them")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if *revoke != (len(rest) > 0) {
			return errors.New("keys --revoke takes the handles of the keys to take back, and nothing else does")
		}
		f, err := open(o, stdout, false, true)
		if err != nil {
			return err
		}
		return cmdKeys(ctx, f, *add, *prefix, *vouch, rest)
	case "offers", "gpus":
		each := fs.Bool("each", false, "each machine on its own line, where a market lists each host's")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		f, err := open(o, stdout, true, false)
		if err != nil {
			return err
		}
		return cmdOffers(ctx, f, *each)
	case "markets":
		note := fs.String("note", "", "with arm or curators: why, kept with it")
		off := fs.Bool("off", false, "with curators: machines there grade only")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		f, err := open(o, stdout, false, false)
		if err != nil {
			return err
		}
		return cmdMarkets(ctx, f, rest, *note, *off)
	case "probe":
		check := fs.String("check", "tools/cage_check.py", "the sandbox's check, tools/cage_check.py in stet's repository")
		wait := fs.Duration("wait", 50*time.Minute, "how long to wait for what the probe finds")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 || o.Market == "" {
			return errors.New("probe takes --market and nothing else")
		}
		f, err := open(o, stdout, true, true)
		if err != nil {
			return err
		}
		return cmdProbe(ctx, f, *check, *wait)
	case "migrate":
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		return cmdMigrate(o, stdout)
	case "status":
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 && cmd != "up" && cmd != "down" && cmd != "ssh" {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		if svc := loadService(o.Home); svc != nil {
			// What the service will rent and end is what it judges by its own settings; any
			// given here are taken over them.
			so := &Options{}
			sfs := flags(cmd, so)
			sfs.SetOutput(io.Discard)
			if _, err := parse(sfs, append(append([]string{}, svc.Args...), args...)); err == nil {
				o, so.Service = so, svc
			}
		}
		f, err := open(o, stdout, true, true)
		if err != nil {
			return err
		}
		return cmdStatus(ctx, f)
	case "scale":
		follow := fs.Bool("follow", false, "keep at it until stopped")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 && cmd != "up" && cmd != "down" && cmd != "ssh" {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		f, err := open(o, stdout, true, true)
		if err != nil {
			return err
		}
		if *follow {
			if err := keepService(fs, o.Home, time.Now()); err != nil {
				return err
			}
		}
		return cmdScale(ctx, f, *follow)
	case "up":
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 && cmd != "up" && cmd != "down" && cmd != "ssh" {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		n := 1
		if len(rest) > 1 {
			return fmt.Errorf("up takes one number of machines, not %q", strings.Join(rest, " "))
		}
		if len(rest) == 1 {
			if _, err := fmt.Sscan(rest[0], &n); err != nil || n < 1 || fmt.Sprint(n) != rest[0] {
				return fmt.Errorf("up takes a number of machines, not %q", rest[0])
			}
		}
		f, err := open(o, stdout, true, true)
		if err != nil {
			return err
		}
		return cmdUp(ctx, f, n)
	case "down":
		force := fs.Bool("force", false, "also end machines that are grading: their jobs are handed back")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 && cmd != "up" && cmd != "down" && cmd != "ssh" {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		f, err := open(o, stdout, true, true)
		if err != nil {
			return err
		}
		return cmdDown(ctx, f, *force, rest)
	case "pause", "resume":
		if rest, err := parse(fs, args); err != nil {
			return err
		} else if len(rest) > 0 {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		f, err := open(o, stdout, false, true)
		if err != nil {
			return err
		}
		return cmdPause(ctx, f, cmd == "pause")
	case "machine":
		last := fs.Int("n", 3, "how many of the latest machines rented with the key")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) != 1 {
			return errors.New("machine takes one key's handle")
		}
		f, err := open(o, stdout, false, false)
		if err != nil {
			return err
		}
		return cmdMachine(f, rest[0], *last)
	case "logs":
		follow := fs.Bool("f", false, "keep printing what it says, as it says it")
		last := fs.Int("n", 60, "how many of its last lines")
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) != 1 {
			return errors.New("logs takes one machine's handle")
		}
		f, err := open(o, stdout, true, false)
		if err != nil {
			return err
		}
		return cmdLogs(ctx, f, rest[0], *follow, *last)
	case "ssh":
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 && cmd != "up" && cmd != "down" && cmd != "ssh" {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		if len(rest) != 1 {
			return errors.New("ssh takes one machine's handle")
		}
		f, err := open(o, stdout, true, false)
		if err != nil {
			return err
		}
		return cmdSSH(ctx, f, rest[0])
	}
	fmt.Fprint(stderr, usage)
	return fmt.Errorf("no command %q", cmd)
}

// open gathers what a command needs: the pool always, the markets with keys and the
// organisers' key when asked for. A pool still in stet-runpod's folder is not started afresh
// beside it: it is copied over first (migrate).
func open(o *Options, out io.Writer, markets, operator bool) (*Fleet, error) {
	f := &Fleet{o: o, st: newStet(o.API), out: out, now: time.Now}
	if err := f.loadKeys(); err != nil {
		return nil, err
	}
	if operator && len(f.keys) == 0 && filepath.Clean(o.Home) == filepath.Clean(defaultHome()) {
		if old, _ := filepath.Glob(filepath.Join(oldHome(), "workers", "*.json")); len(old) > 0 && filepath.Clean(oldHome()) != filepath.Clean(o.Home) {
			return nil, fmt.Errorf("the pool of keys is still stet-runpod's, in %s: `stet-cloud migrate` copies it here when the service moves over; until then, --home %s", oldHome(), oldHome())
		}
	}
	if markets {
		if err := f.refresh(); err != nil {
			return nil, err
		}
		if len(f.markets) == 0 {
			return nil, fmt.Errorf("no market has a key: put them in %s (RUNPOD_API_KEY, VAST_API_KEY, LIUM_API_KEY, SALAD_API_KEY, CLORE_API_KEY)", o.Credentials)
		}
	} else if err := f.refresh(); err != nil {
		return nil, err
	}
	if operator {
		k, err := loadKey(o.OperatorKey)
		if err != nil {
			return nil, fmt.Errorf("the organisers' own stet key (--key): %w", err)
		}
		f.op = k
	}
	return f, nil
}

// --- the settings the service runs with ----------------------------------------------------------
//
// `scale --follow` keeps the settings it was started with, so that `status` judges the work as
// the service does: by its own defaults, a bare `stet-cloud status` would say what those
// would rent and end, not what the service will.

// Service is the settings `scale --follow` was last started with.
type Service struct {
	Args []string `json:"args"` // those given, and nothing else
	At   int64    `json:"at"`
}

func servicePath(home string) string { return filepath.Join(home, "service.json") }

func keepService(fs *flag.FlagSet, home string, at time.Time) error {
	svc := Service{Args: []string{}, At: at.Unix()}
	fs.Visit(func(fl *flag.Flag) {
		if fl.Name != "follow" {
			svc.Args = append(svc.Args, "--"+fl.Name+"="+fl.Value.String())
		}
	})
	raw, err := json.Marshal(svc)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(home, 0o700); err != nil {
		return err
	}
	tmp := servicePath(home) + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, servicePath(home))
}

func loadService(home string) *Service {
	raw, err := os.ReadFile(servicePath(home))
	if err != nil {
		return nil
	}
	var svc Service
	if json.Unmarshal(raw, &svc) != nil {
		return nil
	}
	return &svc
}

// String is the settings as they would be typed.
func (s *Service) String() string {
	var words []string
	for _, a := range s.Args {
		name, value, _ := strings.Cut(a, "=")
		switch {
		case value == "true":
			words = append(words, name)
		case strings.ContainsAny(value, " \t"):
			words = append(words, name, fmt.Sprintf("%q", value))
		default:
			words = append(words, name, value)
		}
	}
	return strings.Join(words, " ")
}
