// Command stet-runpod rents GPUs from RunPod to grade for Stet while the organisers' own
// machines have more to grade than they can get through, and lets them go when there is
// nothing to grade.
//
// Each rented machine runs the stet command line, `stet verify`, with a key of its own
// from a pool kept here. The organisers' own key (--key) vouches for each, by a grant the
// server keeps: from then on the server hands it work as it would any of the organisers'
// machines, and gives it nothing else, the held-out exam least of all. Nothing in the
// server's configuration changes. A machine stops by itself after
// a while with nothing to do and ends its own pod; stet-runpod rents more when work is
// waiting, ends what is idle when nothing is, and ends anything that has gone quiet.
//
// It is built on its own, apart from the stet command line and the server:
//
//	cd runpod && go build -o stet-runpod .
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const usage = `stet-runpod: rent GPUs from RunPod to grade for Stet, as many as the work wants.

  stet-runpod keys [--add N]     the pool of keys the rented machines use; --add makes
                                 more and vouches for them with the organisers' key
                                 (--vouch: any not vouched for; --revoke HANDLE...)
  stet-runpod gpus               what would be rented, cheapest first
  stet-runpod status             what is waiting, what is rented, what it costs
  stet-runpod scale [--follow]   rent and end machines to match the work (once, or
                                 every minute until stopped)
  stet-runpod up [N]             rent N machines now, whatever is waiting (default 1)
  stet-runpod down [--force] [HANDLE...]
                                 end machines that hold no job; --force ends the rest
                                 too, and their jobs are handed back
  stet-runpod logs HANDLE [-f]   what a machine has said: its install, stet verify
  stet-runpod ssh HANDLE         log in to one (rented with --ssh)
  stet-runpod pause | resume     stop drawing rounds, so that the machines turn to the
                                 next exam, or start again (the round under way finishes)

Needs RUNPOD_API_KEY, and the organisers' own stet key (--key), to vouch for machines
and ask what is waiting.
Run any command with -h for its settings.
`

// Options are the settings every command shares.
type Options struct {
	Home         string
	API          string
	RunPodAPI    string
	OperatorKey  string
	Cloud        string
	GPUs         []string
	Image        string
	Disk         int
	MinCUDA      string
	MinRAM       int
	MinCPU       int
	MinGPUMemory int
	MaxPrice     float64 // dollars an hour for one machine; 0: any
	LongMinutes  int     // the longest of the next exam's students a machine takes, minutes; 0: the short ones only
	Idle         int
	SSHKey       string
	DryRun       bool
	Every        time.Duration
	Policy
}

func defaultHome() string {
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
	fs := flag.NewFlagSet("stet-runpod "+name, flag.ContinueOnError)
	fs.StringVar(&o.Home, "home", defaultHome(), "where the pool of keys and the record of spending are kept (STET_RUNPOD_HOME)")
	fs.StringVar(&o.API, "api", envOr("STET_API", "https://api.stetrun.com"), "the Stet API (STET_API)")
	fs.StringVar(&o.RunPodAPI, "runpod-api", runpodAPI, "RunPod's API")
	fs.StringVar(&o.OperatorKey, "key", defaultOperatorKey(), "the organisers' own stet key, to ask what is waiting")
	fs.StringVar(&o.Cloud, "cloud", "any", "any (the cheapest of both), community (strangers' machines: cheapest, but a host could read a machine's key), or secure (RunPod's data centres)")
	gpus := fs.String("gpu", "", "GPU kinds to rent, in order of preference, comma-separated (default: whatever fits, cheapest first)")
	fs.StringVar(&o.Image, "image", "ubuntu:24.04", "the container image; stet is installed on it when the pod starts")
	fs.IntVar(&o.Disk, "disk", 80, "container disk, GB: stet and the season's libraries, and the 60 GB a machine that runs curators keeps free")
	fs.StringVar(&o.MinCUDA, "min-cuda", "13.0", "the oldest CUDA the host's driver may have: the season's PyTorch is built for CUDA 13")
	fs.IntVar(&o.MinRAM, "min-ram", 16, "host memory a GPU, GB: the season's floor for a machine that runs curators (a curator is given 12)")
	fs.IntVar(&o.MinCPU, "min-cpus", 6, "host vCPUs a GPU: the season's floor for a machine that runs curators")
	fs.IntVar(&o.MinGPUMemory, "min-gpu-memory", 12, "GPU memory, GB: a curator is given 8, and a machine that runs them needs 10")
	fs.IntVar(&o.Idle, "idle", 15, "minutes a machine waits with nothing to do before it ends its own pod")
	noEdits := fs.Bool("no-edits", false, "rent machines that only grade, and leave curators to the organisers' own machines")
	ssh := fs.Bool("ssh", false, "let ~/.ssh/id_ed25519.pub log in to the machines rented (for `stet-runpod ssh`)")
	fs.BoolVar(&o.DryRun, "dry-run", false, "say what would be rented and ended, and do none of it")
	fs.IntVar(&o.MaxWorkers, "max", 6, "never more machines than this")
	fs.IntVar(&o.LongMinutes, "long-shadow", 0, "rent machines that also take the next exam's bigger students, given up to this many minutes each, and one for each waiting (0: none)")
	fs.Float64Var(&o.MaxPrice, "max-price", 0, "never rent a machine that costs more than this many dollars an hour, however much work is waiting (0: any)")
	fs.Float64Var(&o.MaxHourly, "max-hourly", 3, "never rent past this many dollars an hour, all machines together")
	fs.Float64Var(&o.DailyBudget, "daily-budget", 0, "rent nothing more once about this many dollars have been spent today, UTC (0: no limit)")
	fs.Float64Var(&o.PerRunner, "per-runner", 2, "with --no-edits: machines that keep pace with one of the organisers' machines running curators")
	fs.IntVar(&o.PerWorker, "per-machine", 6, "jobs waiting (a curator to run is two: the run and its grade) that justify one more machine")
	fs.IntVar(&o.PerShadow, "per-shadow", 4, "the next exam's students waiting to be trained that justify one more machine (0: rent none for them)")
	fs.DurationVar(&o.Every, "every", time.Minute, "with scale --follow: how often to look")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: stet-runpod %s [settings]\n\n", name)
		fs.PrintDefaults()
	}
	o.Grace, o.Silence = 25*time.Minute, 20*time.Minute
	finish := func() error {
		o.LongShadow = o.LongMinutes > 0
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
	return rest, finishers[fs]()
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(0)
		}
		fmt.Fprintf(os.Stderr, "stet-runpod: %v\n", err)
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
	case "gpus":
		rest, err := parse(fs, args)
		if err != nil {
			return err
		}
		if len(rest) > 0 && cmd != "up" && cmd != "down" && cmd != "ssh" {
			return fmt.Errorf("%s takes no %q", cmd, rest[0])
		}
		f, err := open(o, stdout, true, false)
		if err != nil {
			return err
		}
		return cmdGPUs(ctx, f)
	case "status":
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

// open gathers what a command needs: the pool always, RunPod and the organisers' key when
// asked for.
func open(o *Options, out io.Writer, runpod, operator bool) (*Fleet, error) {
	f := &Fleet{o: o, st: newStet(o.API), out: out, now: time.Now}
	if err := f.loadKeys(); err != nil {
		return nil, err
	}
	if runpod {
		key := os.Getenv("RUNPOD_API_KEY")
		if key == "" {
			return nil, errors.New("RUNPOD_API_KEY is not set: make a key at https://www.runpod.io/console/user/settings")
		}
		f.rp = newRunPod(o.RunPodAPI, key)
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
