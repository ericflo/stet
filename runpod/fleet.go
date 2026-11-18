package main

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"
)

//go:embed worker.sh
var workerScript string

// Fleet is the pool of keys, the pods rented with them, and what has been spent.
type Fleet struct {
	o    *Options
	rp   *RunPod
	st   *Stet
	op   *Key // the organisers' own key, for asking what is waiting
	out  io.Writer
	now  func() time.Time
	keys []*Key // the pool, by handle
}

// --- the pool of keys ---------------------------------------------------------------------------

func (f *Fleet) keyDir() string { return filepath.Join(f.o.Home, "workers") }

func (f *Fleet) loadKeys() error {
	paths, err := filepath.Glob(filepath.Join(f.keyDir(), "*.json"))
	if err != nil {
		return err
	}
	f.keys = nil
	for _, p := range paths {
		k, err := loadKey(p)
		if err != nil {
			return err
		}
		if k.Handle == "" {
			k.Handle = strings.TrimSuffix(filepath.Base(p), ".json")
		}
		f.keys = append(f.keys, k)
	}
	sort.Slice(f.keys, func(i, j int) bool { return f.keys[i].Handle < f.keys[j].Handle })
	return nil
}

var handleRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,22}[a-z0-9]$`)

// addKeys makes n more keys, named prefix-01, prefix-02 and on from the last one there is.
func (f *Fleet) addKeys(n int, prefix string) ([]*Key, error) {
	have := map[string]bool{}
	for _, k := range f.keys {
		have[k.Handle] = true
	}
	var made []*Key
	for i := 1; len(made) < n; i++ {
		handle := fmt.Sprintf("%s-%02d", prefix, i)
		if have[handle] {
			continue
		}
		if !handleRe.MatchString(handle) || strings.Contains(handle, "--") {
			return made, fmt.Errorf("%q is not a handle the server takes: 3-24 lowercase letters, digits and single hyphens", handle)
		}
		k, err := newKey(handle)
		if err != nil {
			return made, err
		}
		if err := k.save(filepath.Join(f.keyDir(), handle+".json")); err != nil {
			return made, err
		}
		made = append(made, k)
		f.keys = append(f.keys, k)
	}
	sort.Slice(f.keys, func(i, j int) bool { return f.keys[i].Handle < f.keys[j].Handle })
	return made, nil
}

// workerKeys is the value for STET_WORKER_KEYS: every key in the pool.
func (f *Fleet) workerKeys() string {
	var pubs []string
	for _, k := range f.keys {
		pubs = append(pubs, k.Public)
	}
	return strings.Join(pubs, ",")
}

// lock keeps two of these from renting at once on one pool: two would each see the same
// free keys, and rent two pods for one key. It is held for one look and what follows it,
// so `down` can be run while `scale --follow` is, and waits for its turn.
func (f *Fleet) lock(wait time.Duration) (func(), error) {
	if err := os.MkdirAll(f.o.Home, 0o700); err != nil {
		return nil, err
	}
	fh, err := os.OpenFile(filepath.Join(f.o.Home, "lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	deadline := time.Now().Add(wait)
	for syscall.Flock(int(fh.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) != nil {
		if time.Now().After(deadline) {
			fh.Close()
			return nil, errors.New("another stet-runpod is at work on this pool (" + f.o.Home + ") and has not finished; try again")
		}
		time.Sleep(250 * time.Millisecond)
	}
	return func() { fh.Close() }, nil
}

// --- what is rented, and what it costs ----------------------------------------------------------
//
// RunPod forgets a pod once it is ended, and says nothing of what it cost. So what was
// rented is written down here: each pod while it lives (fleet.json), and each one once it
// is gone (spent.jsonl), with the hours it ran and what it cost an hour. A pod that ended
// itself while nothing was watching is taken to have run until a little after it was last
// seen. These are estimates; RunPod's billing page is the account.

type podRecord struct {
	ID     string    `json:"id"`
	Handle string    `json:"handle"`
	GPU    string    `json:"gpu"`
	Cloud  string    `json:"cloud"`
	Cost   float64   `json:"cost"`
	Start  time.Time `json:"start"`
	Alive  time.Time `json:"alive"` // last seen running
	End    time.Time `json:"end,omitzero"`
}

func (r podRecord) dollarsBetween(from, to time.Time) float64 {
	end := r.End
	if end.IsZero() {
		end = r.Alive
	}
	a, b := maxTime(r.Start, from), minTime(end, to)
	if !b.After(a) {
		return 0
	}
	return b.Sub(a).Hours() * r.Cost
}

func maxTime(a, b time.Time) time.Time {
	if a.After(b) {
		return a
	}
	return b
}

func minTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}

type ledger struct {
	Pods map[string]*podRecord `json:"pods"`
}

func (f *Fleet) readLedger() (*ledger, error) {
	l := &ledger{Pods: map[string]*podRecord{}}
	raw, err := os.ReadFile(filepath.Join(f.o.Home, "fleet.json"))
	if errors.Is(err, os.ErrNotExist) {
		return l, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, l); err != nil {
		return nil, fmt.Errorf("fleet.json: %w", err)
	}
	if l.Pods == nil {
		l.Pods = map[string]*podRecord{}
	}
	return l, nil
}

func (f *Fleet) writeLedger(l *ledger) error {
	raw, err := json.MarshalIndent(l, "", " ")
	if err != nil {
		return err
	}
	path := filepath.Join(f.o.Home, "fleet.json")
	if err := os.WriteFile(path+".tmp", raw, 0o600); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

func (f *Fleet) spentLog() string { return filepath.Join(f.o.Home, "spent.jsonl") }

// note brings the record up to date with the pods RunPod lists now: the live ones were
// alive now, and those it no longer lists are gone.
func (f *Fleet) note(l *ledger, pods []Pod) error {
	now := f.now()
	listed := map[string]*Pod{}
	for i := range pods {
		listed[pods[i].ID] = &pods[i]
	}
	for _, k := range f.keys {
		for i := range pods {
			p := &pods[i]
			if p.Name != k.Handle || !p.live() {
				continue
			}
			r := l.Pods[p.ID]
			if r == nil { // rented by another copy of this tool, or by hand
				r = &podRecord{ID: p.ID, Handle: k.Handle, GPU: p.gpuName(), Cloud: f.o.Cloud, Cost: p.Cost, Start: p.CreatedAt}
				l.Pods[p.ID] = r
			}
			r.Alive, r.Cost = now, p.Cost
		}
	}
	for id, r := range l.Pods {
		if p := listed[id]; p == nil || !p.live() {
			if err := f.retire(l, id, minTime(now, r.Alive.Add(2*f.o.Every))); err != nil {
				return err
			}
		}
	}
	return f.writeLedger(l)
}

// retire moves a pod that has ended from the record of live ones to the log of what was
// spent, which is only ever added to.
func (f *Fleet) retire(l *ledger, id string, end time.Time) error {
	r := l.Pods[id]
	if r == nil {
		return nil
	}
	r.End = end
	line, err := json.Marshal(r)
	if err != nil {
		return err
	}
	fh, err := os.OpenFile(f.spentLog(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer fh.Close()
	if _, err := fh.Write(append(line, '\n')); err != nil {
		return err
	}
	delete(l.Pods, id)
	return nil
}

// spent is about what was spent since a moment: ended pods from the log, and live ones.
func (f *Fleet) spent(l *ledger, since time.Time) (float64, error) {
	now := f.now()
	total := 0.0
	for _, r := range l.Pods {
		r := *r
		r.End = now
		total += r.dollarsBetween(since, now)
	}
	fh, err := os.Open(f.spentLog())
	if errors.Is(err, os.ErrNotExist) {
		return total, nil
	}
	if err != nil {
		return 0, err
	}
	defer fh.Close()
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		var r podRecord
		if json.Unmarshal(sc.Bytes(), &r) == nil {
			total += r.dollarsBetween(since, now)
		}
	}
	return total, sc.Err()
}

func startOfDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

// --- looking -------------------------------------------------------------------------------------

// View is one look at everything: the work, the pool and its pods, and the money.
type View struct {
	Work    *Work
	Workers []*Worker
	Ledger  *ledger
	Hourly  float64 // dollars an hour, every live pod together
	Today   float64 // about what has been spent today (UTC)
}

func (f *Fleet) look(ctx context.Context) (*View, error) {
	work, err := f.st.work(ctx, f.op)
	if err != nil {
		return nil, err
	}
	pods, err := f.rp.pods(ctx)
	if err != nil {
		return nil, err
	}
	l, err := f.readLedger()
	if err != nil {
		return nil, err
	}
	if err := f.note(l, pods); err != nil {
		return nil, err
	}
	v := &View{Work: work, Ledger: l}
	for _, k := range f.keys {
		w := &Worker{Key: k, Machine: work.machine(k.Handle)}
		// The newest pod of that name. Ended ones are not listed at all; one that stopped is,
		// and is the one to end.
		for i := range pods {
			p := &pods[i]
			if p.Name == k.Handle && (w.Pod == nil || p.CreatedAt.After(w.Pod.CreatedAt)) {
				w.Pod = p
			}
		}
		if w.Pod != nil && w.Pod.live() {
			v.Hourly += w.Pod.Cost
		}
		v.Workers = append(v.Workers, w)
	}
	if v.Today, err = f.spent(l, startOfDay(f.now())); err != nil {
		return nil, err
	}
	return v, nil
}

// --- renting and ending --------------------------------------------------------------------------

// Not the kinds a grade can be trusted to: those without bf16 (before Ampere), slices of
// one GPU shared with others, and makers other than NVIDIA (grades are taken on CUDA).
var unfitGPU = regexp.MustCompile(`(?i)V100|\bT4\b|P100|P40|\bP4\b|K80|M60|Quadro|RTX 20|TITAN|GTX|MIG`)

// choices are the GPU kinds to try, cheapest first: those named with --gpu, in that
// order, or every fit kind RunPod has free now.
func (f *Fleet) choices(ctx context.Context) ([]GPUType, error) {
	all, err := f.rp.gpus(ctx, f.o.Cloud, f.o.MinCUDA)
	if err != nil {
		return nil, err
	}
	if len(f.o.GPUs) > 0 {
		var out []GPUType
		for _, want := range f.o.GPUs {
			for _, g := range all {
				if strings.EqualFold(g.ID, want) || strings.EqualFold(g.Name, want) || strings.EqualFold(shortGPU(g.ID), want) {
					out = append(out, g)
				}
			}
		}
		return out, nil
	}
	var out []GPUType
	for _, g := range all {
		if g.Manufacturer == "NVIDIA" && g.Memory >= f.o.MinGPUMemory && g.price(f.o.Cloud) > 0 && g.Availability != "NONE" && g.Availability != "" && !unfitGPU.MatchString(g.ID) {
			out = append(out, g)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].price(f.o.Cloud) < out[j].price(f.o.Cloud) })
	return out, nil
}

func (f *Fleet) spec(k *Key, gpu string, site string) NewPod {
	env := map[string]string{
		"STET_WORKER_KEY":   base64.StdEncoding.EncodeToString(k.file()),
		"STET_API":          f.st.api,
		"STET_INSTALL":      site + "/install.sh",
		"STET_IDLE_MINUTES": fmt.Sprint(f.o.Idle),
	}
	var ports []string
	if f.o.SSHKey != "" {
		env["PUBLIC_KEY"] = f.o.SSHKey
		ports = []string{"22/tcp"}
	}
	return NewPod{
		Name: k.Handle, Image: f.o.Image, Cloud: f.o.Cloud, Disk: f.o.Disk, Env: env, Ports: ports,
		GPU:        NewPodGPU{ID: gpu, Count: 1, MinRAMPerGPU: f.o.MinRAM, MinVCPUPerGPU: f.o.MinCPU, MinCUDAVersion: f.o.MinCUDA},
		Entrypoint: []string{"/bin/bash", "-c"}, Cmd: []string{workerScript},
	}
}

// launch rents up to n machines on keys that have no pod, cheapest kind first, within
// what may be spent an hour. It says what it did as it goes.
func (f *Fleet) launch(ctx context.Context, v *View, n int) (int, error) {
	if n <= 0 {
		return 0, nil
	}
	site, err := f.st.site(ctx)
	if err != nil {
		return 0, err
	}
	gpus, err := f.choices(ctx)
	if err != nil {
		return 0, err
	}
	if len(gpus) == 0 {
		f.say("RunPod has no GPU free now that fits (%s cloud%s); trying again later", strings.ToLower(f.o.Cloud), map[bool]string{true: ", --gpu " + strings.Join(f.o.GPUs, ","), false: ""}[len(f.o.GPUs) > 0])
		return 0, nil
	}
	rented := 0
	for _, w := range v.Workers {
		if rented == n {
			break
		}
		if w.Pod != nil {
			continue
		}
		var worker bool
		if f.o.DryRun {
			// Looked up, not made: a dry run changes nothing anywhere.
			exists, known, err := f.st.lookup(ctx, f.op, w.Key.Handle)
			if err != nil {
				return rented, fmt.Errorf("%s's account: %w", w.Key.Handle, err)
			}
			worker = known || !exists
		} else {
			worker, err = f.st.account(ctx, w.Key)
			if err != nil {
				return rented, fmt.Errorf("%s's account: %w", w.Key.Handle, err)
			}
		}
		if !worker {
			return rented, fmt.Errorf("the Stet server does not take %s as one of the organisers' rented machines yet. Its key goes in STET_WORKER_KEYS (`stet-runpod keys` prints the line), and the server is started again", w.Key.Handle)
		}
		pod, err := f.rent(ctx, v, w.Key, gpus, site)
		if err != nil {
			return rented, err
		}
		if pod == nil {
			break // nothing fits: no point trying the next key
		}
		w.Pod = pod
		rented++
	}
	return rented, nil
}

// rent tries each kind of GPU in turn until RunPod gives one. nil, nil: none could be had
// within what may be spent.
func (f *Fleet) rent(ctx context.Context, v *View, k *Key, gpus []GPUType, site string) (*Pod, error) {
	for _, g := range gpus {
		price := g.price(f.o.Cloud)
		if v.Hourly+price > f.o.MaxHourly+1e-9 {
			if len(f.o.GPUs) == 0 {
				f.say("not renting another machine: at $%.2f an hour it would pass --max-hourly $%.2f", v.Hourly+price, f.o.MaxHourly)
				return nil, nil // cheapest first: the rest cost more
			}
			continue
		}
		if f.o.DryRun {
			f.say("would rent %s on %s (%s cloud, about $%.2f an hour)", k.Handle, shortGPU(g.ID), strings.ToLower(f.o.Cloud), price)
			v.Hourly += price
			return &Pod{ID: "dry-run", Name: k.Handle, Status: "PROVISIONING", Cost: price, CreatedAt: f.now()}, nil
		}
		pod, err := f.rp.create(ctx, f.spec(k, g.ID, site))
		if errors.Is(err, errNoRoom) {
			continue
		}
		if err != nil {
			return nil, err
		}
		v.Hourly += pod.Cost
		v.Ledger.Pods[pod.ID] = &podRecord{ID: pod.ID, Handle: k.Handle, GPU: shortGPU(g.ID), Cloud: f.o.Cloud, Cost: pod.Cost, Start: f.now(), Alive: f.now()}
		if err := f.writeLedger(v.Ledger); err != nil {
			return pod, err
		}
		f.say("rented %s: %s, %s cloud, $%.2f an hour (pod %s)", k.Handle, shortGPU(g.ID), strings.ToLower(f.o.Cloud), pod.Cost, pod.ID)
		return pod, nil
	}
	f.say("RunPod has none of the %d kinds of GPU tried free for %s just now; trying again later", len(gpus), k.Handle)
	return nil, nil
}

// end ends pods and says why.
func (f *Fleet) end(ctx context.Context, v *View, endings []Ending) error {
	for _, e := range endings {
		if e.Worker.Pod == nil {
			continue
		}
		if f.o.DryRun {
			f.say("would end %s: %s", e.Worker.Key.Handle, e.Why)
			continue
		}
		if err := f.rp.terminate(ctx, e.Worker.Pod.ID); err != nil {
			return fmt.Errorf("ending %s: %w", e.Worker.Key.Handle, err)
		}
		if e.Worker.Pod.live() {
			v.Hourly -= e.Worker.Pod.Cost
		}
		if err := f.retire(v.Ledger, e.Worker.Pod.ID, f.now()); err != nil {
			return err
		}
		f.say("ended %s: %s", e.Worker.Key.Handle, e.Why)
		e.Worker.Pod = nil
	}
	return f.writeLedger(v.Ledger)
}

func (f *Fleet) say(format string, args ...any) {
	fmt.Fprintf(f.out, "%s  %s\n", f.now().UTC().Format("15:04:05"), fmt.Sprintf(format, args...))
}
