package main

import (
	"bufio"
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
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

// rename gives a key a new handle, the same prefix and a random end, before it has an
// account or a pod by the old one.
func (f *Fleet) rename(k *Key) error {
	prefix := k.Handle
	if i := strings.LastIndex(prefix, "-"); i > 0 {
		prefix = prefix[:i]
	}
	have := map[string]bool{}
	for _, other := range f.keys {
		have[other.Handle] = true
	}
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	for {
		b := make([]byte, 4)
		if _, err := rand.Read(b); err != nil {
			return err
		}
		for i := range b {
			b[i] = letters[int(b[i])%len(letters)]
		}
		handle := prefix + "-" + string(b)
		if have[handle] || !handleRe.MatchString(handle) {
			continue
		}
		old := filepath.Join(f.keyDir(), k.Handle+".json")
		k.Handle = handle
		if err := k.save(filepath.Join(f.keyDir(), handle+".json")); err != nil {
			return err
		}
		return os.Remove(old)
	}
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
				r = &podRecord{ID: p.ID, Handle: k.Handle, GPU: p.gpuName(), Cost: p.Cost, Start: p.CreatedAt}
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

// Choice is one kind of GPU in one of RunPod's clouds, at its price there.
type Choice struct {
	GPU   GPUType
	Cloud string
	Price float64
}

// clouds are the clouds --cloud lets machines be rented in.
func (f *Fleet) clouds() []string {
	if f.o.Cloud == "ANY" {
		return []string{"COMMUNITY", "SECURE"}
	}
	return []string{f.o.Cloud}
}

// choices are what to try, looked up afresh each time machines are rented: every fit kind
// of GPU RunPod has free now, in every cloud allowed, cheapest first (secure first where
// the price is the same); or those named with --gpu, in that order, each cheapest first.
// For the organisers' bigger students (long), it is what a student costs on each kind that
// counts, not what it costs an hour, under --long-max-price.
func (f *Fleet) choices(ctx context.Context, long bool) ([]Choice, error) {
	ceiling := f.maxPrice(long)
	var all []Choice
	for _, cloud := range f.clouds() {
		gpus, err := f.rp.gpus(ctx, cloud, f.o.MinCUDA)
		if err != nil {
			return nil, err
		}
		for _, g := range gpus {
			if p := g.price(cloud); p > 0 && g.Availability != "NONE" && g.Availability != "" && (ceiling <= 0 || p <= ceiling+1e-9) {
				all = append(all, Choice{g, cloud, p})
			}
		}
	}
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Price != all[j].Price {
			return all[i].Price < all[j].Price
		}
		return all[i].Cloud == "SECURE" && all[j].Cloud != "SECURE"
	})
	var out []Choice
	if len(f.o.GPUs) > 0 {
		for _, want := range f.o.GPUs {
			for _, c := range all {
				if strings.EqualFold(c.GPU.ID, want) || strings.EqualFold(c.GPU.Name, want) || strings.EqualFold(shortGPU(c.GPU.ID), want) {
					out = append(out, c)
				}
			}
		}
		return out, nil
	}
	for _, c := range all {
		if c.GPU.Manufacturer == "NVIDIA" && c.GPU.Memory >= f.o.MinGPUMemory && !unfitGPU.MatchString(c.GPU.ID) {
			out = append(out, c)
		}
	}
	if long {
		byStudent(out)
	}
	return out, nil
}

func (f *Fleet) spec(k *Key, gpu, cloud, site string) NewPod {
	env := map[string]string{
		"STET_WORKER_KEY":   base64.StdEncoding.EncodeToString(k.file()),
		"STET_API":          f.st.api,
		"STET_INSTALL":      site + "/install.sh",
		"STET_IDLE_MINUTES": fmt.Sprint(f.o.Idle),
		"STET_WORKER_EDITS": map[bool]string{true: "1", false: "0"}[f.o.Edits],
	}
	if f.o.LongMinutes > 0 {
		env["STET_FARM_MAX_MINUTES"] = fmt.Sprint(f.o.LongMinutes)
	}
	var ports []string
	if f.o.SSHKey != "" {
		env["PUBLIC_KEY"] = f.o.SSHKey
		ports = []string{"22/tcp"}
	}
	return NewPod{
		Name: k.Handle, Image: f.o.Image, Cloud: cloud, Disk: f.o.Disk, Env: env, Ports: ports,
		GPU:        NewPodGPU{ID: gpu, Count: 1, MinRAMPerGPU: f.o.MinRAM, MinVCPUPerGPU: f.o.MinCPU, MinCUDAVersion: f.o.MinCUDA},
		Entrypoint: []string{"/bin/bash", "-c"}, Cmd: []string{workerScript},
	}
}

// launch rents up to n machines on keys that have no pod, cheapest kind first, within
// what may be spent an hour. It says what it did as it goes.
func (f *Fleet) launch(ctx context.Context, v *View, n, long int) (int, error) {
	if n <= 0 {
		return 0, nil
	}
	site, err := f.st.site(ctx)
	if err != nil {
		return 0, err
	}
	vouched, err := f.st.vouched(ctx, f.op)
	if err != nil {
		return 0, err
	}
	// The machines for the bigger students first, each chosen for what a student costs on
	// it; then the rest, cheapest first.
	rented := 0
	for _, batch := range []struct {
		n    int
		long bool
	}{{min(long, n), true}, {n - min(long, n), false}} {
		got, err := f.launchSome(ctx, v, batch.n, batch.long, site, vouched)
		rented += got
		if err != nil {
			return rented, err
		}
	}
	return rented, nil
}

func (f *Fleet) launchSome(ctx context.Context, v *View, n int, long bool, site string, vouched map[string]Vouched) (int, error) {
	if n <= 0 {
		return 0, nil
	}
	gpus, err := f.choices(ctx, long)
	if err != nil {
		return 0, err
	}
	if len(gpus) == 0 {
		f.say("RunPod has no GPU free now that fits (%s cloud%s%s); trying again later", strings.ToLower(f.o.Cloud), f.ceiling(long), map[bool]string{true: ", --gpu " + strings.Join(f.o.GPUs, ","), false: ""}[len(f.o.GPUs) > 0])
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
		if err := f.ready(ctx, w.Key, vouched); err != nil {
			return rented, err
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

// ready makes sure the server takes a key as one of the organisers' rented machines: the
// organisers' key vouches for it if it does not count yet, and its account is made. A dry
// run says what it would do.
func (f *Fleet) ready(ctx context.Context, k *Key, vouched map[string]Vouched) error {
	if vouched[k.Public].Counts {
		if f.o.DryRun {
			return nil
		}
	} else if f.o.DryRun {
		f.say("would vouch for %s with %s's key", k.Handle, f.op.Handle)
		return nil
	} else {
		if err := f.st.vouch(ctx, f.op, k); err != nil {
			return fmt.Errorf("vouching for %s: %w", k.Handle, err)
		}
		f.say("vouched for %s with %s's key", k.Handle, f.op.Handle)
	}
	worker, err := f.st.account(ctx, k)
	for tries := 0; statusOf(err) == 409 && strings.Contains(err.Error(), "taken") && tries < 3; tries++ {
		// Somebody else has the handle, and handles are anybody's to take. The grant is
		// for the key, not the name, so the key takes another name.
		old := k.Handle
		if err := f.rename(k); err != nil {
			return err
		}
		f.say("%s is somebody else's handle on the server: this key is %s now", old, k.Handle)
		worker, err = f.st.account(ctx, k)
	}
	if err != nil {
		return fmt.Errorf("%s's account: %w", k.Handle, err)
	}
	if !worker {
		return fmt.Errorf("the Stet server does not take %s as one of the organisers' rented machines, though %s vouched for it", k.Handle, f.op.Handle)
	}
	return nil
}

// rent tries each kind of GPU in turn, cheapest first, until RunPod gives one. nil, nil:
// none could be had within what may be spent.
func (f *Fleet) rent(ctx context.Context, v *View, k *Key, gpus []Choice, site string) (*Pod, error) {
	tried, dearer := 0, 0
	for _, c := range gpus {
		g, price := c.GPU, c.Price
		if v.Hourly+price > f.o.MaxHourly+1e-9 {
			dearer++
			continue
		}
		tried++
		if f.o.DryRun {
			f.say("would rent %s on %s (%s cloud, about $%.2f an hour)", k.Handle, shortGPU(g.ID), strings.ToLower(c.Cloud), price)
			v.Hourly += price
			return &Pod{ID: "dry-run", Name: k.Handle, Status: "PROVISIONING", Cost: price, CreatedAt: f.now()}, nil
		}
		pod, err := f.rp.create(ctx, f.spec(k, g.ID, c.Cloud, site))
		if errors.Is(err, errNoRoom) {
			continue
		}
		if err != nil {
			return nil, err
		}
		v.Hourly += pod.Cost
		v.Ledger.Pods[pod.ID] = &podRecord{ID: pod.ID, Handle: k.Handle, GPU: shortGPU(g.ID), Cloud: c.Cloud, Cost: pod.Cost, Start: f.now(), Alive: f.now()}
		if err := f.writeLedger(v.Ledger); err != nil {
			return pod, err
		}
		f.say("rented %s: %s, %s cloud, $%.2f an hour (pod %s)", k.Handle, shortGPU(g.ID), strings.ToLower(c.Cloud), pod.Cost, pod.ID)
		return pod, nil
	}
	switch {
	case tried == 0:
		f.say("not renting another machine: at $%.2f an hour or more it would pass --max-hourly $%.2f", v.Hourly+cheapest(gpus), f.o.MaxHourly)
	case dearer > 0:
		f.say("RunPod has none of the %d kinds of GPU tried free for %s just now, and the other %d would pass --max-hourly $%.2f; trying again later", tried, k.Handle, dearer, f.o.MaxHourly)
	default:
		f.say("RunPod has none of the %d kinds of GPU tried free for %s just now; trying again later", tried, k.Handle)
	}
	return nil, nil
}

func cheapest(gpus []Choice) float64 {
	low := math.Inf(1)
	for _, c := range gpus {
		low = min(low, c.Price)
	}
	return low
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

// release hands back the jobs of machines that are gone. It is said, not stopped for, when it
// cannot be done: the job's lease runs out in the end anyway, and the next look tries again.
func (f *Fleet) release(ctx context.Context, gone []*Worker) {
	for _, wk := range gone {
		if f.o.DryRun {
			f.say("would hand back what %s holds: it is not rented any more", wk.Key.Handle)
			continue
		}
		jobs, students, err := f.st.release(ctx, f.op, wk.Key.Public)
		switch {
		case err != nil:
			f.say("could not hand back what %s holds, though it is not rented any more: %v", wk.Key.Handle, err)
		case jobs+students > 0:
			f.say("handed back what %s held when it was ended: %s", wk.Key.Handle, held(jobs, students))
		}
	}
}

func held(jobs, students int) string {
	var parts []string
	switch jobs {
	case 0:
	case 1:
		parts = append(parts, "1 job")
	default:
		parts = append(parts, fmt.Sprintf("%d jobs", jobs))
	}
	switch students {
	case 0:
	case 1:
		parts = append(parts, "1 of the organisers' students")
	default:
		parts = append(parts, fmt.Sprintf("%d of the organisers' students", students))
	}
	return strings.Join(parts, " and ")
}

func (f *Fleet) say(format string, args ...any) {
	fmt.Fprintf(f.out, "%s  %s\n", f.now().UTC().Format("15:04:05"), fmt.Sprintf(format, args...))
}
