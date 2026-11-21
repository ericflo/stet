package main

import (
	"encoding/base64"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// worker runs the pod's script as RunPod would, with curl, stet and sleep replaced by
// stand-ins that write down what they were asked. Nothing reaches the network.
type workerRun struct {
	dir, home, bin string
	env            []string
}

func newWorkerRun(t *testing.T, stet string) *workerRun {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("no bash")
	}
	dir := t.TempDir()
	w := &workerRun{dir: dir, home: filepath.Join(dir, "root"), bin: filepath.Join(dir, "bin")}
	os.MkdirAll(w.home, 0o755)
	os.MkdirAll(w.bin, 0o755)
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(w.bin, name), []byte("#!/bin/bash\n"+body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// RunPod: a DELETE of this pod with this pod's key is taken.
	write("curl", `echo "$*" >> "$LOG/curl"
case "$*" in
  *"-X DELETE"*"Bearer pod-key"*"/v2/pods/pod-123"*) printf 204 ;;
  *"-X DELETE"*"Bearer container-key"*"console.vast.ai/api/v0/instances/77/"*) printf 200 ;;
  *"169.254.169.254/v1/reallocate"*) echo reallocated ;;
  *install.sh*) echo 'mkdir -p "$UV_TOOL_BIN_DIR" && cp "$STAND_IN" "$UV_TOOL_BIN_DIR/stet"' ;;
  *) printf 401 ;;
esac
`)
	write("sleep", `echo "$*" >> "$LOG/sleep"`)
	write("stet-stand-in", stet)
	k, _ := newKey("stet-rp-01")
	w.env = []string{
		"PATH=" + w.bin + ":/usr/bin:/bin", "LOG=" + dir, "STAND_IN=" + filepath.Join(w.bin, "stet-stand-in"),
		"STET_WORKER_HOME=" + w.home, "STET_WORKER_KEY=" + base64.StdEncoding.EncodeToString(k.file()),
		"STET_API=https://api.stet.test", "STET_INSTALL=https://www.stet.test/install.sh", "STET_IDLE_MINUTES=7",
		"RUNPOD_POD_ID=pod-123", "RUNPOD_API_KEY=pod-key", "STET_WORKER_OPT=" + filepath.Join(dir, "opt"),
	}
	return w
}

func (w *workerRun) cmd() *exec.Cmd {
	c := exec.Command("bash", "-c", workerScript)
	c.Env, c.Dir = w.env, w.dir
	return c
}

func (w *workerRun) log(name string) string {
	raw, _ := os.ReadFile(filepath.Join(w.dir, name))
	return string(raw)
}

func TestTheWorkerGradesUntilIdleAndEndsItsOwnPod(t *testing.T) {
	// The stand-in for stet: what it was run with, whether it can read its key, and
	// whether the key was left in its environment.
	w := newWorkerRun(t, `case "$1" in
  --version) echo "stet 0.0.0" ;;
  verify) echo "$* key=$(stat -c %a "$HOME/.config/stet/key.json") env=${STET_WORKER_KEY:-none}" >> "$LOG/stet" ;;
esac
`)
	out, err := w.cmd().CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	// It installed stet (it had none), ran it to stop when idle, and ended its own pod.
	if !strings.Contains(w.log("curl"), "https://www.stet.test/install.sh") {
		t.Fatalf("stet was not installed from the site:\n%s\n%s", w.log("curl"), out)
	}
	if got := strings.TrimSpace(w.log("stet")); got != "verify --exit-when-idle 7 key=600 env=none" {
		t.Fatalf("stet was run as %q\n%s", got, out)
	}
	if !strings.Contains(w.log("curl"), "-X DELETE -H Authorization: Bearer pod-key https://api.runpod.io/v2/pods/pod-123") {
		t.Fatalf("the pod did not end itself:\n%s", w.log("curl"))
	}
	if !strings.Contains(w.log("sleep"), "infinity") || !strings.Contains(string(out), "this pod is ending") {
		t.Fatalf("it should wait for the end it asked for:\n%s", out)
	}
	key, _ := os.ReadFile(filepath.Join(w.home, ".config", "stet", "key.json"))
	if !strings.Contains(string(key), `"handle":"stet-rp-01"`) {
		t.Fatalf("the key file: %s", key)
	}

	// RunPod starts the container again: it ends the pod again, and grades nothing.
	os.Remove(filepath.Join(w.dir, "stet"))
	if out, err := w.cmd().CombinedOutput(); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if w.log("stet") != "" || strings.Count(w.log("curl"), "-X DELETE") != 2 {
		t.Fatalf("started again, it ran stet (%q) or did not end the pod:\n%s", w.log("stet"), w.log("curl"))
	}
}

func TestAWorkerWithNoKeyGradesNothing(t *testing.T) {
	w := newWorkerRun(t, `echo "$*" >> "$LOG/stet"`)
	for i, e := range w.env {
		if strings.HasPrefix(e, "STET_WORKER_KEY=") {
			w.env[i] = "STET_WORKER_KEY="
		}
	}
	out, err := w.cmd().CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if w.log("stet") != "" || !strings.Contains(string(out), "not given a key") || !strings.Contains(w.log("curl"), "-X DELETE") {
		t.Fatalf("with no key:\n%s", out)
	}
}

// Ended from outside, the pod's process is told to stop, and passes that on to stet, which
// hands back the job it holds.
func TestAWorkerToldToStopTellsStet(t *testing.T) {
	w := newWorkerRun(t, `case "$1" in
  --version) echo "stet 0.0.0" ;;
  verify) trap 'echo "handed the job back" >> "$LOG/stet"; exit 0' TERM; echo started >> "$LOG/stet"; while :; do /bin/sleep 0.05; done ;;
esac
`)
	c := w.cmd()
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for !strings.Contains(w.log("stet"), "started") {
		if time.Now().After(deadline) {
			c.Process.Kill()
			t.Fatal("stet never started")
		}
		time.Sleep(20 * time.Millisecond)
	}
	c.Process.Signal(syscall.SIGTERM)
	done := make(chan error, 1)
	go func() { done <- c.Wait() }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		c.Process.Kill()
		t.Fatal("the worker did not stop when told to")
	}
	if !strings.Contains(w.log("stet"), "handed the job back") {
		t.Fatalf("stet was not told to stop: %q", w.log("stet"))
	}
	if _, err := os.Stat(filepath.Join(w.home, ".stet-worker-done")); err != nil {
		t.Fatal("a worker that was stopped would grade again if its container were started again")
	}
}

// A machine that runs curators runs them in the container sandbox, with stet installed
// where the curator's user can read it; where it cannot run them, it grades.
func TestAWorkerRunsCuratorsAndFallsBackToGrading(t *testing.T) {
	w := newWorkerRun(t, `case "$1" in
  --version) echo "stet 0.0.0" ;;
  verify) echo "$* tools=$UV_TOOL_DIR python=$UV_PYTHON_INSTALL_DIR tmp=$TMPDIR" >> "$LOG/stet"; case "$*" in *--edits*) exit 3 ;; esac ;;
esac
`)
	opt := filepath.Join(w.dir, "opt")
	w.env = append(w.env, "STET_WORKER_EDITS=1")
	out, err := w.cmd().CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	runs := strings.Split(strings.TrimSpace(w.log("stet")), "\n")
	want := []string{
		"verify --edits --sandbox container --exit-when-idle 7 tools=" + opt + "/tools python=" + opt + "/python tmp=" + w.home + "/.cache/stet-tmp",
		"verify --exit-when-idle 7 tools=" + opt + "/tools python=" + opt + "/python tmp=" + w.home + "/.cache/stet-tmp",
	}
	if len(runs) != 2 || runs[0] != want[0] || runs[1] != want[1] {
		t.Fatalf("stet was run as\n%s\nwant\n%s\n%s", strings.Join(runs, "\n"), strings.Join(want, "\n"), out)
	}
	if !strings.Contains(string(out), "grading only") || !strings.Contains(w.log("curl"), "-X DELETE") {
		t.Fatalf("%s", out)
	}
}

// On Vast it ends itself with the key Vast gives the instance; on a market that gives none, it
// waits for stet-cloud to end it, and asks nothing of any market.
func TestTheWorkerEndsItselfWhereItsMarketLetsIt(t *testing.T) {
	stand := `case "$1" in --version) echo "stet 0.0.0" ;; verify) echo "$*" >> "$LOG/stet" ;; esac
`
	w := newWorkerRun(t, stand)
	w.env = append(w.env, "STET_MARKET=vast", "CONTAINER_ID=77", "CONTAINER_API_KEY=container-key")
	out, err := w.cmd().CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if !strings.Contains(w.log("curl"), "-X DELETE -H Authorization: Bearer container-key https://console.vast.ai/api/v0/instances/77/") ||
		!strings.Contains(string(out), "Vast has it; this instance is ending") || strings.Contains(w.log("curl"), "runpod.io") {
		t.Fatalf("on Vast:\n%s\n%s", w.log("curl"), out)
	}
	w = newWorkerRun(t, stand)
	w.env = append(w.env, "STET_MARKET=lium")
	out, err = w.cmd().CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if strings.Contains(w.log("curl"), "DELETE") || !strings.Contains(string(out), "lium gives it no key to end itself with") || !strings.Contains(w.log("sleep"), "infinity") {
		t.Fatalf("on Lium:\n%s\n%s", w.log("curl"), out)
	}
}

// On Salad, a host whose driver is older than 580 is asked to be moved, and grades nothing.
func TestOnSaladAnOldDriverAsksForAnotherMachine(t *testing.T) {
	w := newWorkerRun(t, `case "$1" in --version) echo "stet 0.0.0" ;; verify) echo "$*" >> "$LOG/stet" ;; esac
`)
	if err := os.WriteFile(filepath.Join(w.bin, "nvidia-smi"), []byte("#!/bin/bash\necho 570.86.10\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	w.env = append(w.env, "STET_MARKET=salad")
	out, err := w.cmd().CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if !strings.Contains(w.log("curl"), "169.254.169.254/v1/reallocate") || w.log("stet") != "" || !strings.Contains(string(out), "driver 570.86.10 is older than 580") {
		t.Fatalf("an old driver on Salad:\n%s\n%s", w.log("curl"), out)
	}
}
