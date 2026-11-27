package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptrace"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// --- RunPod's API (REST v2, https://docs.runpod.io/api-reference-v2) ---------------------------

const runpodAPI = "https://api.runpod.io/v2"

type RunPod struct {
	base  string
	key   string
	http  *http.Client
	sleep func(context.Context, time.Duration) error
}

func newRunPod(base, key string) *RunPod {
	return &RunPod{base: strings.TrimRight(base, "/"), key: key, http: &http.Client{Timeout: 60 * time.Second, Transport: timedFor("runpod")}, sleep: sleepCtx}
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// RunPodError is RunPod saying no.
type RunPodError struct {
	Status int
	Detail string
}

func (e *RunPodError) Error() string { return fmt.Sprintf("RunPod said %d: %s", e.Status, e.Detail) }

// errNoRoom is RunPod having no machine of a kind to give: try another kind.
var errNoRoom = errors.New("no machine of that kind is free")

// Pod is a pod as RunPod describes it. Its env is not read: it holds the machine's key.
type Pod struct {
	ID     string  `json:"id"`
	Name   string  `json:"name"`
	Status string  `json:"status"` // PROVISIONING STARTING RUNNING EXITED ERROR TERMINATED
	Cost   float64 `json:"cost"`   // dollars an hour
	GPU    *struct {
		ID     string `json:"id"`
		Count  int    `json:"count"`
		Memory int    `json:"memory"`
	} `json:"gpu"`
	DataCenterID string    `json:"dataCenterId"`
	CreatedAt    time.Time `json:"createdAt"`
	// Runtime is there once its container has started: how long it has been up, seconds.
	Runtime *struct {
		Uptime int `json:"uptime"`
	} `json:"runtime"`
	SSH *struct {
		Direct *struct {
			Command string `json:"command"`
		} `json:"direct"`
		Proxy *struct {
			Command string `json:"command"`
		} `json:"proxy"`
	} `json:"ssh"`
}

func (p *Pod) gpuName() string {
	if p.GPU == nil {
		return "?"
	}
	return shortGPU(p.GPU.ID)
}

// live is a pod that is, or is about to be, billed for its GPU.
func (p *Pod) live() bool {
	return p.Status == "PROVISIONING" || p.Status == "STARTING" || p.Status == "RUNNING"
}

// NewPod is the body of POST /pods.
type NewPod struct {
	Name       string            `json:"name"`
	Image      string            `json:"image"`
	Cloud      string            `json:"cloud"`
	GPU        NewPodGPU         `json:"gpu"`
	Disk       int               `json:"disk"`
	Env        map[string]string `json:"env"`
	Ports      []string          `json:"ports,omitempty"`
	Entrypoint []string          `json:"entrypoint"`
	Cmd        []string          `json:"cmd"`
}

type NewPodGPU struct {
	ID             string `json:"id"`
	Count          int    `json:"count"`
	MinRAMPerGPU   int    `json:"minRamPerGpu,omitempty"`
	MinVCPUPerGPU  int    `json:"minVcpuCountPerGpu,omitempty"`
	MinCUDAVersion string `json:"minCudaVersion,omitempty"`
}

// GPUType is one kind of GPU in RunPod's catalog.
type GPUType struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Manufacturer string `json:"manufacturer"`
	Memory       int    `json:"memory"`
	Secure       bool   `json:"secure"`
	Community    bool   `json:"community"`
	Price        struct {
		Secure    float64 `json:"secure"`
		Community float64 `json:"community"`
	} `json:"price"`
	Availability string `json:"availability"` // NONE LOW MEDIUM HIGH, for the cloud asked about
}

func (g *GPUType) price(cloud string) float64 {
	if cloud == "COMMUNITY" {
		return g.Price.Community
	}
	return g.Price.Secure
}

// do makes one call. A 429 is waited out as RunPod asks; a read that fails on the way is
// asked again. A create is not, here: see create.
func (r *RunPod) do(ctx context.Context, method, path string, body any, into any) (int, error) {
	var raw []byte
	if body != nil {
		var err error
		if raw, err = json.Marshal(body); err != nil {
			return 0, err
		}
	}
	for attempt := 0; ; attempt++ {
		var gotConn, wroteAny, tlsStarted, tlsDone atomic.Bool
		trace := &httptrace.ClientTrace{
			GotConn:           func(httptrace.GotConnInfo) { gotConn.Store(true) },
			WroteHeaderField:  func(string, []string) { wroteAny.Store(true) },
			WroteHeaders:      func() { wroteAny.Store(true) },
			WroteRequest:      func(httptrace.WroteRequestInfo) { wroteAny.Store(true) },
			TLSHandshakeStart: func() { tlsStarted.Store(true) },
			TLSHandshakeDone: func(_ tls.ConnectionState, err error) {
				if err == nil {
					tlsDone.Store(true)
				}
			},
		}
		req, err := http.NewRequestWithContext(httptrace.WithClientTrace(ctx, trace), method, r.base+path, bytes.NewReader(raw))
		if err != nil {
			return 0, err
		}
		req.Header.Set("Authorization", "Bearer "+r.key)
		req.Header.Set("Accept", "application/json")
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := r.http.Do(req)
		if err != nil {
			if method == "GET" && attempt < 3 && ctx.Err() == nil {
				if err := r.sleep(ctx, time.Duration(attempt+1)*2*time.Second); err != nil {
					return 0, err
				}
				continue
			}
			if isProvenPreSendError(err, gotConn.Load(), wroteAny.Load(), tlsStarted.Load(), tlsDone.Load()) {
				return 0, fmt.Errorf("%w: RunPod could not be reached: %w", errRequestNotSent, err)
			}
			return 0, fmt.Errorf("RunPod could not be reached: %w", err)
		}
		data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
		resp.Body.Close()
		if err != nil {
			return resp.StatusCode, err
		}
		switch {
		case resp.StatusCode == 429 && attempt < 6:
			wait, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
			if err := r.sleep(ctx, time.Duration(max(wait, 2))*time.Second); err != nil {
				return 0, err
			}
			continue
		case resp.StatusCode >= 500 && method == "GET" && attempt < 3:
			if err := r.sleep(ctx, time.Duration(attempt+1)*3*time.Second); err != nil {
				return 0, err
			}
			continue
		case resp.StatusCode >= 300:
			var e struct {
				Detail string `json:"detail"`
				Title  string `json:"title"`
			}
			json.Unmarshal(data, &e)
			if e.Detail == "" {
				e.Detail = e.Title
			}
			if e.Detail == "" {
				e.Detail = strings.TrimSpace(string(data))
			}
			return resp.StatusCode, &RunPodError{Status: resp.StatusCode, Detail: e.Detail}
		}
		if into != nil && len(data) > 0 {
			if err := json.Unmarshal(data, into); err != nil {
				return resp.StatusCode, fmt.Errorf("RunPod's answer to %s %s: %w", method, path, err)
			}
		}
		return resp.StatusCode, nil
	}
}

// pods lists every pod on the account.
func (r *RunPod) pods(ctx context.Context) ([]Pod, error) {
	var all []Pod
	cursor := ""
	for {
		q := url.Values{"limit": {"1000"}}
		if cursor != "" {
			q.Set("cursor", cursor)
		}
		var page struct {
			Pods       []Pod `json:"pods"`
			Pagination struct {
				NextCursor  string `json:"nextCursor"`
				HasNextPage bool   `json:"hasNextPage"`
			} `json:"pagination"`
		}
		if _, err := r.do(ctx, "GET", "/pods?"+q.Encode(), nil, &page); err != nil {
			return nil, err
		}
		all = append(all, page.Pods...)
		if !page.Pagination.HasNextPage || page.Pagination.NextCursor == "" {
			return all, nil
		}
		cursor = page.Pagination.NextCursor
	}
}

// create rents a pod. errNoRoom means RunPod has no machine of that kind free now. A
// create that may or may not have gone through (the answer was lost or lacked an ID) is looked
// for by its name before anything is said, skipping any pre-existing knownIDs.
func (r *RunPod) create(ctx context.Context, spec NewPod, knownIDs map[string]bool) (*Pod, error) {
	var pod Pod
	status, err := r.do(ctx, "POST", "/pods", spec, &pod)
	findCreated := func() *Pod {
		pods, lerr := r.pods(ctx)
		if lerr != nil {
			return nil
		}
		for i := range pods {
			if pods[i].Name == spec.Name && pods[i].live() && !knownIDs[pods[i].ID] {
				return &pods[i]
			}
		}
		return nil
	}
	switch {
	case err == nil && (status == 201 || status == 200) && pod.ID != "":
		return &pod, nil
	case err == nil:
		if found := findCreated(); found != nil {
			return found, nil
		}
		return nil, fmt.Errorf("RunPod answered %d to a new pod without a pod ID", status)
	case errors.Is(err, errRequestNotSent):
		return nil, fmt.Errorf("%w: %w", errRentNotSent, err)
	case status == 400 || status == 403:
		// 400: no machine fits; 403: this kind is not open to the account. Either way the
		// next kind may do.
		return nil, fmt.Errorf("%w (%v)", errNoRoom, err)
	case status >= 400 && status < 500 && status != http.StatusRequestTimeout:
		return nil, fmt.Errorf("%w (%v)", errRentRejected, err)
	case status == 0 || status == http.StatusRequestTimeout || status >= 500:
		if found := findCreated(); found != nil {
			return found, nil
		}
	}
	return nil, err
}

// terminate ends a pod for good. One that is already gone is not an error.
func (r *RunPod) terminate(ctx context.Context, id string) error {
	status, err := r.do(ctx, "DELETE", "/pods/"+url.PathEscape(id), nil, nil)
	if status == 404 {
		return nil
	}
	return err
}

// gpus lists the GPU kinds RunPod rents in a cloud, with what it has free now.
func (r *RunPod) gpus(ctx context.Context, cloud, minCUDA string) ([]GPUType, error) {
	q := url.Values{"include": {"AVAILABILITY"}, "product": {"POD"}, "cloud": {cloud}}
	if minCUDA != "" {
		q.Set("minCudaVersion", minCUDA)
	}
	var out struct {
		GPUs []GPUType `json:"gpus"`
	}
	if _, err := r.do(ctx, "GET", "/catalog/gpus?"+q.Encode(), nil, &out); err != nil {
		return nil, err
	}
	return out.GPUs, nil
}

// shortGPU is a GPU's name without the maker's: "RTX A5000", not "NVIDIA RTX A5000".
func shortGPU(id string) string {
	return strings.TrimPrefix(strings.TrimPrefix(id, "NVIDIA "), "GeForce ")
}

// LogLine is one line a pod's container, or RunPod about it, said.
type LogLine struct {
	Source string `json:"source"` // container or system
	Line   string `json:"line"`
	TS     string `json:"ts"`
}

// logs reads what a pod has said (GET /pods/{id}/logs: server-sent events, one line each).
// Without follow it stops once the stream has been quiet for a moment, having sent what
// there was.
func (r *RunPod) logs(ctx context.Context, id string, follow bool, emit func(LogLine)) error {
	req, err := http.NewRequestWithContext(ctx, "GET", r.base+"/pods/"+url.PathEscape(id)+"/logs", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+r.key)
	req.Header.Set("Accept", "text/event-stream")
	resp, err := (&http.Client{}).Do(req) // no timeout: it streams
	if err != nil {
		return fmt.Errorf("RunPod could not be reached: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &RunPodError{Status: resp.StatusCode, Detail: strings.TrimSpace(string(data))}
	}
	lines, done := make(chan string), make(chan struct{})
	defer close(done)
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(resp.Body)
		sc.Buffer(make([]byte, 64<<10), 1<<20)
		for sc.Scan() {
			select {
			case lines <- sc.Text():
			case <-done:
				return
			}
		}
	}()
	for {
		select {
		case l, ok := <-lines:
			if !ok {
				return nil
			}
			if data, ok := strings.CutPrefix(l, "data: "); ok {
				var e LogLine
				if json.Unmarshal([]byte(data), &e) == nil {
					emit(e)
				}
			}
		case <-time.After(3 * time.Second):
			if !follow {
				return nil
			}
		case <-ctx.Done():
			return nil
		}
	}
}
