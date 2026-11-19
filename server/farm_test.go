package main

import (
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
)

// The organisers' own students are farmed out: the organisers' own machine posts a settled
// round's batch, any of the organisers' machines (a rented one too) is handed a student to
// train, one at a time, and its scores are kept for the organisers to add up. Nobody else is
// handed one or may post one, and a task left by a machine goes to another.
func TestTheOrganisersStudentsAreTrainedByTheirMachines(t *testing.T) {
	pubA, privA := newKey()
	pubW, privW := newKey()
	w := newWorld(t, pubA)
	anchor := w.join("anchor-one", privA)
	if code := w.vouch(anchor, pubW); code != 201 {
		t.Fatalf("vouching for a rented machine: HTTP %d", code)
	}
	rented := w.join("rented-one", privW)
	alice := w.join("alice", nil)
	a1 := w.submit(alice, "sieve", "print(1)")
	_ = a1
	w.play(anchor, 1, func(name string, round int) float64 { return 5.3 })
	idx := 0
	w.app.st.view(func(s *State) {
		for i, rd := range s.Rounds {
			if rd.Status == "settled" && i > idx {
				idx = i
			}
		}
	})
	digest := strings.Repeat("ab", 32)
	batch := map[string]any{"round": idx, "scale": "micro", "items_digest": digest, "items": 3, "generator_version": 3,
		"scripts": []string{"unedited", strings.Repeat("cd", 32)}, "items_file": base64.StdEncoding.EncodeToString([]byte("zstd bytes"))}
	if code := rented.do("POST", "/api/v1/admin/farm/batches", batch, nil); code != 403 {
		t.Fatalf("a rented machine posting a batch: HTTP %d", code)
	}
	var made struct {
		Batch string `json:"batch"`
		Tasks int    `json:"tasks"`
	}
	if code := anchor.do("POST", "/api/v1/admin/farm/batches", batch, &made); code != 200 || made.Tasks != 2 {
		t.Fatalf("the organisers' batch: HTTP %d %+v", code, made)
	}
	var work Work
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	if work.FarmWaiting != 2 {
		t.Fatalf("stet-runpod is told of %d students waiting, not 2", work.FarmWaiting)
	}

	type claimed struct {
		Task *struct {
			Batch  string `json:"batch"`
			Index  int    `json:"index"`
			Round  string `json:"round"`
			Script string `json:"script"`
			Items  int    `json:"items"`
		} `json:"task"`
	}
	var c claimed
	if code := alice.do("POST", "/api/v1/farm/claim", map[string]any{}, &c); code != 403 {
		t.Fatalf("a volunteer asking for a student: HTTP %d", code)
	}
	if code := rented.do("POST", "/api/v1/farm/claim", map[string]any{}, &c); code != 200 || c.Task == nil || c.Task.Script != "unedited" || c.Task.Items != 3 {
		t.Fatalf("the rented machine is handed the unedited corpus first: HTTP %d %+v", code, c.Task)
	}
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	for _, m := range work.Machines {
		if m.Handle == "rented-one" && m.Busy != "farm" {
			t.Fatalf("a machine training a student is not shown at work: %+v", m)
		}
	}
	var again claimed
	rented.do("POST", "/api/v1/farm/claim", map[string]any{}, &again)
	if again.Task == nil || again.Task.Index != c.Task.Index {
		t.Fatal("a machine asking again is handed the task it holds")
	}
	if code := rented.do("GET", "/api/v1/farm/batches/"+c.Task.Batch+"/items", nil, nil); code != 200 {
		t.Fatalf("the batch's items: HTTP %d", code)
	}
	task := fmt.Sprintf("/api/v1/farm/batches/%s/tasks/%d", c.Task.Batch, c.Task.Index)
	good := map[string]any{"valid": true, "kept_doc_frac": 1, "train_seconds": 120, "hardware": "RTX 3080 Ti", "item_scores": []float64{-1.2, -0.8, -2.5}}
	if code := anchor.do("POST", task, good, nil); code != 409 {
		t.Fatalf("another machine posting a task it does not hold: HTTP %d", code)
	}
	short := map[string]any{"valid": true, "item_scores": []float64{-1.2}}
	if code := rented.do("POST", task, short, nil); code != 400 {
		t.Fatalf("a result with too few scores: HTTP %d", code)
	}
	if code := rented.do("POST", task, good, nil); code != 200 {
		t.Fatalf("the rented machine's result: HTTP %d", code)
	}
	if code := rented.do("POST", task, good, nil); code != 409 {
		t.Fatalf("a result for a task already done: HTTP %d", code)
	}

	// The second task: the organisers' own machine takes it, hands it back, and it waits again.
	var c2 claimed
	anchor.do("POST", "/api/v1/farm/claim", map[string]any{}, &c2)
	if c2.Task == nil || c2.Task.Script != strings.Repeat("cd", 32) {
		t.Fatalf("the second student: %+v", c2.Task)
	}
	if code := anchor.do("POST", fmt.Sprintf("/api/v1/farm/batches/%s/tasks/%d/release", c2.Task.Batch, c2.Task.Index), map[string]any{}, nil); code != 200 {
		t.Fatalf("handing a task back: HTTP %d", code)
	}
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	if work.FarmWaiting != 1 || work.FarmUnderWay != 0 {
		t.Fatalf("after one done and one handed back: %d waiting, %d under way", work.FarmWaiting, work.FarmUnderWay)
	}

	// Posting the same items again adds only what is new.
	batch["scripts"] = []string{"unedited", strings.Repeat("cd", 32), strings.Repeat("ef", 32)}
	if code := anchor.do("POST", "/api/v1/admin/farm/batches", batch, &made); code != 200 || made.Tasks != 3 {
		t.Fatalf("a batch posted again: HTTP %d %+v", code, made)
	}

	var got struct {
		Tasks []struct {
			Script string    `json:"script"`
			Done   *FarmDone `json:"done"`
			Scores []float64 `json:"item_scores"`
		} `json:"tasks"`
	}
	if code := rented.do("GET", "/api/v1/admin/farm/batches/"+made.Batch, nil, &got); code != 403 {
		t.Fatalf("a rented machine reading the batch: HTTP %d", code)
	}
	if code := anchor.do("GET", "/api/v1/admin/farm/batches/"+made.Batch, nil, &got); code != 200 || len(got.Tasks) != 3 {
		t.Fatalf("the organisers reading the batch: HTTP %d", code)
	}
	if d := got.Tasks[0].Done; d == nil || d.By != "rented-one" || d.Seconds != 120 || len(got.Tasks[0].Scores) != 3 || got.Tasks[0].Scores[2] != -2.5 {
		t.Fatalf("the unedited student, as kept: %+v %v", d, got.Tasks[0].Scores)
	}

	// A bigger student: a scale of its own, held for hours, a repeat, and a script no round
	// holds. It goes only to a machine that asks for long ones.
	big := map[string]any{"round": idx, "scale": "pilot-l", "items_digest": digest, "items": 3, "generator_version": 3,
		"scale_spec": map[string]float64{"n_layer": 10, "d_model": 640, "train_tokens": 6.4e8}, "lease": 4 * 3600,
		"scripts": []string{"unedited", "unedited#2", strings.Repeat("12", 32)}, "items_file": base64.StdEncoding.EncodeToString([]byte("zstd bytes"))}
	var bigMade struct {
		Batch string `json:"batch"`
		Tasks int    `json:"tasks"`
	}
	if code := anchor.do("POST", "/api/v1/admin/farm/batches", big, &bigMade); code != 200 || bigMade.Tasks != 3 || bigMade.Batch == made.Batch {
		t.Fatalf("a batch at a scale of its own: HTTP %d %+v", code, bigMade)
	}
	if code := rented.do("PUT", "/api/v1/admin/farm/scripts/"+strings.Repeat("12", 32), []byte("script bytes"), nil); code != 403 {
		t.Fatalf("a rented machine keeping a script: HTTP %d", code)
	}
	if code := anchor.do("PUT", "/api/v1/admin/farm/scripts/"+strings.Repeat("12", 32), []byte("script bytes"), nil); code != 200 {
		t.Fatalf("the organisers keeping a script: HTTP %d", code)
	}
	if code := rented.do("GET", "/api/v1/farm/scripts/"+strings.Repeat("12", 32), nil, nil); code != 200 {
		t.Fatalf("a rented machine fetching a kept script: HTTP %d", code)
	}
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	if work.FarmLongWaiting != 3 {
		t.Fatalf("long students waiting: %d, not 3", work.FarmLongWaiting)
	}
	// A machine that did not ask for long students finishes the short ones and is then given nothing.
	for range 4 {
		var c claimed
		rented.do("POST", "/api/v1/farm/claim", map[string]any{}, &c)
		if c.Task == nil {
			break
		}
		if c.Task.Batch == bigMade.Batch {
			t.Fatal("a long student was handed to a machine that did not ask for one")
		}
		rented.do("POST", fmt.Sprintf("/api/v1/farm/batches/%s/tasks/%d", c.Task.Batch, c.Task.Index), good, nil)
	}
	var long claimed
	if code := anchor.do("POST", "/api/v1/farm/claim", map[string]any{"max_minutes": 600}, &long); code != 200 || long.Task == nil || long.Task.Batch != bigMade.Batch {
		t.Fatalf("a machine set aside for long students: HTTP %d %+v", code, long.Task)
	}
	// stet-runpod is told which machines train a long one, to count them as rented for those.
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	for _, m := range work.Machines {
		if m.Handle == "anchor-one" && (m.Busy != "farm" || !m.Long) || m.Handle == "rented-one" && m.Long {
			t.Fatalf("which machines train a long student: %+v", work.Machines)
		}
	}

	// Kept across a restart.
	w.check()
	again2 := startWorld(t, w.cfg, w.kv, w.blobs)
	again2.app.st.view(func(s *State) {
		if b := s.Farm[made.Batch]; b == nil || len(b.Tasks) != 3 || b.Tasks[0].Done == nil {
			t.Fatal("the batch was not kept")
		}
	})
}

// A student a rented machine was training goes back to be handed to another when the
// organisers end the machine, without waiting out its lease.
func TestAStudentAnEndedMachineWasTrainingIsHandedBack(t *testing.T) {
	pubA, privA := newKey()
	pubW, privW := newKey()
	w := newWorld(t, pubA)
	anchor := w.join("anchor-one", privA)
	if code := w.vouch(anchor, pubW); code != 201 {
		t.Fatalf("vouching for a rented machine: HTTP %d", code)
	}
	rented := w.join("rented-one", privW)
	alice := w.join("alice", nil)
	w.submit(alice, "sieve", "print(1)")
	w.play(anchor, 1, func(name string, round int) float64 { return 5.3 })
	idx := 0
	w.app.st.view(func(s *State) {
		for i, rd := range s.Rounds {
			if rd.Status == "settled" && i > idx {
				idx = i
			}
		}
	})
	batch := map[string]any{"round": idx, "scale": "micro", "items_digest": strings.Repeat("ab", 32), "items": 3, "generator_version": 3,
		"scripts": []string{"unedited"}, "items_file": base64.StdEncoding.EncodeToString([]byte("zstd bytes"))}
	if code := anchor.do("POST", "/api/v1/admin/farm/batches", batch, nil); code != 200 {
		t.Fatalf("the organisers' batch: HTTP %d", code)
	}
	var c struct {
		Task *struct {
			Batch string `json:"batch"`
			Index int    `json:"index"`
		} `json:"task"`
	}
	if code := rented.do("POST", "/api/v1/farm/claim", map[string]any{}, &c); code != 200 || c.Task == nil {
		t.Fatalf("the rented machine is handed a student: HTTP %d", code)
	}
	var out struct {
		Jobs     int `json:"jobs"`
		Students int `json:"students"`
	}
	if code := anchor.do("POST", "/api/v1/admin/workers/"+rented.pub+"/release", nil, &out); code != 200 || out.Jobs != 0 || out.Students != 1 {
		t.Fatalf("handing back the ended machine's student: HTTP %d %+v", code, out)
	}
	var work Work
	anchor.do("GET", "/api/v1/admin/work", nil, &work)
	if work.FarmWaiting != 1 || work.FarmUnderWay != 0 {
		t.Fatalf("the student is not waiting again: %d waiting, %d under way", work.FarmWaiting, work.FarmUnderWay)
	}
	task := fmt.Sprintf("/api/v1/farm/batches/%s/tasks/%d", c.Task.Batch, c.Task.Index)
	if code := rented.do("POST", task, map[string]any{"valid": true, "item_scores": []float64{-1, -1, -1}}, nil); code != 409 {
		t.Fatalf("a student from the ended machine was taken: HTTP %d", code)
	}
	w.check()
}
