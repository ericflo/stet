package main

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// A kind of GPU is known by the longest name its ID holds as whole words.
func TestTheSpeedOfEachKindOfGPU(t *testing.T) {
	for id, want := range map[string]float64{
		"NVIDIA RTX A4000":                             0.64, // not an A40
		"NVIDIA A40":                                   0.95,
		"NVIDIA L40S":                                  1.9, // not an L40
		"NVIDIA L40":                                   1.6,
		"NVIDIA L4":                                    0.45,
		"NVIDIA GeForce RTX 3080 Ti":                   0.95, // not a 3080
		"NVIDIA GeForce RTX 3080":                      0.85,
		"NVIDIA GeForce RTX 4090":                      1.8,
		"NVIDIA A100-SXM4-80GB":                        2,
		"NVIDIA RTX 6000 Ada Generation":               1.8,
		"NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.6,
		"AMD Instinct MI300X OAM":                      0,
		"NVIDIA RTX 4000 SFF Ada Generation":           0,
	} {
		if got := speed(id); got != want {
			t.Errorf("%s: %v, want %v", id, got, want)
		}
	}
}

// For the bigger students, what a student costs comes first, the quicker card where that is
// about the same, and kinds of unknown speed last.
func TestTheBiggerStudentsAreRentedByWhatAStudentCosts(t *testing.T) {
	kind := func(id string, price float64) Choice { return Choice{GPU: GPUType{ID: id}, Price: price} }
	cs := []Choice{ // RunPod's this morning, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX 4000 SFF Ada Generation", 0.20),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA GeForce RTX 3090", 0.50),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69),
		kind("NVIDIA L40", 0.82), kind("NVIDIA GeForce RTX 4090", 0.89),
	}
	byStudent(cs)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.GPU.ID))
	}
	want := "RTX 3080 Ti, RTX 4090, L40, RTX 3090, RTX 4000 Ada Generation, A40, RTX A6000, L4, RTX 4000 SFF Ada Generation"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}

// By default the fastest card comes first, the cheaper first where two are as fast, and a
// kind of unknown speed last.
func TestMachinesAreRentedFastestFirst(t *testing.T) {
	kind := func(id string, price float64) Choice { return Choice{GPU: GPUType{ID: id}, Price: price} }
	cs := []Choice{ // RunPod's on Nov 20, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX A4000", 0.25), kind("NVIDIA RTX A4500", 0.25),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA Mystery 1", 0.30), kind("NVIDIA RTX PRO 4000 Blackwell", 0.57),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69), kind("NVIDIA GeForce RTX 4090", 0.89),
		kind("NVIDIA L40S", 1.09), kind("NVIDIA RTX 6000 Ada Generation", 1.09),
	}
	byPace(cs)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.GPU.ID))
	}
	want := "L40S, RTX 4090, RTX 6000 Ada Generation, RTX A6000, RTX 3080 Ti, A40, RTX PRO 4000 Blackwell, RTX A4500, RTX A4000, RTX 4000 Ada Generation, L4, Mystery 1"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}

// With --cheapest, what a grade costs comes first: its price over its speed, so a quick card
// dearer by the hour can come before a slow cheap one, and a kind of unknown speed counts as
// half a 3090.
func TestMachinesAreRentedByWhatAGradeCosts(t *testing.T) {
	kind := func(id string, price float64) Choice { return Choice{GPU: GPUType{ID: id}, Price: price} }
	cs := []Choice{ // RunPod's on Nov 20, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX A4000", 0.25), kind("NVIDIA RTX A4500", 0.25),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA Mystery 1", 0.30), kind("NVIDIA RTX PRO 4000 Blackwell", 0.57),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69), kind("NVIDIA GeForce RTX 4090", 0.89),
		kind("NVIDIA L40S", 1.09),
	}
	byGrade(cs)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.GPU.ID))
	}
	want := "RTX 3080 Ti, RTX A4500, RTX A4000, RTX 4000 Ada Generation, RTX 4090, L40S, Mystery 1, A40, RTX A6000, RTX PRO 4000 Blackwell, L4"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}

// For a grade taken before an exam, each kind is rented just in time to have trained by
// then: none before its time, the cheapest a round among those due, and once no kind can be
// on time, the quickest first.
func TestGradersAreRentedJustInTime(t *testing.T) {
	kind := func(id string, price float64) Choice { return Choice{GPU: GPUType{ID: id}, Price: price} }
	cs := []Choice{
		kind("NVIDIA RTX A4000", 0.17), kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA GeForce RTX 3090", 0.22),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA GeForce RTX 5080", 0.39), kind("NVIDIA GeForce RTX 3090", 0.50),
		kind("NVIDIA GeForce RTX 4090", 0.89), kind("NVIDIA L40S", 1.09), kind("NVIDIA L4", 0.69),
	}
	// Leads, with 20 minutes to boot and 36 for a 3090's first grade: L4 100 minutes, the
	// 4000 Ada 80, the A4000 76, the 3080 Ti 58, the 3090s 56, the 5080 46, the 4090 40, the
	// L40S 39.
	at := func(left time.Duration) string {
		var got []string
		for _, c := range forExam(cs, left, 20*time.Minute, 36*time.Minute, time.Minute) {
			got = append(got, fmt.Sprintf("%s $%.2f", shortGPU(c.GPU.ID), c.Price))
		}
		return strings.Join(got, ", ")
	}
	for _, c := range []struct {
		left time.Duration
		want string
	}{
		{90 * time.Minute, ""},
		{80*time.Minute + 30*time.Second, "RTX 4000 Ada Generation $0.28"},
		{77 * time.Minute, "RTX A4000 $0.17, RTX 4000 Ada Generation $0.28"},
		{70 * time.Minute, ""},
		{57 * time.Minute, "RTX 3080 Ti $0.18, RTX 3090 $0.22, RTX 3090 $0.50"},
		{41 * time.Minute, "RTX 5080 $0.39, RTX 4090 $0.89"}, // the L40S not yet
		{39 * time.Minute, "RTX 4090 $0.89, L40S $1.09"},
		{30 * time.Minute, "L40S $1.09, RTX 4090 $0.89, RTX 5080 $0.39, RTX 3090 $0.22, RTX 3090 $0.50, RTX 3080 Ti $0.18, RTX A4000 $0.17, RTX 4000 Ada Generation $0.28, L4 $0.69"},
	} {
		if got := at(c.left); got != c.want {
			t.Errorf("%s before the exam: %q, want %q", c.left, got, c.want)
		}
	}
}
